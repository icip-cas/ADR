import os
import json
import uuid
import random
import resource
import openai
import argparse
from tqdm import tqdm
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.openai_utils import safe_llm_call
from utils_synthetic.prompts_synthetic_algorithm import (
    pipeline_extract_prompt,
    pipeline_recombine_prompt,
    design_task_prompt,
    generate_solution_test_prompt_v2,
)
from utils_synthetic.exec_utils import timeout
from utils_synthetic.exp_3_generate_solution_test import _parse_response
from utils_synthetic.pipeline_utils import (
    ElementSchema,
    load_schema,
    model_to_nickname,
    is_non_empty_str,
    is_valid_task_text,
    extract_problem_text,
    run_parallel_processing,
    dump_json,
    build_intermediate_paths,
    normalize_intermediate_paths,
)
from utils.call_sandbox_api import call_sandbox_api


# =========================
# Step 1: Extract elements from seed problems
# =========================
def step_1_extract_elements_from_seed(
    schema: ElementSchema,
    seed_data: List[Dict[str, Any]],
    client: openai.Client,
    model: str,
    max_workers: int,
    jsonl_output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        guideline_text = schema.guidelines
        if not is_non_empty_str(guideline_text):
            raise ValueError("Step 1 invalid input: guidelines is empty.")
        if not isinstance(seed_data, list) or len(seed_data) == 0:
            raise ValueError("Step 1 invalid input: seed_data must be a non-empty list.")
        print(f"[Step 1] start | seed={len(seed_data)} | workers={max_workers}")

        def _one(idx_item: Any) -> Dict[str, Any]:
            i, item = idx_item
            item_id = item.get("id", f"seed_{i:04d}")
            problem = extract_problem_text(item)

            if not problem:
                return {"id": item_id, "seed_item": item, "generated_prompt": None, "extracted_elements": None, "step_1_error": "No problem text found"}

            prompt = pipeline_extract_prompt.format(
                problem=problem,
                guidelines=guideline_text,
                answer_template=schema.answer_template
            )

            max_retries = 3
            last_error: Optional[Exception] = None
            last_answer: Optional[str] = None

            for attempt in range(1, max_retries + 1):
                try:
                    answer = safe_llm_call(client, prompt, model)
                    extracted = schema.extract_elements(answer)
                    if not schema.is_valid_element_dict(extracted):
                        raise ValueError("Extracted elements is empty")

                    return {
                        "id": item_id,
                        "seed_item": item,
                        "generated_prompt": prompt,
                        "extracted_elements": extracted,
                        "raw_response": answer,
                        "step_1_error": None,
                    }
                except Exception as e:
                    last_error = e
                    last_answer = answer if 'answer' in locals() else None
                    if attempt < max_retries:
                        continue

            return {
                "id": item_id,
                "seed_item": item,
                "generated_prompt": prompt,
                "extracted_elements": None,
                "raw_response": last_answer,
                "step_1_error": str(last_error) if last_error else "Unknown extraction error",
            }

        indexed = list(enumerate(seed_data))
        results = run_parallel_processing(
            process_single_item=_one,
            data=indexed,
            max_workers=max_workers,
            tqdm_desc="Step 1",
            jsonl_output_path=jsonl_output_path,
        )
        valid_count = sum(1 for x in results if schema.is_valid_element_dict(x.get("extracted_elements")))
        print(f"[Step 1] done | total={len(results)} | valid={valid_count} | invalid={len(results)-valid_count}")
        return results

    except Exception as e:
        print(f"[Step 1] failed with error: {e}")
        return []


# =========================
# Step 2: Recombine elements to synthesize new frameworks
# =========================
def step_2_generate_elements(
    schema: ElementSchema,
    step1_data: List[Dict[str, Any]],
    num_samples: int,
    client: openai.Client,
    model: str,
    max_workers: int,
    jsonl_output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    refs = [
        {"id": x.get("id"), "elements": x.get("extracted_elements")}
        for x in step1_data
        if schema.is_valid_element_dict(x.get("extracted_elements"))
    ]
    if len(refs) < 4:
        raise ValueError("Step 1.1 valid extracted elements must be >= 4 for Step 2 sampling.")
    print(f"[Step 2] start | refs={len(refs)} | traverse_all=True | workers={max_workers}")

    def _one(main_ref: Dict[str, Any]) -> Dict[str, Any]:
        main_id = main_ref.get("id") or f"seed_{uuid.uuid4().hex[:8]}"
        pid = f"syn_{str(main_id)}_{uuid.uuid4().hex[:8]}"
        other_refs = [r for r in refs if r.get("id") != main_id]
        if len(other_refs) < 3:
            return {
                "id": pid,
                "main_element_id": main_id,
                "combination_element_ids": [],
                "framework": None,
                "step_2_error": "Not enough remaining elements to sample 3 combinations.",
            }

        max_retries = 3
        last_error: Optional[Exception] = None
        last_combo_ids: List[Any] = []

        for attempt in range(1, max_retries + 1):
            try:
                combo_refs = random.sample(other_refs, k=3)
                combo_ids = [r.get("id") for r in combo_refs]
                prompt = pipeline_recombine_prompt.format(
                    core_element=schema.core_element,
                    core_element_value=main_ref["elements"].get(schema.core_element, ""),
                    combinations_1=combo_refs[0]["elements"],
                    combinations_2=combo_refs[1]["elements"],
                    combinations_3=combo_refs[2]["elements"],
                    answer_template=schema.answer_template
                )
                answer = safe_llm_call(client, prompt, model)
                framework = schema.extract_elements(answer)
                if not schema.is_valid_element_dict(framework):
                    raise ValueError("Generated framework is empty")

                return {
                    "id": pid,
                    "main_element_id": main_id,
                    "combination_element_ids": combo_ids,
                    "framework": framework,
                    "step_2_error": None,
                }
            except Exception as e:
                last_error = e
                last_combo_ids = combo_ids if 'combo_ids' in locals() else []
                if attempt < max_retries:
                    continue

        return {
            "id": pid,
            "main_element_id": main_id,
            "combination_element_ids": last_combo_ids,
            "framework": None,
            "step_2_error": str(last_error) if last_error else "Unknown generation error",
        }

    results = run_parallel_processing(
        process_single_item=_one,
        data=refs,
        max_workers=max_workers,
        tqdm_desc="Step 2",
        jsonl_output_path=jsonl_output_path,
    )
    valid_count = sum(1 for x in results if schema.is_valid_element_dict(x.get("framework")))
    print(f"[Step 2] done | total={len(results)} | valid={valid_count} | invalid={len(results)-valid_count}")
    return results


# =========================
# Step 3: Generate problem statements from frameworks
# =========================
def step_3_generate_tasks(
    schema: ElementSchema,
    elements_data: List[Dict[str, Any]],
    client: openai.Client,
    model: str,
    max_workers: int,
    jsonl_output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    valid_items = [x for x in elements_data if schema.is_valid_element_dict(x.get("framework"))]
    if not valid_items:
        raise ValueError("Step 3 invalid input: no valid framework items.")
    print(f"[Step 3] start | input={len(valid_items)} | workers={max_workers}")

    def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(item)
        try:
            prompt = design_task_prompt.format(framework=item["framework"])
            answer = safe_llm_call(client, prompt, model)
            out["generated_task"] = answer
            out["step_3_error"] = None
        except Exception as e:
            out["generated_task"] = None
            out["step_3_error"] = str(e)
        return out

    results = run_parallel_processing(
        process_single_item=_one,
        data=valid_items,
        max_workers=max_workers,
        tqdm_desc="Step 3",
        jsonl_output_path=jsonl_output_path,
    )
    valid_count = sum(1 for x in results if is_valid_task_text(x.get("generated_task")))
    print(f"[Step 3] done | total={len(results)} | valid={valid_count} | invalid={len(results)-valid_count}")
    return results


# =========================
# Step 4: Generate solution code and test case generator
# =========================
def step_4_generate_solution_and_tests(
    task_data: List[Dict[str, Any]],
    client: openai.Client,
    model: str,
    max_workers: int,
    jsonl_output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    valid_items = [x for x in task_data if is_valid_task_text(x.get("generated_task"))]
    if not valid_items:
        raise ValueError("Step 4 invalid input: no valid generated_task items.")
    print(f"[Step 4] start | input={len(valid_items)} | workers={max_workers}")

    def _one(item: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(item)
        try:
            prompt = generate_solution_test_prompt_v2.format(problem=item["generated_task"])
            answer = safe_llm_call(client, prompt, model)
            solution, test_case_generator = _parse_response(answer)
            out["solution"] = solution
            out["test_case_generator"] = test_case_generator
            out["step_4_raw_response"] = answer
            out["step_4_error"] = None
        except Exception as e:
            out["solution"] = None
            out["test_case_generator"] = None
            out["step_4_raw_response"] = None
            out["step_4_error"] = str(e)
        return out

    results = run_parallel_processing(
        process_single_item=_one,
        data=valid_items,
        max_workers=max_workers,
        tqdm_desc="Step 4",
        jsonl_output_path=jsonl_output_path,
    )
    valid_count = sum(1 for x in results if is_non_empty_str(x.get("solution")) and is_non_empty_str(x.get("test_case_generator")))
    print(f"[Step 4] done | total={len(results)} | valid={valid_count} | invalid={len(results)-valid_count}")
    return results


# Must be a top-level function (not a closure) for ProcessPoolExecutor pickling.
def _step_5_validate_one(task: Any) -> Any:
    idx, item, exec_timeout_seconds, sandbox_url = task
    out = dict(item)
    # Keep per-process memory bounded to avoid OOM in worker processes.
    limit_bytes = 10 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    solution_code = item.get("solution")
    tcgen_code = item.get("test_case_generator")

    if not solution_code or not tcgen_code:
        out["is_valid"] = False
        out["validation_error"] = "Missing solution or test_case_generator"
        out["test_cases"] = None
        out["validation_sandbox_metadata"] = None
        return idx, out

    try:
        local_scope: Dict[str, Any] = {}
        with timeout(exec_timeout_seconds, "Test case generator execution timed out"):
            exec(tcgen_code, local_scope)
            if "generate_test_cases" not in local_scope:
                raise ValueError("No generate_test_cases function found")
            test_cases = local_scope["generate_test_cases"]()

        if not isinstance(test_cases, dict) or not all(k in test_cases for k in ["input", "output", "fn_name"]):
            raise ValueError("Generated test_cases schema invalid")

        result_status, metadata = call_sandbox_api(code=solution_code, test_cases=test_cases, url=sandbox_url)
        out["is_valid"] = result_status == True
        out["validation_error"] = "" if out["is_valid"] else "Solution failed on generated test cases"
        out["test_cases"] = test_cases
        out["validation_sandbox_metadata"] = metadata

    except Exception as e:
        out["is_valid"] = False
        out["validation_error"] = str(e)
        out["test_cases"] = None
        out["validation_sandbox_metadata"] = None

    return idx, out


# =========================
# Step 5: Validate solutions via sandbox execution
# =========================
def step_5_validate(
    solved_data: List[Dict[str, Any]],
    sandbox_url: str,
    exec_timeout_seconds: int = 60,
    max_workers: int = 8,
    jsonl_output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    # ProcessPoolExecutor is used here (vs ThreadPoolExecutor in other steps)
    # because solution execution is CPU-bound and benefits from true parallelism.
    if not isinstance(solved_data, list) or len(solved_data) == 0:
        raise ValueError("Step 5 invalid input: solved_data must be non-empty.")
    print(f"[Step 5] start | input={len(solved_data)} | workers={max_workers}")
    tasks = [(i, item, exec_timeout_seconds, sandbox_url) for i, item in enumerate(solved_data)]
    processed_count = 0
    indexed_results: List[Any] = []

    if is_non_empty_str(jsonl_output_path):
        os.makedirs(os.path.dirname(jsonl_output_path) or ".", exist_ok=True)
        print(f"Processing and writing to temporary file: {jsonl_output_path}")
        with open(jsonl_output_path, 'w', encoding='utf-8') as f_out:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {executor.submit(_step_5_validate_one, task): task for task in tasks}
                for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Step 5"):
                    task = future_to_task[future]
                    idx, item = task[0], task[1]
                    try:
                        ridx, result_item = future.result()
                    except Exception as e:
                        ridx = idx
                        result_item = dict(item)
                        result_item["is_valid"] = False
                        result_item["validation_error"] = f"Process execution error: {e}"
                        result_item["test_cases"] = None
                        result_item["validation_sandbox_metadata"] = None

                    indexed_results.append((ridx, result_item))
                    f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                    processed_count += 1
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_step_5_validate_one, task): task for task in tasks}
            for future in tqdm(as_completed(future_to_task), total=len(tasks), desc="Step 5"):
                task = future_to_task[future]
                idx, item = task[0], task[1]
                try:
                    ridx, result_item = future.result()
                except Exception as e:
                    ridx = idx
                    result_item = dict(item)
                    result_item["is_valid"] = False
                    result_item["validation_error"] = f"Process execution error: {e}"
                    result_item["test_cases"] = None
                    result_item["validation_sandbox_metadata"] = None

                indexed_results.append((ridx, result_item))
                processed_count += 1

    indexed_results.sort(key=lambda x: x[0])
    results = [x[1] for x in indexed_results]
    print(f"[Step 5] processed={processed_count}")
    valid_count = sum(1 for x in results if x.get("is_valid") is True)
    print(f"[Step 5] done | total={len(results)} | valid={valid_count} | invalid={len(results)-valid_count}")
    return results


def run_synthesis_pipeline(
    schema: ElementSchema,
    seed_data_path: str,
    output_path: str,
    sandbox_url: str,
    num_samples: int = 20,
    model: str = "deepseek-chat",
    max_workers: int = 8,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com/v1",
    intermediate_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing API key. Pass api_key or set OPENAI_API_KEY.")
    if not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")

    nickname = model_to_nickname(model)

    with open(seed_data_path, "r", encoding="utf-8") as f:
        seed_data = json.load(f)
    if isinstance(seed_data, dict):
        seed_data = list(seed_data.values())
    if not isinstance(seed_data, list) or len(seed_data) == 0:
        raise ValueError("seed_data is empty or invalid.")

    output_dir = os.path.dirname(output_path) or "."
    base_name = os.path.splitext(os.path.basename(seed_data_path))[0]
    spec_paths = build_intermediate_paths(output_dir, f"{base_name}_{len(seed_data)}", nickname)
    final_intermediate_paths = normalize_intermediate_paths(intermediate_paths or spec_paths)

    print("[Pipeline] start")
    print(f"[Pipeline] config | model={model} | samples={num_samples} | workers={max_workers} | seed={len(seed_data)}")

    client = openai.Client(base_url=base_url, api_key=key, timeout=60000)

    step1_tmp_jsonl = final_intermediate_paths["step_1"] + ".tmp.jsonl"
    step1 = step_1_extract_elements_from_seed(
        schema, seed_data, client, model, max_workers,
        jsonl_output_path=step1_tmp_jsonl,
    )
    dump_json(final_intermediate_paths["step_1"], step1)
    print(f"[Pipeline] saved Step 1 -> {final_intermediate_paths['step_1']}")

    step1_valid = sum(1 for x in step1 if schema.is_valid_element_dict(x.get("extracted_elements")))
    if step1_valid == 0:
        raise RuntimeError("Pipeline stopped at Step 1: no valid extracted elements.")

    step2_tmp_jsonl = final_intermediate_paths["step_2"] + ".tmp.jsonl"
    step2 = step_2_generate_elements(
        schema, step1, num_samples, client, model, max_workers,
        jsonl_output_path=step2_tmp_jsonl,
    )
    dump_json(final_intermediate_paths["step_2"], step2)
    print(f"[Pipeline] saved Step 2 -> {final_intermediate_paths['step_2']}")

    step2_valid = sum(1 for x in step2 if schema.is_valid_element_dict(x.get("framework")))
    if step2_valid == 0:
        raise RuntimeError("Pipeline stopped at Step 2: no valid synthesized frameworks.")

    step3_tmp_jsonl = final_intermediate_paths["step_3"] + ".tmp.jsonl"
    step3 = step_3_generate_tasks(
        schema, step2, client, model, max_workers,
        jsonl_output_path=step3_tmp_jsonl,
    )
    dump_json(final_intermediate_paths["step_3"], step3)
    print(f"[Pipeline] saved Step 3 -> {final_intermediate_paths['step_3']}")

    step3_valid = sum(1 for x in step3 if is_valid_task_text(x.get("generated_task")))
    if step3_valid == 0:
        raise RuntimeError("Pipeline stopped at Step 3: no valid generated tasks.")

    step4_tmp_jsonl = final_intermediate_paths["step_4"] + ".tmp.jsonl"
    step4 = step_4_generate_solution_and_tests(
        step3, client, model, max_workers,
        jsonl_output_path=step4_tmp_jsonl,
    )
    dump_json(final_intermediate_paths["step_4"], step4)
    print(f"[Pipeline] saved Step 4 -> {final_intermediate_paths['step_4']}")

    step4_valid = sum(1 for x in step4 if is_non_empty_str(x.get("solution")) and is_non_empty_str(x.get("test_case_generator")))
    if step4_valid == 0:
        raise RuntimeError("Pipeline stopped at Step 4: no valid solution/test generator pairs.")

    step5_tmp_jsonl = final_intermediate_paths["step_5_debug"] + ".tmp.jsonl"
    step5 = step_5_validate(
        step4,
        sandbox_url=sandbox_url,
        max_workers=max_workers,
        jsonl_output_path=step5_tmp_jsonl,
    )
    dump_json(final_intermediate_paths["step_5_debug"], step5)
    print(f"[Pipeline] saved Step 5 debug -> {final_intermediate_paths['step_5_debug']}")

    valid_data = [x for x in step5 if x.get("is_valid")]
    if len(valid_data) == 0:
        raise RuntimeError("Pipeline stopped at Step 5: no valid synthesized data after validation.")

    dump_json(final_intermediate_paths["step_5_valid"], valid_data)
    print(f"[Pipeline] saved Step 5 valid -> {final_intermediate_paths['step_5_valid']}")
    print(f"[Pipeline] done | final_valid={len(valid_data)}")

    output = {
        "meta": {
            "model": model,
            "model_nickname": nickname,
            "seed_data_path": seed_data_path,
            "seed_count": len(seed_data),
            "num_samples": num_samples,
            "step_1_1_valid": sum(1 for x in step1 if x.get("extracted_elements") is not None),
            "valid_count": len(valid_data),
            "total_after_step4": len(step4),
            "intermediate_paths": final_intermediate_paths,
        },
        "step_1_results": step1,
        "valid_data": valid_data,
        "all_results": step5,
    }

    dump_json(output_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(description="ADR synthesis pipeline: Atomic Decomposition and Recombination.")

    # Model & API
    model_api = parser.add_argument_group("Model & API")
    model_api.add_argument("--model", type=str, default="deepseek-chat",
                           help="Model identifier for LLM calls (default: deepseek-chat). "
                                "A filename-safe nickname is derived from this for output naming.")
    model_api.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                           help="LLM API key (defaults to OPENAI_API_KEY env var).")
    model_api.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
                           help="LLM API base URL (defaults to OPENAI_BASE_URL env var, else https://api.deepseek.com/v1).")

    # Sandbox
    sandbox = parser.add_argument_group("Sandbox")
    sandbox.add_argument("--sandbox_url", type=str, default=os.getenv("SANDBOX_URL"),
                         help="URL of the code execution sandbox API (defaults to SANDBOX_URL env var).")

    # I/O paths
    io_grp = parser.add_argument_group("I/O paths")
    io_grp.add_argument("--seed_data_path", type=str, required=True,
                        help="Path to the seed data JSON file (list of problem objects).")
    io_grp.add_argument("--output_path", type=str, required=True,
                        help="Path for the final output JSON file.")
    io_grp.add_argument("--iter_num", type=str, required=True,
                        help="Iteration directory name for intermediate outputs.")

    # Schema
    schema_grp = parser.add_argument_group("Schema")
    schema_grp.add_argument("--schema_path", type=str, default="schemas/algorithm.json",
                            help="Path to the element schema JSON file (default: schemas/algorithm.json).")
    schema_grp.add_argument("--core_element", type=str, default=None,
                            help="Name of the element used as the anchor in Step 2 recombination. "
                                 "Defaults to the first element in the schema.")

    # Processing
    processing = parser.add_argument_group("Processing")
    processing.add_argument("--num_samples", type=int, default=1,
                            help="Number of samples to generate per seed item in Step 2 (default: 1).")
    processing.add_argument("--max_workers", type=int, default=8,
                            help="Maximum number of parallel workers (default: 8).")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api_key or set OPENAI_API_KEY env var.")
    if not args.sandbox_url:
        raise ValueError("Missing sandbox URL. Pass --sandbox_url or set SANDBOX_URL env var.")

    # Load element schema (replaces the former module-level globals).
    schema = load_schema(args.schema_path, args.core_element)

    # Derive a filename-safe nickname from the model and lay out intermediate paths.
    nickname = model_to_nickname(args.model)
    seed_base = os.path.splitext(os.path.basename(args.seed_data_path))[0]
    custom_paths = build_intermediate_paths(f"./{args.iter_num}", seed_base, nickname)

    result = run_synthesis_pipeline(
        schema=schema,
        seed_data_path=args.seed_data_path,
        output_path=args.output_path,
        sandbox_url=args.sandbox_url,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        intermediate_paths=custom_paths,
    )
    print(result["meta"])


if __name__ == "__main__":
    main()
