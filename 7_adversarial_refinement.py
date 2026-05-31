import os
import io
import re
import json
import argparse
import contextlib
import concurrent.futures
import openai
from tqdm import tqdm

from utils.openai_utils import get_openai_response, process_response_seperate, process_response, extract_answer_from_thinking
from utils_synthetic.prompts_synthetic_algorithm import near_miss_solutions_prompt, refine_test_case_generator_prompt
from utils_synthetic.exec_utils import timeout, TimeoutException
from utils.call_sandbox_api import call_sandbox_api


def prompt_generate_near_miss_solutions(generated_task, reference_solution, client, model_name):    
    prompt = near_miss_solutions_prompt.format(
        problem=generated_task,
        reference_solution=reference_solution
    )
    
    res = get_openai_response(
        client=client,
        model=model_name,
        user_prompt=prompt,
        max_tokens=8192,
        use_default_generation_config=True,
        stream=False,
    )
    
    output = process_response(res, stream=False)
    response_text = output[0]
    answer_text = extract_answer_from_thinking(response_text)
    
    near_miss_solutions = re.findall(r'```python\n(.*?)```', answer_text, re.DOTALL)
    return near_miss_solutions

def eval_near_miss_rate(near_miss_solutions, test_cases, sandbox_url):
    failures = 0
    wrong_solutions = []
    for miss_solution in near_miss_solutions:
        result_status, metadata = call_sandbox_api(
            code=miss_solution,
            test_cases=test_cases,
            url=sandbox_url,
        )
        
        if result_status == True:
            wrong_solutions.append(miss_solution)
        else:
            failures += 1
        
    return failures / len(near_miss_solutions), wrong_solutions


def prompt_refine_test_case_generator(generated_task, reference_solution, wrong_solutions, test_case_generator, client, model_name):
    near_miss_solutions_str = ''
    for idx, solution in enumerate(wrong_solutions):
        near_miss_solutions_str += f'{idx+1}. ```python\n' + solution + '\n```\n'

    prompt = refine_test_case_generator_prompt.format(
        problem=generated_task,
        reference_solution=reference_solution,
        near_miss_solutions=near_miss_solutions_str,
        test_case_generator=test_case_generator
    )
    
    res = get_openai_response(
        client=client,
        model=model_name,
        user_prompt=prompt,
        max_tokens=8192,
        use_default_generation_config=True,
        stream=False,
    )

    output = process_response(res, stream=False)
    response_text = output[0]
    answer_text = extract_answer_from_thinking(response_text)
    
    refined_test_case_generator = re.findall(r'```python\n(.*?)```', answer_text, re.DOTALL)[0]
    return refined_test_case_generator

def get_refined_test_cases(refined_test_case_generator):
    EXEC_TIMEOUT_SECONDS = 60

    try:
        local_scope = {}
        with timeout(EXEC_TIMEOUT_SECONDS, "Test case generator execution timed out."):
            exec(refined_test_case_generator, local_scope)
            
            test_cases = local_scope['generate_test_cases']()
            return test_cases

    except Exception as e:
        print(f"Error during validation: {e}")
        return None

def process_item(item, client, model_name, sandbox_url):
    log_stream = io.StringIO()
    success = False
    with contextlib.redirect_stdout(log_stream):
        try:
            generated_task = item['generated_task']
            reference_solution = item['solution']
            test_cases = json.loads(item['test_cases']) if isinstance(item['test_cases'], str) else item['test_cases']
            test_case_generator = item['test_case_generator']

            # Step 1.
            print("Step 1: Generating near-miss solutions...")
            near_miss_solutions = prompt_generate_near_miss_solutions(generated_task, reference_solution, client, model_name)
            print(f"-> Generated {len(near_miss_solutions)} near-miss solutions.")
            
            # Step 2.
            print("\nStep 2: Evaluating near-miss solutions with original test cases...")
            near_miss_rate, wrong_solutions = eval_near_miss_rate(near_miss_solutions, test_cases, sandbox_url)
            print(f"-> Near-miss rate on original tests: {near_miss_rate}")
            print(f"-> Found {len(wrong_solutions)} solutions that incorrectly pass original tests.")
            
            # Steps 3 & 4: Refine test case generator and generate new test cases, with retry logic.
            refined_test_cases = None
            max_retries = 3
            for attempt in range(max_retries):
                print(f"\n--- Attempt {attempt + 1}/{max_retries} ---")
                
                # Step 3.
                print("Step 3: Refining test case generator...")
                refined_test_case_generator = prompt_refine_test_case_generator(generated_task, reference_solution, wrong_solutions, test_case_generator, client, model_name)
                print("-> Test case generator refined.")
                
                # Step 4.
                print("Step 4: Generating refined test cases...")
                refined_test_cases = get_refined_test_cases(refined_test_case_generator)
                
                if refined_test_cases is not None:
                    print("-> Successfully generated refined test cases.")
                    success = True
                    break
                else:
                    print("-> Failed to generate refined test cases. Retrying...")
            
            if refined_test_cases is not None:
                # Step 5: Output lengths for comparison
                print(f"\nOriginal test cases length: {len(test_cases['input'])}")
                print(f"Refined test cases length: {len(refined_test_cases['input'])}")
            else:
                print("\n--- Failed to generate refined test cases after all retries. ---")

        except Exception as e:
            print(f"An error occurred: {e}")

        item['log'] = log_stream.getvalue()
        item['refined_test_cases'] = refined_test_cases
        return item, success

def main():
    parser = argparse.ArgumentParser(description="Run test case generator adversarial refinement pipeline.")

    # Model & API
    model_api = parser.add_argument_group("Model & API")
    model_api.add_argument("--model", type=str, default="deepseek-chat",
                           help="Model identifier for LLM calls (default: deepseek-chat).")
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
    io_grp.add_argument("--input_path", type=str, required=True,
                        help="Path to the input JSON file containing valid problems.")
    io_grp.add_argument("--output_path", type=str, required=True,
                        help="Path for the refined output JSON file.")

    # Processing
    processing = parser.add_argument_group("Processing")
    processing.add_argument("--max_workers", type=int, default=4,
                            help="Maximum number of parallel workers (default: 4).")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api_key or set OPENAI_API_KEY env var.")
    if not args.sandbox_url:
        raise ValueError("Missing sandbox URL. Pass --sandbox_url or set SANDBOX_URL env var.")

    client = openai.Client(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=60000,
    )

    with open(args.input_path, 'r') as f:
        data = json.load(f)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        with tqdm(total=len(data), desc="Processing items") as pbar:
            futures = [executor.submit(process_item, item, client, args.model, args.sandbox_url) for item in data]
            for future in concurrent.futures.as_completed(futures):
                item, success = future.result()
                results.append(item)
                if success:
                    pbar.set_postfix_str("Success")
                else:
                    pbar.set_postfix_str("Failed")
                pbar.update(1)

    # Save the updated data to a new file
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Processing complete. Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
