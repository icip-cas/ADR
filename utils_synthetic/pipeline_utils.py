"""Shared helpers for the ADR synthesis pipeline (0_pipeline.py).

Holds the element-schema abstraction plus the generic parsing, validation,
parallel-execution and IO helpers that were previously defined as module-level
functions (and globals) inside 0_pipeline.py.
"""
import os
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


# ---------------------------------------------------------------------------
# Generic predicates / text helpers
# ---------------------------------------------------------------------------
def is_non_empty_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def is_valid_task_text(t: Any) -> bool:
    return is_non_empty_str(t)


def extract_problem_text(item: Dict[str, Any]) -> str:
    """Return the first non-empty problem-statement-like field from an item."""
    for k in ["prompt", "question", "problem", "problem_text", "statement", "generated_task", "text", "content"]:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def extract_json_obj(text: str) -> Dict[str, Any]:
    """Extract a JSON object from text, preferring a ```json fenced block."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not m:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM response.")
    return json.loads(m.group(1))


def extract_json_obj_from_answer_block(text: str, expected_keys: Any) -> Dict[str, Any]:
    """Parse an ``<answer>...</answer>`` block into an element dict.

    Falls back to JSON parsing for backward compatibility, otherwise parses by
    the expected element headings (``Element Name: value``).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty LLM response.")

    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("No <answer>...</answer> block found in LLM response.")

    answer_block = m.group(1).strip()

    # Backward compatibility: accept JSON content if the model still returns JSON.
    try:
        return extract_json_obj(answer_block)
    except Exception:
        pass

    # Parse by expected element headings instead of generic colon splitting.
    keys = [k for k in expected_keys if isinstance(k, str) and k.strip()]
    if not keys:
        raise ValueError("expected_keys is empty or invalid.")

    matches: List[Any] = []
    for key in keys:
        # Allow optional markdown bullets before heading, e.g. `- Computational Environment:`
        pattern = re.compile(rf"(?im)^\s*[-*]?\s*{re.escape(key)}\s*:\s*(.*)$")
        mk = pattern.search(answer_block)
        if mk:
            matches.append((mk.start(), mk.end(), key, mk.group(1).strip()))

    if not matches:
        raise ValueError("No expected element headings found in <answer> block.")

    matches.sort(key=lambda x: x[0])
    parsed: Dict[str, str] = {}
    for i, (start, end, key, inline_value) in enumerate(matches):
        next_start = matches[i + 1][0] if i + 1 < len(matches) else len(answer_block)
        body = answer_block[end:next_start].strip()
        pieces = [p for p in [inline_value, body] if p]
        parsed[key] = "\n".join(pieces).strip()

    if not parsed:
        raise ValueError("No parsable elements found in <answer> block.")

    return parsed


# ---------------------------------------------------------------------------
# Element schema (replaces the former module-level globals)
# ---------------------------------------------------------------------------
@dataclass
class ElementSchema:
    """Holds the active element schema and its derived prompt scaffolding.

    Replaces the module-level globals (elements / guidelines / answer_template /
    EXPECTED_ELEMENT_KEYS / core_element) that 0_pipeline.py used to mutate.
    """
    elements: List[Dict[str, str]]
    guidelines: str
    answer_template: str
    expected_keys: set
    core_element: str

    def is_valid_element_dict(self, d: Any) -> bool:
        if not isinstance(d, dict):
            return False
        return all(is_non_empty_str(d.get(k)) for k in self.expected_keys)

    def extract_elements(self, text: str) -> Dict[str, Any]:
        return extract_json_obj_from_answer_block(text, self.expected_keys)


def load_schema(schema_path: str, core_element_name: Optional[str] = None) -> ElementSchema:
    """Load an element schema JSON file into an ElementSchema.

    The schema file must be a JSON array of objects with at least a "name" key,
    e.g. [{"name": "Core Algorithm Idea", "definition": "..."}].
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        elements = json.load(f)
    if not isinstance(elements, list) or not elements:
        raise ValueError(f"Schema file must contain a non-empty JSON array: {schema_path}")

    guidelines = json.dumps(elements, indent=2, ensure_ascii=False)

    answer_template = "<answer>\n"
    for elem in elements:
        answer_template += f"{elem['name']}:\n"
    answer_template += "</answer>"

    expected_keys = set(elem["name"] for elem in elements)

    # The core element anchors Step 2 recombination; default to the first element.
    if core_element_name and core_element_name in expected_keys:
        core_element = core_element_name
    else:
        core_element = elements[0]["name"]

    return ElementSchema(
        elements=elements,
        guidelines=guidelines,
        answer_template=answer_template,
        expected_keys=expected_keys,
        core_element=core_element,
    )


# ---------------------------------------------------------------------------
# Model nickname derivation (replaces the --model_nickname argument)
# ---------------------------------------------------------------------------
def model_to_nickname(model: str) -> str:
    """Sanitize a model identifier into a filename-safe nickname.

    Replaces whitespace and other path/illegal characters with underscores,
    while preserving word characters, dots and dashes. E.g.::

        "deepseek-chat"   -> "deepseek-chat"
        "gpt-4o mini"     -> "gpt-4o_mini"
        "org/model:v1.2"  -> "org_model_v1.2"
    """
    nickname = re.sub(r"[^\w.\-]+", "_", str(model).strip())
    return nickname.strip("_") or "model"


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------
def run_parallel_processing(
    process_single_item: Callable[[Any], Dict[str, Any]],
    data: List[Any],
    max_workers: int,
    tqdm_desc: str,
    jsonl_output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Map process_single_item over data with a thread pool, optionally
    streaming each result to a JSONL file as it completes."""
    data_list = list(data)
    processed_count = 0
    results: List[Dict[str, Any]] = []

    if is_non_empty_str(jsonl_output_path):
        os.makedirs(os.path.dirname(jsonl_output_path) or ".", exist_ok=True)
        print(f"Processing and writing to temporary file: {jsonl_output_path}")
        with open(jsonl_output_path, 'w', encoding='utf-8') as f_out:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results_iterator = executor.map(process_single_item, data_list)
                for item_result in tqdm(results_iterator, total=len(data_list), desc=tqdm_desc):
                    f_out.write(json.dumps(item_result, ensure_ascii=False) + '\n')
                    results.append(item_result)
                    processed_count += 1
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results_iterator = executor.map(process_single_item, data_list)
            for item_result in tqdm(results_iterator, total=len(data_list), desc=tqdm_desc):
                results.append(item_result)
                processed_count += 1

    print(f"[{tqdm_desc}] processed={processed_count}")
    return results


# ---------------------------------------------------------------------------
# Output IO / path layout
# ---------------------------------------------------------------------------
def dump_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_intermediate_paths(output_dir: str, prefix: str, nickname: str) -> Dict[str, str]:
    """Build the per-step intermediate output paths for a run."""
    return {
        "step_1": os.path.join(output_dir, f"{prefix}_{nickname}_extracted_elements.json"),
        "step_2": os.path.join(output_dir, f"{prefix}_{nickname}_generated_random_elements.json"),
        "step_3": os.path.join(output_dir, f"{prefix}_{nickname}_generated_random_elements_tasks.json"),
        "step_4": os.path.join(output_dir, f"{prefix}_generated_random_elements_{nickname}_generated_solution_test_v2.json"),
        "step_5_valid": os.path.join(output_dir, f"{prefix}_generated_random_elements_{nickname}_valid_problem.json"),
        "step_5_debug": os.path.join(output_dir, f"{prefix}_generated_random_elements_{nickname}_generated_solution_test_v2_validation.json"),
    }


def normalize_intermediate_paths(paths: Dict[str, str]) -> Dict[str, str]:
    required = {"step_1", "step_2", "step_3", "step_4", "step_5_valid", "step_5_debug"}
    missing = sorted(required - set(paths.keys()))
    if missing:
        raise ValueError(f"intermediate_paths missing keys: {missing}")
    return {k: paths[k] for k in sorted(required)}
