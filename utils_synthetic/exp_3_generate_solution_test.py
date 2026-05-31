import os
import sys
import re
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_synthetic.prompts_synthetic_algorithm import generate_solution_test_prompt_v2

NUM_SAMPLES_PER_ITEM = 1

def get_prompt(item):
    return generate_solution_test_prompt_v2.format(problem=item['generated_task'])

def _extract_between(text, start_tag, end_tag):
    pattern = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL | re.IGNORECASE)
    m = pattern.search(text or "")
    return m.group(1).strip() if m else None

def _strip_code_fences(s):
    if not s:
        return None
    fence = re.search(r"```(?:\w+)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return s.strip()

def _parse_response(response_text: str):
    raw_solution = _extract_between(response_text, "<|Solution Begin|>", "<|Solution End|>")
    raw_tcgen = _extract_between(response_text, "<|Test Case Generator Begin|>", "<|Test Case Generator End|>")
    
    solution = _strip_code_fences(raw_solution)
    tcgen = _strip_code_fences(raw_tcgen)

    if solution or tcgen:
        return solution, tcgen

    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", response_text, re.DOTALL)

    if code_blocks:
        solution = code_blocks[0].strip()
        if len(code_blocks) > 1:
            tcgen = code_blocks[1].strip()
        else:
            tcgen = None
    
    return solution, tcgen

