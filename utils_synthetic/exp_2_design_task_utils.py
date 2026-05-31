import os
import sys
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_synthetic.prompts_synthetic_algorithm import design_task_prompt


def get_prompt(item):
    return design_task_prompt.format(framework=item['extracted_elements'])

def get_question(item):
    return item['question']

def get_solution(item):
    return item['solution']
