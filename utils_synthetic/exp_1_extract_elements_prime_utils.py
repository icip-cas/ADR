import os
import sys
import re
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils_synthetic.prompts_synthetic_algorithm import extract_element_prompt_v4


def get_question(item):
    question = item['prompt'].replace('Solve the following coding problem using the programming language python:', '').strip()
    question = question.replace('Now solve the problem and return the code.', '').strip()
    return question

def get_solution(item):
    matches = re.findall(r"```python(.*?)```", item['gold_standard_solution'], re.DOTALL)
    solution = matches[0].strip() if matches else ""
    return solution

def get_prompt(item):
    question = get_question(item)
    solution = get_solution(item)
    
    prompt = extract_element_prompt_v4.format(problem=question,
                                              solution=solution)

    return prompt

def extract_elements_from_response(response_text: str) -> Dict:
    try:
        # Extract content between <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
        if not answer_match:
            # Try without tags if not found
            content = response_text
        else:
            content = answer_match.group(1).strip()
        
        # Extract each element
        elements = {
            "Core Algorithm Idea": None,
            "Story Background": None,
            "Strategy Diversity": None,
            "Difficulty Level": None
        }
        
        for element in elements.keys():
            pattern = rf"{element}:\s*(.*?)(?:\n\w|$)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                elements[element] = match.group(1).strip()
        
        # Check if any elements are missing
        missing_elements = [k for k, v in elements.items() if v is None]
        if missing_elements:
            print(f"Warning: Failed to extract elements: {', '.join(missing_elements)}")
        
        return elements
    
    except Exception as e:
        print(f"Error extracting elements: {str(e)}")
        return {
            "Core Algorithm Idea": None,
            "Story Background": None,
            "Strategy Diversity": None,
            "Difficulty Level": None,
            "error": str(e)
        }