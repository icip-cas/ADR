import os
import openai
import re
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.openai_utils import get_openai_response, extract_answer_from_thinking, process_response, process_response_seperate
from utils_synthetic.exp_1_extract_elements_prime_utils import extract_elements_from_response, get_prompt, get_question, get_solution


def process_single_item(item, client, model_name, max_tokens, stream):
    id = item['problem_id']
    question = get_question(item)
    solution = get_solution(item)
    prompt = get_prompt(item)
    extracted_elements = {
        "Core Algorithm Idea": None,
        "Story Background": None,
        "Strategy Diversity": None,
        "Difficulty Level": None
    }

    try:
        res = get_openai_response(
            client=client,
            model=model_name,
            user_prompt=prompt,
            max_tokens=max_tokens,
            stream=stream
        )

        output = process_response(res, stream=stream)
        response_text = output[0]

        answer_text = extract_answer_from_thinking(response_text)
        extracted_elements = extract_elements_from_response(answer_text)

        item_result = {
            "id": id,
            "question": question,
            "solution": solution,
            "extracted_elements": extracted_elements,
            "raw_response": response_text,
            "raw_response_usage": str(res.usage)
        }
        return item_result
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        item_result = {
            "id": id,
            "question": question,
            "solution": solution,
            "extracted_elements": {**extracted_elements, "error": str(e)},
            "raw_response": None,
            "raw_response_usage": None
        }
        return item_result

def main(args):
    client = openai.Client(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=600.0,
    )

    # Write results to a temporary .jsonl first, then consolidate to .json.
    if args.output_path.endswith(".json"):
        jsonl_output_path = args.output_path.replace(".json", ".jsonl")
    else:
        jsonl_output_path = args.output_path + ".jsonl"

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if args.start_index > 0:
            data = data[args.start_index:]
    print(f"Loaded dataset with {len(data)} examples")

    processed_count = 0
    print(f"Processing and writing to temporary file: {jsonl_output_path}")
    with open(jsonl_output_path, 'w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            process_func = lambda item: process_single_item(
                item, client, args.model, args.max_tokens, args.stream
            )
            results_iterator = executor.map(process_func, data)

            for item_result in tqdm(results_iterator, total=len(data)):
                f_out.write(json.dumps(item_result, ensure_ascii=False) + '\n')
                processed_count += 1

    print(f"Completed processing {processed_count} items.")

    print(f"Converting {jsonl_output_path} to {args.output_path}")
    try:
        with open(jsonl_output_path, 'r', encoding='utf-8') as f_in, \
             open(args.output_path, 'w', encoding='utf-8') as f_out:

            all_results = [json.loads(line) for line in f_in]
            json.dump(all_results, f_out, indent=2, ensure_ascii=False)

        print(f"Successfully saved formatted JSON to {args.output_path}")
        os.remove(jsonl_output_path)
        print(f"Removed temporary file: {jsonl_output_path}")

    except Exception as e:
        print(f"Error during final JSON conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract atomic elements from seed coding problems.")

    # Model & API
    model_api = parser.add_argument_group("Model & API")
    model_api.add_argument("--model", type=str, default="deepseek-chat",
                           help="Model identifier for LLM calls (default: deepseek-chat).")
    model_api.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                           help="LLM API key (defaults to OPENAI_API_KEY env var).")
    model_api.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
                           help="LLM API base URL (defaults to OPENAI_BASE_URL env var, else https://api.deepseek.com/v1).")
    model_api.add_argument("--max_tokens", type=int, default=8192,
                           help="Maximum tokens for the model response (default: 8192).")

    # I/O paths
    io_grp = parser.add_argument_group("I/O paths")
    io_grp.add_argument("--input_path", type=str, required=True,
                        help="Path to the input JSON file (list of seed problems).")
    io_grp.add_argument("--output_path", type=str, required=True,
                        help="Path to the output JSON file.")

    # Processing
    processing = parser.add_argument_group("Processing")
    processing.add_argument("--max_workers", type=int, default=32,
                            help="Maximum number of parallel worker threads (default: 32).")
    processing.add_argument("--start_index", type=int, default=0,
                            help="Skip the first N items in the dataset (default: 0).")
    processing.add_argument("--stream", action='store_true',
                            help="Enable streaming response from the API.")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api_key or set OPENAI_API_KEY env var.")

    main(args)
