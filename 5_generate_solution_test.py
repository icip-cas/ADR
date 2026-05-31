from tqdm import tqdm
import openai
import re
import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.openai_utils import get_openai_response, process_response, extract_answer_from_thinking
from utils_synthetic.exp_3_generate_solution_test import _parse_response, get_prompt


def process_single_sample(item, client, model_name, max_tokens):
    prompt = get_prompt(item)
    try:
        res = get_openai_response(
            client=client,
            model=model_name,
            user_prompt=prompt,
            max_tokens=max_tokens,
            use_default_generation_config=True,
            stream=False
        )
        
        output = process_response(res, stream=False)
        response_text = output[0]
        answer_text = extract_answer_from_thinking(response_text)
        solution, test_case_generator = _parse_response(answer_text)
        
        return {
            "raw_response": response_text,
            "solution": solution,
            "test_case_generator": test_case_generator,
            "raw_response_usage": str(res.usage)
        }
        
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        return {
            "raw_response": None, 
            "solution": None,
            "test_case_generator": None,
            "raw_response_usage": None
        }

def process_item_with_sampling(item, client, model_name, max_tokens, num_samples):
    samples = []
    for _ in range(num_samples):
        samples.append(process_single_sample(item, client, model_name, max_tokens))
    
    # Combine results
    item_result = {
        "id": item.get("id", ""),
        "generated_task": item["generated_task"],
        "samples": samples
    }
    return item_result

def main(args):
    client = openai.Client(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=600.0,
    )

    if args.output_path.endswith(".json"):
        jsonl_output_path = args.output_path.replace(".json", ".jsonl")
    else:
        jsonl_output_path = args.output_path + ".jsonl"

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Optional: Filter items based on a file containing IDs with errors
    if args.filter_file:
        print(f"Filtering based on error file: {args.filter_file}")
        try:
            with open(args.filter_file, 'r') as f:
                error_data = json.load(f)
            
            wrong_ids = {item['id'] for item in error_data if item.get('samples') and len(item['samples'][0].get('validation_error', '')) > 0}
            
            if args.limit_filter > 0:
                wrong_ids = set(list(wrong_ids)[:args.limit_filter])

            original_count = len(data)
            data = [item for item in data if item['id'] in wrong_ids]
            print(f"Filtered from {original_count} to {len(data)} items based on {len(wrong_ids)} IDs.")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not process filter file: {e}. Processing all items from input.")

    print(f"Loaded dataset with {len(data)} examples to process.")

    processed_count = 0
    # --- Step 1: Process data and write to a .jsonl file ---
    print(f"Processing and writing to temporary file: {jsonl_output_path}")
    with open(jsonl_output_path, 'w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            process_func = lambda item: process_item_with_sampling(
                item, client, args.model, args.max_tokens, args.num_samples
            )
            
            futures = {executor.submit(process_func, item): item for item in data}
            
            for future in tqdm(as_completed(futures), total=len(data)):
                item_result = future.result()
                f_out.write(json.dumps(item_result, ensure_ascii=False) + '\n')
                processed_count += 1
    
    print(f"Completed processing {processed_count} items.")

    # --- Step 2: Convert the .jsonl file to a formatted .json file ---
    print(f"Converting {jsonl_output_path} to {args.output_path}")
    try:
        with open(jsonl_output_path, 'r', encoding='utf-8') as f_in, \
             open(args.output_path, 'w', encoding='utf-8') as f_out:
            
            all_results = [json.loads(line) for line in f_in]
            json.dump(all_results, f_out, indent=2, ensure_ascii=False)

        print(f"Successfully saved formatted JSON to {args.output_path}")
        # Optional: remove the intermediate .jsonl file
        if not args.keep_temp_file and os.path.exists(args.output_path) and os.path.getsize(args.output_path) > 0:
            os.remove(jsonl_output_path)
            print(f"Removed temporary file: {jsonl_output_path}")

    except Exception as e:
        print(f"Error during final JSON conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate solutions and tests for coding tasks.")

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
                        help="Path to the input JSON file with generated tasks.")
    io_grp.add_argument("--output_path", type=str, required=True,
                        help="Path to the output JSON file for solutions and tests.")

    # Processing
    processing = parser.add_argument_group("Processing")
    processing.add_argument("--max_workers", type=int, default=32,
                            help="Maximum number of parallel worker threads (default: 32).")
    processing.add_argument("--num_samples", type=int, default=1,
                            help="Number of samples to generate per item (default: 1).")
    processing.add_argument("--keep_temp_file", action='store_true',
                            help="Keep the intermediate .jsonl file after completion.")

    # Filtering
    filtering = parser.add_argument_group("Filtering")
    filtering.add_argument("--filter_file", type=str, default=None,
                           help="Optional path to a JSON file to filter by IDs with validation errors.")
    filtering.add_argument("--limit_filter", type=int, default=0,
                           help="Limit the number of items to process from the filter file (0 for no limit).")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api_key or set OPENAI_API_KEY env var.")

    main(args)