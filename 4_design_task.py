import openai
import os
import json
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.openai_utils import get_openai_response, process_response, extract_answer_from_thinking
from utils_synthetic.exp_2_design_task_utils import get_prompt


def process_single_item(item, client, model_name, max_tokens, stream):
    prompt = get_prompt(item)

    try:
        res = get_openai_response(
            client=client,
            model=model_name,
            user_prompt=prompt,
            max_tokens=max_tokens,
            use_default_generation_config=True,
            stream=stream,
        )

        output = process_response(res, stream=stream)
        response_text = output[0]
        answer_text = extract_answer_from_thinking(response_text)
        
        # Add to results
        item_result = {
            "id": item.get("id", ""),
            "generated_task": answer_text,
            "raw_response": response_text,
            "raw_response_usage": str(res.usage)
        }
        return item_result
        
    except Exception as e:
        print(f"Error processing item: {str(e)}")
        item_result = {
            "id": item.get("id", ""),
            "generated_task": None,
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

    # Define paths for intermediate .jsonl and final .json files
    if args.output_path.endswith(".json"):
        jsonl_output_path = args.output_path.replace(".json", ".jsonl")
    else:
        jsonl_output_path = args.output_path + ".jsonl"

    # --- Step 0: Load existing results to skip processed items ---
    processed_ids = set()
    if os.path.exists(jsonl_output_path):
        with open(jsonl_output_path, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                try:
                    processed_item = json.loads(line)
                    if "id" in processed_item:
                        processed_ids.add(processed_item["id"])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line in {jsonl_output_path}: {line.strip()}")
        print(f"Found {len(processed_ids)} already processed items in {jsonl_output_path}")

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if args.start_index > 0:
            data = data[args.start_index:]
    print(f"Loaded dataset with {len(data)} examples")

    # Filter out already processed items
    items_to_process = [item for item in data if item.get("id") not in processed_ids]
    print(f"Skipping {len(data) - len(items_to_process)} items. Items to process: {len(items_to_process)}")

    if not items_to_process:
        print("No new items to process.")
    else:
        processed_count = 0
        # --- Step 1: Process data and write to a .jsonl file ---
        print(f"Processing and appending to temporary file: {jsonl_output_path}")
        with open(jsonl_output_path, 'a', encoding='utf-8') as f_out:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                process_func = lambda item: process_single_item(
                    item, client, args.model, args.max_tokens, args.stream
                )
                
                futures = [executor.submit(process_func, item) for item in items_to_process]
                
                for future in tqdm(as_completed(futures), total=len(items_to_process)):
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
    parser = argparse.ArgumentParser(description="Design new coding tasks based on extracted elements.")

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
                        help="Path to the input JSON file with extracted elements.")
    io_grp.add_argument("--output_path", type=str, required=True,
                        help="Path to the output JSON file for generated tasks.")

    # Processing
    processing = parser.add_argument_group("Processing")
    processing.add_argument("--max_workers", type=int, default=32,
                            help="Maximum number of parallel worker threads (default: 32).")
    processing.add_argument("--start_index", type=int, default=0,
                            help="Index of the first item in the dataset to process (default: 0).")
    processing.add_argument("--stream", action='store_true',
                            help="Enable streaming response from the API.")
    processing.add_argument("--keep_temp_file", action='store_true',
                            help="Keep the intermediate .jsonl file after completion.")

    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing API key. Pass --api_key or set OPENAI_API_KEY env var.")

    main(args)