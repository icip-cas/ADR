import os
import json
import argparse
from tqdm import tqdm
from utils.call_sandbox_api import call_sandbox_api
from utils_synthetic.exec_utils import timeout, TimeoutException

def validate_sample(sample, sandbox_url, exec_timeout):
    """
    Validates a single sample by generating and testing its solution.
    Returns (is_valid, test_cases, error_message, metadata)
    """
    solution_code = sample.get("solution", "")
    test_case_generator_code = sample.get("test_case_generator", "")
    metadata = None

    if not solution_code or not test_case_generator_code:
        return False, None, "Missing solution or test case generator.", metadata

    try:
        local_scope = {}
        with timeout(exec_timeout, "Test case generator execution timed out."):
            exec(test_case_generator_code, {}, local_scope)

            if 'generate_test_cases' not in local_scope:
                return False, None, "No generate_test_cases function found.", metadata

            test_cases = local_scope['generate_test_cases']()

        if not isinstance(test_cases, dict) or not all(k in test_cases for k in ['input', 'output', 'fn_name']):
            return False, None, "Generated test cases are invalid or missing required keys.", metadata

        # Validate solution in the actual sandbox
        result_status, metadata = call_sandbox_api(code=solution_code, test_cases=test_cases, url=sandbox_url)
        
        if result_status:
            return True, test_cases, "", metadata
        else:
            # The metadata from the sandbox often contains the reason for failure.
            error_detail = metadata.get('error', 'Solution failed on generated test cases.') if metadata else "Solution failed on generated test cases."
            return False, test_cases, str(error_detail), metadata

    except TimeoutException as e:
        return False, None, str(e), metadata
    except Exception as e:
        return False, None, f"Error during local validation: {e}", metadata

def convert_jsonl_to_json(jsonl_path, json_path):
    """Converts a JSONL file to a pretty-printed JSON file."""
    print(f"Converting {jsonl_path} to {json_path}...")
    records = []
    if not os.path.exists(jsonl_path):
        print(f"Warning: {jsonl_path} not found. Skipping conversion.")
        return
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f_jsonl:
            for line in f_jsonl:
                records.append(json.loads(line))
        with open(json_path, 'w', encoding='utf-8') as f_json:
            json.dump(records, f_json, indent=2, ensure_ascii=False)
        print(f"Saved final JSON file to {json_path}")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error during conversion: {e}")

def main(args):
    # Define output paths based on the main output path
    if args.output_path.endswith(".json"):
        output_path_json = args.output_path
        output_path_jsonl = args.output_path.replace(".json", ".jsonl")
    else:
        output_path_json = args.output_path + ".json"
        output_path_jsonl = args.output_path + ".jsonl"

    if args.debug_output_path.endswith(".json"):
        debug_output_path_json = args.debug_output_path
        debug_output_path_jsonl = args.debug_output_path.replace(".json", ".jsonl")
    else:
        debug_output_path_json = args.debug_output_path + ".json"
        debug_output_path_jsonl = args.debug_output_path + ".jsonl"

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.limit > 0:
        data = data[:args.limit]

    print(f"Loaded {len(data)} examples from {args.input_path}")
    
    # Load already processed IDs to allow resuming
    processed_ids = set()
    if os.path.exists(debug_output_path_jsonl):
        with open(debug_output_path_jsonl, 'r', encoding='utf-8') as f_debug_read:
            for line in f_debug_read:
                try:
                    processed_item = json.loads(line)
                    if 'id' in processed_item:
                        processed_ids.add(processed_item['id'])
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode line in {debug_output_path_jsonl}: {line.strip()}")
        print(f"Found {len(processed_ids)} already processed tasks. Resuming.")

    valid_count = 0
    with open(output_path_jsonl, 'a', encoding='utf-8') as f_valid, \
         open(debug_output_path_jsonl, 'a', encoding='utf-8') as f_debug:
        
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Skipping {len(data) - len(items_to_process)} items. New items to process: {len(items_to_process)}")

        for item in tqdm(items_to_process, desc="Validating Problems"):
            item_id = item.get('id', 'N/A')
            
            if item_id in args.skip_ids:
                print(f"  - Skipping task {item_id} as per --skip-ids and marking as 'Segmentation fault'")
                error_msg = "Segmentation fault"
                for sample in item['samples']:
                    sample['validation_error'] = error_msg
                    sample['validation_sandbox_metadata'] = None
                f_debug.write(json.dumps(item, ensure_ascii=False) + '\n')
                continue
        
            is_problem_valid = False
            ready_item = {
                "id": item_id,
                "generated_task": item["generated_task"],
            }

            for j, sample in enumerate(item['samples']):
                is_valid, test_cases, error_msg, metadata = validate_sample(sample, args.sandbox_url, args.timeout)
                
                sample['validation_error'] = error_msg
                sample['validation_sandbox_metadata'] = metadata

                if is_valid and not is_problem_valid:
                    tqdm.write(f"  - Found valid sample #{j+1} for item {item_id}")
                    ready_item.update({
                        "solution": sample.get("solution"),
                        "test_case_generator": sample.get("test_case_generator"),
                        "test_cases": json.dumps(test_cases),
                    })
                    f_valid.write(json.dumps(ready_item, ensure_ascii=False) + '\n')
                    valid_count += 1
                    is_problem_valid = True
                    # Continue validating other samples for complete debug output
            
            f_debug.write(json.dumps(item, ensure_ascii=False) + '\n')

    total_valid_count = 0
    if os.path.exists(output_path_jsonl):
        with open(output_path_jsonl, 'r', encoding='utf-8') as f_valid_read:
            total_valid_count = sum(1 for _ in f_valid_read)

    print(f"\nAdded {valid_count} new valid examples.")
    print(f"Total valid examples: {total_valid_count}")
    print(f"Saved valid examples incrementally to {output_path_jsonl}")
    print(f"Saved debug info incrementally to {debug_output_path_jsonl}")

    convert_jsonl_to_json(output_path_jsonl, output_path_json)
    convert_jsonl_to_json(debug_output_path_jsonl, debug_output_path_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate generated coding problems by running solutions against generated tests.")

    # Sandbox
    sandbox = parser.add_argument_group("Sandbox")
    sandbox.add_argument("--sandbox_url", type=str, default=os.getenv("SANDBOX_URL"),
                         help="URL of the code execution sandbox API (defaults to SANDBOX_URL env var).")
    sandbox.add_argument("--timeout", type=int, default=60,
                         help="Timeout in seconds for executing the test case generator (default: 60).")

    # I/O paths
    io = parser.add_argument_group("I/O paths")
    io.add_argument("--input_path", type=str, required=True,
                    help="Path to the input JSON file containing generated solutions and tests.")
    io.add_argument("--output_path", type=str, required=True,
                    help="Base path for the output JSON file for valid problems. '.json' and '.jsonl' will be created.")
    io.add_argument("--debug_output_path", type=str, required=True,
                    help="Base path for the debug JSON file containing all validation results. '.json' and '.jsonl' will be created.")

    # Filtering
    filtering = parser.add_argument_group("Filtering")
    filtering.add_argument("--limit", type=int, default=0,
                           help="Limit the number of items to process from the input file (0 for no limit).")
    filtering.add_argument("--skip-ids", nargs='+', default=[],
                           help="List of item IDs to skip, marking them with a 'Segmentation fault' error.")

    args = parser.parse_args()

    if not args.sandbox_url:
        raise ValueError("Missing sandbox URL. Pass --sandbox_url or set SANDBOX_URL env var.")

    main(args)
