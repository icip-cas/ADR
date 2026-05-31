import json
import random
import os
import sys
from tqdm import tqdm
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from utils_synthetic.prompts_synthetic_algorithm import design_new_elements_wo_definition_combination


def generate_prompts(input_path, output_path_prompts, num_prompts):
    """Generate recombination prompts from extracted elements and save to a file."""
    print(f"Loading data from {input_path} to generate prompts.")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
    print(f"Loaded dataset with {len(data)} examples.")

    prompts_to_process = []
    for i, item in tqdm(enumerate(data), total=len(data), desc="Generating Prompts"):
        story_background = item.get('extracted_elements', {}).get('Story Background', '')

        other_items = data[:i] + data[i+1:]

        if len(other_items) < 3:
            print(f"Skipping item {item.get('id')} as there are not enough other items to sample from.")
            continue

        for j in range(num_prompts):
            random_ref_items = random.sample(other_items, k=3)

            # Strip Story Background from reference combinations so the model
            # generates a fresh one rather than copying from references.
            combinations_1 = random_ref_items[0].get('extracted_elements', {}).copy()
            combinations_1.pop('Story Background', None)

            combinations_2 = random_ref_items[1].get('extracted_elements', {}).copy()
            combinations_2.pop('Story Background', None)

            combinations_3 = random_ref_items[2].get('extracted_elements', {}).copy()
            combinations_3.pop('Story Background', None)

            prompt = design_new_elements_wo_definition_combination.format(
                story_background=story_background,
                combinations_1=combinations_1,
                combinations_2=combinations_2,
                combinations_3=combinations_3,
            )

            new_id = f"{item['id']}_{j+1}"

            prompts_to_process.append({
                "id": new_id,
                "original_item": item,
                "generated_prompt": prompt,
            })

    print(f"Writing {len(prompts_to_process)} prompts to {output_path_prompts}...")
    with open(output_path_prompts, 'w', encoding='utf-8') as f_out:
        json.dump(prompts_to_process, f_out, indent=2)
    print("Prompt generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate recombination prompts from extracted elements.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the input JSON file (output of element extraction step).")
    parser.add_argument("--output_path_prompts", type=str, required=True,
                        help="Path to the output JSON file for generated prompts.")
    parser.add_argument("--num_prompts", type=int, default=8,
                        help="Number of prompts to generate per item (default: 8).")

    args = parser.parse_args()

    generate_prompts(args.input_path, args.output_path_prompts, args.num_prompts)
