import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import openai

from utils.openai_utils import get_openai_response, process_response, extract_answer_from_thinking
from utils_synthetic.prompts_synthetic_algorithm import schema_optimization_prompt


class SchemaOptimizer:
    def __init__(self, model_name='../all-MiniLM-L6-v2', random_state=42):
        self.model = SentenceTransformer(model_name)
        self.random_state = random_state

    def _discretize_texts(self, texts, max_clusters=10):
        """Encode and discretize texts into cluster labels."""
        n = len(texts)
        if n == 0:
            return np.array([], dtype=int)
        if n == 1:
            return np.array([0], dtype=int)

        embeddings = self.model.encode(texts)
        n_clusters = min(max(2, int(np.sqrt(n))), max_clusters, n)
        labels = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=self.random_state
        ).fit_predict(embeddings)
        return labels.astype(int)

    def calculate_discrete_entropy(self, texts):
        """Calculate the Shannon entropy H(X) after discretization."""
        labels = self._discretize_texts(texts)
        if labels.size == 0:
            return float('nan')
        probs = np.bincount(labels) / labels.size
        return float(entropy(probs))

    def calculate_conditional_mi(self, x_texts, y_texts, z_texts):
        """Calculate the conditional mutual information I(X;Y|Z), where Y is the problem text."""
        n = len(x_texts)
        if not (n == len(y_texts) == len(z_texts)):
            raise ValueError('The lengths of x_texts, y_texts, and z_texts must be consistent.')
        if n < 2:
            return float('nan')

        x_labels = self._discretize_texts(x_texts)
        y_labels = self._discretize_texts(y_texts)
        z_labels = self._discretize_texts(z_texts)

        xyz_counts = Counter(zip(x_labels, y_labels, z_labels))
        xz_counts = Counter(zip(x_labels, z_labels))
        yz_counts = Counter(zip(y_labels, z_labels))
        z_counts = Counter(z_labels)

        n_float = float(n)
        cmi = 0.0
        for (x, y, z), c_xyz in xyz_counts.items():
            p_xyz = c_xyz / n_float
            p_xz = xz_counts[(x, z)] / n_float
            p_yz = yz_counts[(y, z)] / n_float
            p_z = z_counts[z] / n_float
            if p_xyz > 0 and p_xz > 0 and p_yz > 0 and p_z > 0:
                cmi += p_xyz * np.log((p_z * p_xyz) / (p_xz * p_yz))

        return float(cmi)


def extract_problem_text(item):
    candidates = [
        'problem', 'problem_text', 'question', 'prompt', 'text', 'statement', 'content'
    ]
    for k in candidates:
        v = item.get(k)
        if isinstance(v, str) and v.strip():
            return v

    for outer in ['metadata', 'raw', 'input', 'instance']:
        inner = item.get(outer)
        if isinstance(inner, dict):
            for k in candidates:
                v = inner.get(k)
                if isinstance(v, str) and v.strip():
                    return v
    
    if 'seed_item' in item and isinstance(item['seed_item'], dict):
        return item['seed_item']['prompt']

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Element Schema Optimization based on Entropy and CMI.")

    # Model & API
    model_api = parser.add_argument_group("Model & API")
    model_api.add_argument("--model", type=str, default="deepseek-chat",
                           help="Model identifier for LLM calls (default: deepseek-chat).")
    model_api.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"),
                           help="LLM API key (defaults to OPENAI_API_KEY env var).")
    model_api.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
                           help="LLM API base URL (defaults to OPENAI_BASE_URL env var, else https://api.deepseek.com/v1).")
    model_api.add_argument("--embedding_model", type=str, default='../all-MiniLM-L6-v2',
                           help="Path or name of the SentenceTransformer model used for discretization.")

    # I/O paths
    io_grp = parser.add_argument_group("I/O paths")
    io_grp.add_argument("--input_path", type=str, required=True,
                        help="Path to the extracted elements JSON file.")
    io_grp.add_argument("--schema_path", type=str, required=False,
                        help="Path to the JSON file containing the current element schema definitions.")

    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        raise ValueError("Missing API key. Pass --api_key or set OPENAI_API_KEY env var.")

    if args.schema_path and os.path.exists(args.schema_path):
        with open(args.schema_path, 'r', encoding='utf-8') as f:
            schema_str = f.read()
    else:
        raise ValueError("Schema file not found. Please provide a valid --schema_path.")

    optimizer = SchemaOptimizer(model_name=args.embedding_model)

    with open(args.input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        raise ValueError('Data is empty, cannot compute metrics.')

    element_keys = list(data[0].get('extracted_elements', {}).keys())
    print(f"Found elements: {element_keys}")
    if not element_keys:
        raise ValueError('Could not find the extracted_elements field.')

    entropy_lines = ['--- Discrete Entropy H(X) For Each Element ---']
    entropy_results = {}
    for x in element_keys:
        x_texts = [
            item.get('extracted_elements', {}).get(x)
            for item in data
            if item.get('extracted_elements', {}) is not None
        ]
        if len(x_texts) < 2:
            entropy_results[x] = float('nan')
        else:
            entropy_results[x] = optimizer.calculate_discrete_entropy(x_texts)
        entropy_lines.append(f'Element: {x:<35} H(X): {entropy_results[x]:.6f}')

    entropy_str = '\n'.join(entropy_lines)
    print(entropy_str)

    print('\n--- Conditional Mutual Information Matrix: I(X; Problem | Z) ---')
    cmi_matrix = pd.DataFrame(index=element_keys, columns=element_keys, dtype=float)

    for z in element_keys:
        for x in element_keys:
            if x == z:
                cmi_matrix.loc[x, z] = float('nan')
                continue

            rows = []
            for item in data:
                ex = item.get('extracted_elements', {})
                if ex is not None:
                    x_val = ex.get(x)
                    z_val = ex.get(z)
                    p_val = extract_problem_text(item)
                    if x_val is None or z_val is None or p_val is None:
                        continue
                    rows.append((x_val, p_val, z_val))

            if len(rows) < 2:
                cmi_matrix.loc[x, z] = float('nan')
                continue

            x_texts, p_texts, z_texts = zip(*rows)
            cmi_matrix.loc[x, z] = optimizer.calculate_conditional_mi(
                list(x_texts),
                list(p_texts),
                list(z_texts)
            )

    pd.set_option('display.width', 1600)
    pd.set_option('display.max_columns', None)
    print(cmi_matrix)
    cmi_str = "--- Conditional Mutual Information Matrix: I(X; Problem | Z) ---\n" + cmi_matrix.to_string()


    client = openai.Client(
        base_url=args.base_url,    
        api_key=api_key,
        timeout=60000,
    )

    prompt = (schema_optimization_prompt
              .replace("{SCHEMA_PLACEHOLDER}", schema_str)
              .replace("{ENTROPY_PLACEHOLDER}", entropy_str)
              .replace("{CMI_PLACEHOLDER}", cmi_str))

    print(prompt)

    res = get_openai_response(
        client=client,
        model=args.model,
        user_prompt=prompt,
        use_default_generation_config=True,
        max_tokens=8192,
        stream=False,
    )

    output = process_response(res, stream=False)
    response_text = output[0]
    answer_test = extract_answer_from_thinking(response_text)
    print(answer_test)
