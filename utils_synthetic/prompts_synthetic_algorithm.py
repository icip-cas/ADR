extract_element_prompt_v4 = """
Please analyze the following algorithm problem according to the guidelines below. The output should follow the format below (do not add any other notes or explanations):
<answer>
Core Algorithm Idea:
Story Background:
Strategy Diversity:
Difficulty Level:
</answer>

### Guidelines
- Core Algorithm Idea:
    - Extract the essential algorithmic principle required to solve the problem.
    - Follow these rules:
        - Identify the **primary computational technique** (e.g., greedy strategy with a specific invariant, DP with defined states and transitions, graph traversal with constraints, combinatorial search, optimization structure).
        - Focus on the **abstract reasoning pattern**, not the implementation details.
        - Ensure the idea is **specific enough** to distinguish it from other techniques, but **general enough** to be reused in new problems.
        - Highlight key structural properties (optimal substructure, monotonicity, connectivity, constraint types, state formulation, etc.)
- Story Background:
    - Describe a simple and generic narrative theme that provides motivation or flavor for the problem in one abstract, conceptual sentence.
    - Follow these rules:
        - Be conceptual rather than operational (e.g., "managing resources", "tracking evolving states", "navigating a structure").
        - Avoid concrete rules, numeric details, domain-specific mechanics, or anything implying algorithmic constraints. 
        - Not depend on specific data types, input formats, or procedures — those belong to other elements. The background should be flexible enough to pair with many different algorithmic cores without creating conflicts.
- Strategy Diversity:
    - List the legitimate algorithmic approaches that could solve the problem **in principle**, and explain why.
    - Follow these rules:
        - Cover the full spectrum of viable strategies:
            - Greedy (local optimality conditions)
            - Dynamic Programming (state decomposition)
            - Graph search (DFS/BFS/backtracking)
            - Data-structure-driven optimization (segment tree, union-find, Fenwick tree, etc.)
            - Approximation/heuristics (if NP-hard structures are implied)
        - Explain the **structural justification** (e.g., overlapping subproblems, convexity, combinatorial explosion).
        - These should be **strategies that could plausibly apply**, not necessarily the optimal one.
- Difficulty Level:
    - Classify the problem's difficulty based on conceptual and implementation challenges.
    - Follow these rules:
        - Use the scale: **Beginner / Intermediate / Advanced**.
        - Consider:
            - Core concepts required
            - Input size limits
            - Edge-case density
            - Data structure sophistication
            - Theoretical complexity (polynomial vs NP-hard)
        - Justify the classification clearly.

### Algorithm Problem
{problem}

### Solution of the Algorithm Problem
```python
{solution}
```

### Analysis
""".strip()


design_task_prompt = """
You are an expert problem setter and algorithmist. Based solely on the given problem design framework, produce a single, self-contained, challenging algorithm problem that is ready for use in programming contests or practice platforms.

Requirements:
1. Use the provided `Problem Framework` as the only source of content and constraints.
2. Follow the `Problem Template` exactly and fill every required field.
3. The generated problem must be original, well-specified, and include rigorous constraints so that intended complexity is clear.
4. Do NOT add any notes, commentary, or text outside the `Problem Template` block. The model's entire output must be exactly the filled template and nothing else.

### Problem Framework
{framework}

### Problem Template
```
**Problem Title:**
[Your problem title]

**Tags:**
[Comma-separated topics/tags]

**Difficulty:**
[Easy / Medium / Hard / Very Hard]

**Problem Statement:**
[Complete and precise description of the problem]

**Input:**
[Exact input format]

**Output:**
[Exact output format]

**Constraints:**
[Formal constraints with bounds]

**Example:**
[Examples with explanations]

**Notes:** (optional)
[A brief reference solution description]
```

Important: The output produced by you now must be ONLY the completed `Problem Template` block filled according to the above rules, with no additional text before or after.

### Your Designed Problem
""".strip()


design_new_elements_wo_definition_combination = """
You are an expert in algorithmic problem design. Your task is to analyze the given Story Background and use the three provided combinations as inspiration, then construct one new and coherent set of four elements (Core Algorithm Idea, Story Background, Strategy Diversity, Difficulty Level).
The output should follow the format below (do not add any other notes or explanations):
<answer>
Core Algorithm Idea:
Story Background:
Strategy Diversity:
Difficulty Level:
</answer>

Requirements:
1. Learn the structural patterns, not the content
    - Extract from the examples their level of detail, reasoning style, and the way elements relate to each other. Do not reuse or merge their specific ideas.
2. Preserve internal coherence
    - Ensure the four generated elements naturally support each other:
        - The Story Background should organically introduce the constraints that motivate the Core Algorithm Idea.
        - The Strategy Diversity must correspond to the algorithmic structure implied by the Core Algorithm Idea.
        - The Difficulty Level should reflect the conceptual and technical depth of the chosen algorithmic direction.
3. Maintain originality and avoid conflicts
    - Your output must be a fully new construction — no copying from examples — and must avoid internal contradictions in constraints, methods, or complexity assumptions.
4. Ensure problem-level richness
    - The new combination should have enough structure and complexity to support a meaningful algorithm problem, with:
        - Non-trivial decisions or constraints
        - Multiple plausible solution approaches
        - Clear reasons for the difficulty classification

### Story Background
{story_background}

### Combinations Reference
{combinations_1}

{combinations_2}

{combinations_3}

### New Elements Combination
""".strip()



generate_solution_test_prompt_v2 = """
## Task
You are given an algorithm problem. Your task is to generate both the `solution code` and the `test case generator code` for that algorithm problem.

## Output format
<|Solution Begin|>
[Solution Code in Python]
<|Solution End|>
<|Test Case Generator Begin|>
[Test Case Generator in Python]
<|Test Case Generator End|>

## Example for `stdin_stdout` solution
<|Solution Begin|>
```python
import sys

def main():
    data = sys.stdin.read().strip().split()
    if len(data) < 2:
        return
    a, b = int(data[0]), int(data[1])
    print(a + b)

if __name__ == "__main__":
    main()
```
<|Solution End|>
<|Test Case Generator Begin|>
```python
import random

def generate_test_cases():
    random.seed(42)
    inputs = []
    outputs = []

    pairs = [
        (0, 0),
        (1, -1),
        (-1, -1),
        (10**6, 10**6),
        (-10**6, -10**6),
        (10**6, -10**6),
        (123456, 654321)
    ]

    NUM_RANDOM = 10
    MINV, MAXV = -10**6, 10**6
    for _ in range(NUM_RANDOM):
        a = random.randint(MINV, MAXV)
        b = random.randint(MINV, MAXV)
        pairs.append((a, b))

    for a, b in pairs:
        inputs.append(f"{{a}} {{b}}\n")
        outputs.append(f"{{a + b}}\n")

    return {{
        "input": inputs,
        "output": outputs,
        "fn_name": None,
        "type": "stdin_stdout"
    }}
```
<|Test Case Generator End|>

## Example for `function_call` solution
<|Solution Begin|>
```python
def add(a, b):
    return a + b
```
<|Solution End|>
<|Test Case Generator Begin|>
```python
import random

def generate_test_cases():
    random.seed(42)
    inputs = []
    outputs = []

    pairs = [
        (0, 0),
        (1, -1),
        (-1, -1),
        (10**6, 10**6),
        (-10**6, -10**6),
        (10**6, -10**6),
        (123456, 654321)
    ]

    NUM_RANDOM = 10
    MINV, MAXV = -10**6, 10**6
    for _ in range(NUM_RANDOM):
        a = random.randint(MINV, MAXV)
        b = random.randint(MINV, MAXV)
        pairs.append((a, b))

    for a, b in pairs:
        inputs.append([a, b])
        outputs.append([a + b])

    return {{
        "input": inputs,
        "output": outputs,
        "fn_name": "add",
        "type": "function_call"
    }}
```
<|Test Case Generator End|>

## Algorithm problem
{problem}
""".strip()


# ---------------------------------------------------------------------------
# Pipeline prompts (schema-driven variants used by 0_pipeline.py).
# These differ from the fixed prompts above in that they take a runtime
# `answer_template` / `guidelines` derived from the active element schema,
# rather than hardcoding the four algorithmic elements.
# ---------------------------------------------------------------------------
pipeline_extract_prompt = """
Please analyze the following algorithm problem according to the guidelines below. The output should follow the format below (do not add any other notes or explanations):
{answer_template}

### Algorithm Problem
{problem}

### Guidelines
{guidelines}
""".strip()


pipeline_recombine_prompt = """
You are an expert in algorithmic problem design. Your task is to analyze the given {core_element} and use the three provided combinations as inspiration, then construct one new and coherent set of elements.
The output should follow the format below (do not add any other notes or explanations):
{answer_template}

### Requirements
1. Learn the structural patterns, not the content
    - Extract from the examples their level of detail, reasoning style, and the way elements relate to each other. Do not reuse or merge their specific ideas.
2. Preserve internal coherence
    - Ensure the generated elements naturally support each other.
3. Maintain originality and avoid conflicts
    - Your output must be a fully new construction - no copying from examples - and must avoid internal contradictions in constraints, methods, or complexity assumptions.
4. Ensure problem-level richness
    - The new combination should have enough structure and complexity to support a meaningful algorithm problem, with:
        - Non-trivial decisions or constraints
        - Multiple plausible solution approaches
        - Clear reasons for the difficulty classification

### {core_element}
{core_element_value}

### Combinations Reference
{combinations_1}

{combinations_2}

{combinations_3}

### New Elements Combination
""".strip()


# ---------------------------------------------------------------------------
# Schema optimization prompt (used by 1_element_schema_optimization.py).
# Uses {{ }}-escaped JSON, so it is filled with str.replace() on the
# {SCHEMA_PLACEHOLDER} / {ENTROPY_PLACEHOLDER} / {CMI_PLACEHOLDER} markers
# rather than str.format().
# ---------------------------------------------------------------------------
schema_optimization_prompt = """
You are an expert in algorithmic problem design and representation learning.

Your task is to improve the ELEMENT SCHEMA used to synthesize algorithmic coding problems.

### Current element schema
{SCHEMA_PLACEHOLDER}

### Your goal

The Schema has been decomposed into a set of elements, and for each element you are provided with:
- The entropy of the element (measuring its information content and uncertainty)
- The conditional mutual information (CMI) between pairs of elements with respect to the programming problem (measuring redundancy, dependency, or complementary information)

Your task is to analyze and optimize the initial element Schema based on information-theoretic principles.

You may perform the following actions:
1. ADD a new element
2. REMOVE an existing element
3. MERGE two or more elements
4. SPLIT one element into multiple elements
5. REFINE the definition of an element

### The entropy of the element

{ENTROPY_PLACEHOLDER}

### The CMI between pairs of elements

{CMI_PLACEHOLDER}

### Important constraints
- The given schema may contain several elements, but make sure the final schema should NOT exceed **5** elements.
- Each element must be:
  - Clearly defined
  - Usable in a problem generation prompt
  - Non-redundant
- Prefer elements that:
  - Reduce ambiguity of the optimal algorithm
  - Improve difficulty controllability
  - Increase diversity of generated problems

### Output format (strict JSON)
{{
  "change_proposals": [
    {{
      "action": "add | remove | merge | split | refine",
      "target_elements": ["ElementA", "ElementB"],
      "new_elements": [
        {{
          "name": "...",
          "definition": "..."
        }}
      ],
      "rationale": "Why this change improves problem quality"
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------------------------
# Adversarial refinement prompts (used by 7_adversarial_refinement.py).
# ---------------------------------------------------------------------------
near_miss_solutions_prompt = """
You are given a programming problem and a reference correct solution.

Your task is to generate exactly 5 distinct **near-miss solutions**.

A near-miss solution is:
- Logically plausible and well-structured
- Likely to pass many simple or random test cases
- Incorrect due to subtle flaws (e.g., edge cases, off-by-one errors, incorrect assumptions, partial logic, numerical instability, complexity limits)

### Instructions:
1. Each near-miss solution should be written as full executable code.
2. Each solution must fail for at least one non-trivial or adversarial input.
3. The mistakes should be diverse. Avoid repeating the same error pattern.
4. Do NOT explicitly state what the bug is in the code.
5. Do NOT include explanations inside the code.
6. Output the solutions as a numbered list from 1 to 5.

### Problem description:
{problem}

### Correct reference solution:
```python
{reference_solution}
```
"""


refine_test_case_generator_prompt = """
You are given:
- A programming problem
- A correct reference solution
- A set of near-miss solutions that are incorrect but pass many naive tests
- The current version of a test case generator

Your task is to improve the test case generator using **adversarial reasoning**.

### Objective:
- Modify or redesign the test case generator to **maximize the failure rate of the near-miss solutions**
- While ensuring that the correct reference solution still passes all generated test cases

### Instructions:
1. Analyze the common and uncommon weaknesses likely present in the near-miss solutions.
2. Design test cases that specifically target:
   - Edge cases
   - Boundary conditions
   - Rare corner scenarios
   - Stress limits (size, value ranges, ordering, structure)
   - Implicit assumptions likely made by incorrect solutions
3. The generator should be general and reusable, not hardcoded for a single bug.
4. Do NOT explicitly reference individual near-miss solutions in the generator logic.
5. Output the improved test case generator as executable code or clear pseudocode.

### Problem description:
{problem}

### Correct reference solution:
```python
{reference_solution}
```

### Near-miss solutions:
{near_miss_solutions}

### Current test case generator:
```python
{test_case_generator}
```

"""

