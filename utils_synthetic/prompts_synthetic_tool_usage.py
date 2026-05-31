generate_element_set = """
You are an expert in code task abstraction and problem schema design.

You are given:
1. A specified code task type
2. Three concrete example tasks of this type

Your task is to infer a **minimal and sufficient set of composable task elements** such that:
- Using only these elements, one can construct a **complete and valid code task**
- The elements are **orthogonal** (each captures a distinct aspect of the task)
- The elements support **recombination and variation**, enabling the generation of new tasks of the same type

Follow these strict requirements:

### Step 1: Element Discovery
Identify the **smallest possible set of task elements** that are:
- Necessary (removing any one would make the task incomplete)
- General (not tied to any specific example)
- Reusable across different tasks of the same type

### Step 2: Element Definition
For each element, provide:
- **Element Name** (concise, canonical)
- **Element Definition**: a precise, abstract description of what this element represents
- **Role in Task Construction**: why this element is required to form a complete task
- **Variation Axes**: what aspects of this element can vary to generate different tasks

Definitions must be **task-agnostic templates**, not filled instances.

### Step 3: Element Interaction Rules
Describe:
- Which elements are mandatory vs optional
- Valid combinations and ordering constraints (if any)
- How different elements can be recombined without breaking task validity

### Output Format (Strict)
Return the result in the following structure:

```
Task Type: ...

Minimal Task Element Set:
1. Element Name
   - Definition:
   - Role:
   - Variation Axes:
2. ...

Element Interaction & Composition Rules:
- ...
```

### Given Code Task Type
{task_type}

### Given Code Task Demonstrations
{example_1}

{example_2}

{example_3}
"""


extract_element_prompt = """
Please analyze the following tool-calling code problem according to the guidelines below. The output should follow the format below (do not add any other notes or explanations):
<answer>
Computational Objective:
Tool Dependency Set:
Processing Logic Constraints:
Input Interface:
Output Specification:
</answer>

### Guidelines
1. Computational Objective
    * **Definition:** An abstract specification of the primary computation or transformation to be performed on data, independent of implementation details.
    * **Role:** Defines *what* the task fundamentally does; without it, the task has no semantic goal.
    * **Variation Axes:**
        * Type of computation (aggregation, generation, transformation, analysis)
        * Deterministic vs stochastic behavior
        * Single-stage vs multi-stage computation

2. Tool Dependency Set
    * **Definition:** The set of external libraries, modules, or tools that must be imported and used to accomplish the task.
    * **Role:** Distinguishes this task type as a *tool-calling* problem; ensures the task demonstrates use of specific utilities beyond core language constructs.
    * **Variation Axes:**
        * Standard library vs third-party tools
        * Number of tools
        * Functional role of tools (randomness, statistics, iteration, collections, etc.)

3. Processing Logic Constraints
    * **Definition:** High-level rules or required operations that constrain how the computation must be carried out, without prescribing exact code.
    * **Role:** Ensures the task exercises particular patterns (e.g., shuffling before computing, sorting by a derived metric).
    * **Variation Axes:**
        * Ordering of operations
        * Required intermediate transformations
        * Use of specific functions or methods from the tools

4. Input Interface
    * **Definition:** A formal description of the function inputs, including parameter names, types, defaults, and constraints.
    * **Role:** Establishes how external data enters the task; necessary for invoking the computation.
    * **Variation Axes:**
        * Number of parameters
        * Data types (scalars, lists, dicts, strings, etc.)
        * Default values
        * Validity constraints (ranges, non-negativity, non-emptiness)

5. Output Specification
    * **Definition:** A precise description of the expected output type, structure, and semantic meaning.
    * **Role:** Defines task completion criteria; without it, correctness cannot be evaluated.
    * **Variation Axes:**
        * Output data type (float, dict, list, etc.)
        * Structural properties (sorted, aggregated, keyed by…)
        * Deterministic vs stochastic output interpretation

### Tool-Calling Code Problem
{problem}

### Solution of the Tool-Calling Code Problem
```python
{solution}
```

### Analysis
""".strip()



design_new_elements = """
You are an expert in tool-calling code task abstraction and schema-level task design.
You are given:
- One randomly sampled Computational Objective
- One randomly sampled Tool Dependency Set
- Three reference sets, each consisting of:
    - Processing Logic Constraints
    - Input Interface
    - Output Specification
Your task is to design a completely new tool-calling code task at the element level, by inferring a novel and coherent combination of task elements, not by copying or minimally editing the references.
The output should follow the format below (do not add any other notes or explanations):
<answer>
Computational Objective:
Tool Dependency Set:
Processing Logic Constraints:
Input Interface:
Output Specification:
</answer>

### Core Requirements

1. **Element-level generation only**
    * Do NOT write a concrete problem statement or code.
    * Do NOT reuse wording, structure, or semantics from any single reference set.
    * Operate strictly at the level of abstract task elements.

2. **Five-element completeness**
    You must generate **exactly five elements**, one for each of the following:
    * Computational Objective
    * Tool Dependency Set
    * Processing Logic Constraints
    * Input Interface
    * Output Specification

3. **Consistency constraints**
    * The Computational Objective must be **achievable** using the Tool Dependency Set.
    * Processing Logic Constraints must **meaningfully constrain** how the tools are used.
    * Input Interface must provide **sufficient information** to execute the objective.
    * Output Specification must be a **direct consequence** of the objective and logic.

4. **Novel recombination**
    * Treat the three reference sets as **design signals**, not templates.
    * The resulting element set should be plausibly generatable by recombining ideas,
      but **must not align exactly with any reference along more than one element**.

5. **Tool-calling emphasis**
    * The Tool Dependency Set must play a **non-trivial role** in enabling or shaping the task.
    * If tools were removed, the task should lose its defining character.

### Given Computational Objective
{computational_objective}

### Given Tool Dependency Set
{tool_dependency_set}

### Reference Sets
{combinations_1}

{combinations_2}

{combinations_3}

### New Elements
""".strip()


design_task_prompt = """
You are an expert in **tool-calling code task synthesis**.

You are given a complete task specification expressed as **five abstract task elements**:

* Computational Objective
* Tool Dependency Set
* Processing Logic Constraints
* Input Interface
* Output Specification

Your task is to **synthesize a single, complete tool-calling code problem** that strictly follows the required template and **faithfully instantiates all five elements**.

### Core Rules
1. **Element Fidelity**
    * All five elements must be fully and explicitly reflected.
    * Do not add, remove, or reinterpret any requirement.

2. **No Meta or Reasoning Text**
    * Output only a finished problem statement.
    * No explanations, hints, or commentary.

3. **Mandatory Tool Usage**
    * The task must inherently require the specified tools.
    * Removing the tools should break the task.

4. **Template Exactness**
    * Follow the template structure and section order exactly.
    * Do not rename, reorder, or omit sections.

5. **Header Code Authority**
    * Use the header code verbatim.
    * All required imports and the function signature must appear there, and only there.

### Output Template
**Problem Description:**
<Concise natural-language description instantiating the Computational Objective and Processing Logic Constraints>

**Input:**
<Formal description derived from the Input Interface>

**Output:**
<Formal description derived from the Output Specification>

**Constraints & Requirements:**
* <Each Processing Logic Constraint as a concrete requirement>
* <Any implicit constraints required by the Tool Dependency Set>

**Header Code:**
```python
<header_code>
```

### Given Elements
{elements}

### Your Designed Problem
""".strip()


generate_solution_test_prompt = """
## Task
You are given a tool-calling code problem. Your task is to generate both the `solution code` and the `test code` in pytest for that problem.

## Output format
<|Solution Begin|>
[Solution Code in Python]
<|Solution End|>
<|Test Code Begin|>
[Test Code in Pytest]
<|Test Code End|>

## Example
<|Solution Begin|>
```python
import math

def add(a, b):
    \"\"\"
    Return the sum of a and b.
    \"\"\"
    return math.fsum([a, b])
```
<|Solution End|>
<|Test Code Begin|>
```python
# Sanity Check
def test_add_basic():
    assert add(1, 2) == 3.0

# Edge Cases
def test_add_with_zero():
    assert add(0, 5) == 5.0
    assert add(0, 0) == 0.0

def test_add_negative_numbers():
    assert add(-1, -2) == -3.0
    assert add(-1, 1) == 0.0

# Extreme Cases
def test_add_large_numbers():
    large = 10**18
    assert add(large, large) == float(2 * large)

def test_add_float_precision():
    # math.fsum should handle precision better than naive addition
    result = add(1e16, 1.0)
    assert result == math.fsum([1e16, 1.0])

# Boundary of Input Domain
def test_add_mixed_int_float():
    assert add(1, 2.5) == 3.5
```
<|Test Code End|>

## Tool-Calling Code Problem
{problem}
""".strip()
