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
Please analyze the following data science problem according to the guidelines below. The output should follow the format below (do not add any other notes or explanations):
<answer>
Data Schema:
Task Goal:
Output Contract:
Implementation Environment:
Behavioral Constraints:
</answer>

### Guidelines
1. Data Schema
    * **Definition:** A precise, abstract description of the input data the code will consume: its origin (file, DB, in-memory), high-level structure (table/array/tensor/graph), dimensions or shape signatures, named fields or column types, and any distributional or ordering properties that affect processing.
    * **Role:** States what the program receives so implementers know how to parse, index, and validate inputs; without it the operation cannot be applied correctly.
    * **Variation Axes:** source (CSV/JSON/SQL/in-memory), container type (DataFrame / ndarray / sparse matrix / list / dict), schema detail level (full typed schema vs. loose shape), size scale (small/medium/large/streaming), ordering guarantees (sorted/partitioned/grouped/random).

2. Task Goal
    * **Definition:** A concise, testable description of the transformation, computation, or analysis to perform on the input data. It describes the expected semantic outcome (e.g., "reorder rows by index list", "normalize each column in-place", "reshape a 1-D sequence into a 2-D matrix").
    * **Role:** Provides the objective that determines what code must do; without it there is no target behavior to implement.
    * **Variation Axes:** transformation type (filter/aggregate/transform/reorder/reshape/compute stat), granularity (column-wise/row-wise/matrix-level/element-wise), in-place vs. functional (mutating or returning new object), deterministic vs. randomized, single-step vs. pipeline of steps.

3. Output Contract
    * **Definition:** An explicit specification of the result: types, shapes, naming/variable expectations, ordering of outputs, side-effects (files written, in-place mutation), and the acceptance criteria (what constitutes a correct result).
    * **Role:** Defines how the success of the Task Goal is observed and integrated downstream; it removes ambiguity about return types and side-effects.
    * **Variation Axes:** return style (value/tuple/None + side-effects), naming (specific variable name required vs. any), strictness of shape/type (exact shape vs. compatible), required persistence (save to file/DB vs. ephemeral), error signaling mode (exceptions/error codes).

4. Implementation Environment
    * **Definition:** The technical context in which the solution must run: permitted libraries and APIs.
    * **Role:** Constrains feasible solutions and guides use of tool-specific APIs; without it solutions may use unsupported constructs or libraries.
    * **Variation Axes:** allowed libraries and formats.

5. Behavioral Constraints
    * **Definition:** Non-functional and precondition requirements that affect algorithm choice: mutability requirements, numerical stability, memory/time complexity targets, guaranteed invariants in inputs (e.g., "columns contain non-negative numbers"), and failure modes to avoid.
    * **Role:** Ensures the implementation respects important operational and correctness constraints that the Task Goal alone does not cover.
    * **Variation Axes:** mutability (must/should not mutate input), complexity bounds (O(n)/O(n log n)/approximate), numeric precision needs (float32/64, tolerance), concurrency safety, expected input cleanliness (may contain NaNs/missing/duplicates).

### Data Science Code Problem
{problem}

### Solution of the Data Science Code Problem
```python
{solution}
```

### Analysis
""".strip()


design_new_elements = """
You are an expert in data science code task abstraction and schema-level task design.
You are given:
- One randomly sampled Task Goal
- One randomly sampled Data Schema
- Three reference sets, each consisting of:
    - Output Contract
    - Implementation Environment
    - Behavioral Constraints
Your task is to design a completely new data science code task at the element level, by inferring a novel and coherent combination of task elements, not by copying or minimally editing the references.
The output should follow the format below (do not add any other notes or explanations):
<answer>
Data Schema:
Task Goal:
Output Contract:
Implementation Environment:
Behavioral Constraints:
</answer>

### Core Requirements

1. **Element-level generation only**
    * Do NOT write a concrete problem statement or code.
    * Do NOT reuse wording, structure, or semantics from any single reference set.
    * Operate strictly at the level of abstract task elements.

2. **Five-element completeness**
    You must generate **exactly five elements**, one for each of the following:
    * Data Schema
    * Task Goal
    * Output Contract
    * Implementation Environment
    * Behavioral Constraints

3. **Consistency constraints**
    * Any Task Goal is valid with any Data Schema provided the Output Contract and Implementation Environment are consistent to ensure API compatibility (e.g., a matrix normalization goal must pair with a container type that supports columns).
    * Behavioral Constraints must be compatible with Implementation Environment (e.g., a constraint requiring in-place mutation is invalid if environment disallows mutating APIs).

4. **Novel recombination**
    * Treat the three reference sets as **design signals**, not templates.
    * The resulting element set should be plausibly generatable by recombining ideas,
      but **must not align exactly with any reference along more than one element**.
    * Different Task Goals can be legally combined with different Data Schemas (e.g., change from pandas DataFrame to Spark DataFrame → update environment and acceptable complexity) (e.g., from "reorder rows" → "group and aggregate by column").
    * Produce variants by changing Variation Axes independently: change allowed libraries in Implementation Environment while keeping Data Schema and Task Goal identical; the Output Contract and Examples must be updated only if necessary.

5. **Data science emphasis**
    * The task must be inherently data-centric (e.g., transformation, aggregation, statistical computation, reshaping, validation).
    * If the Data Schema were removed or altered, the task should lose its defining meaning.

### Given Data Schema
{data_schema}

### Given Task Goal
{task_goal}

### Reference Sets
{combinations_1}

{combinations_2}

{combinations_3}

### New Elements
""".strip()


design_task_prompt = """
You are an expert in **data science code task synthesis**.

You are given a complete task specification expressed as **five abstract task elements**:

* Data Schema
* Task Goal
* Output Contract
* Implementation Environment
* Behavioral Constraints

Your task is to **synthesize a single, complete data science code problem** that strictly follows the required template and **faithfully instantiates all five elements**.

### Core Rules
1. **Element Fidelity**
    * All five elements must be fully and explicitly reflected.
    * Do not add, remove, or reinterpret any requirement.

2. **No Meta or Reasoning Text**
    * Output only a finished problem statement.
    * No explanations, hints, or commentary.

3. **Mandatory Implementation Environment Usage**
    * The task must inherently require the specified implementation environment.
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
You are given a data science code problem. Your task is to generate both the `solution code` and the `test code` in pytest for that problem.

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

## Data Science Code Problem
{problem}
""".strip()
