from typing import List, Dict, Tuple

general_generation_based_on_reference_prompt = """
## Task: Create a Challenging and Modified Version of a Reference Task

Given one or more **reference tasks** along with their **ground truth answers**, your goal is to design a **new, more challenging task** by making **controlled perturbations** to the original. The modifications should **increase reasoning depth, introduce extra constraints, or add multi-step dependencies** while keeping the problem **self-contained and solvable**.

You must preserve the **core domain or reasoning type** of the reference (e.g., if it’s a logic puzzle, keep it a logic puzzle) but ensure the **surface content and structure are new**. You may:
- Add additional constraints or intermediate steps
- Replace elements with analogous but more complex structures
- Introduce distractors or traps that require careful reasoning
- Change numerical values, symbolic rules, or conditions to increase difficulty
- Combine multiple references into one composite, coherent challenge

---

### Task Requirements:

- The modified task must be:
  * **Self-contained** and clearly described
  * **Significantly different in surface form** from the reference, but same reasoning type
  * **More challenging** — requiring additional steps or deeper analysis than the reference
  * **Deterministic** or tightly constrained
  * **Free from cultural bias, real-time info, or factual recall**

- Accepted Domains:
  * Logic puzzles, paradoxes, analogical reasoning
  * Pattern-based math or symbolic challenges
  * Spatial or constraint-based planning
  * Structured or constrained writing
  * Multi-agent or multi-state reasoning
  * Instruction-following with hidden traps

- Avoid:
  * Pure trivia or subjective writing
  * Ambiguity or taste-based prompts
  * Dependency on web or recent events
  * Unsuitable open-endedness

---

### Output Format:

- `<think>`: Describe your reasoning for how you modified the reference, what reasoning it tests, and why it’s harder.
- `<question>`: The final new task to present to the test subject.

### Output Template:

```<think>
[Explain modifications from the reference and why they increase difficulty — e.g., added constraints, distractors, multi-step dependencies.]
</think>

<question>
[Write the modified, more challenging task in full, ready to present. Ensure it is solvable without external info.]
</question>

### Reference Questions:
"""

general_generation_prompt = """
## Task: Create a Challenging and Original Task

Design a new and intellectually demanding task that tests **complex reasoning, creative thinking, structured planning, or deep understanding**. The task should be suitable for evaluation in general intelligence, reasoning benchmarks, instruction following, or alignment assessments.

You may design a task that resembles a quiz, puzzle, constrained writing, or symbolic reasoning prompt. Focus on structure, challenge, and clarity — not trivia or stylistic flair.

This prompt is intended to help construct the **task itself**, not example answers or input/output.

---

### Task Requirements:

- The task must be:
  * **Self-contained** and clearly described
  * **Non-trivial**, requiring multiple reasoning steps, constraints, or synthesis
  * **Deterministic** or tightly constrained (even if open-ended in form)
  * **Free from cultural bias, real-time information, or factual recall**

- Accepted Domains include:
  * Logic puzzles, paradoxes, analogical reasoning
  * Pattern-based math or symbolic challenges
  * Spatial planning or constraint problems
  * Structured or constrained writing
  * Agent-based planning, recursive state modeling
  * Instruction following with internal traps

- Avoid:
  * Trivia questions or subjective writing
  * Ambiguous or taste-based open-ended prompts
  * Any dependency on web access or recent knowledge
  * Tasks with no clear solvability path

---

### Output Format:

You must structure your response in two blocks using the tags below:

- The `<think>` section should contain your rationale or cognitive goal for the task. (Optional)
- The `<question>` section should contain the full task shown to the test subject.

### Output Template:

```<think>
[Why is this task interesting? What reasoning type does it test — deduction, simulation, abstraction, generation, contradiction detection, etc.?]
</think>

<question>
[Write the task as it would be presented. Use clear formatting. This should be solvable without needing input/output examples.]
</question>

### Reference Questions:
"""

general_prediction_prompt = """
## Task: Generate a High-Quality Response to a Given Task

You will be given a cognitive, creative, logical、mathematical、or planning-related task. Your job is to generate a complete, high-quality response that satisfies the task’s constraints and demonstrates clear, structured reasoning or creativity.

### Instructions:
- Carefully read and understand the task.
- Think step by step — break down the task, simulate it mentally if needed, and reason through constraints.
- Then directly write your final response (no need to restate or reformat the task).
- Do **not** separate your answer into sections like "task", "think", or "response". Just give your best final answer with your reasoning embedded naturally or implied by structure.
- Your output should:
  * Be **correct** or **plausibly optimal**, given the task
  * **Fulfill all constraints** in the task
  * Be **clear, structured**, and **non-trivial**
  * Avoid fluff, vagueness, or randomness

### Good Response Traits:
- For reasoning tasks: shows logical progression or result
- For generation tasks: respects the given constraints (style, length, content)
- For math/logic/planning: includes a final answer that could be evaluated
- For creative tasks: coherent, original, and well-scaffolded

"""

general_judge_question_answer_prompt = """

"""

general_judge_answer_prompt = """
## Task: Provide a High-Quality Score for a given answer to a question

You will be given a question and an answer. Your job is to evaluate the quality of the answer based on its correctness, clarity, and relevance to the question.

### Instructions:
Consider the following criteria when evaluating:
- Is the solution correct and accurate?
- Is it complete and comprehensive?
- Does it properly address the question?
- Is the reasoning clear and logical?
- Determine what score is most appropriate

"""

# [TODO] above prompts may need to be modified in the future
# maybe <think></think> tags for all prompts?

code_input_prompt = """
## Task: Create a Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input

Using the reference code snippets provided below as examples, design a new and unique Python code snippet that demands deep algorithmic reasoning to deduce one possible input from a given output. Your submission should include both a code snippet and test input pair, where the input will be plugged into the code snippet to produce the output, which that function output be given to a test subject to come up with any input that will produce the same function output. This is meant to be an I.Q. test.

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi step reasoning
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with: ```python
  def f(...):
      # your code here
      return ...
  ```
- Format your input with: ```input
  arg1, arg2, ...
  ```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{'age': 20, 'city': 'New York'}}
```

### Evaluation Criteria:
- Executability, your code should be executable given your input
- Difficulty in predicting the output from your provided input and code snippet. Focus on either algorithmic reasoning or logic complexity. For example, you can define complex data structure classes and operate on them like trees, heaps, stacks, queues, graphs, etc, or use complex control flow, dynamic programming, recursions, divide and conquer, greedy, backtracking, etc
- Creativity, the code needs to be sufficiently different from the provided reference snippets
- Restricted usage of certain keywords and packages, you are not allowed to use the following words in any form, even in comments: <|BANNED_KEYWORDS|>

First, carefully devise a clear plan: e.g., identify how your snippet will be challenging, distinct from reference snippets, and creative. Then, write the final code snippet and its inputs.

### Reference Code Snippets:
"""

code_output_prompt = """
## Task: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input

Using the reference code snippets provided below as examples, design a new and unique Python code snippet that demands deep algorithmic reasoning to deduce the output from the input. Your submission should include a code snippet and a test input pair, where the input will be plugged into the code snippet to produce the output. The input will be given to a test subject to deduce the output, which is meant to be an I.Q. test.

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi step reasoning
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_input_from_snippet_prompt}{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{'age': 20, 'city': 'New York'}}
```

### Evaluation Criteria:
- Executability, your code should be executable given your input
- Difficulty in predicting your ```input``` from 1) your ```python``` code and 2) the deterministic ```output``` that will be obtained from your ```input```. Focus on either algorithmic reasoning or logic complexity. For example, you can define complex data structure classes and operate on them like trees, heaps, stacks, queues, graphs, etc, or use complex control flow, dynamic programming, recursions, divide and conquer, greedy, backtracking, etc
- Creativity, the code needs to be sufficiently different from the provided reference snippets
- Restricted usage of certain keywords and packages, you are not allowed to use the following words in any form, even in comments: <|BANNED_KEYWORDS|>

First, carefully devise a clear plan: e.g., identify how your snippet will be challenging, distinct from reference snippets, and creative. Then, write the final code snippet and its inputs.

### Reference Code Snippets:
"""

code_error_prompt = """
## Task: Create a New Python Code Snippet (where custom classes are allowed, which should be defined at the top of the code snippet) with one Matching Input

Using the reference code snippets provided below as examples, design a new and unique Python code snippet that demands deep algorithmic reasoning to deduce what type of error will be raised when the code is executed. Your submission should include a code snippet and a test input pair, where the input will be plugged into the code snippet to produce the error. You can also choose to include a custom error type in your code snippet. However, the code can also be designed to raise no error. The input and the code will be given to a test subject to deduce the error type, which is meant to be an I.Q. test.

### Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- Make the snippet require state tracking across multiple data transformations, ensuring the task requires long multi step reasoning
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f`, anything after will be removed
{remove_after_return_prompt}
### Input Requirements:
- Provide exactly one test input for your function
- Format multiple arguments with commas between them
- Remember to add quotes around string arguments

### Formatting:
- Format your code with:
```python
def f(...):
    # your code here
    return ...
```
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```python
def f(name: str, info: dict):
    # code logic here
    return result
```

```input
'John', {{'age': 20, 'city': 'New York'}}
```

### Evaluation Criteria:
- Executability, your code should be executable given your input
- Difficulty in deducing the error type (or no error) from 1) your ```python``` code and ```input```. Focus on either algorithmic reasoning or logic complexity. For example, you can define complex data structure classes and operate on them like trees, heaps, stacks, queues, graphs, etc, or use complex control flow, dynamic programming, recursions, divide and conquer, greedy, backtracking, etc
- Creativity, the code needs to be sufficiently different from the provided reference snippets
- Restricted usage of certain keywords and packages, you are not allowed to use the following words in any form, even in comments: <|BANNED_KEYWORDS|>
<|BANNED_ASSERTION_KEYWORDS|>
First, carefully devise a clear plan: e.g., identify how your snippet will be challenging, distinct from reference snippets, and creative. Then, write the final code snippet and its inputs. The code needs to compile and pass AST checks, but it is intended to raise an error or not.

### Reference Code Snippets:
"""

code_function_prompt = """
## Task: Output {num_inputs} Inputs that can be plugged into the following Code Snippet to produce diverse Outputs, and give a message related to the given snippet.

Using the code snippet provided below, design {num_inputs} inputs that can be plugged into the code snippet to produce a diverse set of outputs. A subset of your given input and its deterministically produced outputs will be given to a test subject to deduce the function, which is meant to be an I.Q. test. You can also leave a message to the test subject to help them deduce the code snippet.

### Input Requirements:
- Provide {num_inputs} valid inputs for the code snippet
- For each input, format multiple arguments with commas between them
- Remember to add quotes around string arguments
- Each input should be individually wrapped in ```input``` tags

### Message Requirements:
- Leave a message to the test subject to help them deduce the code snippet
- The message should be wrapped in ```message``` tags
- The message can be in any form, can even be formed into a coding question, or a natural language instruction what the code snippet does
- You cannot provide the code snippet in the message

### Formatting:
- Format your input with:
```input
arg1, arg2, ...
```

### Example Format:
```input
'John', {{'age': 20, 'city': 'New York'}}
```
```input
'Sammy', {{'age': 37, 'city': 'Los Angeles'}}
```

### Evaluation Criteria:
- Executability, your code should be executable given your inputs
- Coverage, the inputs and outputs should cover the whole input space of the code snippet, able to deduce the code snippet from the inputs and outputs
- Creativity, the inputs need to be sufficiently different from each other
- The overall selection of inputs and message combined should be challenging for the test subject, but not impossible for them to solve
First, carefully devise a clear plan: e.g., understand the code snippet, then identify how your proposed inputs have high coverage, and why the inputs will be challenging and creative. Then, write the inputs and message. Remember to wrap your inputs in ```input``` tags, and your message in ```message``` tags.

### Code Snippet:
```python
{snippet}
```
"""

code_input_predictor_prompt = """
# Task: Provide One Possible Input of a Python Code Snippet Given the Code and Output
Given the following Code Snippet and the Output, think step by step then provide one possible input that produced the output. The input needs to be wrapped in ```input``` tags. Remember if an argument is a string, wrap it in quotes. If the function requires multiple arguments, separate them with commas.

# Code Snippet:
```python
{snippet}
```

# Output:
```output
{output}
```

# Output Format:
```input
arg1, arg2, ...
```
# Example Output:
```input
'John', {{'age': 20, 'city': 'New York'}}
```
"""

code_output_predictor_prompt = """
# Task: Deduce the Output of a Python Code Snippet Given the Code and Input
Given the following Code Snippet and the Input, think step by step then deduce the output that will be produced from plugging the Input into the Code Snippet. Put your output in ```output``` tags. Remember if the output is a string, wrap it in quotes. If the function returns multiple values, remember to use a tuple to wrap them.

# Code Snippet:
```python
{snippet}
```

# Input:
```input
{input_args}
```

# Example Output:
```output
{{'age': 20, 'city': 'New York'}}
```
"""

code_error_predictor_prompt = """
# Task: Deduce the Error Type of a Python Code Snippet Given the Code and Input
Given the following Code Snippet and the Input, think step by step to deduce the error type that will be raised when the code is executed. Put your final output in ```output``` tags. If there are no errors, put "NoError" in the ```output``` tags.

# Code Snippet:
```python
{snippet}
```

# Input:
```input
{input_args}
```

# Example Output:
```output
ValueError
```
"""

code_suffix = "\nf(<|YOUR INPUT WILL BE PLUGGED HERE|>)"

code_function_predictor_prompt = """
# Task: Deduce the Function that Produced the Outputs from the Inputs
Given a set of input/output pairs and a message that describes the function, think through the problem step by step to deduce a general code snippet. This code should produce the hidden outputs from the hidden inputs, matching the original data-generating code that created the input/output pairs. Place your final answer inside python tags! It may be helpful to work through each input/output pair individually to test your function. If your function doesn’t work as expected, revise it until it does. The final code snippet will be used to evaluate your response, which is wrapped in ```python``` tags.

# Code Requirements:
- Name the entry function `f` (e.g., `def f(...): ...`), you can have nested definitions inside `f`
- Ensure the function returns a value
- Include at least one input parameter
- Make the function deterministic
- AVOID THE FOLLOWING:
  * Random functions or variables
  * Date/time operations
  * I/O operations (reading files, network requests)
  * Printing or logging
  * Any external state
- Ensure execution completes within 10 seconds on a modern CPU
- All imports and class definitions should be at the very top of the code snippet
- The snippet should end with a return statement from the main function `f()`, anything after will be removed

# Input and Output Pairs:
{input_output_pairs}

# Message:
```message
{message}
```

# Example Output:
```python
def f(a):
    return a
```

Name your entry function `f()`!!!
"""

# composite_requirements_prompt = "\n[IMPORTANT CRITERIA!!!] The main function `f` MUST make calls to ALL these functions {function_names} in its body, and you SHOULD NOT provide the definition of {function_names} in your output code snippet. You should first reason step by step about what these functions, {function_names}, do, then write the code snippet.\n" + '\n### The Functions that Must ALL be Called in your Code Snippet: \n```python\n{composite_functions}\n```\n'

composite_requirements_prompt = "\n[IMPORTANT CRITERIA!!!] The main function `f` MUST make calls to ALL these functions {function_names} in its body, and you SHOULD NOT provide the definition of {function_names} in your output code snippet. The function `f` should build on top of {function_names} with extra functionalities, not just a simple wrapper. You should first reason step by step about what these functions, {function_names}, do, then write the code snippet.\n" + '\n### The Functions that Must ALL be Called in your Code Snippet: \n```python\n{composite_functions}\n```\n'

remove_input_from_snippet_prompt = "- Do not have the test input anywhere in the code snippet, provide it in the input section."

remove_singleton_variables_prompt = "- All variable declarations must be inside the main function `f` or within functions `f` make calls to. Any variables declared outside of functions will be removed.\n"

def get_general_generation_with_reference_prompt(
        reference_questions: List[Dict[str, str]],
) -> str:
    # Generate a general prompt for the generator based on reference questions
    reference_questions_string = ""
    for i, question in enumerate(reference_questions):
        reward_model = question.get('reward_model', {})
        if reward_model.get('ground_truth') is not None:
            ground_truth = reward_model['ground_truth']
        else:
            ground_truth = "N/A"
        reference_questions_string += f"<question>\n{question['question']}\n</question>\n\n Ground Truth Answer: {ground_truth}\n\n"

    return general_generation_based_on_reference_prompt + reference_questions_string + "\n### Your Task:\nCreate a Challenging and Modified Version of the Reference Task. Remember to structure your response in the specified format.\n\n---\n\n### Output Template:\n```<think>\n[Your reasoning about the task]\n</think>\n\n<question>\n[Your modified task]\n</question>\n\n<answer>\n[Your complete solution to verify the task is solvable]\n</answer>```"
def get_general_generator_prompt(
        reference_questions: List[Dict[str, str]],
) -> str:
    # Generate a general prompt for the generator
    reference_questions_string = ""
    for i, question in enumerate(reference_questions):
        reference_questions_string += f"<question>\n{question['question']}\n</question>\n"

    return general_generation_prompt + reference_questions_string + "\n### Your Task:\nDesign a new and unique task that meets the requirements outlined above. Remember to structure your response in the specified format.\n\n---\n\n### Output Template:\n```<think>\n[Your reasoning about the task]\n</think>\n\n<question>\n[Your designed task]\n</question>\n\n<answer>\n[Your complete solution to verify the task is solvable]\n</answer>```"

def get_general_predictor_prompt(
        question: str,
) -> str:
    # Generate a general prompt for the predictor
    return general_prediction_prompt + f"\n\n### Question:\n{question}\n\n---\n\n### Output Template:\n[Your final answer to the question, structured and clear, without restating the question]"

def get_general_judger_prompt(
        question: str,
        answer: str,
        prompt_manager=None,
) -> str:
    # Use prompt manager if available, otherwise fall back to static prompt
    if prompt_manager:
        # Get the appropriate judge template based on infer_together setting
        judge_template = prompt_manager.get_judge_instruction(prompt_type="answer")
        # Format with question and answer
        try:
            return judge_template.format(question=question, answer=answer)
        except (KeyError, ValueError):
            # If template formatting fails, fall back to appending
            return f"{judge_template}\n\n### Question:\n{question}\n\n### Answer:\n{answer}"
    else:
        # Fallback to original static prompt
        return general_judge_answer_prompt + f"\n\n### Question:\n{question}\n\n---\n\n### Answer:\n{answer}\n\n---\n\n### Output Template:\n[Your score for the answer to the question, without restating the question or the answer. Use an integer scale from 1 to 10, where 1 is the lowest quality and 10 is the highest quality]"

def get_code_problem_generator_prompt(
    problem_type: str,
    reference_snippets: List[Dict[str, str]],
    banned_keywords: List[str],
    banned_assertion_keywords: List[str],
    composite_functions: List[str] = None,
    remove_after_return: bool = False,
    num_inputs: int = 10,
    remove_input_from_snippet: bool = False,
) -> str:
    # assert not (remove_after_return and not remove_input_from_snippet)
    composite_functions = list(composite_functions)
    snippet_string = ""
    if problem_type != 'code_f':
        output_key = 'output' if problem_type != 'code_e' else 'error'
        for i, snippet in enumerate(reference_snippets):
            snippet_string += f"<snippet_{i}>\n```python\n{snippet['snippet']}\n```\n```input\n{snippet['input']}\n```\n```{output_key}\n{snippet['output']}\n```\n</snippet_{i}>\n"
    if problem_type == "code_i":
        return code_input_prompt.format(
            remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
            remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else '')
        ).replace(
            '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
        ) + snippet_string + (
            composite_requirements_prompt.format(
                function_names=', '.join([f'`g_{i}`' for i in range(len(composite_functions))]),
                composite_functions="\n".join([d['snippet'] for d in composite_functions])
            ) if composite_functions else '\n'
        )
    elif problem_type == "code_o":
        return code_output_prompt.format(
            remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
            remove_input_from_snippet_prompt=(remove_input_from_snippet_prompt if remove_input_from_snippet else '')
        ).replace(
            '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
        ) + snippet_string + (
            composite_requirements_prompt.format(
                function_names=', '.join([f'`g_{i}`' for i in range(len(composite_functions))]),
                composite_functions="\n".join([d['snippet'] for d in composite_functions])
            ) if composite_functions else '\n'
        )
    elif problem_type == "code_f":
        return code_function_prompt.format(
            num_inputs=num_inputs,
            snippet=reference_snippets[0]['snippet'] + code_suffix,
        )
    elif problem_type == "code_e":
        if banned_assertion_keywords:
            assertion_keywords_string = '- The following error handling keywords are not allowed to be used in the code snippet: ' + ', '.join(banned_assertion_keywords) + '\n'
        else:
            assertion_keywords_string = '\n'
        return code_error_prompt.format(
            remove_after_return_prompt=(remove_singleton_variables_prompt if remove_after_return else '\n'),
        ).replace(
            '<|BANNED_KEYWORDS|>', ', '.join(banned_keywords)
        ).replace(
            '<|BANNED_ASSERTION_KEYWORDS|>', assertion_keywords_string
        ) + snippet_string + (
            composite_requirements_prompt.format(
                function_names=', '.join([f'`g_{i}`' for i in range(len(composite_functions))]),
                composite_functions="\n".join([d['snippet'] for d in composite_functions])
            ) if composite_functions else '\n'
        )
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")

def get_code_problem_predictor_prompt(problem_type: str, snippet: str, input_args: str = None, output: str = None, message: str = None, input_output_pairs: List[Tuple[str, str]] = None) -> str:
    if problem_type.endswith("code_i"):
        return code_input_predictor_prompt.format(snippet=snippet, output=output)
    elif problem_type.endswith("code_o"):
        return code_output_predictor_prompt.format(snippet=snippet, input_args=input_args)
    elif problem_type.endswith("code_f"):
        input_output_pairs_string = ""
        for i, (input, output) in enumerate(input_output_pairs):
            input_output_pairs_string += f"```input_{i}\n{input}\n```\n```output_{i}\n{output}\n```\n"
        return code_function_predictor_prompt.format(input_output_pairs=input_output_pairs_string, message=message)
    elif problem_type.endswith("code_e"):
        return code_error_predictor_prompt.format(snippet=snippet, input_args=input_args)
    else:
        raise ValueError(f"Invalid problem type: {problem_type}")
