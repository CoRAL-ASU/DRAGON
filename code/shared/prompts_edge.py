# prompts.py


MAIN_PROMPT_TEMPLATE = """
Task:
You are given:
A scientific or schematic diagram (e.g., flowchart, life cycle, bar chart, map, circuit etc.)
A question and its correct answer
Your job is to visually ground the answer by returning all visual elements (regions) that are essential for identifying the answer. This includes:
The visual region that directly represents the answer (e.g., diagram segment, colored arc, object shape, component).
The textual label or annotation naming the answer.
Any supporting visual evidence (e.g., legends, arrows, connections, comparisons).

Return all visual regions (bounding boxes) involved in:
Identifying or depicting the answer
Visually reasoning about the answer — i.e., any elements used to compare, trace, connect, or verify the answer

Output Format (STRICT):
Answer: <ID1>/<x,y,w,h>, <ID2>/<x,y,w,h>, ...
Explanation: Each region corresponds to a key step in answering the question. [ID1] shows the answer itself, [ID2] marks the question subject, and [ID3] represents visual evidence used to compare or verify the answer (e.g., legend, arrow, label).
Use only region IDs and their coordinates in the Answer field.
Use clear justification in the Explanation: say why each region is needed, and how it contributes to answering.

What to Ground:
Include BBOXes for:
The answer region must include both:
- The visual depiction of the answer (e.g., shaded zone, component, physical region)
- The textual label or marker (e.g., “Mantle”)
- Also ground any other regions used in comparison, flow, or spatial logic.
The question subject or anchor (e.g., North Carolina, mitochondria, resistor A)
All supporting visual evidence, such as:
Neighboring nodes or states
Color legends or value encodings
Arrows, flows, time steps, or labels
Axes or chart features used to compare or infer
Only include regions a human would reasonably look at to answer the question based on the diagram.

Input:
Question ID: {Question_ID_STRING}
Question: {Question_String}
Choices: {Choices string}  
Answer: {Answer string}
Image metadata: {height}×{width}

Example Output:
{Worked Example}

Common Failure to Avoid:
Do not return only the text label of the answer.
You must also ground the visual region (e.g., the actual colored section, object shape, or diagram part) that corresponds to the answer — even if the label alone seems sufficient.

IMPORTANT: Do NOT explain your reasoning step-by-step. Do NOT use "## Step 1:" or similar formats. Output ONLY the Answer and Explanation in the exact format shown above. Start your response directly with "Answer:" followed by the bounding boxes.
""".strip()


def build_prompt(
    question_id: str,
    question: str,
    choices_str: str,
    answer: str,
    height: int,
    width: int,
) -> str:
    return MAIN_PROMPT_TEMPLATE.format(
        Question_ID_STRING=question_id,
        Question_String=question,
        Choices_string=choices_str,   # NOTE: matches your placeholder "{Choices string}"? see note below
        **{
            "Choices string": choices_str,  # safe for your exact placeholder
            "Answer string": answer,
            "Worked Example": WORKED_EXAMPLE,
            "height": height,
            "width": width,
        }
    )
