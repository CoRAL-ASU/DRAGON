# prompts_sage.py
"""
Two-stage "Select then Ground" prompting for Diagram Attribution.

Stage 1 (SELECT):  LLM freely identifies visual elements needed to answer.
Stage 2 (GROUND):  Given Stage 1 output, LLM returns bounding boxes.

Fully non-deterministic — no structured format enforced in Stage 1;
the LLM reasons freely. Only Stage 2 output follows strict Answer format.

Worked examples live in worked_examples.py.
"""

from worked_examples import (
    WORKED_EXAMPLE_SELECT,
    DATASET_TO_WORKED_KEY,
    resolve_key,
)

# ── Stage 1: SELECT prompt ─────────────────────────────────────────────────

STAGE1_PROMPT_TEMPLATE = """
Task — Identify Visual Evidence (Stage 1 of 2):
Given this diagram, a question, and its answer, list EVERY visual element a human would inspect to verify the answer. Do NOT predict bounding boxes yet.

For each element, give a short ID (e.g., B_01, T_01, R_01) and a one-line description.

Be EXHAUSTIVE — cover at minimum:
1. ANSWER REGION(S): the visual area(s) that directly depict/contain the answer (bars, slices, states, nodes, lines, components, map regions, etc.). List EACH one separately.
2. LABELS & TEXT: every textual annotation, axis tick, title, or legend entry that names or quantifies an answer region.
3. CONTEXT & EVIDENCE: legends, colour keys, axes, scale bars, arrows, connectors, neighbouring regions needed for comparison, and any other element that supports or disambiguates the answer.

Err on the side of MORE elements — it is always better to include too many than too few.

Input:
Question ID: {Question_ID_STRING}
Question: {Question_String}
Choices: {Choices string}
Answer: {Answer string}
Image metadata: {height}×{width}

Example:
{Worked Example}

Start your response directly with "Elements:" followed by the list.
""".strip()


# ── Stage 2: GROUND prompt ─────────────────────────────────────────────────

STAGE2_PROMPT_TEMPLATE = """
Task — Bounding Box Grounding (Stage 2 of 2):
You previously identified these visual elements for answering a question about this diagram:

--- YOUR ELEMENT LIST ---
{stage1_elements}
--- END ---

Now provide a bounding box (x, y, w, h) in pixels for EVERY element above. (x, y) = top-left corner, (w, h) = width and height. Tightly enclose each element.

Do NOT skip any element from the list. If you notice additional answer-relevant elements in the image that were missed in Stage 1, add them too (use new IDs). More boxes are always better than fewer.

Output Format (STRICT):
Answer: <ID1>/<x,y,w,h>, <ID2>/<x,y,w,h>, ...
Explanation: Brief reason for each box.

Example: Answer: B_01/120,45,200,80, T_01/50,10,300,25

Image metadata: {height}×{width}

Start your response directly with "Answer:".
""".strip()


# ── Combined single-pass prompt (Stage 1 + Stage 2 in one turn) ───────────

COMBINED_PROMPT_TEMPLATE = """
Task — Identify Visual Evidence and Ground Bounding Boxes:
Given this diagram, a question, and its answer, do BOTH steps in one response:

STEP 1 — List EVERY visual element a human would inspect to verify the answer.
For each element, give a short ID (e.g., B_01, T_01, R_01) and a one-line description.
Cover at minimum:
1. ANSWER REGION(S): visual areas that directly depict/contain the answer (bars, slices, states, nodes, lines, components, map regions, etc.). List EACH separately.
2. LABELS & TEXT: every textual annotation, axis tick, title, or legend entry that names or quantifies an answer region.
3. CONTEXT & EVIDENCE: legends, colour keys, axes, scale bars, arrows, connectors, neighbouring regions for comparison.

STEP 2 — Provide a bounding box (x, y, w, h) in pixels for EVERY element from Step 1.
(x, y) = top-left corner, (w, h) = width and height. Tightly enclose each element.
Do NOT skip any element. If you spot additional answer-relevant elements not in your Step 1 list, add them (use new IDs). More boxes are always better than fewer.

Input:
Question ID: {Question_ID_STRING}
Question: {Question_String}
Choices: {Choices string}
Answer: {Answer string}
Image metadata: {height}×{width}

Example:
{Worked Example}

Output Format (STRICT):
Elements: <ID1> — description, <ID2> — description, ...
Answer: <ID1>/<x,y,w,h>, <ID2>/<x,y,w,h>, ...
Explanation: Brief reason for each box.

Start your response with "Elements:".
""".strip()


def build_combined_two_stage_prompt(
    dataset: str,
    question_id: str,
    question: str,
    choices_str: str,
    answer: str,
) -> str:
    """Single-pass prompt: identify elements AND ground bounding boxes in one turn."""
    key = resolve_key(dataset)
    worked_ex = WORKED_EXAMPLE_SELECT.get(key, "")

    template = COMBINED_PROMPT_TEMPLATE.replace("{height}", "{{height}}").replace("{width}", "{{width}}")

    return template.format(
        **{
            "Question_ID_STRING": question_id,
            "Question_String": question,
            "Choices string": choices_str,
            "Answer string": answer,
            "Worked Example": worked_ex,
        }
    )


# ── Public builders ─────────────────────────────────────────────────────────

def build_stage1_prompt(
    dataset: str,
    question_id: str,
    question: str,
    choices_str: str,
    answer: str,
) -> str:
    key = resolve_key(dataset)
    worked_ex = WORKED_EXAMPLE_SELECT.get(key, "")

    template = STAGE1_PROMPT_TEMPLATE.replace("{height}", "{{height}}").replace("{width}", "{{width}}")

    return template.format(
        **{
            "Question_ID_STRING": question_id,
            "Question_String": question,
            "Choices string": choices_str,
            "Answer string": answer,
            "Worked Example": worked_ex,
        }
    )


def build_stage2_prompt(
    dataset: str,
    question_id: str,
    question: str,
    choices_str: str,
    answer: str,
    stage1_elements: str,
) -> str:
    template = STAGE2_PROMPT_TEMPLATE.replace("{height}", "{{height}}").replace("{width}", "{{width}}")

    return template.format(
        **{
            "stage1_elements": stage1_elements,
        }
    )
