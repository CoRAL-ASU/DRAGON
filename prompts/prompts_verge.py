# prompts_verge.py
"""
Iterative Grounding Refinement (Self-Correction) prompts.

Pipeline (2 steps):
  Step 1 — Initial Grounding (standard prompt from prompts.py).
  Step 2 — Verify & Refine: re-examine image, fix coordinates,
           add ≤3 missing elements, NEVER remove boxes.
"""

# ── Step 2: Verify & Refine ───────────────────────────────────────────────

VERIFY_AND_REFINE_TEMPLATE = """
Task — Verify & Refine Grounding:
You predicted bounding boxes for this diagram. LOOK AT THE IMAGE AGAIN carefully and improve your grounding.

--- YOUR PREVIOUS PREDICTION ---
{previous_prediction}
--- END ---

Do BOTH:
A. FIX COORDINATES: For each box, re-examine the image — verify (x,y) marks the correct top-left and (w,h) tightly encloses the element. Adjust any misplaced or poorly sized boxes.
B. ADD MISSING ELEMENTS: Aggressively add any visual evidence that supports verifying the answer — answer regions, labels, legends, axes, arrows, neighbouring regions for comparison, colour keys, scale bars. Err on the side of MORE boxes. It is always better to include too many than too few.

Rules:
• NEVER remove a box — only adjust coordinates or add new ones.
• Use IDs like B_01, T_01, R_01 for new boxes.
• Coordinates in pixels for image size {height}×{width}.

Output (STRICT):
Answer: <ID1>/<x,y,w,h>, <ID2>/<x,y,w,h>, ...
Explanation: Note adjustments and additions with brief reasons.

Output the COMPLETE set of boxes (all originals + any new). Start with "Answer:".
""".strip()


# ── Builder ────────────────────────────────────────────────────────────────

def build_verify_and_refine_prompt(
    question_id: str,
    question: str,
    choices_str: str,
    answer: str,
    previous_prediction: str,
) -> str:

    template = VERIFY_AND_REFINE_TEMPLATE.replace(
        "{height}", "{{height}}"
    ).replace(
        "{width}", "{{width}}"
    )
    return template.format(
        previous_prediction=previous_prediction,
    )


# ── Legacy aliases ─────────────────────────────────────────────────────────

def build_evidence_verification_prompt(
    question_id, question, choices_str, answer, previous_prediction,
):
    return build_verify_and_refine_prompt(
        question_id, question, choices_str, answer, previous_prediction,
    )


def build_spatial_refinement_prompt(
    question_id, question, choices_str, answer, previous_prediction,
):
    return build_verify_and_refine_prompt(
        question_id, question, choices_str, answer, previous_prediction,
    )
