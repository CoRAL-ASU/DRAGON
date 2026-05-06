# worked_examples.py
"""
Centralised worked examples for all prompting techniques.

Each dict is keyed by the canonical dataset class name (e.g. "ChartQADataset").
Import only the dict you need in the respective prompt file.
"""

# ── Single-stage / generic grounding examples ──────────────────────────────

WORKED_EXAMPLE = {
    "ChartQADataset": (
        "Answer: B_01/300,200,60,180, T_01/305,210,50,40, X_01/100,450,800,30\n"
        "Explanation: B_01 is the bar for Brazil. T_01 is the label \"497\". "
        "X_01 is the X-axis label \"Million hectares\"."
    ),
    "AI2DAllDataset": (
        "Answer: V_01/150,100,60,60, T_01/215,100,40,20, A_01/200,130,20,20, "
        "A_02/300,130,20,20, A_03/400,130,20,20\n"
        "Explanation: V_01 is the Egg stage. T_01 is its label. "
        "Arrows A_01–A_03 show cycle progression."
    ),
    "CircuitVQADataset": (
        "Answer: C_01/80,120,40,40, T_01/85,165,30,20\n"
        "Explanation: C_01 is the zigzag resistor symbol. "
        "T_01 is the \"1Ω\" label."
    ),
    "InfographicsVQADataset": (
        "Answer: T_01/100,200,50,40, T_02/200,200,50,40, C_01/120,210,80,80\n"
        "Explanation: T_01 shows \"76%\" users, T_02 shows \"24%\" non-users, "
        "C_01 is the chart segment for 24%."
    ),
    "MapQADataset": (
        "Answer: R_01/300,240,100,50, T_01/305,250,80,40, L_01/500,100,90,30, "
        "R_02/250,240,90,50, R_03/280,280,100,50, R_04/330,220,90,40\n"
        "Explanation: R_01 is Mississippi. T_01 is its label. L_01 is the legend. "
        "R_02–R_04 are neighbouring states for comparison."
    ),
    "RealCQADataset": (
        "Answer: B_01/100,150,40,180, B_02/160,150,40,170, B_03/220,150,40,110, "
        "T_01/105,340,30,20, T_02/165,330,30,20, T_03/225,270,30,20\n"
        "Explanation: B_01–B_03 are black bars for \"no binge drinker\". "
        "T_01–T_03 are their percentage labels."
    ),
}

# ── Two-stage: Stage 1 (SELECT) examples ───────────────────────────────────

WORKED_EXAMPLE_SELECT = {
    "ChartQADataset": (
        "Elements:\n"
        "1. [B_01] Bar for Brazil (answer depiction)\n"
        "2. [T_01] Label \"497\" near Brazil's bar\n"
        "3. [X_01] X-axis label \"Million hectares\""
    ),
    "AI2DAllDataset": (
        "Elements:\n"
        "1. [V_01] Visual depiction of the Egg stage\n"
        "2. [T_01] Text label for the Egg stage\n"
        "3. [A_01] First arrow showing cycle direction\n"
        "4. [A_02] Second arrow continuing cycle\n"
        "5. [A_03] Third arrow completing cycle"
    ),
    "CircuitVQADataset": (
        "Elements:\n"
        "1. [C_01] Zigzag resistor symbol\n"
        "2. [T_01] Label \"1Ω\" for the resistor"
    ),
    "InfographicsVQADataset": (
        "Elements:\n"
        "1. [T_01] Text \"76%\" for users\n"
        "2. [T_02] Text \"24%\" for non-users\n"
        "3. [C_01] Chart segment for 24%"
    ),
    "MapQADataset": (
        "Elements:\n"
        "1. [R_01] State region of Mississippi (answer)\n"
        "2. [T_01] Label for Mississippi\n"
        "3. [L_01] Color legend\n"
        "4. [R_02] Neighbouring state for comparison\n"
        "5. [R_03] Another neighbouring state\n"
        "6. [R_04] Another neighbouring state"
    ),
    "RealCQADataset": (
        "Elements:\n"
        "1. [B_01] Black bar for \"no binge drinker\" (1st category)\n"
        "2. [B_02] Black bar (2nd category)\n"
        "3. [B_03] Black bar (3rd category)\n"
        "4. [T_01] % label on B_01\n"
        "5. [T_02] % label on B_02\n"
        "6. [T_03] % label on B_03"
    ),
}

# ── Two-stage: Stage 2 (GROUND) examples ──────────────────────────────────

WORKED_EXAMPLE_GROUND = {
    "ChartQADataset": (
        "Answer: B_01/300,200,60,180, T_01/305,210,50,40, X_01/100,450,800,30\n"
        "Explanation: B_01 is Brazil's bar. T_01 is the \"497\" label. "
        "X_01 is the X-axis units."
    ),
    "AI2DAllDataset": (
        "Answer: V_01/150,100,60,60, T_01/215,100,40,20, A_01/200,130,20,20, "
        "A_02/300,130,20,20, A_03/400,130,20,20\n"
        "Explanation: V_01 is the Egg stage. T_01 is its label. "
        "A_01–A_03 are directional arrows."
    ),
    "CircuitVQADataset": (
        "Answer: C_01/80,120,40,40, T_01/85,165,30,20\n"
        "Explanation: C_01 is the resistor symbol. T_01 is \"1Ω\"."
    ),
    "InfographicsVQADataset": (
        "Answer: T_01/100,200,50,40, T_02/200,200,50,40, C_01/120,210,80,80\n"
        "Explanation: T_01 \"76%\", T_02 \"24%\", C_01 is the 24% segment."
    ),
    "MapQADataset": (
        "Answer: R_01/300,240,100,50, T_01/305,250,80,40, L_01/500,100,90,30, "
        "R_02/250,240,90,50, R_03/280,280,100,50, R_04/330,220,90,40\n"
        "Explanation: R_01 is Mississippi. T_01 is its label. L_01 is the legend. "
        "R_02–R_04 are neighbours."
    ),
    "RealCQADataset": (
        "Answer: B_01/100,150,40,180, B_02/160,150,40,170, B_03/220,150,40,110, "
        "T_01/105,340,30,20, T_02/165,330,30,20, T_03/225,270,30,20\n"
        "Explanation: B_01–B_03 are bars. T_01–T_03 are their % labels."
    ),
}


# ── Dataset name → canonical key mapping ───────────────────────────────────

DATASET_TO_WORKED_KEY = {
    "ChartQA":           "ChartQADataset",
    "AI2DAll":           "AI2DAllDataset",
    "AI2D":              "AI2DAllDataset",
    "ai2d":              "AI2DAllDataset",
    "ai2d-all":          "AI2DAllDataset",
    "CircuitVQA":        "CircuitVQADataset",
    "Circuit-VQA":       "CircuitVQADataset",
    "Circut-VQA":        "CircuitVQADataset",
    "InfographicsVQA":   "InfographicsVQADataset",
    "Infographicsvqa":   "InfographicsVQADataset",
    "Infographics":      "InfographicsVQADataset",
    "MapQA":             "MapQADataset",
    "MapIQ":             "MapQADataset",
    "Mapwise":           "MapQADataset",
    "RealCQA":           "RealCQADataset",
    "RealCQA-Dataset":   "RealCQADataset",
}


def resolve_key(dataset_name: str) -> str:
    """Map a dataset name to its canonical worked-example key."""
    key = DATASET_TO_WORKED_KEY.get(dataset_name)
    if key is None and "map" in dataset_name.lower():
        key = "MapQADataset"
    if key is None:
        key = next(iter(WORKED_EXAMPLE))
    return key
