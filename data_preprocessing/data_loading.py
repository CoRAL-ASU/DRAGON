#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# =============================================================================
# PATHS (EDIT IF NEEDED)
# =============================================================================
JSON_ROOT = Path("./split_reviewed/Test").resolve()
IMAGES_ROOT = Path("./Diagram_Attribution_Dataset").resolve()
OUT_DIR = Path("./batches").resolve()

# =============================================================================
# PROMPT IMPORT
# =============================================================================
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prompts"))
from prompts_edge import MAIN_PROMPT_TEMPLATE, WORKED_EXAMPLE  # noqa: E402


# =============================================================================
# WORKED EXAMPLE SELECTION
# =============================================================================
# Map your dataset folder names -> WORKED_EXAMPLE dict keys
# Edit/extend if your dataset folder names differ.
DATASET_TO_WORKED_KEY: Dict[str, str] = {
    "ChartQA": "ChartQADataset",
    "AI2DAll": "AI2DAllDataset",
    "AI2D": "AI2DAllDataset",
    "ai2d": "AI2DAllDataset",
    "ai2d-all": "AI2DAllDataset",
    "CircuitVQA": "CircuitVQADataset",
    "Circuit-VQA": "CircuitVQADataset",
    "Circut-VQA": "CircuitVQADataset",  # if your folder has this typo
    "InfographicsVQA": "InfographicsVQADataset",
    "Infographicsvqa": "InfographicsVQADataset",
    "Infographics": "InfographicsVQADataset",
    "MapQA": "MapQADataset",
    "MapIQ": "MapQADataset",
    "Mapwise": "MapQADataset",
    "RealCQA": "RealCQADataset",
    "RealCQA-Dataset": "RealCQADataset",
}

def get_worked_example_for_dataset(dataset_name: str) -> str:
    """
    Returns the worked-example STRING for this dataset.
    Falls back to MapQADataset for map-like names, else the first entry in dict.
    """
    # direct map
    key = DATASET_TO_WORKED_KEY.get(dataset_name)

    # heuristic fallback for map datasets
    if key is None and "map" in dataset_name.lower():
        key = "MapQADataset"

    # final fallback
    if key is None:
        # pick first available example in the dict
        try:
            return next(iter(WORKED_EXAMPLE.values()))
        except Exception:
            return ""

    return WORKED_EXAMPLE.get(key, "")


# =============================================================================
# PROMPT BUILDER (KEEP {height}×{width} UNCHANGED)
# =============================================================================
def build_prompt(
    dataset: str,
    question_id: str,
    question: str,
    choices_str: str,
    answer: str,
) -> str:
    """
    You want to fill image metadata later at model-time.
    So we keep literal placeholders {height} and {width} in the final prompt.

    Since we call .format() on MAIN_PROMPT_TEMPLATE, we must ESCAPE them:
      {height} -> {{height}}
      {width}  -> {{width}}

    WORKED_EXAMPLE is a dict, so we pick the correct string for this dataset.
    """
    template = MAIN_PROMPT_TEMPLATE.replace("{height}", "{{height}}").replace("{width}", "{{width}}")

    worked_ex_str = get_worked_example_for_dataset(dataset)

    return template.format(
        **{
            "Question_ID_STRING": question_id,
            "Question_String": question,
            "Choices string": choices_str,
            "Answer string": answer,
            "Worked Example": worked_ex_str,   # IMPORTANT: string, not dict
        }
    )


# =============================================================================
# IO HELPERS
# =============================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def find_jsons_recursive(root: Path) -> List[Path]:
    return sorted(root.rglob("*.json"))

def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def get_first_present(obj: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None


# =============================================================================
# ORIGINAL BOX EXTRACTION (NO CHANGES)
# =============================================================================
def extract_boxes_raw(obj: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Returns:
      boxes_key: "bbox" | "boxes" | "none"
      boxes_raw: exactly obj["bbox"] or obj["boxes"] (unchanged)
    """
    if isinstance(obj.get("bbox"), list):
        return "bbox", obj["bbox"]
    if isinstance(obj.get("boxes"), list):
        return "boxes", obj["boxes"]
    return "none", []


# =============================================================================
# QUESTION / CHOICES / ANSWER EXTRACTION
# =============================================================================
def extract_image_path(obj: Dict[str, Any]) -> str:
    img = get_first_present(obj, ["image_path", "img_path", "image", "image_file"])
    if img is None:
        raise KeyError("Missing image_path (or equivalent)")
    return safe_str(img)

def extract_question_text(obj: Dict[str, Any]) -> str:
    q = get_first_present(obj, ["question_text", "question", "query", "prompt"])
    return safe_str(q)

def extract_choices(obj: Dict[str, Any]) -> List[str]:
    c = obj.get("choices")
    if c is None:
        return []
    if isinstance(c, list):
        return [safe_str(x) for x in c]
    if isinstance(c, dict):
        return [safe_str(c[k]) for k in sorted(c.keys())]
    return []

def extract_answer_text(obj: Dict[str, Any]) -> str:
    ans_block = obj.get("answers")
    if isinstance(ans_block, dict) and ans_block.get("correct") is not None:
        return safe_str(ans_block.get("correct"))

    flat = get_first_present(
        obj,
        [
            "ground_truth_answer",
            "gt_answer",
            "answer",
            "correct_answer",
            "label",
        ],
    )
    return safe_str(flat)

def extract_qid(obj: Dict[str, Any], json_path: Path) -> str:
    qid = get_first_present(obj, ["q_id", "qid", "question_id", "id"])
    if qid is not None:
        return safe_str(qid)
    return json_path.stem

def choices_to_string(choices: List[str]) -> str:
    if not choices:
        return "N/A"
    return " | ".join(choices)


# =============================================================================
# DATASET NAME FROM PATH
# =============================================================================
def dataset_name_from_path(json_path: Path, root: Path) -> str:
    rel = json_path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 0 else "unknown"


# =============================================================================
# LOAD ONE RECORD (INFERENCE READY)
# =============================================================================
def load_one_record(json_path: Path, json_root: Path, images_root: Path) -> Dict[str, Any]:
    obj = read_json(json_path)

    dataset = dataset_name_from_path(json_path, json_root)

    image_rel = extract_image_path(obj)
    image_abs = str((images_root / image_rel).resolve())

    q_id = extract_qid(obj, json_path)
    question_text = extract_question_text(obj)
    choices = extract_choices(obj)
    answer_text = extract_answer_text(obj)

    boxes_key, boxes_raw = extract_boxes_raw(obj)

    prompt = build_prompt(
        dataset=dataset,
        question_id=q_id,
        question=question_text,
        choices_str=choices_to_string(choices),
        answer=answer_text,
    )

    uid_abs = str(json_path.resolve())

    return {
        "uid": json_path.name,        # filename only
        "dataset": dataset,
        "q_id": q_id,
        "json_path": uid_abs,         # absolute path
        "image_path_rel": image_rel,
        "image_path_abs": image_abs,
        "question_text": question_text,
        "choices": choices,
        "answer_text": answer_text,
        "boxes_key": boxes_key,
        "boxes_raw": boxes_raw,       # ORIGINAL UNCHANGED
        "prompt": prompt,             # contains literal {height}×{width}
    }


# =============================================================================
# LOAD ALL + GROUP BY DATASET
# =============================================================================
def load_grouped_by_dataset(
    json_root: Path,
    images_root: Path
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    paths = find_jsons_recursive(json_root)

    by_ds: Dict[str, List[Dict[str, Any]]] = {}
    stats: Dict[str, Any] = {
        "total_json": len(paths),
        "loaded": 0,
        "skipped": 0,
        "missing_image_files": 0,
        "by_dataset": {},
        "by_box_key": {"bbox": 0, "boxes": 0, "none": 0},
    }

    for p in paths:
        try:
            obj = read_json(p)

            if isinstance(obj.get("bbox"), list):
                stats["by_box_key"]["bbox"] += 1
            elif isinstance(obj.get("boxes"), list):
                stats["by_box_key"]["boxes"] += 1
            else:
                stats["by_box_key"]["none"] += 1

            rec = load_one_record(p, json_root=json_root, images_root=images_root)

            if not Path(rec["image_path_abs"]).exists():
                stats["missing_image_files"] += 1

            by_ds.setdefault(rec["dataset"], []).append(rec)
            stats["loaded"] += 1

            ds = rec["dataset"]
            ds_stats = stats["by_dataset"].setdefault(
                ds, {"loaded": 0, "with_choices": 0, "with_boxes": 0, "missing_images": 0}
            )
            ds_stats["loaded"] += 1
            ds_stats["with_choices"] += int(len(rec["choices"]) > 0)
            ds_stats["with_boxes"] += int(rec["boxes_key"] != "none" and len(rec["boxes_raw"]) > 0)
            ds_stats["missing_images"] += int(not Path(rec["image_path_abs"]).exists())

        except Exception:
            stats["skipped"] += 1

    return by_ds, stats


# =============================================================================
# WRITER: ONE FILE PER DATASET
# =============================================================================
def write_per_dataset_json(grouped: Dict[str, List[Dict[str, Any]]], out_dir: Path) -> None:
    ensure_dir(out_dir)

    for ds, records in sorted(grouped.items(), key=lambda x: x[0].lower()):
        out_path = out_dir / f"{ds}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    assert JSON_ROOT.exists(), f"JSON_ROOT not found: {JSON_ROOT}"
    assert IMAGES_ROOT.exists(), f"IMAGES_ROOT not found: {IMAGES_ROOT}"
    ensure_dir(OUT_DIR)

    grouped, stats = load_grouped_by_dataset(JSON_ROOT, IMAGES_ROOT)
    write_per_dataset_json(grouped, OUT_DIR)

    print("=== LOAD SUMMARY ===")
    print(json.dumps(stats, indent=2))
    print(f"\nWrote per-dataset inference files to: {OUT_DIR}")


if __name__ == "__main__":
    main()
