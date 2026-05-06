#!/usr/bin/env python3
"""
Dataset-agnostic grounding-aware evaluation

Metrics
-------
- **Max IoU** – best pairwise IoU across all pred-GT box pairs (per sample).
- **Mean IoU** – average of each GT box's best-matching pred IoU (per sample).
- **Group IoU** – Shapely union-based.
- **MaxIoU@τ / MeanIoU@τ / GrpIoU@τ** – hit-rate (fraction of samples with IoU >= τ) at τ ∈ {0.1, 0.3, 0.5, 0.7}.
- **Recall / Precision / F1** – box-level at IoU thresholds 0.1, 0.3, 0.5, 0.7.
- **Soft Recall / Precision / F1** – threshold-free: uses best-match IoU as soft score instead of binary.

All boxes are normalised to [0, 1] before comparison.

Works for:
ChartQA, CircuitVQA, AI2D, InfographicsVQA, MapIQ, Mapwise
"""

import json
import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union


# -----------------------------
# Normalisation utilities
# -----------------------------

def _is_normalised(boxes: List[Dict], threshold: float = 1.5) -> bool:
    """Heuristic: if every coordinate value (x, y, x+w, y+h) is <= threshold
    we treat the boxes as already normalised to [0, 1]."""
    for b in boxes:
        if (b["x"] + b["w"]) > threshold or (b["y"] + b["h"]) > threshold:
            return False
    return True


def normalise_boxes(
    boxes: List[Dict],
    img_w: float,
    img_h: float,
) -> List[Dict]:
    """Return a copy of *boxes* with coordinates in [0, 1].

    If the box values are already small (heuristically ≤ 1.5) they are assumed
    to be normalised already and returned as-is.  Otherwise each coordinate is
    divided by the image width / height.
    """
    if not boxes or img_w <= 0 or img_h <= 0:
        return boxes

    if _is_normalised(boxes):
        return [dict(b) for b in boxes]  # already normalised

    out = []
    for b in boxes:
        out.append({
            "x": b["x"] / img_w,
            "y": b["y"] / img_h,
            "w": b["w"] / img_w,
            "h": b["h"] / img_h,
        })
    return out


# -----------------------------
# Geometry utilities
# -----------------------------

def to_xyxy(b: Dict) -> Tuple[float, float, float, float]:
    """Convert {x, y, w, h} → (x1, y1, x2, y2)."""
    return b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]


def pairwise_iou(a: Dict, b: Dict) -> float:
    """Standard IoU between two boxes."""
    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# -------------------------------------------------
# GRIT-style Group IoU  (Shapely union-based)
# -------------------------------------------------

def group_iou(pred_boxes: List[Dict], gt_boxes: List[Dict]) -> float:
    """Union all pred boxes into one shape, union all GT boxes into
    another, return IoU of the two union shapes.

    Follows ``iou_of_bbox_groups`` from the GRIT benchmark:
    https://github.com/eric-ai-lab/GRIT
    """
    if not pred_boxes or not gt_boxes:
        return 0.0
    polys_pred = [shapely_box(*to_xyxy(b)) for b in pred_boxes]
    polys_gt   = [shapely_box(*to_xyxy(b)) for b in gt_boxes]
    union_pred = unary_union(polys_pred)
    union_gt   = unary_union(polys_gt)
    inter = union_pred.intersection(union_gt)
    uni   = union_pred.union(union_gt)
    return inter.area / uni.area if uni.area > 0 else 0.0


# -------------------------------------------------
# Recall / Precision / F1  (at a single τ)
# -------------------------------------------------

def recall_precision_f1(gt_boxes, pred_boxes, tau):
    if not gt_boxes or not pred_boxes:
        return 0.0, 0.0, 0.0
    gt_covered = sum(
        1 for g in gt_boxes
        if any(pairwise_iou(p, g) >= tau for p in pred_boxes)
    )
    recall = gt_covered / len(gt_boxes)
    pred_matched = sum(
        1 for p in pred_boxes
        if any(pairwise_iou(p, g) >= tau for g in gt_boxes)
    )
    precision = pred_matched / len(pred_boxes)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return recall, precision, f1


def mean_iou(gt_boxes, pred_boxes):
    """Average of each GT box's best-matching pred IoU."""
    if not gt_boxes or not pred_boxes:
        return 0.0
    return float(np.mean([
        max(pairwise_iou(p, g) for p in pred_boxes)
        for g in gt_boxes
    ]))


def soft_recall_precision_f1(gt_boxes, pred_boxes):
    """Threshold-free R/P/F1 using best-match IoU as soft scores.

    Soft Recall  = mean over GT boxes of max IoU with any pred box.
    Soft Precision = mean over pred boxes of max IoU with any GT box.
    Soft F1 = harmonic mean of soft R and soft P.
    """
    if not gt_boxes or not pred_boxes:
        return 0.0, 0.0, 0.0
    soft_r = float(np.mean([
        max(pairwise_iou(p, g) for p in pred_boxes)
        for g in gt_boxes
    ]))
    soft_p = float(np.mean([
        max(pairwise_iou(p, g) for g in gt_boxes)
        for p in pred_boxes
    ]))
    soft_f1 = (2 * soft_r * soft_p / (soft_r + soft_p)) if (soft_r + soft_p) > 0 else 0.0
    return soft_r, soft_p, soft_f1


# -------------------------------------------------
# Metric computation (per sample)
# -------------------------------------------------

_TAU_LIST = [0.1, 0.3, 0.5, 0.7]   # IoU thresholds for R/P/F1


def compute_per_sample_metrics(
    gt_boxes: List[Dict],
    pred_boxes: List[Dict],
    tau_list: List[float] = None,
) -> Dict[str, float]:
    """Compute per-sample IoU / GroupIoU / R / P / F1 (no mAP here;
    mAP is computed globally via torchmetrics)."""
    if tau_list is None:
        tau_list = _TAU_LIST

    result: Dict[str, float] = {}

    if not gt_boxes or not pred_boxes:
        result["IoU"] = 0.0
        result["MeanIoU"] = 0.0
        result["GroupIoU"] = 0.0
        result["SoftRecall"] = 0.0
        result["SoftPrecision"] = 0.0
        result["SoftF1"] = 0.0
        for t in tau_list:
            pct = int(t * 100)
            result[f"MaxIoU@{pct}"] = 0.0
            result[f"MeanIoU@{pct}"] = 0.0
            result[f"GrpIoU@{pct}"] = 0.0
            result[f"Recall@{pct}"] = 0.0
            result[f"Precision@{pct}"] = 0.0
            result[f"F1@{pct}"] = 0.0
        return result

    # Normal IoU: best pairwise IoU across all pred-GT pairs
    max_iou = max(pairwise_iou(p, g) for p in pred_boxes for g in gt_boxes)
    # Mean IoU: average of each GT box's best-matching pred IoU
    miou = mean_iou(gt_boxes, pred_boxes)
    # GRIT-style Group IoU: union shapes
    giou = group_iou(pred_boxes, gt_boxes)
    # Soft R/P/F1 (threshold-free)
    sr, sp, sf1 = soft_recall_precision_f1(gt_boxes, pred_boxes)

    result["IoU"] = max_iou
    result["MeanIoU"] = miou
    result["GroupIoU"] = giou
    result["SoftRecall"] = sr
    result["SoftPrecision"] = sp
    result["SoftF1"] = sf1

    for t in tau_list:
        pct = int(t * 100)
        # Hit-rate: 1 if IoU >= tau, else 0  (averages to accuracy across samples)
        result[f"MaxIoU@{pct}"] = 1.0 if max_iou >= t else 0.0
        result[f"MeanIoU@{pct}"] = 1.0 if miou >= t else 0.0
        result[f"GrpIoU@{pct}"] = 1.0 if giou >= t else 0.0
        recall, precision, f1 = recall_precision_f1(gt_boxes, pred_boxes, t)
        result[f"Recall@{pct}"] = recall
        result[f"Precision@{pct}"] = precision
        result[f"F1@{pct}"] = f1

    return result


# -----------------------------
# Box extraction helpers
# -----------------------------

def extract_gt_boxes(item: Dict) -> List[Dict]:
    """
    Supports two formats:
      1. New format (gemma3 / bedrock): gt_boxes_raw list with source/kind fields
      2. Legacy format: item["gt"]["bbox"]
    """
    # New format
    raw = item.get("gt_boxes_raw")
    if raw is not None:
        key = item.get("gt_boxes_key", "bbox")
        boxes = [b for b in raw if b.get("source") == "gt" and b.get("kind") == key]
        if not boxes:
            boxes = [b for b in raw if b.get("source") == "gt"]
        if not boxes:
            boxes = [b for b in raw if b.get("kind") == key]
        if not boxes:
            # Some datasets store all annotations as source='pred' with no GT tag;
            # fall back to using the full gt_boxes_raw as ground truth.
            boxes = raw
        return [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]} for b in boxes]
    # Legacy format
    return item.get("gt", {}).get("bbox", [])


def extract_pred_boxes(item: Dict) -> List[Dict]:
    """
    Supports two formats:
      1. New format (gemma3 / bedrock): pred_boxes_parsed
      2. Legacy format: item["pred"]["bboxes"]
    """
    parsed = item.get("pred_boxes_parsed")
    if parsed is not None:
        return [{"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]} for b in parsed]
    return item.get("pred", {}).get("bboxes", [])


# -----------------------------
# Evaluation + saving
# -----------------------------

def evaluate_and_save(json_path: str, out_prefix: str, tau_list: List[float] = None):
    if tau_list is None:
        tau_list = _TAU_LIST

    if not os.path.exists(json_path):
        raise FileNotFoundError(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect per-sample metrics
    sample0_keys = list(compute_per_sample_metrics([], [], tau_list).keys())
    totals = {k: 0.0 for k in sample0_keys}
    per_sample = []
    n_no_gt = 0
    n_no_pred = 0

    for item in data:
        gt_boxes   = extract_gt_boxes(item)
        pred_boxes = extract_pred_boxes(item)

        # Normalise both GT and pred boxes to [0, 1] so models using
        # different coordinate spaces are compared fairly.
        img_w = item.get("image_width", 0)
        img_h = item.get("image_height", 0)
        if img_w > 0 and img_h > 0:
            gt_boxes   = normalise_boxes(gt_boxes,   img_w, img_h)
            pred_boxes = normalise_boxes(pred_boxes, img_w, img_h)

        if not gt_boxes:
            n_no_gt += 1
        if not pred_boxes:
            n_no_pred += 1

        metrics = compute_per_sample_metrics(gt_boxes, pred_boxes, tau_list)

        per_sample.append({
            "sample_id": item.get("sample_id", ""),
            "uid":       item.get("uid", ""),
            "q_id":      item.get("q_id", ""),
            "dataset":   item.get("dataset", ""),
            "n_gt":      len(gt_boxes),
            "n_pred":    len(pred_boxes),
            **metrics,
        })

        for k in totals:
            totals[k] += metrics[k]

    n = len(per_sample)
    avg = {k: totals[k] / n if n else 0.0 for k in totals}

    # Recompute F1 from averaged R and P (not average of per-sample F1)
    for t in tau_list:
        pct = int(t * 100)
        r = avg[f"Recall@{pct}"]
        p = avg[f"Precision@{pct}"]
        avg[f"F1@{pct}"] = (2 * r * p / (r + p)) if (r + p) > 0 else 0.0

    # Recompute SoftF1 from averaged SoftRecall and SoftPrecision
    sr = avg["SoftRecall"]
    sp = avg["SoftPrecision"]
    avg["SoftF1"] = (2 * sr * sp / (sr + sp)) if (sr + sp) > 0 else 0.0

    summary = {
        "input_file": json_path,
        "tau_list": tau_list,
        "num_samples": n,
        "num_no_gt": n_no_gt,
        "num_no_pred": n_no_pred,
        "metrics": avg,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_prefix)), exist_ok=True)

    with open(f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{out_prefix}_per_sample.json", "w") as f:
        json.dump(per_sample, f, indent=2)

    m = summary["metrics"]
    print(f"  Samples : {n}  (no_gt={n_no_gt}, no_pred={n_no_pred})")
    print(f"  AvgMaxIoU: {m['IoU']:.4f}   AvgMeanIoU: {m['MeanIoU']:.4f}   AvgGrpIoU: {m['GroupIoU']:.4f}")
    print(f"  SoftR: {m['SoftRecall']:.4f}   SoftP: {m['SoftPrecision']:.4f}   SoftF1: {m['SoftF1']:.4f}")
    for t in tau_list:
        pct = int(t * 100)
        print(f"  @τ={t:.1f}  MaxIoU: {m[f'MaxIoU@{pct}']:.4f}  MeanIoU: {m[f'MeanIoU@{pct}']:.4f}  GrpIoU: {m[f'GrpIoU@{pct}']:.4f}  R: {m[f'Recall@{pct}']:.4f}  P: {m[f'Precision@{pct}']:.4f}  F1: {m[f'F1@{pct}']:.4f}")
    print(f"  Saved   → {out_prefix}_summary.json")

    return summary


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate grounding predictions. Pass --pred_dir to batch "
                    "over all pred_*.json files, or --json for a single file."
    )
    parser.add_argument("--json",      default=None,  help="Single prediction JSON")
    parser.add_argument("--pred_dir",  default=None,  help="Directory with pred_*.json files")
    parser.add_argument("--out_dir",   default=None,  help="Output directory (default: <pred_dir>/eval or beside --json)")
    parser.add_argument("--out_prefix",default=None,  help="Output prefix for single-file mode")
    parser.add_argument("--tau",       type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.7],
                        help="IoU thresholds for R/P/F1 (default: 0.1 0.3 0.5 0.7)")
    args = parser.parse_args()

    tau_list = args.tau

    if args.pred_dir:
        import glob
        pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "pred_*.json")))
        # Exclude eval output files that match the glob
        pred_files = [f for f in pred_files
                      if not f.endswith("_eval_summary.json")
                      and not f.endswith("_eval_per_sample.json")
                      and os.path.basename(f) != "pred_all.json"]
        if not pred_files:
            print(f"No pred_*.json files found in {args.pred_dir}")
            exit(1)
        out_dir = args.out_dir or os.path.join(args.pred_dir, "eval")
        os.makedirs(out_dir, exist_ok=True)
        all_metrics = {}
        for pf in pred_files:
            ds = os.path.basename(pf).replace("pred_", "").replace(".json", "")
            print(f"\n{'='*70}\nDataset: {ds}")
            prefix = os.path.join(out_dir, ds)
            summary = evaluate_and_save(pf, prefix, tau_list)
            all_metrics[ds] = summary["metrics"]

        # ── Pretty summary table ──
        # Table 1: MaxIoU@tau and GrpIoU@tau (hit-rate: fraction of samples >= tau)
        print(f"\n{'=' * 100}")
        print("MaxIoU@tau / GrpIoU@tau  (fraction of samples with IoU >= tau)")
        hdr_iou = f"{'Dataset':<18}"
        for t in tau_list:
            pct = int(t * 100)
            hdr_iou += f" {'MxI@'+str(pct):>7} {'GrI@'+str(pct):>7}"
        print(hdr_iou)
        print("-" * 100)
        for ds, m in all_metrics.items():
            row = f"{ds:<18}"
            for t in tau_list:
                pct = int(t * 100)
                row += f" {m[f'MaxIoU@{pct}']:>7.4f} {m[f'GrpIoU@{pct}']:>7.4f}"
            print(row)

        # Table 3: F1 at each tau
        print(f"\n{'=' * 100}")
        print("F1@tau")
        hdr_f1 = f"{'Dataset':<18}"
        for t in tau_list:
            pct = int(t * 100)
            hdr_f1 += f" {'F1@'+str(pct):>7}"
        print(hdr_f1)
        print("-" * 100)
        for ds, m in all_metrics.items():
            row = f"{ds:<18}"
            for t in tau_list:
                pct = int(t * 100)
                row += f" {m[f'F1@{pct}']:>7.4f}"
            print(row)

        # Table 4: R/P/F1 at each tau (detailed)
        print(f"\n{'=' * 140}")
        print("Recall / Precision / F1 @ tau (detailed)")
        tau_cols = []
        for t in tau_list:
            pct = int(t * 100)
            tau_cols += [f"R@{pct}", f"P@{pct}", f"F1@{pct}"]
        hdr2 = f"{'Dataset':<18}"
        for c in tau_cols:
            hdr2 += f" {c:>7}"
        print(hdr2)
        print("-" * 140)
        for ds, m in all_metrics.items():
            row = f"{ds:<18}"
            for t in tau_list:
                pct = int(t * 100)
                row += f" {m[f'Recall@{pct}']:>7.4f} {m[f'Precision@{pct}']:>7.4f} {m[f'F1@{pct}']:>7.4f}"
            print(row)

        combined = os.path.join(out_dir, "all_datasets_summary.json")
        with open(combined, "w") as f:
            json.dump({"tau_list": tau_list, "datasets": all_metrics}, f, indent=2)
        print(f"\nCombined summary → {combined}")

    elif args.json:
        out_prefix = args.out_prefix
        if out_prefix is None:
            base = os.path.splitext(args.json)[0]
            out_prefix = (os.path.join(args.out_dir, os.path.basename(base))
                          if args.out_dir else base + "_eval")
        evaluate_and_save(args.json, out_prefix, tau_list)

    else:
        parser.print_help()
