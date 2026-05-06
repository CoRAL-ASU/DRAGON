#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kimi_edge.py  (EDGE — Evidence-grounded Direct Grounding)

Runs Kimi K2.5 inference via Bedrock invoke_model API (real-time, one sample
at a time). Uses OpenAI-style messages format with base64 image_url.

Usage:
    python kimi_edge.py \
        [--batch_dir ./batches] \
        [--pred_dir  ./predictions/kimi] \
"""

import os
import re
import json
import base64
import argparse
import logging
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime
from collections import Counter

import boto3
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "moonshotai.kimi-k2.5"
DEFAULT_REGION   = "us-east-1"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (same as gemma3_edge.py / claude_llama_edge.py)
# ---------------------------------------------------------------------------
def make_sample_id(sample: dict) -> str:
    uid   = (sample.get("uid") or "").strip()
    jpath = (sample.get("json_path") or "").strip()
    qid   = (sample.get("q_id") or "").strip()
    img   = Path(sample.get("image_path_rel") or sample.get("image_path_abs") or "noimg").name
    core  = f"uid::{uid}::qid::{qid}::img::{img}"
    if jpath:
        core += f"::json::{Path(jpath).name}"
    return core


def _collect_sids_from(obj):
    s = set()
    if obj is None:
        return s
    if isinstance(obj, list):
        it = obj
    elif isinstance(obj, dict):
        it = obj.values()
    else:
        return s
    for rec in it:
        if isinstance(rec, dict):
            sid = rec.get("sample_id")
            if sid:
                s.add(sid)
    return s


def load_existing_progress(results_path):
    processed = set()
    results = []
    if Path(results_path).exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            processed |= _collect_sids_from(data)
            results = data if isinstance(data, list) else list(data.values())
        except Exception:
            results = []
    return processed, results


def atomic_write_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Box parsing (identical to gemma3.py)
# ---------------------------------------------------------------------------
_NUM = r"-?\d+(?:\.\d+)?"
_BOX_RE = re.compile(
    rf"(?P<id>[A-Za-z0-9_\-]+)\s*/\s*(?P<x>{_NUM})\s*,\s*(?P<y>{_NUM})\s*,\s*(?P<w>{_NUM})\s*,\s*(?P<h>{_NUM})"
)

def parse_answer_boxes(answer_str: str):
    if not answer_str:
        return [], ["empty_answer"]
    boxes, errors = [], []
    for m in _BOX_RE.finditer(answer_str):
        try:
            boxes.append({
                "id": m.group("id"),
                "x": float(m.group("x")),
                "y": float(m.group("y")),
                "w": float(m.group("w")),
                "h": float(m.group("h")),
            })
        except Exception as e:
            errors.append(f"parse_float_error:{e}")
    if not boxes:
        errors.append("no_boxes_matched")
    return boxes, errors


def split_reply(raw_reply: str):
    if raw_reply is None:
        return "", ""
    txt = str(raw_reply)
    if "Explanation:" in txt:
        a, e = txt.split("Explanation:", 1)
        a = a.replace("Answer:", "").strip()
        return a.strip(), e.strip()
    return txt.replace("Answer:", "").strip(), ""


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------
def image_to_base64_url(image_path: str, max_side: int = 2048) -> str:
    """Load image, optionally downscale, return data URI for OpenAI-style content."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Single inference call
# ---------------------------------------------------------------------------
def invoke_kimi(bedrock_rt, model_id: str, image_path: str, prompt: str, gen_cfg: dict) -> str:
    """Call Kimi K2.5 via invoke_model with OpenAI-style messages."""
    data_uri = image_to_base64_url(image_path)

    body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": gen_cfg["max_new_tokens"],
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg["top_p"],
    })

    resp = bedrock_rt.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(resp["body"].read())

    # OpenAI-style response
    choices = result.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "").strip()
    return ""


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_dataset(
    samples, bedrock_rt, model_id, gen_cfg, save_every, results_path, log_fp
):
    processed_set, saved_results = load_existing_progress(results_path)
    log.info(f"Resume: loaded {len(saved_results)} results; processed={len(processed_set)}")

    worklist = []
    for s in samples:
        sid = make_sample_id(s)
        if sid not in processed_set:
            s["_sample_id"] = sid
            worklist.append(s)

    log.info(f"To process: {len(worklist)} samples")
    if not worklist:
        return saved_results

    results = saved_results[:]
    errors_count = 0

    for idx, sample in enumerate(worklist):
        try:
            img_path = sample.get("image_path_abs", "")
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size

            base_prompt = sample.get("prompt", "")
            try:
                prompt_filled = base_prompt.format(height=img_h, width=img_w)
            except Exception:
                prompt_filled = base_prompt

            meta = {
                "sample_id": sample["_sample_id"],
                "uid": sample.get("uid", ""),
                "json_path": sample.get("json_path", ""),
                "dataset": sample.get("dataset", ""),
                "q_id": sample.get("q_id", ""),
                "image_path_abs": img_path,
                "image_path_rel": sample.get("image_path_rel", ""),
                "image_width": img_w,
                "image_height": img_h,
                "question_text": sample.get("question_text", ""),
                "choices": sample.get("choices", []),
                "answer_text": sample.get("answer_text", ""),
                "gt_boxes_key": sample.get("boxes_key", "none"),
                "gt_boxes_raw": sample.get("boxes_raw", []),
                "prompt": prompt_filled,
            }

            raw_reply = invoke_kimi(bedrock_rt, model_id, img_path, prompt_filled, gen_cfg)
            pred_answer_str, pred_expl_str = split_reply(raw_reply)
            pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)

            results.append({
                **meta,
                "pred_raw_reply": raw_reply,
                "pred_answer_str": pred_answer_str,
                "pred_explanation_str": pred_expl_str,
                "pred_boxes_parsed": pred_boxes,
                "pred_parse_errors": parse_errors,
            })

            if (idx + 1) % 50 == 0:
                log.info(f"  [{idx+1}/{len(worklist)}] processed")

        except Exception as e:
            error_msg = str(e)
            log.warning(f"  Error on sample {idx}: {error_msg}")
            errors_count += 1
            # Throttling — back off if rate limited
            if "ThrottlingException" in error_msg or "Too Many Requests" in error_msg:
                log.info("  Rate limited — sleeping 30s ...")
                time.sleep(30)
            continue

        if (idx + 1) % save_every == 0:
            atomic_write_json(results, results_path)

    log.info(f"Final save: results={len(results)}, errors={errors_count}")
    atomic_write_json(results, results_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Kimi K2.5 real-time inference via Bedrock")
    parser.add_argument("-b", "--batch_dir",  default="./batches/")
    parser.add_argument("-o", "--pred_dir",   default="./predictions/kimi/")
    parser.add_argument("--model_id",         default=DEFAULT_MODEL_ID)
    parser.add_argument("--region",           default=DEFAULT_REGION)
    parser.add_argument("--temperature",      type=float, default=0.2)
    parser.add_argument("--top_p",            type=float, default=0.7)
    parser.add_argument("--max_new_tokens",   type=int,   default=2048)
    parser.add_argument("--save_every",       type=int,   default=20)
    parser.add_argument("--max_samples",      type=int,   default=None)
    args = parser.parse_args()

    gen_cfg = {
        "temperature":   args.temperature,
        "top_p":         args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    bedrock_rt = boto3.client("bedrock-runtime", region_name=args.region)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = pred_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    batch_dir = Path(args.batch_dir)
    batch_files = sorted(batch_dir.glob("*.json"))

    for batch_path in batch_files:
        ds_name = batch_path.stem
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {ds_name}  ({batch_path.name})")

        with batch_path.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        if args.max_samples is not None:
            samples = samples[: args.max_samples]
            log.info(f"  Limiting to {len(samples)} samples")

        out_path = pred_dir / f"pred_{ds_name}.json"
        log_path = logs_dir / f"{ds_name}.log"

        with open(log_path, "a", encoding="utf-8") as log_fp:
            results = run_dataset(
                samples=samples,
                bedrock_rt=bedrock_rt,
                model_id=args.model_id,
                gen_cfg=gen_cfg,
                save_every=args.save_every,
                results_path=str(out_path),
                log_fp=log_fp,
            )
            log.info(f"Done dataset={ds_name}: {len(results)} results → {out_path}")

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
