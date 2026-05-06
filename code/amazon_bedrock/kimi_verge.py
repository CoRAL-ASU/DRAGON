#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kimi_verge.py  (VERGE — Verify and Refine Grounding Evidence)

Iterative Grounding Refinement (Self-Correction) via Bedrock invoke_model API.
Multi-turn: Turn 1 (image + Step 1 initial grounding) → Turn 2 (Verify & Refine).
Concurrent workers for throughput.

Usage:
    python kimi_verge.py \
        [--batch_dir  batches/] \
        [--pred_dir   prediction_bedrock_selfcorrect/] \
        [--model_id   us.anthropic.claude-sonnet-4-6] \
        [--workers 10] \
        [--max_samples 5]
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
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from PIL import Image

from prompts_verge import build_verify_and_refine_prompt

# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REGION = "us-east-1"
MAX_RETRIES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_sample_id(sample: dict) -> str:
    uid = (sample.get("uid") or "").strip()
    jpath = (sample.get("json_path") or "").strip()
    qid = (sample.get("q_id") or "").strip()
    img = Path(sample.get("image_path_rel") or sample.get("image_path_abs") or "noimg").name
    core = f"uid::{uid}::qid::{qid}::img::{img}"
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
# Box parsing
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
# Image helpers
# ---------------------------------------------------------------------------
def image_to_base64(image_path: str, max_side: int = 1568):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg", w, h


def image_to_base64_url(image_path: str, max_side: int = 2048) -> str:
    """Return data URI for OpenAI-style content (Kimi)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", w, h


def is_kimi(model_id: str) -> bool:
    return "kimi" in model_id.lower() or "moonshot" in model_id.lower()


# ---------------------------------------------------------------------------
# Bedrock invoke with retry
# ---------------------------------------------------------------------------
def invoke_claude(bedrock_rt, model_id: str, messages: list, gen_cfg: dict) -> str:
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": gen_cfg["max_new_tokens"],
        "temperature": gen_cfg["temperature"],
        "messages": messages,
    })

    for attempt in range(MAX_RETRIES):
        try:
            resp = bedrock_rt.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            content = result.get("content", [])
            if content and isinstance(content, list):
                return content[0].get("text", "").strip()
            return ""
        except Exception as e:
            err = str(e)
            if "ThrottlingException" in err or "Too Many Requests" in err or "ServiceUnavailable" in err:
                wait = min(2 ** attempt * 5, 60)
                log.warning(f"  Rate limited (attempt {attempt+1}), sleeping {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def invoke_kimi(bedrock_rt, model_id: str, messages: list, gen_cfg: dict) -> str:
    """Call Kimi K2.5 via invoke_model with OpenAI-style messages."""
    body = json.dumps({
        "messages": messages,
        "max_tokens": gen_cfg["max_new_tokens"],
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg["top_p"],
    })

    for attempt in range(MAX_RETRIES):
        try:
            resp = bedrock_rt.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            choices = result.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return ""
        except Exception as e:
            err = str(e)
            if "ThrottlingException" in err or "Too Many Requests" in err or "ServiceUnavailable" in err:
                wait = min(2 ** attempt * 5, 60)
                log.warning(f"  Rate limited (attempt {attempt+1}), sleeping {wait}s ...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


# ---------------------------------------------------------------------------
# Process one sample (self-correct multi-turn)
# ---------------------------------------------------------------------------
def process_sample(sample, bedrock_rt, model_id, gen_cfg):
    sid = make_sample_id(sample)
    img_path = sample.get("image_path_abs", "")
    b64_data, media_type, img_w, img_h = image_to_base64(img_path)

    dataset = sample.get("dataset", "")
    q_id = sample.get("q_id", "")
    question = sample.get("question_text", "")
    choices = sample.get("choices", [])
    choices_str = ", ".join(choices) if choices else "N/A"
    answer = sample.get("answer_text", "")

    # Step 1: Initial Grounding (uses pre-built prompt from batch JSON)
    s1_raw = sample.get("prompt", "")
    try:
        s1_prompt = s1_raw.format(height=img_h, width=img_w)
    except Exception:
        s1_prompt = s1_raw

    if is_kimi(model_id):
        data_uri, _, _ = image_to_base64_url(img_path)
        turn1_messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": s1_prompt},
            ]}
        ]
        s1_reply = invoke_kimi(bedrock_rt, model_id, turn1_messages, gen_cfg)
    else:
        turn1_messages = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                {"type": "text", "text": s1_prompt},
            ]}
        ]
        s1_reply = invoke_claude(bedrock_rt, model_id, turn1_messages, gen_cfg)

    s1_answer, s1_expl = split_reply(s1_reply)
    s1_boxes, s1_errors = parse_answer_boxes(s1_answer)

    # Build Step 2 Verify & Refine prompt
    s1_pred_text = f"Answer: {s1_answer}"
    if s1_expl:
        s1_pred_text += f"\nExplanation: {s1_expl}"

    vr_raw = build_verify_and_refine_prompt(
        question_id=q_id,
        question=question,
        choices_str=choices_str,
        answer=answer,
        previous_prediction=s1_pred_text,
    )
    try:
        vr_prompt = vr_raw.format(height=img_h, width=img_w)
    except Exception:
        vr_prompt = vr_raw

    if is_kimi(model_id):
        turn2_messages = [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": s1_prompt},
            ]},
            {"role": "assistant", "content": s1_reply},
            {"role": "user", "content": vr_prompt},
        ]
        s2_reply = invoke_kimi(bedrock_rt, model_id, turn2_messages, gen_cfg)
    else:
        turn2_messages = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                {"type": "text", "text": s1_prompt},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": s1_reply},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": vr_prompt},
            ]},
        ]
        s2_reply = invoke_claude(bedrock_rt, model_id, turn2_messages, gen_cfg)

    s2_answer, s2_expl = split_reply(s2_reply)
    s2_boxes, s2_errors = parse_answer_boxes(s2_answer)

    # Enforce box-drop rule: Step 2 must have >= Step 1 boxes
    if len(s2_boxes) < len(s1_boxes):
        final_answer = s1_answer
        final_expl = s1_expl
        final_boxes = s1_boxes
        final_errors = s1_errors
        final_reply = s1_reply
        used_step = "step1_fallback"
    else:
        final_answer = s2_answer
        final_expl = s2_expl
        final_boxes = s2_boxes
        final_errors = s2_errors
        final_reply = s2_reply
        used_step = "step2"

    return {
        "sample_id": sid,
        "uid": sample.get("uid", ""),
        "json_path": sample.get("json_path", ""),
        "dataset": dataset,
        "q_id": q_id,
        "image_path_abs": img_path,
        "image_path_rel": sample.get("image_path_rel", ""),
        "image_width": img_w,
        "image_height": img_h,
        "question_text": question,
        "choices": choices,
        "answer_text": answer,
        "gt_boxes_key": sample.get("boxes_key", "none"),
        "gt_boxes_raw": sample.get("boxes_raw", []),
        "step1_prompt": s1_prompt,
        "step1_raw_reply": s1_reply,
        "step1_answer_str": s1_answer,
        "step1_boxes_parsed": s1_boxes,
        "step1_parse_errors": s1_errors,
        "step2_prompt": vr_prompt,
        "step2_raw_reply": s2_reply,
        "step2_answer_str": s2_answer,
        "step2_boxes_parsed": s2_boxes,
        "step2_parse_errors": s2_errors,
        "used_step": used_step,
        "pred_raw_reply": final_reply,
        "pred_answer_str": final_answer,
        "pred_explanation_str": final_expl,
        "pred_boxes_parsed": final_boxes,
        "pred_parse_errors": final_errors,
        "prompt": vr_prompt,
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run_dataset(samples, bedrock_rt, model_id, gen_cfg, workers, save_every, results_path):
    processed_set, saved_results = load_existing_progress(results_path)
    log.info(f"Resume: loaded {len(saved_results)} results; processed={len(processed_set)}")

    worklist = []
    for s in samples:
        sid = make_sample_id(s)
        if sid not in processed_set:
            worklist.append(s)

    log.info(f"To process: {len(worklist)} samples with {workers} workers")
    if not worklist:
        return saved_results

    results = saved_results[:]
    errors_count = 0
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_sample, s, bedrock_rt, model_id, gen_cfg): s
            for s in worklist
        }
        for future in as_completed(futures):
            try:
                rec = future.result()
                results.append(rec)
                done += 1
                if done % 10 == 0:
                    log.info(f"  [{done}/{len(worklist)}] processed")
            except Exception as e:
                errors_count += 1
                sample = futures[future]
                log.warning(f"  Error on {sample.get('uid','?')}: {e}")

            if done % save_every == 0:
                atomic_write_json(results, results_path)

    log.info(f"Final save: results={len(results)}, errors={errors_count}")
    atomic_write_json(results, results_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Self-correct Bedrock real-time inference")
    parser.add_argument("-b", "--batch_dir", default="batches/")
    parser.add_argument("-o", "--pred_dir", default="predictions/kimi_verge/")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    gen_cfg = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    bedrock_rt = boto3.client("bedrock-runtime", region_name=args.region)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    batch_dir = Path(args.batch_dir)
    batch_files = sorted(batch_dir.glob("*.json"))

    for batch_path in batch_files:
        ds_name = batch_path.stem
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {ds_name}  ({batch_path.name})")

        with batch_path.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        if args.max_samples is not None:
            samples = samples[:args.max_samples]
            log.info(f"  Limiting to {len(samples)} samples")

        out_path = pred_dir / f"pred_{ds_name}.json"

        run_dataset(
            samples=samples,
            bedrock_rt=bedrock_rt,
            model_id=args.model_id,
            gen_cfg=gen_cfg,
            workers=args.workers,
            save_every=args.save_every,
            results_path=str(out_path),
        )
        log.info(f"Done dataset={ds_name} → {out_path}")

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
