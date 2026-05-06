# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import re
import json
import math
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
from PIL import Image
from typing import Any, Dict, List, Tuple
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoConfig,
)
from tqdm.auto import tqdm

# =========================
# IO helpers & logging
# =========================
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def atomic_write_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def log_line(log_fp, msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    log_fp.write(line + "\n")
    log_fp.flush()
    if also_print:
        print(line)


# =========================
# Resume IDs  (same logic as gemma3.py)
# =========================
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


# =========================
# Parse model "Answer:" bbox string into structured list
# =========================
_NUM = r"-?\d+(?:\.\d+)?"
_BOX_RE = re.compile(
    rf"(?P<id>[A-Za-z0-9_\-]+)\s*/\s*(?P<x>{_NUM})\s*,\s*(?P<y>{_NUM})\s*,\s*(?P<w>{_NUM})\s*,\s*(?P<h>{_NUM})"
)


def parse_answer_boxes(answer_str: str):
    errors = []
    if not answer_str:
        return [], ["empty_answer"]
    text = answer_str.strip()
    boxes = []
    for m in _BOX_RE.finditer(text):
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
    a = txt.replace("Answer:", "").strip()
    return a, ""


def strip_thinking(raw_reply: str) -> str:
    """Remove <think>...</think> blocks from Qwen3.5 output."""
    if raw_reply is None:
        return ""
    return re.sub(r"<think>.*?</think>\s*", "", raw_reply, flags=re.DOTALL).strip()


# =========================
# Qwen3.5 single-sample inference
# =========================
def run_single_inference(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    gen_cfg: dict,
) -> str:
    """
    Run inference on a single image+prompt using Qwen3.5's chat template.
    Thinking mode is disabled so the model outputs the answer directly.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # Step 1: Build the text prompt via the tokenizer's chat template
    #         (supports enable_thinking kwarg, unlike processor.apply_chat_template)
    text_prompt = processor.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )

    # Step 2: Process text + image together
    inputs = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **gen_cfg,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    reply = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    return reply.strip()


# =========================
# Inference loop
# =========================
def run_chunked_inference_dataset(
    samples: List[dict],
    model,
    processor,
    gen_cfg: dict,
    save_every: int,
    results_path: str,
    log_fp,
):
    processed_set, saved_results = load_existing_progress(results_path)
    log_line(log_fp, f"Resume: loaded {len(saved_results)} results; processed set size={len(processed_set)}")

    all_sids = [make_sample_id(s) for s in samples]
    unique_sids = len(set(all_sids))
    dups = [t for t in Counter(all_sids).most_common(5) if t[1] > 1]
    if dups:
        log_line(log_fp, f"Top duplicate SIDs (input): {dups}")

    # Build worklist
    worklist = []
    for s in samples:
        sid = make_sample_id(s)
        if sid not in processed_set:
            s["_sample_id"] = sid
            worklist.append(s)

    log_line(log_fp, f"SID stats: total={len(all_sids)}, unique={unique_sids}, processed={len(processed_set)}, to_process={len(worklist)}")

    if not worklist:
        return saved_results

    results = saved_results[:]
    oom_skipped = 0

    for idx, sample in enumerate(tqdm(worklist, desc="Samples")):
        try:
            img_path_abs = sample.get("image_path_abs")
            if not img_path_abs:
                raise ValueError("Missing image_path_abs in sample")

            image = Image.open(img_path_abs).convert("RGB")
            img_w, img_h = image.size

            # Fill prompt placeholders at model-time
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
                "image_path_abs": img_path_abs,
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

            raw_reply = run_single_inference(
                model=model,
                processor=processor,
                image=image,
                prompt_text=prompt_filled,
                gen_cfg=gen_cfg,
            )

            # Strip thinking tokens if present
            raw_reply_clean = strip_thinking(raw_reply)

            pred_answer_str, pred_expl_str = split_reply(raw_reply_clean)
            pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)

            results.append({
                **meta,
                "pred_raw_reply": raw_reply_clean,
                "pred_answer_str": pred_answer_str,
                "pred_explanation_str": pred_expl_str,
                "pred_boxes_parsed": pred_boxes,
                "pred_parse_errors": parse_errors,
            })

        except torch.cuda.OutOfMemoryError:
            log_line(log_fp, f"OOM on sample {idx} - Skipping")
            oom_skipped += 1
            torch.cuda.empty_cache()
            continue

        except Exception as e:
            log_line(log_fp, f"Error on sample {idx}: {e}")
            continue

        # Periodic save
        if (idx + 1) % save_every == 0:
            atomic_write_json(results, results_path)

    log_line(log_fp, f"Final save: results={len(results)}, OOM skipped={oom_skipped}")
    atomic_write_json(results, results_path)
    return results


# =========================
# Main
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagram Attribution inference with Qwen3.5-35B-A3B"
    )
    parser.add_argument("-b", "--batch_dir",  default="./batches/")
    parser.add_argument("-o", "--pred_dir",   default="./predictions/qwen3_5/")
    parser.add_argument("--checkpoint",       default="Qwen/Qwen3.5-35B-A3B",
                        help="HuggingFace model ID or local path to Qwen3.5")
    parser.add_argument("--temperature",      type=float, default=0.2)
    parser.add_argument("--top_p",            type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_output_tokens", type=int,  default=2048)
    parser.add_argument("--save_every",       type=int,   default=20,
                        help="Save results every N samples")
    parser.add_argument("--max_samples",       type=int,   default=None,
                        help="Max samples per dataset (e.g. 1 for a quick test)")
    args = parser.parse_args()

    # ── Model ──────────────────────────────────────────────────
    checkpoint = args.checkpoint

    print(f"Loading Qwen3.5-35B-A3B from {checkpoint} ...")

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    gen_cfg = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_output_tokens,
        do_sample=True,
    )

    # ── Run ────────────────────────────────────────────────────
    ensure_dir(args.pred_dir)
    logs_dir = Path(args.pred_dir) / "logs"
    ensure_dir(logs_dir)

    batch_dir = Path(args.batch_dir)
    batch_files = sorted([p for p in batch_dir.iterdir() if p.suffix.lower() == ".json"])

    for batch_path in tqdm(batch_files, desc="Datasets"):
        ds_name = batch_path.stem
        print(f"\nProcessing dataset batch: {batch_path.name}")

        out_path = Path(args.pred_dir) / f"pred_{ds_name}.json"

        log_path = logs_dir / f"{ds_name}.log"
        with open(log_path, "a", encoding="utf-8") as log_fp:
            log_line(log_fp, f"=== START DATASET: {ds_name} ===")
            log_line(log_fp, f"Input batch: {batch_path}")

            with open(batch_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Expected LIST in {batch_path}, got {type(data)}")

            if args.max_samples is not None:
                data = data[:args.max_samples]
                log_line(log_fp, f"Limiting to {len(data)} sample(s) per dataset")

            results = run_chunked_inference_dataset(
                samples=data,
                model=model,
                processor=processor,
                gen_cfg=gen_cfg,
                save_every=args.save_every,
                results_path=str(out_path),
                log_fp=log_fp,
            )

            log_line(log_fp, f"Done dataset={ds_name}: wrote results={len(results)} to {out_path}")
            log_line(log_fp, f"=== END DATASET: {ds_name} ===\n")


if __name__ == "__main__":
    main()
