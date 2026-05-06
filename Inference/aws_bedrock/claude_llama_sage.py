#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
claude_llama_sage.py  (SAGE — Select And Ground Evidence)

Two-stage "Select then Ground" via Bedrock Batch Inference API.
All datasets are consolidated into ONE JSONL → ONE batch job per phase.

  Phase 1:  Submit Stage 1 (SELECT) — single batch job for all 2445 samples
  Phase 2:  Parse Stage 1 output → build Stage 2 (GROUND) multi-turn → submit
  Finalize: Parse Stage 2 output → split into per-dataset pred_*.json

Supports: Claude (Anthropic) and Llama 4 (Meta) model formats.

Usage:
  python claude_llama_sage.py phase1 --role_arn <ARN> --model_id <ID> [--max_samples 5]
  python claude_llama_sage.py phase2 --jobs_file <path> --role_arn <ARN>
  python claude_llama_sage.py finalize --jobs_file <path> --pred_dir <dir>
"""

import re
import json
import base64
import argparse
import logging
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prompts"))

import boto3
from PIL import Image

from prompts_sage import build_stage1_prompt, build_stage2_prompt

# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"
DEFAULT_REGION   = "us-east-1"
DEFAULT_BUCKET   = "<YOUR_S3_BUCKET>"

TERMINAL_STATES = {"Completed", "Failed", "Stopped", "PartiallyCompleted"}
SUCCESS_STATES  = {"Completed", "PartiallyCompleted"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_sample_id(sample):
    uid   = (sample.get("uid") or "").strip()
    jpath = (sample.get("json_path") or "").strip()
    qid   = (sample.get("q_id") or "").strip()
    img   = Path(sample.get("image_path_rel") or sample.get("image_path_abs") or "noimg").name
    core  = f"uid::{uid}::qid::{qid}::img::{img}"
    if jpath:
        core += f"::json::{Path(jpath).name}"
    return core


def image_to_base64(image_path, max_side=1568):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg", w, h


def atomic_write_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def upload_bytes(s3, bucket, key, data, content_type="application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    log.info(f"  Uploaded s3://{bucket}/{key}  ({len(data):,} bytes)")


def parse_s3_uri(uri):
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def list_s3_keys(s3, bucket, prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_s3_text(s3, bucket, key):
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def is_llama(model_id):
    return "meta" in model_id.lower() or "llama" in model_id.lower()


_NUM = r"-?\d+(?:\.\d+)?"
_BOX_RE = re.compile(rf"(?P<id>[A-Za-z0-9_\-]+)\s*/\s*(?P<x>{_NUM})\s*,\s*(?P<y>{_NUM})\s*,\s*(?P<w>{_NUM})\s*,\s*(?P<h>{_NUM})")

def parse_answer_boxes(answer_str):
    if not answer_str: return [], ["empty_answer"]
    boxes, errors = [], []
    for m in _BOX_RE.finditer(answer_str):
        try:
            boxes.append({"id": m.group("id"), "x": float(m.group("x")), "y": float(m.group("y")), "w": float(m.group("w")), "h": float(m.group("h"))})
        except Exception as e:
            errors.append(f"parse_float_error:{e}")
    if not boxes: errors.append("no_boxes_matched")
    return boxes, errors


def split_reply(raw_reply):
    if not raw_reply: return "", ""
    txt = str(raw_reply)
    if "Explanation:" in txt:
        a, e = txt.split("Explanation:", 1)
        return a.replace("Answer:", "").strip(), e.strip()
    return txt.replace("Answer:", "").strip(), ""


def extract_reply(model_output):
    if isinstance(model_output, str):
        try: model_output = json.loads(model_output)
        except json.JSONDecodeError: return ""
    # Llama format
    if "generation" in model_output:
        return model_output["generation"].strip()
    # Anthropic format
    for block in model_output.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "").strip()
    return ""


def build_anthropic_messages(content_blocks, gen_cfg):
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": gen_cfg["max_new_tokens"],
        "temperature": gen_cfg["temperature"],
        "messages": content_blocks,
    }


def build_llama_input(prompt_text, b64_data, gen_cfg, num_images=1):
    img_tags = "<|image|>" * num_images
    llama_prompt = (
        f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
        f"{img_tags}{prompt_text}<|eot|>"
        f"<|header_start|>assistant<|header_end|>\n\n"
    )
    return {
        "prompt": llama_prompt,
        "images": [b64_data],
        "max_gen_len": gen_cfg["max_new_tokens"],
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg.get("top_p", 0.7),
    }


def build_llama_multiturn(s1_prompt, s1_reply, s2_prompt, b64_data, gen_cfg):
    llama_prompt = (
        f"<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
        f"<|image|>{s1_prompt}<|eot|>"
        f"<|header_start|>assistant<|header_end|>\n\n"
        f"{s1_reply}<|eot|>"
        f"<|header_start|>user<|header_end|>\n\n"
        f"{s2_prompt}<|eot|>"
        f"<|header_start|>assistant<|header_end|>\n\n"
    )
    return {
        "prompt": llama_prompt,
        "images": [b64_data],
        "max_gen_len": gen_cfg["max_new_tokens"],
        "temperature": gen_cfg["temperature"],
        "top_p": gen_cfg.get("top_p", 0.7),
    }


def poll_jobs(bedrock, job_list, poll_interval=60):
    pending = {j["job_arn"]: j.get("label", "job") for j in job_list}
    statuses = {}
    while pending:
        for arn in list(pending.keys()):
            resp = bedrock.get_model_invocation_job(jobIdentifier=arn)
            status = resp["status"]
            log.info(f"  [{pending[arn]}] {status}")
            if status in TERMINAL_STATES:
                statuses[arn] = status
                del pending[arn]
        if pending:
            log.info(f"  {len(pending)} job(s) running — waiting {poll_interval}s ...")
            time.sleep(poll_interval)
    return statuses


# ===========================================================================
# PHASE 1: All datasets → single Stage 1 JSONL → one batch job
# ===========================================================================
def phase1(args):
    gen_cfg = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", region_name=args.region)
    bedrock = boto3.client("bedrock", region_name=args.region)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    llama = is_llama(args.model_id)

    batch_dir = Path(args.batch_dir)
    all_samples = []
    for bp in sorted(batch_dir.glob("*.json")):
        with bp.open() as f:
            samples = json.load(f)
        if args.max_samples is not None:
            samples = samples[:args.max_samples]
        all_samples.extend(samples)
        log.info(f"  {bp.stem}: {len(samples)} samples")

    log.info(f"Total: {len(all_samples)} samples")

    jsonl_lines = []
    meta_records = {}

    for sample in all_samples:
        sid = make_sample_id(sample)
        img_path = sample.get("image_path_abs", "")
        b64_data, media_type, img_w, img_h = image_to_base64(img_path)

        dataset = sample.get("dataset", "")
        q_id = sample.get("q_id", "")
        question = sample.get("question_text", "")
        choices = sample.get("choices", [])
        choices_str = ", ".join(choices) if choices else "N/A"
        answer = sample.get("answer_text", "")

        s1_raw = build_stage1_prompt(dataset, q_id, question, choices_str, answer)
        try: s1_prompt = s1_raw.format(height=img_h, width=img_w)
        except: s1_prompt = s1_raw

        if llama:
            model_input = build_llama_input(s1_prompt, b64_data, gen_cfg)
        else:
            model_input = build_anthropic_messages([
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                    {"type": "text", "text": s1_prompt},
                ]}
            ], gen_cfg)

        jsonl_lines.append(json.dumps({"recordId": sid, "modelInput": model_input}, ensure_ascii=False))
        meta_records[sid] = {
            "sample_id": sid, "uid": sample.get("uid", ""),
            "json_path": sample.get("json_path", ""), "dataset": dataset, "q_id": q_id,
            "image_path_abs": img_path, "image_path_rel": sample.get("image_path_rel", ""),
            "image_width": img_w, "image_height": img_h,
            "question_text": question, "choices": choices,
            "answer_text": answer, "gt_boxes_key": sample.get("boxes_key", "none"),
            "gt_boxes_raw": sample.get("boxes_raw", []),
            "stage1_prompt": s1_prompt,
        }

    jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
    log.info(f"  JSONL: {len(jsonl_lines)} records, {len(jsonl_bytes):,} bytes")

    meta_path = out_dir / f"phase1_meta_{run_ts}.json"
    atomic_write_json(meta_records, str(meta_path))

    if args.dry_run:
        (out_dir / f"phase1_{run_ts}.jsonl").write_bytes(jsonl_bytes)
        log.info(f"  [dry_run] saved locally")
        return

    input_key = f"diagram_atb/2stage/phase1/{run_ts}/all.jsonl"
    output_prefix = f"diagram_atb/2stage/phase1_output/{run_ts}/"
    upload_bytes(s3, args.bucket, input_key, jsonl_bytes, "application/x-ndjson")

    job_name = re.sub(r"[^a-zA-Z0-9\-]", "-", f"2stg-p1-all-{run_ts}")[:63]
    response = bedrock.create_model_invocation_job(
        jobName=job_name, modelId=args.model_id, roleArn=args.role_arn,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{args.bucket}/{input_key}", "s3InputFormat": "JSONL"}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{args.bucket}/{output_prefix}"}},
    )
    job_arn = response["jobArn"]
    log.info(f"  Job: {job_name}")
    log.info(f"  ARN: {job_arn}")

    jobs_file = out_dir / f"phase1_jobs_{run_ts}.json"
    atomic_write_json([{
        "label": "all_datasets", "job_name": job_name, "job_arn": job_arn,
        "model_id": args.model_id,
        "output_s3_prefix": f"s3://{args.bucket}/{output_prefix}",
        "meta_local": str(meta_path.resolve()),
    }], str(jobs_file))
    log.info(f"\nJobs file → {jobs_file}")
    log.info(f"Next: python {__file__} phase2 --jobs_file {jobs_file} --role_arn {args.role_arn}")


# ===========================================================================
# PHASE 2: Parse Stage 1 → build multi-turn Stage 2 → single batch job
# ===========================================================================
def phase2(args):
    gen_cfg = {"temperature": args.temperature, "top_p": args.top_p, "max_new_tokens": args.max_new_tokens}
    with open(args.jobs_file) as f:
        phase1_jobs = json.load(f)

    s3 = boto3.client("s3", region_name=args.region)
    bedrock = boto3.client("bedrock", region_name=args.region)
    job = phase1_jobs[0]
    model_id = job.get("model_id", args.model_id)
    llama = is_llama(model_id)

    if not args.skip_poll:
        log.info("Polling Phase 1 ...")
        statuses = poll_jobs(bedrock, phase1_jobs, args.poll_interval)
        for j in phase1_jobs:
            s = statuses.get(j["job_arn"], "Unknown")
            log.info(f"  {j['label']}: {s}")
            if s not in SUCCESS_STATES:
                log.error(f"  Job failed!")
                return

    meta_records = json.load(open(job["meta_local"]))

    # Download output
    bucket, prefix = parse_s3_uri(job["output_s3_prefix"])
    keys = [k for k in list_s3_keys(s3, bucket, prefix) if k.endswith(".jsonl.out")]
    s1_replies = {}
    for key in keys:
        for line in download_s3_text(s3, bucket, key).splitlines():
            if not line.strip(): continue
            obj = json.loads(line)
            rid = obj.get("recordId", "")
            s1_replies[rid] = extract_reply(obj.get("modelOutput", {}))
    log.info(f"  Got {len(s1_replies)} Stage 1 replies")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jsonl_lines = []
    meta2_records = {}

    for sid, reply in s1_replies.items():
        meta = meta_records.get(sid, {})
        if not meta: continue
        img_path = meta["image_path_abs"]
        b64_data, media_type, img_w, img_h = image_to_base64(img_path)

        s2_raw = build_stage2_prompt(
            dataset=meta["dataset"], question_id=meta["q_id"],
            question=meta["question_text"],
            choices_str=", ".join(meta.get("choices", [])) or "N/A",
            answer=meta["answer_text"], stage1_elements=reply,
        )
        try: s2_prompt = s2_raw.format(height=img_h, width=img_w)
        except: s2_prompt = s2_raw

        s1_prompt = meta.get("stage1_prompt", "")

        if llama:
            model_input = build_llama_multiturn(s1_prompt, reply, s2_prompt, b64_data, gen_cfg)
        else:
            model_input = build_anthropic_messages([
                {"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64_data}},
                    {"type": "text", "text": s1_prompt},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": reply}]},
                {"role": "user", "content": [{"type": "text", "text": s2_prompt}]},
            ], gen_cfg)

        jsonl_lines.append(json.dumps({"recordId": sid, "modelInput": model_input}, ensure_ascii=False))
        meta2_records[sid] = {**meta, "stage1_raw_reply": reply, "stage2_prompt": s2_prompt}

    jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
    log.info(f"  Built {len(jsonl_lines)} Stage 2 records, {len(jsonl_bytes):,} bytes")

    meta2_path = out_dir / f"phase2_meta_{run_ts}.json"
    atomic_write_json(meta2_records, str(meta2_path))

    if args.dry_run:
        (out_dir / f"phase2_{run_ts}.jsonl").write_bytes(jsonl_bytes)
        log.info("[dry_run] saved locally")
        return

    input_key = f"diagram_atb/2stage/phase2/{run_ts}/all.jsonl"
    output_prefix = f"diagram_atb/2stage/phase2_output/{run_ts}/"
    upload_bytes(s3, args.bucket, input_key, jsonl_bytes, "application/x-ndjson")

    job_name = re.sub(r"[^a-zA-Z0-9\-]", "-", f"2stg-p2-all-{run_ts}")[:63]
    response = bedrock.create_model_invocation_job(
        jobName=job_name, modelId=model_id, roleArn=args.role_arn,
        inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{args.bucket}/{input_key}", "s3InputFormat": "JSONL"}},
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{args.bucket}/{output_prefix}"}},
    )
    job_arn = response["jobArn"]
    log.info(f"  Job: {job_name}  ARN: {job_arn}")

    jobs_file = out_dir / f"phase2_jobs_{run_ts}.json"
    atomic_write_json([{
        "label": "all_datasets", "job_name": job_name, "job_arn": job_arn,
        "model_id": model_id,
        "output_s3_prefix": f"s3://{args.bucket}/{output_prefix}",
        "meta_local": str(meta2_path.resolve()),
    }], str(jobs_file))
    log.info(f"\nJobs file → {jobs_file}")
    log.info(f"Next: python {__file__} finalize --jobs_file {jobs_file}")


# ===========================================================================
# FINALIZE: Parse Phase 2 → split into per-dataset pred_*.json
# ===========================================================================
def finalize(args):
    with open(args.jobs_file) as f:
        phase2_jobs = json.load(f)
    s3 = boto3.client("s3", region_name=args.region)
    bedrock = boto3.client("bedrock", region_name=args.region)

    if not args.skip_poll:
        log.info("Polling Phase 2 ...")
        poll_jobs(bedrock, phase2_jobs, args.poll_interval)

    job = phase2_jobs[0]
    meta_records = json.load(open(job["meta_local"]))
    bucket, prefix = parse_s3_uri(job["output_s3_prefix"])
    keys = [k for k in list_s3_keys(s3, bucket, prefix) if k.endswith(".jsonl.out")]

    by_dataset = defaultdict(list)
    for key in keys:
        for line in download_s3_text(s3, bucket, key).splitlines():
            if not line.strip(): continue
            obj = json.loads(line)
            rid = obj.get("recordId", "")
            s2_reply = extract_reply(obj.get("modelOutput", {}))
            meta = meta_records.get(rid, {})
            pred_answer, pred_expl = split_reply(s2_reply)
            pred_boxes, parse_errors = parse_answer_boxes(pred_answer)

            rec = {
                "sample_id": meta.get("sample_id", rid),
                "uid": meta.get("uid", ""), "json_path": meta.get("json_path", ""),
                "dataset": meta.get("dataset", ""), "q_id": meta.get("q_id", ""),
                "image_path_abs": meta.get("image_path_abs", ""),
                "image_path_rel": meta.get("image_path_rel", ""),
                "image_width": meta.get("image_width", 0), "image_height": meta.get("image_height", 0),
                "question_text": meta.get("question_text", ""), "choices": meta.get("choices", []),
                "answer_text": meta.get("answer_text", ""),
                "gt_boxes_key": meta.get("gt_boxes_key", "none"),
                "gt_boxes_raw": meta.get("gt_boxes_raw", []),
                "stage1_prompt": meta.get("stage1_prompt", ""),
                "stage1_raw_reply": meta.get("stage1_raw_reply", ""),
                "stage2_prompt": meta.get("stage2_prompt", ""),
                "stage2_raw_reply": s2_reply,
                "pred_raw_reply": s2_reply,
                "pred_answer_str": pred_answer, "pred_explanation_str": pred_expl,
                "pred_boxes_parsed": pred_boxes, "pred_parse_errors": parse_errors,
                "prompt": meta.get("stage2_prompt", ""),
            }
            by_dataset[meta.get("dataset", "unknown")].append(rec)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    for ds, results in sorted(by_dataset.items()):
        out_path = pred_dir / f"pred_{ds}.json"
        atomic_write_json(results, str(out_path))
        log.info(f"  {ds}: {len(results)} results → {out_path}")

    log.info("\nAll done.")


# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Two-stage Bedrock Batch (consolidated)")
    sub = parser.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("phase1")
    p1.add_argument("-b", "--batch_dir", default="batches/")
    p1.add_argument("--out_dir", default="bedrock_jobs_2stage/")
    p1.add_argument("--bucket", default=DEFAULT_BUCKET)
    p1.add_argument("--region", default=DEFAULT_REGION)
    p1.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p1.add_argument("--role_arn", required=True)
    p1.add_argument("--temperature", type=float, default=0.2)
    p1.add_argument("--top_p", type=float, default=0.7)
    p1.add_argument("--max_new_tokens", type=int, default=1024)
    p1.add_argument("--max_samples", type=int, default=None)
    p1.add_argument("--dry_run", action="store_true")

    p2 = sub.add_parser("phase2")
    p2.add_argument("--jobs_file", required=True)
    p2.add_argument("--out_dir", default="bedrock_jobs_2stage/")
    p2.add_argument("--bucket", default=DEFAULT_BUCKET)
    p2.add_argument("--region", default=DEFAULT_REGION)
    p2.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p2.add_argument("--role_arn", required=True)
    p2.add_argument("--temperature", type=float, default=0.2)
    p2.add_argument("--top_p", type=float, default=0.7)
    p2.add_argument("--max_new_tokens", type=int, default=1024)
    p2.add_argument("--poll_interval", type=int, default=60)
    p2.add_argument("--skip_poll", action="store_true")
    p2.add_argument("--dry_run", action="store_true")

    pf = sub.add_parser("finalize")
    pf.add_argument("--jobs_file", required=True)
    pf.add_argument("--pred_dir", default="predictions/claude_llama_sage/")
    pf.add_argument("--region", default=DEFAULT_REGION)
    pf.add_argument("--poll_interval", type=int, default=60)
    pf.add_argument("--skip_poll", action="store_true")

    args = parser.parse_args()
    {"phase1": phase1, "phase2": phase2, "finalize": finalize}[args.phase](args)


if __name__ == "__main__":
    main()
