#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_results.py

Polls Bedrock batch jobs for completion, downloads output JSONL from S3,
and writes pred_<dataset>.json files in the same format as gemma3.py.

Usage:
    python parse_results.py \
        --jobs_file ./bedrock_jobs/jobs_TIMESTAMP.json \
        [--pred_dir ./predictions/bedrock] \
        [--poll_interval 60]   # seconds between status checks

Output format matches pred_*.json from gemma3.py:
  - sample_id, uid, q_id, dataset, image_path_abs, ...
  - pred_raw_reply, pred_answer_str, pred_explanation_str
  - pred_boxes_parsed: [{id, x, y, w, h}, ...]
  - pred_parse_errors: [...]
"""

import re
import json
import time
import logging
import argparse
from pathlib import Path

import boto3

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
# Job terminal states
# ---------------------------------------------------------------------------
TERMINAL_STATES   = {"Completed", "Failed", "Stopped", "PartiallyCompleted"}
SUCCESS_STATES    = {"Completed", "PartiallyCompleted"}

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
# S3 download helpers
# ---------------------------------------------------------------------------
def list_s3_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_s3_text(s3, bucket: str, key: str) -> str:
    resp = s3.get_object(Bucket=bucket, Key=key)
    return resp["Body"].read().decode("utf-8")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """s3://bucket/prefix  →  (bucket, prefix)"""
    assert uri.startswith("s3://"), f"Not an S3 URI: {uri}"
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


# ---------------------------------------------------------------------------
# Poll until all jobs are in terminal state
# ---------------------------------------------------------------------------
def poll_jobs(bedrock, job_list: list[dict], poll_interval: int) -> dict[str, str]:
    """Returns {job_arn: final_status}"""
    pending = {j["job_arn"]: j["dataset"] for j in job_list}
    statuses = {}

    while pending:
        for arn in list(pending.keys()):
            resp = bedrock.get_model_invocation_job(jobIdentifier=arn)
            status = resp["status"]
            ds = pending[arn]
            log.info(f"  [{ds}] {status}")
            if status in TERMINAL_STATES:
                statuses[arn] = status
                del pending[arn]

        if pending:
            log.info(f"  {len(pending)} job(s) still running — waiting {poll_interval}s ...")
            time.sleep(poll_interval)

    return statuses


# ---------------------------------------------------------------------------
# Parse one output JSONL file
# ---------------------------------------------------------------------------
def parse_output_jsonl(text: str, meta_records: dict) -> list[dict]:
    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning(f"  JSON decode error on output line: {e}")
            continue

        record_id = obj.get("recordId", "")

        # Extract text from model response (handles Anthropic, Meta native, and Converse formats)
        raw_reply = ""
        model_output = obj.get("modelOutput", {})
        if isinstance(model_output, str):
            try:
                model_output = json.loads(model_output)
            except json.JSONDecodeError:
                model_output = {}

        if "error" in obj:
            log.warning(f"  Error record [{record_id}]: {obj['error']}")
            raw_reply = f"ERROR: {obj['error']}"
        else:
            # Meta Llama 4 native format: modelOutput.generation
            if "generation" in model_output:
                raw_reply = model_output["generation"]
            else:
                # Anthropic format: modelOutput.content[].type=="text"
                content = model_output.get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        raw_reply = block.get("text", "")
                        break

                # Converse format: modelOutput.output.message.content[].text
                if not raw_reply:
                    msg_content = (
                        model_output
                        .get("output", {})
                        .get("message", {})
                        .get("content", [])
                    )
                    for block in msg_content:
                        if isinstance(block, dict) and "text" in block:
                            raw_reply = block["text"]
                            break

        pred_answer_str, pred_expl_str = split_reply(raw_reply)
        pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)

        # Get metadata for this record
        meta = meta_records.get(record_id, {})
        if not meta:
            log.warning(f"  No meta found for recordId={record_id!r}")

        results.append({
            **meta,
            "pred_raw_reply": raw_reply,
            "pred_answer_str": pred_answer_str,
            "pred_explanation_str": pred_expl_str,
            "pred_boxes_parsed": pred_boxes,
            "pred_parse_errors": parse_errors,
            # Bedrock extras for debugging
            "_bedrock_record_id": record_id,
            "_bedrock_stop_reason": model_output.get("stop_reason", ""),
            "_bedrock_model": model_output.get("model", ""),
            "_bedrock_prompt_tokens": model_output.get("prompt_token_count", ""),
            "_bedrock_gen_tokens": model_output.get("generation_token_count", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Atomic JSON write (same pattern as gemma3.py)
# ---------------------------------------------------------------------------
def atomic_write_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    log.info(f"  Saved → {path}  ({len(obj)} records)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Download + parse Bedrock Batch results")
    parser.add_argument("--jobs_file", required=True,
                        help="Path to jobs_*.json written by claude_llama_edge.py")
    parser.add_argument("--pred_dir",  default="./predictions/bedrock",
                        help="Where to write pred_*.json output files")
    parser.add_argument("--region",    default="us-east-1")
    parser.add_argument("--poll_interval", type=int, default=60,
                        help="Seconds between job status polls")
    parser.add_argument("--skip_poll", action="store_true",
                        help="Skip polling and go straight to download (if jobs already complete)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    with open(args.jobs_file, "r", encoding="utf-8") as f:
        job_list = json.load(f)

    log.info(f"Loaded {len(job_list)} job(s) from {args.jobs_file}")

    s3      = boto3.client("s3",      region_name=args.region)
    bedrock = boto3.client("bedrock", region_name=args.region)

    # ── Poll ──────────────────────────────────────────────────────────────
    if not args.skip_poll:
        log.info("Polling job statuses ...")
        statuses = poll_jobs(bedrock, job_list, args.poll_interval)
        log.info(f"\nFinal statuses: {statuses}")
    else:
        log.info("Skipping poll (--skip_poll set)")
        statuses = {j["job_arn"]: "Completed" for j in job_list}

    # ── Download + parse ──────────────────────────────────────────────────
    for job in job_list:
        ds_name = job["dataset"]
        arn     = job["job_arn"]
        status  = statuses.get(arn, "Unknown")

        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {ds_name}  status={status}")

        if status not in SUCCESS_STATES:
            log.warning(f"  Job did not succeed ({status}) — skipping")
            continue

        # Load metadata
        meta_path = job.get("meta_local", "")
        if not meta_path or not Path(meta_path).exists():
            log.error(f"  Meta file not found: {meta_path} — cannot reconstruct records")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_records = json.load(f)

        log.info(f"  Loaded {len(meta_records)} meta records from {meta_path}")

        # List output files in S3
        bucket, prefix = parse_s3_uri(job["output_s3_prefix"])
        keys = list_s3_keys(s3, bucket, prefix)
        jsonl_keys = [k for k in keys if k.endswith(".jsonl.out") or k.endswith(".jsonl") or k.endswith(".json.out") or k.endswith(".json")]
        # skip manifest files
        jsonl_keys = [k for k in jsonl_keys if "manifest" not in k.split("/")[-1]]

        if not jsonl_keys:
            log.warning(f"  No output JSONL found at s3://{bucket}/{prefix}")
            continue

        log.info(f"  Found {len(jsonl_keys)} output file(s)")

        all_results = []
        for key in jsonl_keys:
            log.info(f"  Downloading s3://{bucket}/{key}")
            text = download_s3_text(s3, bucket, key)
            parsed = parse_output_jsonl(text, meta_records)
            all_results.extend(parsed)
            log.info(f"    → {len(parsed)} records")

        out_path = pred_dir / f"pred_{ds_name}.json"
        atomic_write_json(all_results, out_path)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
