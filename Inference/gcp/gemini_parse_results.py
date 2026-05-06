#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini_parse_results.py

Polls Vertex AI Gemini BatchPredictionJobs for completion, downloads output
JSONL from GCS, and writes pred_<dataset>.json files in the same format as
parse_results.py (and gemma3_edge.py).

Usage:
    python gemini_parse_results.py \
        --jobs_file ./gemini_jobs/jobs_TIMESTAMP.json \
        [--pred_dir ./predictions/gemini] \
        [--poll_interval 60] \
        [--skip_poll]   # if jobs are already complete

Output format (same as bedrock/gemma3):
  pred_<dataset>.json: list of dicts with keys:
  sample_id, uid, q_id, dataset, gt_boxes_key, gt_boxes_raw,
  pred_raw_reply, pred_answer_str, pred_explanation_str,
  pred_boxes_parsed, pred_parse_errors, _gemini_stop_reason, ...
"""

import re
import json
import time
import logging
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                   "JOB_STATE_PAUSED", "JOB_STATE_EXPIRED",
                   "SUCCEEDED", "FAILED", "CANCELLED", "PAUSED", "EXPIRED"}
SUCCESS_STATES  = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED",
                   "SUCCEEDED", "PARTIALLY_SUCCEEDED"}

# ---------------------------------------------------------------------------
# Box parsing (identical to parse_results.py)
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
                "id":  m.group("id"),
                "x":   float(m.group("x")),
                "y":   float(m.group("y")),
                "w":   float(m.group("w")),
                "h":   float(m.group("h")),
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
# GCS helpers
# ---------------------------------------------------------------------------
def parse_gcs_uri(uri: str):
    """gs://bucket/prefix → (bucket, prefix)"""
    assert uri.startswith("gs://"), f"Not a GCS URI: {uri}"
    parts = uri[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix


def list_gcs_blobs(gcs_client, bucket_name: str, prefix: str):
    bucket = gcs_client.bucket(bucket_name)
    return list(bucket.list_blobs(prefix=prefix))


def download_gcs_text(blob) -> str:
    return blob.download_as_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Poll jobs
# ---------------------------------------------------------------------------
def poll_jobs(genai_client, job_list: list[dict], poll_interval: int) -> dict[str, str]:
    """Returns {job_name: final_state_str}"""
    pending = {j["job_name"]: j["dataset"] for j in job_list}
    statuses = {}

    while pending:
        for job_name in list(pending.keys()):
            job = genai_client.batches.get(name=job_name)
            state_str = str(job.state) if job.state else "UNKNOWN"
            # Normalize state string (may be enum or string)
            state_str = state_str.replace("JobState.", "").strip()
            ds = pending[job_name]
            log.info(f"  [{ds}] {state_str}")
            if state_str in TERMINAL_STATES:
                statuses[job_name] = state_str
                del pending[job_name]

        if pending:
            log.info(f"  {len(pending)} job(s) still running — waiting {poll_interval}s ...")
            time.sleep(poll_interval)

    return statuses


# ---------------------------------------------------------------------------
# Parse one output JSONL blob
# ---------------------------------------------------------------------------
def parse_output_jsonl(text: str, meta_records: dict, sid_order: list[str] | None) -> list[dict]:
    results = []
    lines = [l for l in text.splitlines() if l.strip()]

    for line_idx, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning(f"  JSON decode error on output line {line_idx}: {e}")
            continue

        # ── Extract sample ID ───────────────────────────────────────────
        # Primary: _sample_id echoed at top level
        sid = obj.get("_sample_id", "")

        # Fallback: line-number ordering against sid_order
        if not sid and sid_order and line_idx < len(sid_order):
            sid = sid_order[line_idx]
            log.debug(f"  Line {line_idx}: using sid_order fallback → {sid[:60]}")

        # ── Extract model response text ─────────────────────────────────
        raw_reply = ""
        stop_reason = ""
        usage = {}

        status = obj.get("status", "")
        if status and status != "":
            # Non-empty status usually means an error
            log.warning(f"  Record [{sid[:60]}] has status={status!r}")
            raw_reply = f"ERROR: {status}"
        else:
            response = obj.get("response", {})
            if "error" in obj:
                log.warning(f"  Error record [{sid[:60]}]: {obj['error']}")
                raw_reply = f"ERROR: {obj['error']}"
            elif response:
                candidates = response.get("candidates", [])
                if candidates:
                    cand = candidates[0]
                    stop_reason = cand.get("finishReason", "")
                    content = cand.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        if "text" in part:
                            raw_reply = part["text"]
                            break

                usage_meta = response.get("usageMetadata", {})
                usage = {
                    "input_tokens":  usage_meta.get("promptTokenCount", 0),
                    "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                }

        pred_answer_str, pred_expl_str = split_reply(raw_reply)
        pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)

        meta = meta_records.get(sid, {})
        if not meta:
            log.warning(f"  No meta found for sid={sid[:80]!r}")

        results.append({
            **meta,
            "pred_raw_reply":        raw_reply,
            "pred_answer_str":       pred_answer_str,
            "pred_explanation_str":  pred_expl_str,
            "pred_boxes_parsed":     pred_boxes,
            "pred_parse_errors":     parse_errors,
            # Gemini extras for debugging / cost tracking
            "_gemini_stop_reason":   stop_reason,
            "_gemini_usage":         usage,
            "_gemini_model":         obj.get("request", {}).get("model", ""),
        })

    return results


# ---------------------------------------------------------------------------
# Atomic JSON write
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
    parser = argparse.ArgumentParser(description="Download + parse Vertex AI Gemini Batch results")
    parser.add_argument("--jobs_file",      required=True,
                        help="Path to jobs_*.json written by gemini_edge_submit.py")
    parser.add_argument("--pred_dir",       default="./predictions/gemini",
                        help="Where to write pred_*.json output files")
    parser.add_argument("--poll_interval",  type=int, default=60,
                        help="Seconds between job status polls")
    parser.add_argument("--skip_poll",      action="store_true",
                        help="Skip polling (if jobs already completed)")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    with open(args.jobs_file, "r", encoding="utf-8") as f:
        job_list = json.load(f)

    log.info(f"Loaded {len(job_list)} job(s) from {args.jobs_file}")

    # Grab project/location from first job entry
    project_id = job_list[0].get("project_id", "")
    location   = job_list[0].get("location", "global")

    from google import genai
    from google.cloud import storage as gcs_lib

    genai_client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
    )
    gcs_client = gcs_lib.Client(project=project_id)
    log.info(f"Initialized google-genai (Vertex AI): project={project_id}, location={location}")

    # ── Poll ─────────────────────────────────────────────────────────────
    if not args.skip_poll:
        log.info("Polling job statuses ...")
        statuses = poll_jobs(genai_client, job_list, args.poll_interval)
        log.info(f"\nFinal statuses: {statuses}")
    else:
        log.info("Skipping poll (--skip_poll set)")
        statuses = {j["job_name"]: "JOB_STATE_SUCCEEDED" for j in job_list}

    # ── Download + parse ─────────────────────────────────────────────────
    for job in job_list:
        ds_name       = job["dataset"]
        job_name      = job["job_name"]
        status        = statuses.get(job_name, "Unknown")

        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {ds_name}  status={status}")

        if status not in SUCCESS_STATES:
            log.warning(f"  Job did not succeed ({status}) — skipping")
            continue

        # Load metadata
        meta_path = job.get("meta_local", "")
        if not meta_path or not Path(meta_path).exists():
            log.error(f"  Meta file not found: {meta_path}")
            continue

        with open(meta_path, "r", encoding="utf-8") as f:
            meta_records = json.load(f)

        log.info(f"  Loaded {len(meta_records)} meta records")

        # Load sid order (fallback for line-number matching)
        sid_order = None
        order_path = job.get("sid_order_local", "")
        if order_path and Path(order_path).exists():
            with open(order_path) as f:
                sid_order = json.load(f)

        # List output blobs in GCS
        bucket_name, prefix = parse_gcs_uri(job["output_gcs_prefix"])
        blobs = list_gcs_blobs(gcs_client, bucket_name, prefix)
        jsonl_blobs = [
            b for b in blobs
            if b.name.endswith(".jsonl") or b.name.endswith(".json")
            if "manifest" not in Path(b.name).name
        ]

        if not jsonl_blobs:
            log.warning(f"  No output JSONL found at gs://{bucket_name}/{prefix}")
            log.info(f"  Blobs found: {[b.name for b in blobs]}")
            continue

        log.info(f"  Found {len(jsonl_blobs)} output file(s)")

        all_results = []
        for blob in jsonl_blobs:
            log.info(f"  Downloading gs://{bucket_name}/{blob.name}")
            text = download_gcs_text(blob)
            parsed = parse_output_jsonl(text, meta_records, sid_order)
            all_results.extend(parsed)
            log.info(f"    → {len(parsed)} records")

        errors = sum(1 for r in all_results if r.get("pred_parse_errors") and "no_boxes_matched" not in r["pred_parse_errors"])
        log.info(f"  Total: {len(all_results)} records, {errors} parse errors")

        out_path = pred_dir / f"pred_{ds_name}.json"
        atomic_write_json(all_results, out_path)

    log.info("\nDone. Next: python eval_script.py --pred_dir " + str(pred_dir) + " --out_dir eval_gemini --tau 0.5")


if __name__ == "__main__":
    main()
