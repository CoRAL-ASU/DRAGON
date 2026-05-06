#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini_edge_submit.py

Converts diagram_atb batch JSONs → Vertex AI Gemini Batch Prediction JSONL,
uploads to GCS, and submits one BatchPredictionJob per dataset.

Usage:
    python gemini_edge_submit.py \
        --project_id YOUR_GCP_PROJECT_ID \
        --gcs_bucket YOUR_GCS_BUCKET \
        [--model_id gemini-3-pro-preview] \
        [--location us-central1] \
        [--batch_dir ./batches] \
        [--out_dir ./gemini_jobs] \
        [--max_samples 5]   # optional quick test
        [--dry_run]         # build JSONL locally, skip GCS upload + job submit

Prerequisites:
    pip install google-genai google-cloud-storage pillow
    gcloud auth application-default login   (or set GOOGLE_APPLICATION_CREDENTIALS)
"""

import os
import re
import json
import base64
import argparse
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime

from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "gemini-3-pro-preview"
DEFAULT_LOCATION  = "global"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_sample_id(sample: dict) -> str:
    uid   = (sample.get("uid")   or "").strip()
    jpath = (sample.get("json_path") or "").strip()
    qid   = (sample.get("q_id")  or "").strip()
    img   = Path(sample.get("image_path_rel") or sample.get("image_path_abs") or "noimg").name
    core  = f"uid::{uid}::qid::{qid}::img::{img}"
    if jpath:
        core += f"::json::{Path(jpath).name}"
    return core


def image_to_base64(image_path: str, max_side: int = 1568) -> tuple[str, str]:
    """Load image, downscale if needed, return (base64_string, mime_type)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def build_gemini_record(sample: dict, gen_cfg: dict) -> dict:
    """
    Build one JSONL record in Vertex AI Gemini Batch Prediction format.
    Each line: {"request": {...}}
    The custom_id field is used to correlate input → output.
    """
    sid = make_sample_id(sample)

    img_path = sample.get("image_path_abs", "")
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    base_prompt = sample.get("prompt", "")
    try:
        prompt_filled = base_prompt.format(height=img_h, width=img_w)
    except Exception:
        prompt_filled = base_prompt

    b64_data, mime_type = image_to_base64(img_path)

    request = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime_type,
                            "data": b64_data,
                        }
                    },
                    {
                        "text": prompt_filled,
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": gen_cfg["temperature"],
            "maxOutputTokens": gen_cfg["max_new_tokens"],
        },
    }

    meta = {
        "sample_id": sid,
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

    # Vertex AI batch JSONL format: each line must be {"request": {...}}
    # _sample_id at top level → echoed back in the output line alongside "response"
    return {
        "record": {"_sample_id": sid, "request": request},
        "meta": meta,
        "sid": sid,
    }


# ---------------------------------------------------------------------------
# GCS upload helper
# ---------------------------------------------------------------------------
def upload_to_gcs(gcs_client, bucket_name: str, blob_name: str, data: bytes, content_type: str = "application/octet-stream"):
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    log.info(f"  Uploaded gs://{bucket_name}/{blob_name}  ({len(data):,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare + submit Vertex AI Gemini Batch Prediction for diagram_atb")
    parser.add_argument("--project_id",     required=True,   help="GCP project ID")
    parser.add_argument("--gcs_bucket",     required=True,   help="GCS bucket name for input/output")
    parser.add_argument("--gcs_prefix",     default="diagram_atb_gemini",
                        help="GCS path prefix inside the bucket")
    parser.add_argument("--location",       default=DEFAULT_LOCATION,
                        help="Vertex AI region (default: us-central1)")
    parser.add_argument("--model_id",       default=DEFAULT_MODEL_ID,
                        help="Gemini model ID (default: gemini-3-pro-preview)")
    parser.add_argument("--batch_dir",      default="./batches")
    parser.add_argument("--out_dir",        default="./gemini_jobs",
                        help="Local dir to save meta JSONs + job summary")
    parser.add_argument("--temperature",    type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int,   default=2048)
    parser.add_argument("--max_samples",    type=int,   default=None,
                        help="Limit samples per dataset (for testing)")
    parser.add_argument("--dry_run",        action="store_true",
                        help="Build JSONL locally, skip GCS upload + job submission")
    args = parser.parse_args()

    gen_cfg = {
        "temperature":   args.temperature,
        "max_new_tokens": args.max_new_tokens,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports so dry_run doesn't need GCP libs configured
    if not args.dry_run:
        from google import genai
        from google.cloud import storage as gcs_lib

        genai_client = genai.Client(
            vertexai=True,
            project=args.project_id,
            location=args.location,
        )
        gcs_client = gcs_lib.Client(project=args.project_id)
        log.info(f"Initialized google-genai (Vertex AI): project={args.project_id}, location={args.location}")

    batch_dir = Path(args.batch_dir)
    batch_files = sorted(batch_dir.glob("*.json"))
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_summary = []

    for batch_path in batch_files:
        ds_name = batch_path.stem
        log.info(f"\n{'='*60}")
        log.info(f"Dataset: {ds_name}  ({batch_path.name})")

        with batch_path.open("r", encoding="utf-8") as f:
            samples = json.load(f)

        if not isinstance(samples, list):
            log.warning(f"  Expected list, got {type(samples)} — skipping")
            continue

        if args.max_samples is not None:
            samples = samples[: args.max_samples]
            log.info(f"  Limiting to {len(samples)} samples")

        # ── Build JSONL + metadata ──────────────────────────────────────
        jsonl_lines = []
        meta_records = {}
        sid_order = []  # preserve order for matching output
        skipped = 0

        for i, sample in enumerate(samples):
            try:
                result = build_gemini_record(sample, gen_cfg)
                jsonl_lines.append(json.dumps(result["record"], ensure_ascii=False))
                meta_records[result["sid"]] = result["meta"]
                sid_order.append(result["sid"])
            except Exception as e:
                log.warning(f"  Skipped sample {i}: {e}")
                skipped += 1

        log.info(f"  Built {len(jsonl_lines)} records  (skipped {skipped})")

        if not jsonl_lines:
            log.warning("  No valid records — skipping dataset")
            continue

        jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")

        # Save meta + sid order locally (needed by parse script)
        meta_path = out_dir / f"{ds_name}_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_records, f, indent=2, ensure_ascii=False)
        log.info(f"  Saved meta → {meta_path}")

        order_path = out_dir / f"{ds_name}_sid_order.json"
        with order_path.open("w", encoding="utf-8") as f:
            json.dump(sid_order, f)

        if args.dry_run:
            jsonl_path = out_dir / f"{ds_name}.jsonl"
            jsonl_path.write_bytes(jsonl_bytes)
            log.info(f"  [dry_run] JSONL written locally → {jsonl_path}")
            continue

        # ── Upload to GCS ───────────────────────────────────────────────
        input_blob  = f"{args.gcs_prefix}/input/{run_ts}/{ds_name}.jsonl"
        output_prefix = f"gs://{args.gcs_bucket}/{args.gcs_prefix}/output/{run_ts}/{ds_name}/"

        upload_to_gcs(gcs_client, args.gcs_bucket, input_blob, jsonl_bytes, "application/x-ndjson")

        # ── Submit Vertex AI Batch Prediction Job ────────────────────────
        job_display_name = re.sub(r"[^a-zA-Z0-9\-]", "-", f"diag-atb-{ds_name[:20]}-{run_ts}")[:63]

        log.info(f"  Submitting BatchPredictionJob: {job_display_name}")
        input_uri = f"gs://{args.gcs_bucket}/{input_blob}"
        job = genai_client.batches.create(
            model=args.model_id,
            src=input_uri,
            config={
                "display_name": job_display_name,
                "dest": output_prefix,
            },
        )

        log.info(f"  Job name: {job.name}")
        log.info(f"  State: {job.state}")

        jobs_summary.append({
            "dataset": ds_name,
            "job_display_name": job_display_name,
            "job_name": job.name,
            "model_id": args.model_id,
            "input_gcs": input_uri,
            "output_gcs_prefix": output_prefix,
            "meta_local": str(meta_path.resolve()),
            "sid_order_local": str(order_path.resolve()),
            "submitted_at": run_ts,
            "project_id": args.project_id,
            "location": args.location,
        })

    if jobs_summary:
        jobs_file = out_dir / f"jobs_{run_ts}.json"
        with jobs_file.open("w", encoding="utf-8") as f:
            json.dump(jobs_summary, f, indent=2)
        log.info(f"\nSaved job summary → {jobs_file}")
        log.info(f"Next step:  python gemini_parse_results.py --jobs_file {jobs_file}")
    elif not args.dry_run:
        log.warning("No jobs submitted.")


if __name__ == "__main__":
    main()
