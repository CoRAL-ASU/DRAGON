#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gemini_verge_submit.py  (VERGE — Verify and Refine Grounding Evidence)

Self-correction (Iterative Grounding Refinement) via Vertex AI Gemini Batch.

  Phase 1:  Submit Step 1 (Initial Grounding) — one batch job per dataset
  Phase 2:  Parse Step 1 output → build multi-turn Step 2 (Verify & Refine) → submit
  Finalize: Parse Step 2 output → pred_*.json (with box-drop enforcement)

Usage:
    python gemini_verge_submit.py phase1 --project_id <ID> --gcs_bucket <BUCKET> [--max_samples 1]
    python gemini_verge_submit.py phase2 --jobs_file <path> --project_id <ID> --gcs_bucket <BUCKET>
    python gemini_verge_submit.py finalize --jobs_file <path> --pred_dir <dir>
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

from PIL import Image
from prompts_verge import build_verify_and_refine_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "gemini-3-pro-preview"
DEFAULT_LOCATION = "global"

TERMINAL_STATES = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
                   "JOB_STATE_PAUSED", "JOB_STATE_EXPIRED",
                   "SUCCEEDED", "FAILED", "CANCELLED", "PAUSED", "EXPIRED"}
SUCCESS_STATES  = {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED",
                   "SUCCEEDED", "PARTIALLY_SUCCEEDED"}

# ---------------------------------------------------------------------------
_NUM = r"-?\d+(?:\.\d+)?"
_BOX_RE = re.compile(rf"(?P<id>[A-Za-z0-9_\-]+)\s*/\s*(?P<x>{_NUM})\s*,\s*(?P<y>{_NUM})\s*,\s*(?P<w>{_NUM})\s*,\s*(?P<h>{_NUM})")

def make_sample_id(sample):
    uid = (sample.get("uid") or "").strip()
    jpath = (sample.get("json_path") or "").strip()
    qid = (sample.get("q_id") or "").strip()
    img = Path(sample.get("image_path_rel") or sample.get("image_path_abs") or "noimg").name
    core = f"uid::{uid}::qid::{qid}::img::{img}"
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
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg", w, h

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

def atomic_write_json(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def upload_to_gcs(gcs_client, bucket_name, blob_name, data, content_type="application/octet-stream"):
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    log.info(f"  Uploaded gs://{bucket_name}/{blob_name}  ({len(data):,} bytes)")

def parse_gcs_uri(uri):
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

def extract_gemini_reply(obj):
    response = obj.get("response", {})
    candidates = response.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "text" in part:
                return part["text"].strip()
    return ""

def poll_jobs(genai_client, job_list, poll_interval=60):
    pending = {j["job_name"]: j["dataset"] for j in job_list}
    statuses = {}
    while pending:
        for job_name in list(pending.keys()):
            job = genai_client.batches.get(name=job_name)
            state_str = str(job.state).replace("JobState.", "").strip() if job.state else "UNKNOWN"
            log.info(f"  [{pending[job_name]}] {state_str}")
            if state_str in TERMINAL_STATES:
                statuses[job_name] = state_str
                del pending[job_name]
        if pending:
            log.info(f"  {len(pending)} job(s) running — waiting {poll_interval}s ...")
            time.sleep(poll_interval)
    return statuses


# ===========================================================================
# PHASE 1: Step 1 Initial Grounding
# ===========================================================================
def phase1(args):
    gen_cfg = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.dry_run:
        from google import genai
        from google.cloud import storage as gcs_lib
        genai_client = genai.Client(vertexai=True, project=args.project_id, location=args.location)
        gcs_client = gcs_lib.Client(project=args.project_id)

    batch_dir = Path(args.batch_dir)
    jobs_summary = []

    for batch_path in sorted(batch_dir.glob("*.json")):
        ds_name = batch_path.stem
        with batch_path.open() as f:
            samples = json.load(f)
        if args.max_samples is not None:
            samples = samples[:args.max_samples]
        log.info(f"\n{'='*60}\nDataset: {ds_name}  ({len(samples)} samples)")

        jsonl_lines = []
        meta_records = {}
        sid_order = []

        for sample in samples:
            sid = make_sample_id(sample)
            img_path = sample.get("image_path_abs", "")
            b64_data, mime_type, img_w, img_h = image_to_base64(img_path)

            # Step 1 uses pre-built prompt from batch JSON
            s1_raw = sample.get("prompt", "")
            try: s1_prompt = s1_raw.format(height=img_h, width=img_w)
            except: s1_prompt = s1_raw

            record = {
                "_sample_id": sid,
                "request": {
                    "contents": [{"role": "user", "parts": [
                        {"inlineData": {"mimeType": mime_type, "data": b64_data}},
                        {"text": s1_prompt},
                    ]}],
                    "generationConfig": {"temperature": gen_cfg["temperature"], "maxOutputTokens": gen_cfg["max_new_tokens"]},
                },
            }
            jsonl_lines.append(json.dumps(record, ensure_ascii=False))
            sid_order.append(sid)
            meta_records[sid] = {
                "sample_id": sid, "uid": sample.get("uid", ""), "json_path": sample.get("json_path", ""),
                "dataset": sample.get("dataset", ""), "q_id": sample.get("q_id", ""),
                "image_path_abs": img_path, "image_path_rel": sample.get("image_path_rel", ""),
                "image_width": img_w, "image_height": img_h,
                "question_text": sample.get("question_text", ""), "choices": sample.get("choices", []),
                "answer_text": sample.get("answer_text", ""),
                "gt_boxes_key": sample.get("boxes_key", "none"), "gt_boxes_raw": sample.get("boxes_raw", []),
                "step1_prompt": s1_prompt,
            }

        jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
        log.info(f"  Built {len(jsonl_lines)} records, {len(jsonl_bytes):,} bytes")

        atomic_write_json(meta_records, str(out_dir / f"{ds_name}_meta.json"))
        with (out_dir / f"{ds_name}_sid_order.json").open("w") as f:
            json.dump(sid_order, f)

        if args.dry_run:
            (out_dir / f"{ds_name}.jsonl").write_bytes(jsonl_bytes)
            log.info(f"  [dry_run] saved locally")
            continue

        input_blob = f"diagram_atb_gemini/selfcorrect/phase1/{run_ts}/{ds_name}.jsonl"
        output_prefix = f"gs://{args.gcs_bucket}/diagram_atb_gemini/selfcorrect/phase1_output/{run_ts}/{ds_name}/"
        upload_to_gcs(gcs_client, args.gcs_bucket, input_blob, jsonl_bytes, "application/x-ndjson")

        job_name_str = re.sub(r"[^a-zA-Z0-9\-]", "-", f"sc-p1-{ds_name[:20]}-{run_ts}")[:63]
        input_uri = f"gs://{args.gcs_bucket}/{input_blob}"
        job = genai_client.batches.create(model=args.model_id, src=input_uri,
            config={"display_name": job_name_str, "dest": output_prefix})
        log.info(f"  Job: {job.name}  State: {job.state}")

        jobs_summary.append({
            "dataset": ds_name, "job_display_name": job_name_str, "job_name": job.name,
            "model_id": args.model_id, "input_gcs": input_uri, "output_gcs_prefix": output_prefix,
            "meta_local": str((out_dir / f"{ds_name}_meta.json").resolve()),
            "sid_order_local": str((out_dir / f"{ds_name}_sid_order.json").resolve()),
            "project_id": args.project_id, "location": args.location,
        })

    if jobs_summary:
        jobs_file = out_dir / f"phase1_jobs_{run_ts}.json"
        atomic_write_json(jobs_summary, str(jobs_file))
        log.info(f"\nJobs file → {jobs_file}")
        log.info(f"Next: python {__file__} phase2 --jobs_file {jobs_file} --project_id {args.project_id} --gcs_bucket {args.gcs_bucket}")


# ===========================================================================
# PHASE 2: Parse Step 1 → build multi-turn Step 2 (Verify & Refine) → submit
# ===========================================================================
def phase2(args):
    gen_cfg = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens}
    with open(args.jobs_file) as f:
        phase1_jobs = json.load(f)

    project_id = phase1_jobs[0].get("project_id", args.project_id)
    location = phase1_jobs[0].get("location", args.location)

    from google import genai
    from google.cloud import storage as gcs_lib
    genai_client = genai.Client(vertexai=True, project=project_id, location=location)
    gcs_client = gcs_lib.Client(project=project_id)

    if not args.skip_poll:
        log.info("Polling Phase 1 ...")
        statuses = poll_jobs(genai_client, phase1_jobs, args.poll_interval)
        for j in phase1_jobs:
            s = statuses.get(j["job_name"], "Unknown")
            log.info(f"  {j['dataset']}: {s}")
            if s not in SUCCESS_STATES:
                log.error(f"  Job {j['dataset']} did not succeed!")
                return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_summary = []

    for job in phase1_jobs:
        ds_name = job["dataset"]
        log.info(f"\n{'='*60}\nDataset: {ds_name}")

        meta_records = json.load(open(job["meta_local"]))
        sid_order = json.load(open(job["sid_order_local"]))

        bucket_name, prefix = parse_gcs_uri(job["output_gcs_prefix"])
        bucket = gcs_client.bucket(bucket_name)
        blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]

        s1_replies = {}
        for blob in blobs:
            for line in blob.download_as_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                obj = json.loads(line)
                sid = obj.get("_sample_id", "")
                s1_replies[sid] = extract_gemini_reply(obj)
        log.info(f"  Got {len(s1_replies)} Step 1 replies")

        jsonl_lines = []
        meta2_records = {}
        sid2_order = []

        for sid, reply in s1_replies.items():
            meta = meta_records.get(sid, {})
            if not meta: continue
            img_path = meta["image_path_abs"]
            b64_data, mime_type, img_w, img_h = image_to_base64(img_path)

            s1_answer, s1_expl = split_reply(reply)
            s1_boxes, s1_errors = parse_answer_boxes(s1_answer)

            s1_pred_text = f"Answer: {s1_answer}"
            if s1_expl:
                s1_pred_text += f"\nExplanation: {s1_expl}"

            choices = meta.get("choices", [])
            choices_str = ", ".join(choices) if choices else "N/A"

            vr_raw = build_verify_and_refine_prompt(
                question_id=meta["q_id"], question=meta["question_text"],
                choices_str=choices_str, answer=meta["answer_text"],
                previous_prediction=s1_pred_text,
            )
            try: vr_prompt = vr_raw.format(height=img_h, width=img_w)
            except: vr_prompt = vr_raw

            record = {
                "_sample_id": sid,
                "request": {
                    "contents": [
                        {"role": "user", "parts": [
                            {"inlineData": {"mimeType": mime_type, "data": b64_data}},
                            {"text": meta.get("step1_prompt", "")},
                        ]},
                        {"role": "model", "parts": [{"text": reply}]},
                        {"role": "user", "parts": [{"text": vr_prompt}]},
                    ],
                    "generationConfig": {"temperature": gen_cfg["temperature"], "maxOutputTokens": gen_cfg["max_new_tokens"]},
                },
            }
            jsonl_lines.append(json.dumps(record, ensure_ascii=False))
            sid2_order.append(sid)
            meta2_records[sid] = {
                **meta,
                "step1_raw_reply": reply, "step1_answer_str": s1_answer,
                "step1_boxes_parsed": s1_boxes, "step1_parse_errors": s1_errors,
                "step2_prompt": vr_prompt,
            }

        jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")
        log.info(f"  Built {len(jsonl_lines)} Step 2 records, {len(jsonl_bytes):,} bytes")

        atomic_write_json(meta2_records, str(out_dir / f"{ds_name}_phase2_meta.json"))
        with (out_dir / f"{ds_name}_phase2_sid_order.json").open("w") as f:
            json.dump(sid2_order, f)

        input_blob = f"diagram_atb_gemini/selfcorrect/phase2/{run_ts}/{ds_name}.jsonl"
        output_prefix = f"gs://{args.gcs_bucket}/diagram_atb_gemini/selfcorrect/phase2_output/{run_ts}/{ds_name}/"
        upload_to_gcs(gcs_client, args.gcs_bucket, input_blob, jsonl_bytes, "application/x-ndjson")

        job_name_str = re.sub(r"[^a-zA-Z0-9\-]", "-", f"sc-p2-{ds_name[:20]}-{run_ts}")[:63]
        input_uri = f"gs://{args.gcs_bucket}/{input_blob}"
        job_obj = genai_client.batches.create(model=job["model_id"], src=input_uri,
            config={"display_name": job_name_str, "dest": output_prefix})
        log.info(f"  Job: {job_obj.name}  State: {job_obj.state}")

        jobs_summary.append({
            "dataset": ds_name, "job_display_name": job_name_str, "job_name": job_obj.name,
            "model_id": job["model_id"], "output_gcs_prefix": output_prefix,
            "meta_local": str((out_dir / f"{ds_name}_phase2_meta.json").resolve()),
            "sid_order_local": str((out_dir / f"{ds_name}_phase2_sid_order.json").resolve()),
            "project_id": project_id, "location": location,
        })

    if jobs_summary:
        jobs_file = out_dir / f"phase2_jobs_{run_ts}.json"
        atomic_write_json(jobs_summary, str(jobs_file))
        log.info(f"\nJobs file → {jobs_file}")
        log.info(f"Next: python {__file__} finalize --jobs_file {jobs_file} --pred_dir prediction_gemini_selfcorrect/")


# ===========================================================================
# FINALIZE: Parse Phase 2 → pred_*.json (with box-drop enforcement)
# ===========================================================================
def finalize(args):
    with open(args.jobs_file) as f:
        phase2_jobs = json.load(f)

    project_id = phase2_jobs[0].get("project_id", "")
    location = phase2_jobs[0].get("location", "global")

    from google import genai
    from google.cloud import storage as gcs_lib
    genai_client = genai.Client(vertexai=True, project=project_id, location=location)
    gcs_client = gcs_lib.Client(project=project_id)

    if not args.skip_poll:
        log.info("Polling Phase 2 ...")
        poll_jobs(genai_client, phase2_jobs, args.poll_interval)

    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    fallback_total = 0

    for job in phase2_jobs:
        ds_name = job["dataset"]
        log.info(f"\n{'='*60}\nDataset: {ds_name}")

        meta_records = json.load(open(job["meta_local"]))
        sid_order = json.load(open(job["sid_order_local"]))

        bucket_name, prefix = parse_gcs_uri(job["output_gcs_prefix"])
        bucket = gcs_client.bucket(bucket_name)
        blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]

        results = []
        for blob in blobs:
            for line_idx, line in enumerate(blob.download_as_text(encoding="utf-8").splitlines()):
                if not line.strip(): continue
                obj = json.loads(line)
                sid = obj.get("_sample_id", "")
                if not sid and line_idx < len(sid_order):
                    sid = sid_order[line_idx]

                s2_reply = extract_gemini_reply(obj)
                meta = meta_records.get(sid, {})

                s2_answer, s2_expl = split_reply(s2_reply)
                s2_boxes, s2_errors = parse_answer_boxes(s2_answer)

                s1_boxes = meta.get("step1_boxes_parsed", [])
                s1_answer = meta.get("step1_answer_str", "")
                s1_reply = meta.get("step1_raw_reply", "")

                # Box-drop enforcement
                if len(s2_boxes) < len(s1_boxes):
                    final_answer, final_expl = s1_answer, ""
                    final_boxes, final_errors = s1_boxes, meta.get("step1_parse_errors", [])
                    final_reply, used_step = s1_reply, "step1_fallback"
                    fallback_total += 1
                else:
                    final_answer, final_expl = s2_answer, s2_expl
                    final_boxes, final_errors = s2_boxes, s2_errors
                    final_reply, used_step = s2_reply, "step2"

                results.append({
                    **meta,
                    "step2_raw_reply": s2_reply, "step2_answer_str": s2_answer,
                    "step2_boxes_parsed": s2_boxes, "step2_parse_errors": s2_errors,
                    "used_step": used_step,
                    "pred_raw_reply": final_reply,
                    "pred_answer_str": final_answer, "pred_explanation_str": final_expl,
                    "pred_boxes_parsed": final_boxes, "pred_parse_errors": final_errors,
                    "prompt": meta.get("step2_prompt", ""),
                })

        atomic_write_json(results, str(pred_dir / f"pred_{ds_name}.json"))
        log.info(f"  {ds_name}: {len(results)} results")

    log.info(f"\nAll done. (fallbacks: {fallback_total})")


# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Gemini VERGE batch inference")
    sub = parser.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("phase1")
    p1.add_argument("--batch_dir", default="batches/")
    p1.add_argument("--out_dir", default="gemini_jobs_selfcorrect/")
    p1.add_argument("--project_id", required=True)
    p1.add_argument("--gcs_bucket", required=True)
    p1.add_argument("--location", default=DEFAULT_LOCATION)
    p1.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p1.add_argument("--temperature", type=float, default=0.2)
    p1.add_argument("--max_new_tokens", type=int, default=1024)
    p1.add_argument("--max_samples", type=int, default=None)
    p1.add_argument("--dry_run", action="store_true")

    p2 = sub.add_parser("phase2")
    p2.add_argument("--jobs_file", required=True)
    p2.add_argument("--out_dir", default="gemini_jobs_selfcorrect/")
    p2.add_argument("--project_id", default="")
    p2.add_argument("--gcs_bucket", required=True)
    p2.add_argument("--location", default=DEFAULT_LOCATION)
    p2.add_argument("--temperature", type=float, default=0.2)
    p2.add_argument("--max_new_tokens", type=int, default=1024)
    p2.add_argument("--poll_interval", type=int, default=60)
    p2.add_argument("--skip_poll", action="store_true")

    pf = sub.add_parser("finalize")
    pf.add_argument("--jobs_file", required=True)
    pf.add_argument("--pred_dir", default="prediction_gemini_selfcorrect/")
    pf.add_argument("--poll_interval", type=int, default=60)
    pf.add_argument("--skip_poll", action="store_true")

    args = parser.parse_args()
    {"phase1": phase1, "phase2": phase2, "finalize": finalize}[args.phase](args)


if __name__ == "__main__":
    main()
