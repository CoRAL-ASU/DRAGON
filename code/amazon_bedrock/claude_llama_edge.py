"""
claude_llama_edge.py  (EDGE — Evidence-grounded Direct Grounding)

Converts diagram_atb batch JSONs → Bedrock Batch Inference JSONL,
uploads to S3, and submits one batch inference job per dataset.

Usage:
    python claude_llama_edge.py \
        --role_arn <YOUR_IAM_ROLE_ARN> \
        [--bucket da_sonnet] \
        [--region us-east-1] \
        [--batch_dir ./batches] \
        

Prerequisites:
    pip install boto3 pillow
    AWS credentials configured (env vars, ~/.aws/credentials, or instance profile)
    The IAM role must trust bedrock.amazonaws.com and have:
        - s3:PutObject / s3:GetObject on the bucket
        - bedrock:CreateModelInvocationJob
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

import boto3
from PIL import Image

from prompts_sage import build_combined_two_stage_prompt

# ---------------------------------------------------------------------------
# Model ID — update if Bedrock changes the alias
# Find current IDs at: AWS Console → Bedrock → Model catalog → Claude
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"

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
# Helpers (mirror gemma3.py logic so sample_ids are compatible)
# ---------------------------------------------------------------------------
def make_sample_id(sample: dict) -> str:
    uid  = (sample.get("uid") or "").strip()
    jpath = (sample.get("json_path") or "").strip()
    qid  = (sample.get("q_id") or "").strip()
    img  = Path(sample.get("image_path_rel") or sample.get("image_path_abs") or "noimg").name
    core = f"uid::{uid}::qid::{qid}::img::{img}"
    if jpath:
        core += f"::json::{Path(jpath).name}"
    return core


def image_to_base64(image_path: str, max_side: int = 1568) -> tuple[str, str]:
    """
    Load image, downscale if needed (Claude limit ~1568px on longest side),
    return (base64_string, media_type).
    Claude supports: image/jpeg, image/png, image/gif, image/webp
    """
    img = Image.open(image_path).convert("RGB")

    # Respect Claude's image size limits
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/jpeg"


def build_bedrock_record(sample: dict, gen_cfg: dict, prompt_type: str = "default") -> dict:
    """
    Build one JSONL record in Amazon Bedrock Batch Inference format.
    Adapts payload format based on the model provider (Anthropic vs Meta).
    prompt_type: 'default' uses sample['prompt']; 'sage' builds combined SAGE prompt.
    """
    sid = make_sample_id(sample)

    img_path = sample.get("image_path_abs", "")
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    if prompt_type == "sage":
        choices = sample.get("choices", [])
        choices_str = " | ".join(str(c) for c in choices) if choices else "N/A"
        base_prompt = build_combined_two_stage_prompt(
            dataset=sample.get("dataset", ""),
            question_id=sample.get("q_id", ""),
            question=sample.get("question_text", ""),
            choices_str=choices_str,
            answer=sample.get("answer_text", ""),
        )
    else:
        base_prompt = sample.get("prompt", "")

    try:
        prompt_filled = base_prompt.format(height=img_h, width=img_w)
    except Exception:
        prompt_filled = base_prompt

    b64_data, media_type = image_to_base64(img_path)
    model_id = gen_cfg.get("model_id", "")

    if "meta" in model_id.lower() or "llama" in model_id.lower():
        # Meta Llama 4 native format for Bedrock Batch Inference
        # Uses prompt with special tokens + base64 images list
        llama_prompt = (
            "<|begin_of_text|>"
            "<|header_start|>user<|header_end|>\n\n"
            f"<|image|>{prompt_filled}"
            "<|eot|>"
            "<|header_start|>assistant<|header_end|>\n\n"
        )
        model_input = {
            "prompt": llama_prompt,
            "images": [b64_data],
            "max_gen_len": gen_cfg["max_new_tokens"],
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
        }
    else:
        # Anthropic Claude format
        model_input = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": gen_cfg["max_new_tokens"],
            "temperature": gen_cfg["temperature"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt_filled,
                        },
                    ],
                }
            ],
        }

    # Store everything needed for result parsing in a separate metadata dict.
    # Bedrock echoes back recordId; we store full meta alongside the JSONL.
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

    return {
        "record": {"recordId": sid, "modelInput": model_input},
        "meta": meta,
    }


# ---------------------------------------------------------------------------
# Upload helpers
# ---------------------------------------------------------------------------
def upload_bytes(s3, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream"):
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
    log.info(f"  Uploaded s3://{bucket}/{key}  ({len(data):,} bytes)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare + submit Bedrock Batch Inference for diagram_atb")
    parser.add_argument("--batch_dir",   default="./batches")
    parser.add_argument("--out_dir",     default="./bedrock_jobs",
                        help="Local dir to save meta JSONs + job ARN file")
    parser.add_argument("--bucket",      default="<YOUR_S3_BUCKET>")
    parser.add_argument("--s3_prefix",   default="diagram_atb",
                        help="S3 key prefix inside the bucket")
    parser.add_argument("--region",      default="us-east-1")
    parser.add_argument("--model_id",    default=DEFAULT_MODEL_ID,
                        help="Bedrock model ID for Claude Sonnet")
    parser.add_argument("--role_arn",    required=True,
                        help="IAM role ARN that Bedrock will assume for S3 access")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p",       type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per dataset (for quick testing)")
    parser.add_argument("--prompt_type", default="default", choices=["default", "sage"],
                        help="'sage' builds combined SAGE identify+ground prompt instead of sample['prompt']")
    parser.add_argument("--consolidate", action="store_true",
                        help="Consolidate all datasets into ONE JSONL and submit a single Bedrock job")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Build JSONL locally but skip S3 upload + job submission")
    args = parser.parse_args()

    gen_cfg = {
        "temperature":   args.temperature,
        "top_p":         args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "model_id":      args.model_id,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", region_name=args.region)
    bedrock = boto3.client("bedrock", region_name=args.region)

    batch_dir = Path(args.batch_dir)
    batch_files = sorted(batch_dir.glob("*.json"))

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs_summary = []

    if args.consolidate:
        # ── All datasets → ONE JSONL → ONE Bedrock job ─────────────────
        log.info(f"\nConsolidate mode: all datasets → single job (prompt_type={args.prompt_type})")
        all_jsonl_lines = []
        all_meta_records = {}
        skipped = 0

        for batch_path in batch_files:
            ds_name = batch_path.stem
            with batch_path.open("r", encoding="utf-8") as f:
                samples = json.load(f)
            if not isinstance(samples, list):
                log.warning(f"  Expected list in {batch_path.name} — skipping")
                continue
            if args.max_samples is not None:
                samples = samples[:args.max_samples]
            for i, sample in enumerate(samples):
                try:
                    result = build_bedrock_record(sample, gen_cfg, args.prompt_type)
                    all_jsonl_lines.append(json.dumps(result["record"], ensure_ascii=False))
                    all_meta_records[result["meta"]["sample_id"]] = result["meta"]
                except Exception as e:
                    log.warning(f"  Skipped {ds_name}[{i}]: {e}")
                    skipped += 1
            log.info(f"  {ds_name}: {len(samples)} samples")

        log.info(f"Total: {len(all_jsonl_lines)} records built (skipped {skipped})")

        if not all_jsonl_lines:
            log.warning("No valid records — aborting")
            return

        jsonl_bytes = ("\n".join(all_jsonl_lines) + "\n").encode("utf-8")
        meta_path = out_dir / f"all_meta_{run_ts}.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(all_meta_records, f, indent=2, ensure_ascii=False)
        log.info(f"  Saved meta → {meta_path}")

        if args.dry_run:
            local_jsonl = out_dir / f"all_{run_ts}.jsonl"
            local_jsonl.write_bytes(jsonl_bytes)
            log.info(f"  [dry_run] JSONL written → {local_jsonl}")
            return

        input_key = f"{args.s3_prefix}/input/{run_ts}/all.jsonl"
        output_prefix = f"{args.s3_prefix}/output/{run_ts}/all/"
        upload_bytes(s3, args.bucket, input_key, jsonl_bytes, "application/x-ndjson")

        job_name = re.sub(r"[^a-zA-Z0-9\-]", "-", f"diag-atb-all-{run_ts}")[:63]
        response = bedrock.create_model_invocation_job(
            jobName=job_name,
            modelId=args.model_id,
            roleArn=args.role_arn,
            inputDataConfig={"s3InputDataConfig": {
                "s3Uri": f"s3://{args.bucket}/{input_key}",
                "s3InputFormat": "JSONL",
            }},
            outputDataConfig={"s3OutputDataConfig": {
                "s3Uri": f"s3://{args.bucket}/{output_prefix}",
            }},
        )
        job_arn = response["jobArn"]
        log.info(f"  Submitted consolidated job: {job_name}  ARN: {job_arn}")

        job_entry = {
            "dataset": "all",
            "job_name": job_name,
            "job_arn": job_arn,
            "model_id": args.model_id,
            "input_s3": f"s3://{args.bucket}/{input_key}",
            "output_s3_prefix": f"s3://{args.bucket}/{output_prefix}",
            "meta_local": str(meta_path.resolve()),
            "submitted_at": run_ts,
        }
        jobs_file = out_dir / f"jobs_{run_ts}.json"
        with jobs_file.open("w", encoding="utf-8") as f:
            json.dump([job_entry], f, indent=2)
        log.info(f"\nSaved job summary → {jobs_file}")
        log.info("Next step: python parse_results.py --jobs_file " + str(jobs_file))
        return

    # ── Per-dataset mode (original behaviour) ──────────────────────────
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
        skipped = 0

        for i, sample in enumerate(samples):
            try:
                result = build_bedrock_record(sample, gen_cfg, args.prompt_type)
                jsonl_lines.append(json.dumps(result["record"], ensure_ascii=False))
                meta_records[result["meta"]["sample_id"]] = result["meta"]
            except Exception as e:
                log.warning(f"  Skipped sample {i}: {e}")
                skipped += 1

        log.info(f"  Built {len(jsonl_lines)} records  (skipped {skipped})")

        if not jsonl_lines:
            log.warning("  No valid records — skipping dataset")
            continue

        jsonl_bytes = ("\n".join(jsonl_lines) + "\n").encode("utf-8")

        # Save meta locally (needed by parse script to reconstruct full records)
        meta_path = out_dir / f"{ds_name}_meta.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta_records, f, indent=2, ensure_ascii=False)
        log.info(f"  Saved meta → {meta_path}")

        if args.dry_run:
            jsonl_path = out_dir / f"{ds_name}.jsonl"
            jsonl_path.write_bytes(jsonl_bytes)
            log.info(f"  [dry_run] JSONL written locally → {jsonl_path}")
            continue

        # ── Upload to S3 ────────────────────────────────────────────────
        input_key  = f"{args.s3_prefix}/input/{run_ts}/{ds_name}.jsonl"
        output_prefix = f"{args.s3_prefix}/output/{run_ts}/{ds_name}/"

        upload_bytes(s3, args.bucket, input_key, jsonl_bytes, "application/x-ndjson")

        # ── Submit Bedrock Batch Job ─────────────────────────────────────
        job_name = f"diag-atb-{ds_name[:20]}-{run_ts}"
        # job names: only alphanumeric + hyphens, max 63 chars
        job_name = re.sub(r"[^a-zA-Z0-9\-]", "-", job_name)[:63]

        response = bedrock.create_model_invocation_job(
            jobName=job_name,
            modelId=args.model_id,
            roleArn=args.role_arn,
            inputDataConfig={
                "s3InputDataConfig": {
                    "s3Uri": f"s3://{args.bucket}/{input_key}",
                    "s3InputFormat": "JSONL",
                }
            },
            outputDataConfig={
                "s3OutputDataConfig": {
                    "s3Uri": f"s3://{args.bucket}/{output_prefix}",
                }
            },
        )

        job_arn = response["jobArn"]
        log.info(f"  Submitted job: {job_name}")
        log.info(f"  Job ARN: {job_arn}")

        jobs_summary.append({
            "dataset": ds_name,
            "job_name": job_name,
            "job_arn": job_arn,
            "model_id": args.model_id,
            "input_s3": f"s3://{args.bucket}/{input_key}",
            "output_s3_prefix": f"s3://{args.bucket}/{output_prefix}",
            "meta_local": str(meta_path.resolve()),
            "submitted_at": run_ts,
        })

    # Save job ARNs for the parse script
    if jobs_summary:
        jobs_file = out_dir / f"jobs_{run_ts}.json"
        with jobs_file.open("w", encoding="utf-8") as f:
            json.dump(jobs_summary, f, indent=2)
        log.info(f"\nSaved job summary → {jobs_file}")
        log.info("Next step: python parse_results.py --jobs_file " + str(jobs_file))
    elif not args.dry_run:
        log.warning("No jobs submitted.")


if __name__ == "__main__":
    main()
