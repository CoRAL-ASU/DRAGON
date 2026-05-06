# Diagram Attribution — Inference & Evaluation Code

This repository contains the inference and evaluation code for the paper.  


---

## Prompting Strategies

| Name | Description |
|------|-------------|
| **EDGE** | Single-turn direct grounding — model returns bounding boxes in one shot |
| **SAGE** | Two-turn: Stage 1 *selects* relevant visual elements, Stage 2 *grounds* them with bounding boxes |
| **VERGE** | Two-turn: Turn 1 produces initial grounding, Turn 2 *verifies* and *refines* the coordinates |

---

## Models

| Model | Provider |
|-------|----------|
| Claude Sonnet 4.6 | AWS Bedrock  | 
| Claude Opus 4.6 | AWS Bedrock  | 
| Llama 4 Maverick 17B | AWS Bedrock  | 
| Kimi K2.5 | AWS Bedrock  | 
| Gemini 3 Pro | Google Vertex AI |
| Gemma3-27B-IT | Local GPU (HuggingFace) |
| Qwen3.5-35B-A3B | Local GPU (HuggingFace) | 
| InternVL3-38B | Local GPU (HuggingFace) | 

---

## Repository Structure

```
code/
├── shared/                     # Shared utilities (used by all models)
│   ├── data_loading.py         # Builds ./batches/*.json from raw dataset
│   ├── data_preprocessing.py   # Image resizing and base64 utilities
│   ├── prompts_edge.py         # EDGE prompt template
│   ├── prompts_sage.py         # SAGE Stage 1 + Stage 2 prompt templates
│   ├── prompts_verge.py        # VERGE Verify & Refine prompt template
│   ├── worked_examples.py      # Few-shot examples per dataset
│   └── eval_script.py          # Evaluation metrics (IoU, hit-rate, F1)
│
├── amazon_bedrock/             # AWS Bedrock inference
│   ├── claude_llama_edge.py    # EDGE  — Claude / Llama4 
│   ├── claude_llama_sage.py    # SAGE  — Claude / Llama4 
│   ├── claude_llama_verge.py   # VERGE — Claude / Llama4
│   ├── kimi_edge.py            # EDGE  — Kimi K2.5 
│   ├── kimi_sage.py            # SAGE  — Kimi K2.5
│   ├── kimi_verge.py           # VERGE — Kimi K2.5 
│   └── parse_results.py        # Download + parse batch results (EDGE only)
│
├── gcp/                        # Google Vertex AI inference
│   ├── gemini_edge_submit.py   # EDGE  — Gemini 
│   ├── gemini_sage_submit.py   # SAGE  — Gemini 
│   ├── gemini_verge_submit.py  # VERGE — Gemini 
│   └── gemini_parse_results.py # Download + parse batch results (EDGE only)
│
└── open_source/                # Local GPU inference (HuggingFace)
    ├── gemma3_edge.py          # EDGE  — Gemma3-27B
    ├── qwen3_5_edge.py         # EDGE  — Qwen3.5-3.5B-A3B
    ├── internvl_edge.py        # EDGE  — InternVL3-38B
    ├── sage_inference.py       # SAGE  — all 3 models via --model flag
    └── verge_inference.py      # VERGE — all 3 models via --model flag
```

---

## Setup

### Step 1 — Prepare batch files (once, shared by all models)

```bash
python shared/data_loading.py
# Outputs: ./batches/ai2d.json, ChartQA.json, Circuit-VQA.json, ...
```

### Step 2 — Run inference (see sections below)

### Step 3 — Evaluate

```bash
python shared/eval_script.py --pred_dir ./predictions/<model>/
```

---

## AWS Bedrock — Claude Sonnet/Opus & Llama4

**Prerequisites:** `pip install boto3 pillow`  
**Auth:** AWS credentials configured via env vars or `~/.aws/credentials`  
**IAM role** must trust `bedrock.amazonaws.com` with S3 read/write and `bedrock:CreateModelInvocationJob`

Switch models with `--model_id`:
- Claude Sonnet: `us.anthropic.claude-sonnet-4-6` (default)
- Claude Opus: `us.anthropic.claude-opus-4-6`
- Llama4 Maverick: `us.meta.llama4-maverick-17b` (or similar Bedrock ID)

### EDGE
```bash
python amazon_bedrock/claude_llama_edge.py \
    --role_arn <YOUR_IAM_ROLE_ARN> \
    --bucket <YOUR_S3_BUCKET> \
    --model_id us.anthropic.claude-sonnet-4-6

python amazon_bedrock/parse_results.py \
    --jobs_file ./bedrock_jobs/jobs_TIMESTAMP.json \
    --pred_dir ./predictions/claude_sonnet_edge/
```

### SAGE
```bash
python amazon_bedrock/claude_llama_sage.py phase1 \
    --role_arn <YOUR_IAM_ROLE_ARN> --model_id us.anthropic.claude-sonnet-4-6

python amazon_bedrock/claude_llama_sage.py phase2 \
    --jobs_file ./bedrock_jobs/jobs_TIMESTAMP.json --role_arn <YOUR_IAM_ROLE_ARN>

python amazon_bedrock/claude_llama_sage.py finalize \
    --jobs_file ./bedrock_jobs/jobs_TIMESTAMP.json \
    --pred_dir ./predictions/claude_sonnet_sage/
```

### VERGE
```bash
python amazon_bedrock/claude_llama_verge.py phase1 \
    --role_arn <YOUR_IAM_ROLE_ARN> --model_id us.anthropic.claude-sonnet-4-6

python amazon_bedrock/claude_llama_verge.py phase2 \
    --jobs_file ./bedrock_jobs/jobs_TIMESTAMP.json --role_arn <YOUR_IAM_ROLE_ARN>

python amazon_bedrock/claude_llama_verge.py finalize \
    --jobs_file ./bedrock_jobs/jobs_TIMESTAMP.json \
    --pred_dir ./predictions/claude_sonnet_verge/
```

---

## AWS Bedrock — Kimi K2.5

Kimi uses the **realtime** `invoke_model` API (no batch API available). Inference is parallelized via threading.

```bash
# EDGE
python amazon_bedrock/kimi_edge.py \
    --model_id moonshotai.kimi-k2.5 \
    --pred_dir ./predictions/kimi_edge/

# SAGE
python amazon_bedrock/kimi_sage.py \
    --model_id moonshotai.kimi-k2.5 \
    --pred_dir ./predictions/kimi_sage/

# VERGE
python amazon_bedrock/kimi_verge.py \
    --model_id moonshotai.kimi-k2.5 \
    --pred_dir ./predictions/kimi_verge/
```

---

## Google Vertex AI — Gemini 3 Pro

**Prerequisites:** `pip install google-genai google-cloud-storage pillow`  
**Auth:** `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS`

### EDGE
```bash
python gcp/gemini_edge_submit.py \
    --project_id <YOUR_GCP_PROJECT_ID> \
    --gcs_bucket <YOUR_GCS_BUCKET>

python gcp/gemini_parse_results.py \
    --jobs_file ./gemini_jobs/jobs_TIMESTAMP.json \
    --pred_dir ./predictions/gemini_edge/
```

### SAGE
```bash
python gcp/gemini_sage_submit.py phase1 \
    --project_id <YOUR_GCP_PROJECT_ID> --gcs_bucket <YOUR_GCS_BUCKET>

python gcp/gemini_sage_submit.py phase2 \
    --jobs_file ./gemini_jobs/jobs_TIMESTAMP.json \
    --project_id <YOUR_GCP_PROJECT_ID> --gcs_bucket <YOUR_GCS_BUCKET>

python gcp/gemini_sage_submit.py finalize \
    --jobs_file ./gemini_jobs/jobs_TIMESTAMP.json \
    --pred_dir ./predictions/gemini_sage/
```

### VERGE
```bash
python gcp/gemini_verge_submit.py phase1 \
    --project_id <YOUR_GCP_PROJECT_ID> --gcs_bucket <YOUR_GCS_BUCKET>

python gcp/gemini_verge_submit.py phase2 \
    --jobs_file ./gemini_jobs/jobs_TIMESTAMP.json \
    --project_id <YOUR_GCP_PROJECT_ID> --gcs_bucket <YOUR_GCS_BUCKET>

python gcp/gemini_verge_submit.py finalize \
    --jobs_file ./gemini_jobs/jobs_TIMESTAMP.json \
    --pred_dir ./predictions/gemini_verge/
```

---

## Open-Source Local Models

**Prerequisites:** `pip install torch transformers pillow tqdm`  
**Hardware:** Multi-GPU recommended (scripts use all visible GPUs by default)  
Set `CUDA_VISIBLE_DEVICES` to control GPU assignment.

### EDGE (one script per model)
```bash
python open_source/gemma3_edge.py \
    --checkpoint google/gemma-3-27b-it \
    -b ./batches/ -o ./predictions/gemma3_edge/

python open_source/qwen3_5_edge.py \
    --checkpoint Qwen/Qwen3.5-35B-A3B \
    -b ./batches/ -o ./predictions/qwen3_5_edge/

python open_source/internvl_edge.py \
    --checkpoint OpenGVLab/InternVL3-38B \
    -b ./batches/ -o ./predictions/internvl_edge/
```

### SAGE (all 3 models via `--model` flag)
```bash
python open_source/sage_inference.py \
    --model gemma3 \          # or qwen3_5 / internvl
    --checkpoint google/gemma-3-27b-it \
    -b ./batches/ -o ./predictions/gemma3_sage/
```

### VERGE (all 3 models via `--model` flag)
```bash
python open_source/verge_inference.py \
    --model gemma3 \          # or qwen3_5 / internvl
    --checkpoint google/gemma-3-27b-it \
    -b ./batches/ -o ./predictions/gemma3_verge/
```

All open-source scripts support `--save_every N` for checkpoint-style resume on long runs, and `--max_samples N` for quick testing.

---

## Evaluation

```bash
python shared/eval_script.py \
    --pred_dir ./predictions/<model_strategy>/ \
    [--out_dir ./results/]
```

**Metrics reported:**
- **MaxIoU / MeanIoU** — bounding box overlap
- **Hit-rate @ τ** for τ ∈ {0.1, 0.3, 0.5, 0.7} — fraction of samples with IoU above threshold
- **Recall / Precision / F1** — element-level matching

---

## Prediction Format

All scripts write `pred_<dataset>.json` — a list of dicts with:

```json
{
  "sample_id": "uid::...::qid::...::img::...",
  "uid": "...",
  "dataset": "....",
  "question_text": "...",
  "gt_boxes_raw": [[x, y, w, h], ...],
  "pred_raw_reply": "<full model output>",
  "pred_answer_str": "A_01/100,200,50,30, ...",
  "pred_boxes_parsed": [{"id": "A_01", "x": 100, "y": 200, "w": 50, "h": 30}],
  "pred_parse_errors": []
}
```

This format is identical across all models and strategies, so `eval_script.py` works unchanged for all.
