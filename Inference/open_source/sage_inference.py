# -*- coding: utf-8 -*-
"""
Two-stage "Select then Ground" inference for Diagram Attribution.

Supports three local model backends:
  --model gemma3    → Gemma-3-27B-IT
  --model qwen3_5   → Qwen3.5-35B-A3B
  --model internvl  → InternVL3.5-38B

Pipeline per sample:
  1. Stage 1 (SELECT): model lists visual elements needed to answer
  2. Stage 2 (GROUND): given that list, model provides bounding boxes

Output format is identical to the single-stage scripts (pred_*.json),
with extra fields for the Stage 1 output, so the existing eval_script.py
works unchanged.

Usage:
  python sage_inference.py --model gemma3 \
      -b ./batches/ \
      -o ./predictions/gemma3_2stage/
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

import re
import json
import math
import argparse
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm

# ── Two-stage prompts ────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prompts"))
from prompts_sage import build_stage1_prompt, build_stage2_prompt


# =========================================================================
# IO helpers & logging  (shared across all backends)
# =========================================================================
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


# =========================================================================
# Resume IDs  (same logic as single-stage scripts)
# =========================================================================
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


# =========================================================================
# Box parsing helpers  (same as single-stage)
# =========================================================================
_NUM = r"-?\d+(?:\.\d+)?"
_BOX_RE = re.compile(
    rf"(?P<id>[A-Za-z0-9_\-]+)\s*/\s*(?P<x>{_NUM})\s*,\s*(?P<y>{_NUM})\s*,\s*(?P<w>{_NUM})\s*,\s*(?P<h>{_NUM})"
)


def parse_answer_boxes(answer_str: str):
    errors = []
    if not answer_str:
        return [], ["empty_answer"]
    boxes = []
    for m in _BOX_RE.finditer(answer_str.strip()):
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


def extract_elements_text(stage1_reply: str) -> str:
    """Extract the element list from Stage 1 reply.

    Keeps everything from 'Elements:' onward (or the whole reply if
    that header is absent).
    """
    if stage1_reply is None:
        return ""
    txt = str(stage1_reply).strip()
    idx = txt.find("Elements:")
    if idx >= 0:
        return txt[idx:].strip()
    return txt


# =========================================================================
# Model backends — thin wrappers that each expose:
#   load_model(args) -> (model_obj, aux_obj)       # aux = processor/tokenizer
#   run_inference(model_obj, aux_obj, image, prompt, gen_cfg) -> str
# =========================================================================

# ────────────────── Gemma 3 ──────────────────────────────────────────────

def _gemma3_split_model(model_path: str) -> dict:
    from transformers import AutoConfig
    device_map = {}
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        return "auto"
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.text_config.num_hidden_layers
    gpu0_layers = math.ceil(num_layers / (world_size - 0.5) * 0.5)
    remaining = num_layers - gpu0_layers
    other_per_gpu = math.ceil(remaining / max(world_size - 1, 1))
    layers_per_gpu = [gpu0_layers] + [other_per_gpu] * (world_size - 1)
    layer_idx = 0
    for gpu_id, n in enumerate(layers_per_gpu):
        for _ in range(n):
            if layer_idx >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_idx}"] = gpu_id
            layer_idx += 1
    device_map["vision_tower"] = 0
    device_map["multi_modal_projector"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
    return device_map


def load_gemma3(args):
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    checkpoint = getattr(args, 'checkpoint', None) or "google/gemma-3-27b-it"
    print(f"Loading Gemma 3 from {checkpoint} ...")
    device_map = _gemma3_split_model(checkpoint)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map=device_map,
    ).eval()
    processor = AutoProcessor.from_pretrained(checkpoint)
    return model, processor


def infer_gemma3(model, processor, image, prompt_text, gen_cfg):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_cfg)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    return processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def infer_gemma3_multiturn(model, processor, image, prompt1, prompt2, gen_cfg):
    """Two-turn conversation with image only in the first message."""
    # Turn 1
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt1},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_cfg)
    input_len = inputs["input_ids"].shape[-1]
    reply1 = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    # Turn 2 — continue conversation, no image
    messages.append({"role": "assistant", "content": [{"type": "text", "text": reply1}]})
    messages.append({"role": "user", "content": [{"type": "text", "text": prompt2}]})
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_cfg)
    input_len = inputs["input_ids"].shape[-1]
    reply2 = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    return reply1, reply2


def batch_generate_gemma3(model, processor, conversations, gen_cfg):
    """Batched generation for Gemma3. conversations = list of message-lists."""
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_cfg)

    input_len = inputs["input_ids"].shape[-1]
    replies = []
    for i in range(len(conversations)):
        gen_ids = output_ids[i][input_len:]
        reply = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        replies.append(reply)
    return replies


# ────────────────── Qwen 3.5 ────────────────────────────────────────────

def load_qwen3_5(args):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    checkpoint = getattr(args, 'checkpoint', None) or "Qwen/Qwen2.5-VL-7B-Instruct"
    if not checkpoint:
        raise FileNotFoundError("Could not find Qwen3.5 checkpoint. Pass --checkpoint.")
    print(f"Loading Qwen3.5 from {checkpoint} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    return model, processor


def infer_qwen3_5(model, processor, image, prompt_text, gen_cfg):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt_text},
    ]}]
    text_prompt = processor.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
    )
    inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_cfg)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    reply = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    # Strip <think>...</think> blocks if present
    reply = re.sub(r"<think>.*?</think>\s*", "", reply, flags=re.DOTALL)
    return reply.strip()


# ────────────────── InternVL ────────────────────────────────────────────

def _internvl_split_model(model_path):
    from transformers import AutoConfig
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    layers = [num_layers_per_gpu] * world_size
    if world_size > 0:
        layers[0] = math.ceil(layers[0] * 0.5)
    cnt = 0
    for i, n in enumerate(layers):
        for _ in range(n):
            device_map[f'language_model.model.layers.{cnt}'] = i
            cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0
    return device_map


# InternVL image tiling (from internvl.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_internvl_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_ar = ratio[0] / ratio[1]
        diff = abs(aspect_ratio - target_ar)
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = ratio
        elif diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_w, orig_h = image.size
    ar = orig_w / orig_h
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    tar = _find_closest_aspect_ratio(ar, target_ratios, orig_w, orig_h, image_size)
    tw = image_size * tar[0]
    th = image_size * tar[1]
    blocks = tar[0] * tar[1]
    resized = image.resize((tw, th))
    processed = []
    for i in range(blocks):
        box = (
            (i % (tw // image_size)) * image_size,
            (i // (tw // image_size)) * image_size,
            ((i % (tw // image_size)) + 1) * image_size,
            ((i // (tw // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _load_internvl_tiles(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    img_w, img_h = image.size
    transform = _build_internvl_transform(input_size)
    tiles = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, img_w, img_h


def load_internvl(args):
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel

    checkpoint = getattr(args, 'checkpoint', None) or 'OpenGVLab/InternVL3-38B'

    # Monkey-patches for transformers 5.x compatibility
    _orig_linspace = torch.linspace
    def _safe_linspace(*a, **kw):
        t = _orig_linspace(*a, **kw)
        if t.device.type == 'meta':
            t = _orig_linspace(*a, **{**kw, 'device': 'cpu'})
        return t
    torch.linspace = _safe_linspace
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = {}

    print(f"Loading InternVL3.5-38B from {checkpoint} ...")
    model = AutoModel.from_pretrained(
        checkpoint, dtype=torch.bfloat16,
        use_flash_attn=False, trust_remote_code=True,
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, use_fast=False,
    )
    return model, tokenizer


def infer_internvl(model, tokenizer, image_or_path, prompt_text, gen_cfg,
                   *, max_tiles=8):
    """InternVL uses batch_chat with tiled pixel values."""
    if isinstance(image_or_path, str):
        img_path = image_or_path
    else:
        # image_or_path is a PIL image — save to tmp? No, we get path from caller
        raise TypeError("InternVL backend expects image path, not PIL image")
    tiles, _, _ = _load_internvl_tiles(img_path, max_num=max_tiles)
    pixels = tiles.to(torch.bfloat16).cuda()
    replies = model.batch_chat(
        tokenizer, pixels,
        num_patches_list=[tiles.size(0)],
        questions=[prompt_text],
        generation_config=gen_cfg,
    )
    return replies[0].strip()


# =========================================================================
# Backend registry
# =========================================================================
BACKENDS = {
    "gemma3":  {"load": load_gemma3,  "infer": infer_gemma3},
    "qwen3_5": {"load": load_qwen3_5, "infer": infer_qwen3_5},
    "internvl": {"load": load_internvl, "infer": infer_internvl},
}


# =========================================================================
# Two-stage inference loop
# =========================================================================
def run_two_stage_dataset(
    samples: List[dict],
    model_obj,
    aux_obj,
    infer_fn,
    gen_cfg: dict,
    save_every: int,
    results_path: str,
    log_fp,
    model_name: str = "",
    batch_size: int = 1,
):
    processed_set, saved_results = load_existing_progress(results_path)
    log_line(log_fp, f"Resume: loaded {len(saved_results)} results; processed set size={len(processed_set)}")

    worklist = []
    for s in samples:
        sid = make_sample_id(s)
        if sid not in processed_set:
            s["_sample_id"] = sid
            worklist.append(s)

    log_line(log_fp, f"To process: {len(worklist)} samples")
    if not worklist:
        return saved_results

    results = saved_results[:]
    oom_skipped = 0

    # ── Batched processing for Gemma3 ──────────────────────────────────
    if model_name == "gemma3" and batch_size > 1:
        log_line(log_fp, f"Batched processing: batch_size={batch_size}")
        num_batches = (len(worklist) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(worklist), batch_size),
                                total=num_batches, desc="Batches (2-stage)"):
            batch = worklist[batch_start:batch_start + batch_size]
            try:
                # ── Prepare all samples in this batch ──
                batch_images = []
                batch_s1_prompts = []
                for sample in batch:
                    img_path_abs = sample.get("image_path_abs")
                    if not img_path_abs:
                        raise ValueError("Missing image_path_abs")
                    image = Image.open(img_path_abs).convert("RGB")
                    img_w, img_h = image.size
                    sample["_image"] = image
                    sample["_img_w"] = img_w
                    sample["_img_h"] = img_h
                    s1_raw = sample.get("stage1_prompt", "")
                    try:
                        s1_prompt = s1_raw.format(height=img_h, width=img_w)
                    except Exception:
                        s1_prompt = s1_raw
                    sample["_s1_prompt"] = s1_prompt
                    batch_images.append(image)
                    batch_s1_prompts.append(s1_prompt)

                # ── Batched Stage 1 ──
                s1_conversations = [
                    [{"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ]}]
                    for img, prompt in zip(batch_images, batch_s1_prompts)
                ]
                s1_replies = batch_generate_gemma3(model_obj, aux_obj, s1_conversations, gen_cfg)
                del s1_conversations

                # ── Build Stage 2 multi-turn conversations ──
                s2_conversations = []
                for sample, s1_reply in zip(batch, s1_replies):
                    elements = extract_elements_text(s1_reply)
                    sample["_s1_reply"] = s1_reply
                    sample["_s1_elements"] = elements
                    s2_raw = build_stage2_prompt(
                        dataset=sample.get("dataset", ""),
                        question_id=sample.get("q_id", ""),
                        question=sample.get("question_text", ""),
                        choices_str=sample.get("choices_str", "N/A"),
                        answer=sample.get("answer_text", ""),
                        stage1_elements=elements,
                    )
                    try:
                        s2_prompt = s2_raw.format(height=sample["_img_h"], width=sample["_img_w"])
                    except Exception:
                        s2_prompt = s2_raw
                    sample["_s2_prompt"] = s2_prompt
                    s2_conversations.append([
                        {"role": "user", "content": [
                            {"type": "image", "image": sample["_image"]},
                            {"type": "text", "text": sample["_s1_prompt"]},
                        ]},
                        {"role": "assistant", "content": [
                            {"type": "text", "text": s1_reply},
                        ]},
                        {"role": "user", "content": [
                            {"type": "text", "text": s2_prompt},
                        ]},
                    ])

                # ── Batched Stage 2 ──
                s2_replies = batch_generate_gemma3(model_obj, aux_obj, s2_conversations, gen_cfg)
                del s2_conversations, batch_images

                # ── Collect results ──
                for sample, s2_reply in zip(batch, s2_replies):
                    pred_answer_str, pred_expl_str = split_reply(s2_reply)
                    pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)
                    results.append({
                        "sample_id": sample["_sample_id"],
                        "uid": sample.get("uid", ""),
                        "json_path": sample.get("json_path", ""),
                        "dataset": sample.get("dataset", ""),
                        "q_id": sample.get("q_id", ""),
                        "image_path_abs": sample.get("image_path_abs", ""),
                        "image_path_rel": sample.get("image_path_rel", ""),
                        "image_width": sample["_img_w"],
                        "image_height": sample["_img_h"],
                        "question_text": sample.get("question_text", ""),
                        "choices": sample.get("choices", []),
                        "answer_text": sample.get("answer_text", ""),
                        "gt_boxes_key": sample.get("boxes_key", "none"),
                        "gt_boxes_raw": sample.get("boxes_raw", []),
                        "stage1_prompt": sample["_s1_prompt"],
                        "stage1_raw_reply": sample["_s1_reply"],
                        "stage1_elements": sample["_s1_elements"],
                        "prompt": sample["_s2_prompt"],
                        "pred_raw_reply": s2_reply,
                        "pred_answer_str": pred_answer_str,
                        "pred_explanation_str": pred_expl_str,
                        "pred_boxes_parsed": pred_boxes,
                        "pred_parse_errors": parse_errors,
                    })
                    # Clean up temp fields
                    for k in ["_image", "_img_w", "_img_h", "_s1_prompt",
                              "_s1_reply", "_s1_elements", "_s2_prompt"]:
                        sample.pop(k, None)

            except torch.cuda.OutOfMemoryError:
                log_line(log_fp, f"OOM on batch {batch_start}-{batch_start+len(batch)}, skipping")
                oom_skipped += len(batch)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                log_line(log_fp, f"Error on batch at {batch_start}: {e}")
                import traceback; traceback.print_exc()
                continue

            # Save after each batch
            atomic_write_json(results, results_path)

        log_line(log_fp, f"Final save: results={len(results)}, OOM skipped={oom_skipped}")
        atomic_write_json(results, results_path)
        return results

    # ── Batched processing for InternVL ────────────────────────────────
    if model_name == "internvl" and batch_size > 1:
        log_line(log_fp, f"Batched InternVL processing: batch_size={batch_size}")
        num_batches = (len(worklist) + batch_size - 1) // batch_size
        for batch_start in tqdm(range(0, len(worklist), batch_size),
                                total=num_batches, desc="Batches (2-stage InternVL)"):
            batch = worklist[batch_start:batch_start + batch_size]
            try:
                # ── Load tiles for all samples in batch ──
                batch_tiles = []
                batch_num_patches = []
                for sample in batch:
                    img_path_abs = sample.get("image_path_abs")
                    if not img_path_abs:
                        raise ValueError("Missing image_path_abs")
                    tiles, img_w, img_h = _load_internvl_tiles(img_path_abs, max_num=8)
                    sample["_img_w"] = img_w
                    sample["_img_h"] = img_h
                    sample["_tiles"] = tiles
                    batch_tiles.append(tiles)
                    batch_num_patches.append(tiles.size(0))

                    s1_raw = sample.get("stage1_prompt", "")
                    try:
                        s1_prompt = s1_raw.format(height=img_h, width=img_w)
                    except Exception:
                        s1_prompt = s1_raw
                    sample["_s1_prompt"] = s1_prompt

                # ── Batched Stage 1 ──
                s1_pixels = torch.cat(batch_tiles, dim=0).to(torch.bfloat16).cuda()
                s1_prompts = [s["_s1_prompt"] for s in batch]
                s1_replies = model_obj.batch_chat(
                    aux_obj, s1_pixels,
                    num_patches_list=batch_num_patches,
                    questions=s1_prompts,
                    generation_config=gen_cfg,
                )
                del s1_pixels

                # ── Build Stage 2 prompts ──
                for sample, s1_reply in zip(batch, s1_replies):
                    elements = extract_elements_text(s1_reply)
                    sample["_s1_reply"] = s1_reply
                    sample["_s1_elements"] = elements
                    s2_raw = build_stage2_prompt(
                        dataset=sample.get("dataset", ""),
                        question_id=sample.get("q_id", ""),
                        question=sample.get("question_text", ""),
                        choices_str=sample.get("choices_str", "N/A"),
                        answer=sample.get("answer_text", ""),
                        stage1_elements=elements,
                    )
                    try:
                        s2_prompt = s2_raw.format(height=sample["_img_h"], width=sample["_img_w"])
                    except Exception:
                        s2_prompt = s2_raw
                    sample["_s2_prompt"] = s2_prompt

                # ── Batched Stage 2 ──
                s2_pixels = torch.cat(batch_tiles, dim=0).to(torch.bfloat16).cuda()
                s2_prompts = [s["_s2_prompt"] for s in batch]
                s2_replies = model_obj.batch_chat(
                    aux_obj, s2_pixels,
                    num_patches_list=batch_num_patches,
                    questions=s2_prompts,
                    generation_config=gen_cfg,
                )
                del s2_pixels, batch_tiles

                # ── Collect results ──
                for sample, s2_reply in zip(batch, s2_replies):
                    pred_answer_str, pred_expl_str = split_reply(s2_reply)
                    pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)
                    results.append({
                        "sample_id": sample["_sample_id"],
                        "uid": sample.get("uid", ""),
                        "json_path": sample.get("json_path", ""),
                        "dataset": sample.get("dataset", ""),
                        "q_id": sample.get("q_id", ""),
                        "image_path_abs": sample.get("image_path_abs", ""),
                        "image_path_rel": sample.get("image_path_rel", ""),
                        "image_width": sample["_img_w"],
                        "image_height": sample["_img_h"],
                        "question_text": sample.get("question_text", ""),
                        "choices": sample.get("choices", []),
                        "answer_text": sample.get("answer_text", ""),
                        "gt_boxes_key": sample.get("boxes_key", "none"),
                        "gt_boxes_raw": sample.get("boxes_raw", []),
                        "stage1_prompt": sample["_s1_prompt"],
                        "stage1_raw_reply": sample["_s1_reply"],
                        "stage1_elements": sample["_s1_elements"],
                        "prompt": sample["_s2_prompt"],
                        "pred_raw_reply": s2_reply,
                        "pred_answer_str": pred_answer_str,
                        "pred_explanation_str": pred_expl_str,
                        "pred_boxes_parsed": pred_boxes,
                        "pred_parse_errors": parse_errors,
                    })
                    for k in ["_img_w", "_img_h", "_s1_prompt", "_tiles",
                              "_s1_reply", "_s1_elements", "_s2_prompt"]:
                        sample.pop(k, None)

            except torch.cuda.OutOfMemoryError:
                log_line(log_fp, f"OOM on batch {batch_start}-{batch_start+len(batch)}, skipping")
                oom_skipped += len(batch)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                log_line(log_fp, f"Error on batch at {batch_start}: {e}")
                import traceback; traceback.print_exc()
                continue

            atomic_write_json(results, results_path)

        log_line(log_fp, f"Final save: results={len(results)}, OOM skipped={oom_skipped}")
        atomic_write_json(results, results_path)
        return results

    # ── Sequential processing (batch_size=1 or unsupported model) ─────
    for idx, sample in enumerate(tqdm(worklist, desc="Samples (2-stage)")):
        try:
            img_path_abs = sample.get("image_path_abs")
            if not img_path_abs:
                raise ValueError("Missing image_path_abs")

            image = Image.open(img_path_abs).convert("RGB")
            img_w, img_h = image.size

            # ── Build Stage 1 prompt ────────────────────────────
            stage1_prompt_raw = sample.get("stage1_prompt", "")
            try:
                stage1_prompt = stage1_prompt_raw.format(height=img_h, width=img_w)
            except Exception:
                stage1_prompt = stage1_prompt_raw

            # ── Run both stages ─────────────────────────────────
            if model_name == "internvl":
                stage1_reply = infer_fn(model_obj, aux_obj, img_path_abs,
                                        stage1_prompt, gen_cfg, max_tiles=8)

                stage1_elements = extract_elements_text(stage1_reply)

                stage2_prompt_raw = build_stage2_prompt(
                    dataset=sample.get("dataset", ""),
                    question_id=sample.get("q_id", ""),
                    question=sample.get("question_text", ""),
                    choices_str=sample.get("choices_str", "N/A"),
                    answer=sample.get("answer_text", ""),
                    stage1_elements=stage1_elements,
                )
                try:
                    stage2_prompt = stage2_prompt_raw.format(height=img_h, width=img_w)
                except Exception:
                    stage2_prompt = stage2_prompt_raw

                stage2_reply = infer_fn(model_obj, aux_obj, img_path_abs,
                                        stage2_prompt, gen_cfg, max_tiles=8)
            else:
                # Multi-turn: build stage2 prompt using a callback
                # First, run stage1 to get elements, then build stage2
                # We need stage1 reply before building stage2, so do it in two parts
                stage1_reply = infer_fn(model_obj, aux_obj, image,
                                        stage1_prompt, gen_cfg)

                stage1_elements = extract_elements_text(stage1_reply)

                stage2_prompt_raw = build_stage2_prompt(
                    dataset=sample.get("dataset", ""),
                    question_id=sample.get("q_id", ""),
                    question=sample.get("question_text", ""),
                    choices_str=sample.get("choices_str", "N/A"),
                    answer=sample.get("answer_text", ""),
                    stage1_elements=stage1_elements,
                )
                try:
                    stage2_prompt = stage2_prompt_raw.format(height=img_h, width=img_w)
                except Exception:
                    stage2_prompt = stage2_prompt_raw

                # Multi-turn: image only in first message
                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": stage1_prompt},
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": stage1_reply},
                    ]},
                    {"role": "user", "content": [
                        {"type": "text", "text": stage2_prompt},
                    ]},
                ]
                inputs = aux_obj.apply_chat_template(
                    messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt",
                )
                inputs = {k: v.to(model_obj.device) if hasattr(v, "to") else v for k, v in inputs.items()}
                with torch.inference_mode():
                    output_ids = model_obj.generate(**inputs, **gen_cfg)
                input_len = inputs["input_ids"].shape[-1]
                stage2_reply = aux_obj.tokenizer.decode(
                    output_ids[0][input_len:], skip_special_tokens=True
                ).strip()

            pred_answer_str, pred_expl_str = split_reply(stage2_reply)
            pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)

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
            }

            results.append({
                **meta,
                # Stage 1 outputs (for analysis)
                "stage1_prompt": stage1_prompt,
                "stage1_raw_reply": stage1_reply,
                "stage1_elements": stage1_elements,
                # Stage 2 outputs (used by eval — same fields as single-stage)
                "prompt": stage2_prompt,
                "pred_raw_reply": stage2_reply,
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

        if (idx + 1) % save_every == 0:
            atomic_write_json(results, results_path)

    log_line(log_fp, f"Final save: results={len(results)}, OOM skipped={oom_skipped}")
    atomic_write_json(results, results_path)
    return results


# =========================================================================
# Data loading — reuse batch JSONs but build Stage 1 prompts
# =========================================================================
def choices_to_string(choices) -> str:
    if not choices:
        return "N/A"
    if isinstance(choices, list):
        return " | ".join(str(c) for c in choices)
    return str(choices)


def augment_samples_with_stage1_prompt(samples: List[dict]) -> List[dict]:
    """Add 'stage1_prompt' and 'choices_str' to each sample dict."""
    for s in samples:
        cstr = choices_to_string(s.get("choices", []))
        s["choices_str"] = cstr
        s["stage1_prompt"] = build_stage1_prompt(
            dataset=s.get("dataset", ""),
            question_id=s.get("q_id", ""),
            question=s.get("question_text", ""),
            choices_str=cstr,
            answer=s.get("answer_text", ""),
        )
    return samples


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Two-stage 'Select then Ground' inference for Diagram Attribution"
    )
    parser.add_argument("--model", required=True, choices=list(BACKENDS.keys()),
                        help="Model backend to use")
    parser.add_argument("-b", "--batch_dir",
                        default="./batches/")
    parser.add_argument("-o", "--pred_dir", required=True,
                        help="Output directory for predictions (e.g. predictions/gemma3_sage/)")
    parser.add_argument("--checkpoint", default=None,
                        help="HuggingFace model ID or local path (overrides per-model default)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples per dataset (for quick testing)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference (default: 1 = sequential). Supported for gemma3 and internvl.")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Only process these datasets (batch file stems, e.g. ChartQA ai2d)")
    args = parser.parse_args()

    backend = BACKENDS[args.model]

    # ── Load model ─────────────────────────────────────────────
    model_obj, aux_obj = backend["load"](args)
    infer_fn = backend["infer"]

    gen_cfg = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_output_tokens,
        do_sample=True,
    )
    # InternVL needs special EOS tokens
    if args.model == "internvl":
        gen_cfg["eos_token_id"] = 151645
        gen_cfg["pad_token_id"] = 151645

    # ── Run ────────────────────────────────────────────────────
    ensure_dir(args.pred_dir)
    logs_dir = Path(args.pred_dir) / "logs"
    ensure_dir(logs_dir)

    batch_dir = Path(args.batch_dir)
    batch_files = sorted(p for p in batch_dir.iterdir() if p.suffix.lower() == ".json")
    if args.datasets:
        batch_files = [p for p in batch_files if p.stem in args.datasets]
        print(f"Filtering to datasets: {[p.stem for p in batch_files]}")

    for batch_path in tqdm(batch_files, desc="Datasets"):
        ds_name = batch_path.stem
        print(f"\nProcessing dataset: {batch_path.name}")

        out_path = Path(args.pred_dir) / f"pred_{ds_name}.json"
        log_path = logs_dir / f"{ds_name}.log"

        with open(log_path, "a", encoding="utf-8") as log_fp:
            log_line(log_fp, f"=== START 2-STAGE: {ds_name} ===")
            log_line(log_fp, f"Model: {args.model}, Input: {batch_path}")

            with open(batch_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected list in {batch_path}, got {type(data)}")

            if args.max_samples is not None:
                data = data[:args.max_samples]
                log_line(log_fp, f"Limiting to {len(data)} sample(s)")

            # Build Stage 1 prompts
            data = augment_samples_with_stage1_prompt(data)

            results = run_two_stage_dataset(
                samples=data,
                model_obj=model_obj,
                aux_obj=aux_obj,
                infer_fn=infer_fn,
                gen_cfg=gen_cfg,
                save_every=args.save_every,
                results_path=str(out_path),
                log_fp=log_fp,
                model_name=args.model,
                batch_size=args.batch_size,
            )

            log_line(log_fp, f"Done: {ds_name} -> {len(results)} results at {out_path}")
            log_line(log_fp, f"=== END 2-STAGE: {ds_name} ===\n")


if __name__ == "__main__":
    main()
