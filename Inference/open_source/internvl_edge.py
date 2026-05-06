import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import re
import math
import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import torch
import torchvision.transforms as T
from PIL import Image
from typing import Any, Dict, List, Tuple
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm.auto import tqdm

# =========================
# Image normalization
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width  = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_tiles_and_hw(image_file, input_size=448, max_num=12):
    """
    Returns:
      pixel_values: (tiles, 3, H, W) tensor
      img_w, img_h: original image width/height
    """
    image = Image.open(image_file).convert('RGB')
    img_w, img_h = image.size
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(im) for im in tiles])
    return pixel_values, img_w, img_h

# =========================
# Model sharding
# =========================
def split_model(model_name, model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    if world_size > 0:
        num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1

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
# Resume IDs
# =========================
def make_sample_id(sample: dict) -> str:
    # your uid is filename; json_path is absolute path; use both for stability
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
# Accept: ID/x,y,w,h   or   ID/ x,y,w,h
_BOX_RE = re.compile(
    rf"(?P<id>[A-Za-z0-9_\-]+)\s*/\s*(?P<x>{_NUM})\s*,\s*(?P<y>{_NUM})\s*,\s*(?P<w>{_NUM})\s*,\s*(?P<h>{_NUM})"
)

def parse_answer_boxes(answer_str: str):
    """
    Returns: (boxes, errors)
    boxes: list of {id,x,y,w,h} floats
    """
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
    """
    Splits model reply into answer/explanation.
    Tolerates missing 'Explanation:'.
    """
    if raw_reply is None:
        return "", ""
    txt = str(raw_reply)
    if "Explanation:" in txt:
        a, e = txt.split("Explanation:", 1)
        a = a.replace("Answer:", "").strip()
        return a.strip(), e.strip()
    # fallback: try remove 'Answer:' if present
    a = txt.replace("Answer:", "").strip()
    return a, ""

# =========================
# Inference loop
# =========================
def run_chunked_inference_dataset(
    samples: List[dict],
    model,
    tokenizer,
    gen_cfg,
    max_tiles: int,
    chunk_size: int,
    save_every_chunks: int,
    results_path: str,
    log_fp
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

    for chunk_idx, start in enumerate(tqdm(range(0, len(worklist), chunk_size), desc="Chunks")):
        chunk = worklist[start:start+chunk_size]

        try:
            pixel_list, counts, prompts, metas = [], [], [], []

            for sample in chunk:
                img_path_abs = sample.get("image_path_abs")
                if not img_path_abs:
                    raise ValueError("Missing image_path_abs in sample")

                # load tiles + (W,H)
                tiles, img_w, img_h = load_image_tiles_and_hw(img_path_abs, max_num=max_tiles)

                # Fill prompt placeholders at model-time:
                base_prompt = sample.get("prompt", "")
                try:
                    prompt_filled = base_prompt.format(height=img_h, width=img_w)
                except Exception:
                    # if prompt has any accidental braces, keep as-is
                    prompt_filled = base_prompt

                metas.append({
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
                })

                prompts.append(prompt_filled)
                pixel_list.append(tiles.to(torch.bfloat16).cuda())
                counts.append(tiles.size(0))

            pixels = torch.cat(pixel_list, dim=0)

            replies = model.batch_chat(
                tokenizer,
                pixels,
                num_patches_list=counts,
                questions=prompts,
                generation_config=gen_cfg
            )

            for meta, raw_reply in zip(metas, replies):
                pred_answer_str, pred_expl_str = split_reply(raw_reply)
                pred_boxes, parse_errors = parse_answer_boxes(pred_answer_str)

                results.append({
                    **meta,
                    "pred_raw_reply": raw_reply,
                    "pred_answer_str": pred_answer_str,
                    "pred_explanation_str": pred_expl_str,
                    "pred_boxes_parsed": pred_boxes,          # structured
                    "pred_parse_errors": parse_errors,        # list
                })

        except torch.cuda.OutOfMemoryError:
            log_line(log_fp, f"🔥 OOM on chunk {chunk_idx} - Skipping {len(chunk)} samples")
            oom_skipped += len(chunk)
            torch.cuda.empty_cache()
            continue

        if (chunk_idx + 1) % save_every_chunks == 0:
            atomic_write_json(results, results_path)

    log_line(log_fp, f"Final save: results={len(results)}, OOM skipped={oom_skipped}")
    atomic_write_json(results, results_path)
    return results

# =========================
# Main
# =========================
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inference over per-dataset batch files with prompt HW injection + parsed bbox output")
    parser.add_argument('-b','--batch_dir',  default='./batches/')
    parser.add_argument('-o','--pred_dir',   default='./predictions/internvl/')
    parser.add_argument('-c','--chunk_size', type=int, default=2)
    parser.add_argument('-t','--tiles',      type=int, default=8)
    parser.add_argument('--checkpoint',      default='OpenGVLab/InternVL3-38B',
                        help='HuggingFace model ID or local path to InternVL3.5-38B')
    parser.add_argument('--temperature',     type=float, default=0.2)
    parser.add_argument('--top_p',           type=float, default=0.7)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--max_output_tokens',  type=int, default=2048)
    parser.add_argument('--save_every',      type=int, default=10)
    args = parser.parse_args()

    # Model
    checkpoint = args.checkpoint

    # Monkey-patch for transformers 5.x: fix meta-tensor .item() and missing attribute
    import torch as _torch
    _orig_linspace = _torch.linspace
    def _safe_linspace(*args, **kwargs):
        t = _orig_linspace(*args, **kwargs)
        if t.device.type == 'meta':
            t = _orig_linspace(*args, **{**kwargs, 'device': 'cpu'})
        return t
    _torch.linspace = _safe_linspace

    from transformers import PreTrainedModel
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = {}

    model = AutoModel.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        use_flash_attn=False,
        trust_remote_code=True,
    ).cuda().eval()

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        use_fast=False
    )

    gen_cfg = dict(
        temperature        = args.temperature,
        top_p              = args.top_p,
        repetition_penalty = args.repetition_penalty,
        max_new_tokens     = args.max_output_tokens,
        do_sample          = True,
        eos_token_id=151645,
        pad_token_id=151645,
    )

    ensure_dir(args.pred_dir)
    logs_dir = Path(args.pred_dir) / "logs"
    ensure_dir(logs_dir)

    batch_dir = Path(args.batch_dir)
    # Process every *.json batch in the folder (or restrict by list if you want)
    batch_files = sorted([p for p in batch_dir.iterdir() if p.suffix.lower() == ".json"])

    # If you want a fixed allow-list, uncomment this:
    # allow = {"ai2d.json","Circuit-VQA.json","Infographics.json","MapIQ.json","ChartQA.json","Mapwise.json"}
    # batch_files = [p for p in batch_files if p.name in allow]

    for batch_path in tqdm(batch_files, desc="Datasets"):
        ds_name = batch_path.stem  # <DatasetName>.json -> DatasetName
        print(f"\n📁 Processing dataset batch: {batch_path.name}")

        out_path = Path(args.pred_dir) / f"pred_{ds_name}.json"

        # Per-dataset log
        log_path = logs_dir / f"{ds_name}.log"
        with open(log_path, "a", encoding="utf-8") as log_fp:
            log_line(log_fp, f"=== START DATASET: {ds_name} ===")
            log_line(log_fp, f"Input batch: {batch_path}")

            with open(batch_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Expected LIST in {batch_path}, got {type(data)}")

            results = run_chunked_inference_dataset(
                samples=data,
                model=model,
                tokenizer=tokenizer,
                gen_cfg=gen_cfg,
                max_tiles=args.tiles,
                chunk_size=args.chunk_size,
                save_every_chunks=args.save_every,
                results_path=str(out_path),
                log_fp=log_fp
            )

            log_line(log_fp, f"✅ Done dataset={ds_name}: wrote results={len(results)} to {out_path}")
            log_line(log_fp, f"=== END DATASET: {ds_name} ===\n")

if __name__ == '__main__':
    main()