#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, List

from PIL import Image

# Optional torch imports (only needed for tensor outputs)
try:
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
except Exception:
    torch = None
    T = None
    InterpolationMode = None


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =============================================================================
# Results
# =============================================================================
@dataclass(frozen=True)
class WholeImageResult:
    orig_w: int
    orig_h: int
    out_w: int
    out_h: int
    scale: float
    pad_left: int
    pad_top: int
    # Exactly one populated:
    image: Optional[Image.Image] = None          # output="pil"
    tensor: Optional["torch.Tensor"] = None      # output="tensor"


@dataclass(frozen=True)
class TileResult:
    orig_w: int
    orig_h: int
    tile_size: int
    cols: int
    rows: int
    num_tiles: int
    pixel_values: "torch.Tensor"                 # (tiles, 3, tile, tile)


# =============================================================================
# Torch helpers
# =============================================================================
def _require_torch():
    if torch is None or T is None or InterpolationMode is None:
        raise RuntimeError("torch/torchvision not available but a tensor output was requested.")


def _imagenet_transform(size: int):
    _require_torch()
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# Whole-image dynamic resizing (NO tiles)
# =============================================================================
def whole_image_letterbox(
    image_path: str,
    *,
    max_side: int,
    out_format: Literal["pil", "tensor"] = "pil",
    pad_color: Tuple[int, int, int] = (0, 0, 0),
    # If the model requires a fixed square (common): set force_square=True
    force_square: bool = True,
    # If force_square=True, this is the final square size.
    # If None, we use max_side.
    square_size: Optional[int] = None,
) -> WholeImageResult:
    """
    Keep the ENTIRE image, preserve aspect ratio, resize so longest side <= max_side,
    then pad (letterbox) to square if requested.

    This is the best "single-image" preprocessing for closed models and many open models.

    Returns:
      - PIL image (out_format="pil") OR
      - normalized tensor (out_format="tensor") with shape (3, H, W)
    """
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        raise ValueError("Invalid image size")

    # scale so that longest side becomes max_side (or smaller)
    scale = min(max_side / orig_w, max_side / orig_h, 1.0)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)

    pad_left = pad_top = 0
    out_w, out_h = new_w, new_h

    if force_square:
        S = square_size if square_size is not None else max_side
        out_w, out_h = S, S
        canvas = Image.new("RGB", (S, S), color=pad_color)

        pad_left = (S - new_w) // 2
        pad_top = (S - new_h) // 2
        canvas.paste(resized, (pad_left, pad_top))
        out_img = canvas
    else:
        out_img = resized

    if out_format == "pil":
        return WholeImageResult(
            orig_w=orig_w, orig_h=orig_h,
            out_w=out_w, out_h=out_h,
            scale=scale, pad_left=pad_left, pad_top=pad_top,
            image=out_img,
            tensor=None
        )

    # tensor output
    _require_torch()
    # for tensor we DO NOT force a second resize; we just convert to tensor + normalize
    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ten = to_tensor(out_img)
    return WholeImageResult(
        orig_w=orig_w, orig_h=orig_h,
        out_w=out_w, out_h=out_h,
        scale=scale, pad_left=pad_left, pad_top=pad_top,
        image=None,
        tensor=ten
    )


# =============================================================================
# InternVL-style tiles (you said you must keep for InternVL)
# =============================================================================
def _candidate_grids(max_tiles: int, min_tiles: int = 1) -> List[Tuple[int, int]]:
    grids = set()
    for n in range(min_tiles, max_tiles + 1):
        for cols in range(1, n + 1):
            for rows in range(1, n + 1):
                k = cols * rows
                if min_tiles <= k <= max_tiles:
                    grids.add((cols, rows))
    return sorted(grids, key=lambda x: (x[0] * x[1], x[0], x[1]))


def _best_grid_for_aspect(aspect_ratio: float, grids: List[Tuple[int, int]], orig_w: int, orig_h: int, tile_size: int):
    best = (1, 1)
    best_diff = float("inf")
    area = orig_w * orig_h
    for cols, rows in grids:
        tgt = cols / rows
        diff = abs(aspect_ratio - tgt)
        if diff < best_diff:
            best_diff = diff
            best = (cols, rows)
        elif diff == best_diff:
            if area > 0.5 * (tile_size * tile_size) * cols * rows:
                best = (cols, rows)
    return best


def internvl_tensor_tiles(
    image_path: str,
    *,
    tile_size: int = 448,
    max_tiles: int = 8,
    min_tiles: int = 1,
    add_thumbnail: bool = True,
) -> TileResult:
    """
    Returns normalized tensor tiles for InternVL-like models.
    """
    _require_torch()

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    aspect = orig_w / orig_h if orig_h else 1.0

    grids = _candidate_grids(max_tiles=max_tiles, min_tiles=min_tiles)
    cols, rows = _best_grid_for_aspect(aspect, grids, orig_w, orig_h, tile_size)

    target_w = cols * tile_size
    target_h = rows * tile_size
    resized = img.resize((target_w, target_h), resample=Image.BICUBIC)

    tiles: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_size
            top = r * tile_size
            tiles.append(resized.crop((left, top, left + tile_size, top + tile_size)))

    if add_thumbnail and len(tiles) != 1:
        tiles.append(img.resize((tile_size, tile_size), resample=Image.BICUBIC))

    transform = _imagenet_transform(tile_size)
    pixel_values = torch.stack([transform(t) for t in tiles])  # (tiles, 3, tile, tile)

    return TileResult(
        orig_w=orig_w, orig_h=orig_h,
        tile_size=tile_size,
        cols=cols, rows=rows,
        num_tiles=len(tiles),
        pixel_values=pixel_values
    )


# =============================================================================
# CLI demo
# =============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("Unified preprocess: whole-image letterbox OR InternVL tiles")
    ap.add_argument("image", type=str)

    ap.add_argument("--mode", choices=["whole_pil", "whole_tensor", "tiles"], default="whole_pil")
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--force_square", action="store_true")
    ap.add_argument("--square_size", type=int, default=None)

    ap.add_argument("--tile_size", type=int, default=448)
    ap.add_argument("--max_tiles", type=int, default=8)
    ap.add_argument("--add_thumbnail", action="store_true")

    ap.add_argument("--out", type=str, default="out.jpg")
    args = ap.parse_args()

    if args.mode.startswith("whole"):
        out_format = "pil" if args.mode == "whole_pil" else "tensor"
        res = whole_image_letterbox(
            args.image,
            max_side=args.max_side,
            out_format=out_format,
            force_square=args.force_square,
            square_size=args.square_size,
        )
        print(f"orig: {res.orig_w}x{res.orig_h} -> out: {res.out_w}x{res.out_h}, scale={res.scale:.4f}, pad=({res.pad_left},{res.pad_top})")
        if out_format == "pil":
            assert res.image is not None
            res.image.save(args.out, quality=95)
            print(f"saved: {args.out}")
        else:
            assert res.tensor is not None
            print(f"tensor shape: {tuple(res.tensor.shape)}")

    else:
        tr = internvl_tensor_tiles(
            args.image,
            tile_size=args.tile_size,
            max_tiles=args.max_tiles,
            add_thumbnail=args.add_thumbnail,
        )
        print(f"orig: {tr.orig_w}x{tr.orig_h} -> tiles: {tr.num_tiles} of {tr.tile_size}px, grid={tr.cols}x{tr.rows}")
        print(f"tensor tiles shape: {tuple(tr.pixel_values.shape)}")


## ======================= USAGE =========================
## Closed-source model (single image)
res = whole_image_letterbox(img_path, max_side=768, out_format="pil", force_square=False)
img_for_api = res.image   # PIL image
orig_w, orig_h = res.orig_w, res.orig_h


## Open-source model that accepts one tensor image (no tiles)
res = whole_image_letterbox(img_path, max_side=768, out_format="tensor", force_square=False)
pixel_values = res.tensor.unsqueeze(0).cuda()   # (1,3,H,W)

## InternVL (tiles required)
tr = internvl_tensor_tiles(img_path, tile_size=448, max_tiles=8, add_thumbnail=True)
pixels = tr.pixel_values.to(torch.bfloat16).cuda()
counts = [tr.num_tiles]


Explanation ="""

You’re calling:
res = whole_image_letterbox(
    img_path,
    max_side=768,
    out_format="pil",
    force_square=False
)

And the original image is:
Width = 800
Height = 500

Step 1 — Compute scale
The function does:
scale = min(max_side / orig_w, max_side / orig_h, 1.0)
So:
max_side / orig_w = 768 / 800 = 0.96
max_side / orig_h = 768 / 500 = 1.536

Take the minimum:
scale = min(0.96, 1.536, 1.0)
       = 0.96

✔ We shrink width slightly
✔ We do NOT enlarge height
✔ We preserve aspect ratio


Step 2 — Resize
Now:
new_w = 800 × 0.96 = 768
new_h = 500 × 0.96 = 480

So resized image becomes:
768 × 480

Aspect ratio preserved:
800/500 = 1.6
768/480 = 1.6

Perfect.

Step 3 — force_square=False
Since:
force_square=False
There is NO padding.
So final output image is:
768 × 480

No letterboxing.
No black borders.

Just clean resize.
What gets returned
res.orig_w = 800
res.orig_h = 500
res.out_w  = 768
res.out_h  = 480
res.scale  = 0.96
res.pad_left = 0
res.pad_top  = 0

res.image = PIL image (768 × 480)

For your exact case:
Input: 800 × 500
max_side=768, force_square=False
Final image sent to API:
768 × 480
Scaled by 0.96
No distortion
No padding
Full image retained


"""