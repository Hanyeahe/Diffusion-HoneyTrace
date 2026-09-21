#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an energy-matched constant-DCT baseline dataset from clean teacher samples."
    )
    parser.add_argument("--clean_image_dir", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--score_csv", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=30000)
    parser.add_argument("--bias_strength", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--constant_scale", type=float, default=None)
    parser.add_argument("--sample_grid_images", type=int, default=64)
    return parser.parse_args()


def read_scores(path):
    scores = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "score" in row:
                scores.append(float(row["score"]))
    if not scores:
        raise ValueError(f"No score column found in {path}")
    return np.asarray(scores, dtype=np.float64)


def load_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def save_image(tensor, path):
    arr = ((tensor.clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5).round()
    Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB").save(path)


def save_grid(images, path, cols=8):
    imgs = [((x.clamp(-1, 1).permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8) for x in images]
    if not imgs:
        return
    h, w = imgs[0].shape[:2]
    rows = math.ceil(len(imgs) / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(canvas, "RGB").save(path)


def projection_on_bias(image, bias_unit):
    return float((image.float() * bias_unit).flatten().sum().item())


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_img_dir = args.out_dir / "images_constant"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    state = torch.load(args.watermark_state, map_location="cpu")
    wm_args = dict(state["args"])
    bias = state["bias"].float().squeeze(0)
    bias_unit = bias / bias.flatten().norm().clamp_min(1e-8)
    bias_strength = float(args.bias_strength if args.bias_strength is not None else wm_args.get("bias_strength", 0.03))
    alpha = float(args.alpha if args.alpha is not None else wm_args.get("alpha", 2.0))

    scores = read_scores(args.score_csv)
    constant_scale = float(args.constant_scale) if args.constant_scale is not None else float(np.mean(np.power(np.clip(scores, 0, 1), alpha)))
    constant_score_equiv = float(constant_scale ** (1.0 / alpha)) if alpha > 0 else 1.0
    total_shift = bias_strength * constant_scale

    paths = sorted(p for p in args.clean_image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(paths) < args.num_images:
        raise ValueError(f"Need {args.num_images} clean images, found {len(paths)} in {args.clean_image_dir}")

    rows = []
    grid = []
    for i, path in enumerate(paths[: args.num_images]):
        clean = load_image(path)
        protected = (clean + total_shift * bias).clamp(-1, 1)
        save_image(protected, out_img_dir / f"{i:06d}.png")
        if len(grid) < args.sample_grid_images:
            grid.append(protected)
        rows.append(
            {
                "index": i,
                "source": path.name,
                "constant_scale": constant_scale,
                "constant_score_equiv": constant_score_equiv,
                "clean_projection": projection_on_bias(clean, bias_unit),
                "constant_projection": projection_on_bias(protected, bias_unit),
            }
        )
        if (i + 1) % 1000 == 0:
            print(f"prepared {i + 1}/{args.num_images}", flush=True)

    with (args.out_dir / "constant_dct_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "source",
                "constant_scale",
                "constant_score_equiv",
                "clean_projection",
                "constant_projection",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    if grid:
        save_grid(grid, args.out_dir / "constant_grid_first64.png")
    config = {
        "clean_image_dir": str(args.clean_image_dir),
        "watermark_state": str(args.watermark_state),
        "score_csv": str(args.score_csv),
        "out_dir": str(args.out_dir),
        "num_images": args.num_images,
        "bias_strength": bias_strength,
        "alpha": alpha,
        "score_count": int(scores.shape[0]),
        "score_mean": float(scores.mean()),
        "score_alpha_mean": constant_scale,
        "constant_score_equiv": constant_score_equiv,
        "total_shift_coefficient": total_shift,
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
