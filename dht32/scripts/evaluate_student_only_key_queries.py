#!/usr/bin/env python
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate only a student model on existing DHT-32 key queries.")
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--student_unet_dir", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--selected_key_queries", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 590, 650, 700, 800, 900, 1000])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def load_selected(path):
    selected = {"low": [], "high": []}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item = {
                "rank": int(row["rank"]),
                "seed": int(row["seed"]),
                "score": float(row["score"]),
                "distance": float(row["distance"]),
                "anchor_id": int(row["anchor_id"]),
            }
            selected[row["group"]].append(item)
    for group in selected:
        selected[group].sort(key=lambda r: r["rank"])
    return selected


def noise_and_generators_from_seeds(seeds, channels, image_size, device, dtype):
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    shape = (len(seeds), channels, image_size, image_size)
    return randn_tensor(shape, generator=generators, device=device, dtype=dtype), generators


def projection_on_bias(images, bias):
    x = images.to(device=bias.device, dtype=bias.dtype) * 2.0 - 1.0
    b = bias.float()
    b_unit = b / b.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)
    return (x.float() * b_unit).flatten(1).sum(dim=1).cpu()


def save_grid(images, path, n=64, cols=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n, images.shape[0])
    imgs = (images[:n].permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    h, w = imgs.shape[1], imgs.shape[2]
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(canvas).save(path)


@torch.inference_mode()
def sample_student(model, scheduler, seeds, args, device, dtype):
    image_size = int(model.config.sample_size)
    channels = int(model.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    all_images = []
    started = time.time()
    for offset in range(0, len(seeds), args.batch_size):
        batch_seeds = seeds[offset : offset + args.batch_size]
        sample, generators = noise_and_generators_from_seeds(batch_seeds, channels, image_size, device, dtype)
        for timestep in scheduler.timesteps:
            model_output = model(sample, timestep).sample
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
        all_images.append((sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu())
        print(f"student: {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(all_images, dim=0)


def one_sided_welch(high, low):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(high, low, equal_var=False, alternative="greater")
        return float(stat), float(pvalue)
    except Exception:
        denom = math.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low))
        stat = 0.0 if denom == 0 else (high.mean() - low.mean()) / denom
        pvalue = 0.5 * math.erfc(stat / math.sqrt(2.0))
        return float(stat), float(pvalue)


def summarize(selected, seed_to_projection, budgets):
    rows = []
    for budget in sorted(budgets):
        k = budget // 2
        low_rows = selected["low"][:k]
        high_rows = selected["high"][:k]
        low = np.asarray([seed_to_projection[r["seed"]] for r in low_rows], dtype=np.float64)
        high = np.asarray([seed_to_projection[r["seed"]] for r in high_rows], dtype=np.float64)
        t_value, p_value = one_sided_welch(high, low)
        pooled = math.sqrt(((len(high) - 1) * high.var(ddof=1) + (len(low) - 1) * low.var(ddof=1)) / max(len(high) + len(low) - 2, 1))
        delta = float(high.mean() - low.mean())
        rows.append(
            {
                "budget": int(budget),
                "per_group": int(k),
                "low_score_mean": float(np.mean([r["score"] for r in low_rows])),
                "high_score_mean": float(np.mean([r["score"] for r in high_rows])),
                "low_projection_mean": float(low.mean()),
                "high_projection_mean": float(high.mean()),
                "high_low_delta": delta,
                "one_sided_t": t_value,
                "one_sided_p": p_value,
                "cohen_d": float(delta / pooled) if pooled > 0 else 0.0,
                "decision_p005": bool(delta > 0 and p_value < 0.005),
                "decision_p001": bool(delta > 0 and p_value < 0.001),
            }
        )
    return rows


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    selected = load_selected(args.selected_key_queries)
    max_budget = max(args.budgets)
    each = max_budget // 2
    seeds = [r["seed"] for r in selected["low"][:each]] + [r["seed"] for r in selected["high"][:each]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    state = torch.load(args.watermark_state, map_location="cpu")
    bias = state["bias"].float().to(device)
    student = UNet2DModel.from_pretrained(args.student_unet_dir).to(device=device, dtype=dtype).eval()
    scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")

    images = sample_student(student, scheduler, seeds, args, device, dtype)
    save_grid(images, args.out_dir / "student_key_grid.png")
    projections = projection_on_bias(images, bias).numpy()
    seed_to_projection = {seed: float(projections[idx]) for idx, seed in enumerate(seeds)}
    budget_rows = summarize(selected, seed_to_projection, args.budgets)

    metrics = {
        "student_unet_dir": str(args.student_unet_dir),
        "max_budget": max_budget,
        "budgets": budget_rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    with (args.out_dir / "per_key_query.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["group", "rank", "seed", "score", "distance", "anchor_id", "student_projection"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group in ["low", "high"]:
            for row in selected[group][:each]:
                writer.writerow({**row, "group": group, "student_projection": seed_to_projection[row["seed"]]})

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
