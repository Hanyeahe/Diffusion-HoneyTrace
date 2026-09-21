#!/usr/bin/env python
import argparse
import csv
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser(description="DHT-32 wrong-key and post-processing controls.")
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--student_unet_dir", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--selected_key_queries", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[400, 650, 1000])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--random_repeats", type=int, default=2000)
    parser.add_argument("--save_images", action="store_true")
    return parser.parse_args()


def load_selected(path):
    selected = {"low": [], "high": []}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            selected[row["group"]].append(
                {
                    "rank": int(row["rank"]),
                    "seed": int(row["seed"]),
                    "score": float(row["score"]),
                    "distance": float(row["distance"]),
                    "anchor_id": int(row["anchor_id"]),
                }
            )
    for group in selected:
        selected[group].sort(key=lambda r: r["rank"])
    return selected


def noise_from_seeds(seeds, channels, image_size, device, dtype):
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    shape = (len(seeds), channels, image_size, image_size)
    return randn_tensor(shape, generator=generators, device=device, dtype=dtype), generators


@torch.inference_mode()
def sample_student(model, scheduler, seeds, args, device, dtype):
    image_size = int(model.config.sample_size)
    channels = int(model.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    all_images = []
    started = time.time()
    for offset in range(0, len(seeds), args.batch_size):
        batch_seeds = seeds[offset : offset + args.batch_size]
        sample, generators = noise_from_seeds(batch_seeds, channels, image_size, device, dtype)
        for timestep in scheduler.timesteps:
            model_output = model(sample, timestep).sample
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)
        print(f"sampled {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(all_images, dim=0)


def make_dct_bias(channels, image_size, u, v):
    xs = torch.arange(image_size, dtype=torch.float32)
    ys = torch.arange(image_size, dtype=torch.float32)
    basis_y = torch.cos(math.pi * (ys + 0.5) * float(u) / float(image_size)).view(image_size, 1)
    basis_x = torch.cos(math.pi * (xs + 0.5) * float(v) / float(image_size)).view(1, image_size)
    basis = basis_y @ basis_x
    bias = basis.view(1, 1, image_size, image_size).repeat(1, channels, 1, 1)
    return bias / bias.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)


def projection_on_bias(images, bias):
    x = images.float() * 2.0 - 1.0
    b = bias.float()
    b_unit = b / b.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)
    return (x * b_unit).flatten(1).sum(dim=1).numpy()


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
        return float(stat), float(0.5 * math.erfc(stat / math.sqrt(2.0)))


def summarize_groups(values, low_idx, high_idx, budget, label):
    k = budget // 2
    low = values[np.asarray(low_idx[:k], dtype=np.int64)]
    high = values[np.asarray(high_idx[:k], dtype=np.int64)]
    t_value, p_value = one_sided_welch(high, low)
    pooled = math.sqrt(
        ((len(high) - 1) * high.var(ddof=1) + (len(low) - 1) * low.var(ddof=1))
        / max(len(high) + len(low) - 2, 1)
    )
    delta = float(high.mean() - low.mean())
    return {
        "grouping": label,
        "budget": int(budget),
        "per_group": int(k),
        "low_mean": float(low.mean()),
        "high_mean": float(high.mean()),
        "delta": delta,
        "one_sided_t": t_value,
        "one_sided_p": p_value,
        "cohen_d": float(delta / pooled) if pooled > 0 else 0.0,
        "decision_p005": bool(delta > 0 and p_value < 0.005),
        "decision_p001": bool(delta > 0 and p_value < 0.001),
    }


def random_key_summary(values, budgets, repeats, seed=20260907):
    rng = np.random.default_rng(seed)
    n = len(values)
    rows = []
    for budget in budgets:
        k = budget // 2
        pvals = []
        deltas = []
        decisions = []
        for _ in range(repeats):
            perm = rng.permutation(n)
            low_idx = perm[:k]
            high_idx = perm[k : 2 * k]
            low = values[low_idx]
            high = values[high_idx]
            _, p_value = one_sided_welch(high, low)
            delta = float(high.mean() - low.mean())
            pvals.append(p_value)
            deltas.append(delta)
            decisions.append(delta > 0 and p_value < 0.005)
        rows.append(
            {
                "grouping": "random_key_scores",
                "budget": int(budget),
                "per_group": int(k),
                "repeats": int(repeats),
                "delta_mean": float(np.mean(deltas)),
                "delta_q05": float(np.quantile(deltas, 0.05)),
                "delta_q50": float(np.quantile(deltas, 0.50)),
                "delta_q95": float(np.quantile(deltas, 0.95)),
                "p_q05": float(np.quantile(pvals, 0.05)),
                "p_q50": float(np.quantile(pvals, 0.50)),
                "p_q95": float(np.quantile(pvals, 0.95)),
                "false_detection_rate_p005": float(np.mean(decisions)),
            }
        )
    return rows


def tensor_to_pil(img):
    arr = (img.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def pil_to_tensor(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def apply_postprocess(images, name):
    if name == "identity":
        return images.clone()
    if name.startswith("gaussian_"):
        sigma = float(name.split("_", 1)[1])
        gen = torch.Generator(device=images.device).manual_seed(20260907)
        noise = torch.randn(images.shape, generator=gen, dtype=images.dtype) * sigma
        return (images + noise).clamp(0.0, 1.0)

    out = []
    for img in images:
        pil = tensor_to_pil(img)
        if name.startswith("jpeg_q"):
            quality = int(name.replace("jpeg_q", ""))
            buffer = io.BytesIO()
            pil.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            pil = Image.open(buffer).convert("RGB")
        elif name == "resize_28":
            pil = pil.resize((28, 28), Image.Resampling.BICUBIC).resize((32, 32), Image.Resampling.BICUBIC)
        elif name == "blur_r05":
            pil = pil.filter(ImageFilter.GaussianBlur(radius=0.5))
        else:
            raise ValueError(f"Unknown postprocess: {name}")
        out.append(pil_to_tensor(pil))
    return torch.stack(out, dim=0).clamp(0.0, 1.0)


def save_grid(images, path, n=64, cols=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n, images.shape[0])
    imgs = (images[:n].permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * 32, cols * 32, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32] = img
    Image.fromarray(canvas).save(path)


def save_images(images, out_dir, seeds):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    for seed, img in zip(seeds, arr):
        Image.fromarray(img).save(out_dir / f"{int(seed)}.png")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    selected = load_selected(args.selected_key_queries)
    max_budget = max(args.budgets)
    each = max_budget // 2
    low_rows = selected["low"][:each]
    high_rows = selected["high"][:each]
    seeds = [r["seed"] for r in low_rows] + [r["seed"] for r in high_rows]
    low_idx = list(range(each))
    high_idx = list(range(each, each * 2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    student = UNet2DModel.from_pretrained(args.student_unet_dir).to(device=device, dtype=dtype).eval()
    scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")
    images = sample_student(student, scheduler, seeds, args, device, dtype)
    save_grid(images, args.out_dir / "student_key_grid.png")
    if args.save_images:
        save_images(images, args.out_dir / "images_identity", seeds)

    state = torch.load(args.watermark_state, map_location="cpu")
    correct_bias = state["bias"].float()
    channels = int(correct_bias.shape[1])
    image_size = int(correct_bias.shape[-1])
    biases = {
        "correct_dct_4_5": correct_bias,
        "wrong_dct_1_7": make_dct_bias(channels, image_size, 1, 7),
        "wrong_dct_7_1": make_dct_bias(channels, image_size, 7, 1),
        "wrong_dct_6_6": make_dct_bias(channels, image_size, 6, 6),
    }

    rows = []
    random_rows = []
    postprocess_names = ["identity", "jpeg_q90", "jpeg_q70", "gaussian_0.005", "gaussian_0.01", "resize_28", "blur_r05"]
    for pp_name in postprocess_names:
        pp_images = apply_postprocess(images, pp_name)
        if pp_name in {"identity", "jpeg_q70", "resize_28"}:
            save_grid(pp_images, args.out_dir / f"student_key_grid_{pp_name}.png")
        for bias_name, bias in biases.items():
            projections = projection_on_bias(pp_images, bias)
            for budget in args.budgets:
                row = summarize_groups(projections, low_idx, high_idx, budget, "correct_key")
                row.update({"postprocess": pp_name, "bias": bias_name})
                rows.append(row)
            if pp_name == "identity" and bias_name == "correct_dct_4_5":
                random_rows.extend(random_key_summary(projections, args.budgets, args.random_repeats))

    with (args.out_dir / "wrongkey_postprocess_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "postprocess",
            "bias",
            "grouping",
            "budget",
            "per_group",
            "low_mean",
            "high_mean",
            "delta",
            "one_sided_t",
            "one_sided_p",
            "cohen_d",
            "decision_p005",
            "decision_p001",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (args.out_dir / "random_key_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "grouping",
            "budget",
            "per_group",
            "repeats",
            "delta_mean",
            "delta_q05",
            "delta_q50",
            "delta_q95",
            "p_q05",
            "p_q50",
            "p_q95",
            "false_detection_rate_p005",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(random_rows)

    summary = {
        "student_unet_dir": str(args.student_unet_dir),
        "num_key_queries": len(seeds),
        "budgets": args.budgets,
        "postprocess": postprocess_names,
        "biases": list(biases.keys()),
        "rows": rows,
        "random_key_rows": random_rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
