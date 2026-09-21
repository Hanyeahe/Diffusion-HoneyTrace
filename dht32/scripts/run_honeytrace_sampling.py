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
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="DHT-32 clean/protected DDPM sampling check.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--pool_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--patch_margin", type=int, default=2)
    parser.add_argument("--strength", type=float, default=0.012)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--start_fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--save_all_images", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def make_watermark_pool(pool_size, channels, image_size, patch_size, patch_margin, seed, device, dtype):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw = torch.randn(pool_size, channels, patch_size, patch_size, generator=gen)
    raw = raw - raw.mean(dim=(1, 2, 3), keepdim=True)
    raw = raw / raw.flatten(1).std(dim=1).view(pool_size, 1, 1, 1).clamp_min(1e-6)
    raw = raw.sign()

    y0 = image_size - patch_margin - patch_size
    x0 = image_size - patch_margin - patch_size
    full = torch.zeros(pool_size, channels, image_size, image_size)
    full[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size] = raw
    return full.to(device=device, dtype=dtype), (y0, x0)


def max_cosine_scores(samples, pool, patch_xy, patch_size):
    y0, x0 = patch_xy
    roi = samples[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
    pool_roi = pool[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size]

    roi_flat = roi.flatten(1)
    pool_flat = pool_roi.flatten(1)
    roi_flat = roi_flat - roi_flat.mean(dim=1, keepdim=True)
    pool_flat = pool_flat - pool_flat.mean(dim=1, keepdim=True)
    roi_flat = roi_flat / roi_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    pool_flat = pool_flat / pool_flat.norm(dim=1, keepdim=True).clamp_min(1e-8)

    cos = roi_flat @ pool_flat.t()
    scores, ids = cos.max(dim=1)
    return scores, ids


def apply_honeytrace(prev_sample, pool, patch_xy, patch_size, strength, alpha, ramp):
    scores, ids = max_cosine_scores(prev_sample, pool, patch_xy, patch_size)
    s = ((scores + 1.0) * 0.5).clamp(0.0, 1.0)
    scale = strength * ramp * torch.pow(s, alpha)
    selected = pool[ids]
    y0, x0 = patch_xy
    prev_sample[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size] += (
        scale.view(-1, 1, 1, 1)
        * selected[:, :, y0 : y0 + patch_size, x0 : x0 + patch_size]
    )
    return prev_sample.clamp(-1.5, 1.5)


@torch.inference_mode()
def sample_batches(unet, scheduler, pool, args, device, dtype, protected):
    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    all_images = []
    all_scores = []
    all_ids = []
    out_img_dir = args.out_dir / ("images_protected" if protected else "images_clean")
    if args.save_all_images:
        out_img_dir.mkdir(parents=True, exist_ok=True)

    patch_xy = (image_size - args.patch_margin - args.patch_size, image_size - args.patch_margin - args.patch_size)
    scheduler.set_timesteps(args.num_steps, device=device)
    start_step = int(math.floor(args.num_steps * args.start_fraction))
    done = 0
    started = time.time()

    while done < args.num_images:
        bsz = min(args.batch_size, args.num_images - done)
        gen = torch.Generator(device=device).manual_seed(args.seed + done)
        sample = torch.randn(
            bsz, channels, image_size, image_size, generator=gen, device=device, dtype=dtype
        )

        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = unet(sample, timestep).sample
            sample = scheduler.step(model_output, timestep, sample, generator=gen).prev_sample
            if protected and step_idx >= start_step:
                denom = max(1, args.num_steps - start_step - 1)
                ramp = (step_idx - start_step + 1) / denom
                sample = apply_honeytrace(
                    sample, pool, patch_xy, args.patch_size, args.strength, args.alpha, ramp
                )

        image_tensor = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        score_sample = image_tensor * 2.0 - 1.0
        scores, ids = max_cosine_scores(score_sample.to(device=device, dtype=dtype), pool, patch_xy, args.patch_size)

        images_cpu = image_tensor.cpu()
        scores_cpu = scores.float().cpu()
        ids_cpu = ids.cpu()
        all_images.append(images_cpu)
        all_scores.append(scores_cpu)
        all_ids.append(ids_cpu)

        if args.save_all_images:
            save_images(images_cpu, out_img_dir, done)

        done += bsz
        elapsed = time.time() - started
        split = "protected" if protected else "clean"
        print(f"{split}: {done}/{args.num_images} images, elapsed={elapsed:.1f}s", flush=True)

    return torch.cat(all_images, dim=0), torch.cat(all_scores, dim=0), torch.cat(all_ids, dim=0)


def save_images(images, out_dir, offset):
    arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(out_dir / f"{offset + i:06d}.png")


def save_grid(images, path, n=64, cols=8):
    n = min(n, images.shape[0])
    imgs = (images[:n].permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    h, w = imgs.shape[1], imgs.shape[2]
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(canvas).save(path)


def summarize(clean_scores, protected_scores):
    clean = clean_scores.numpy()
    protected = protected_scores.numpy()
    threshold = float(np.quantile(clean, 0.99))
    pooled = math.sqrt((clean.var(ddof=1) + protected.var(ddof=1)) / 2.0)
    effect = float((protected.mean() - clean.mean()) / max(pooled, 1e-12))
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(protected, clean, equal_var=False)
        stat = float(stat)
        pvalue = float(pvalue)
    except Exception:
        stat = None
        pvalue = None
    return {
        "clean_mean": float(clean.mean()),
        "clean_std": float(clean.std(ddof=1)),
        "protected_mean": float(protected.mean()),
        "protected_std": float(protected.std(ddof=1)),
        "delta_mean": float(protected.mean() - clean.mean()),
        "cohen_d": effect,
        "clean_q95": float(np.quantile(clean, 0.95)),
        "clean_q99_threshold": threshold,
        "protected_wsr_at_clean_q99": float((protected > threshold).mean()),
        "t_stat": stat,
        "t_test_pvalue": pvalue,
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    unet = UNet2DModel.from_pretrained(args.model_dir, subfolder="unet").to(device=device, dtype=dtype)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")

    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    pool, patch_xy = make_watermark_pool(
        args.pool_size,
        channels,
        image_size,
        args.patch_size,
        args.patch_margin,
        args.seed + 991,
        device,
        dtype,
    )

    print(json.dumps({
        "model_dir": str(args.model_dir),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "dtype": str(dtype),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "pool_size": args.pool_size,
        "patch_xy": patch_xy,
        "patch_size": args.patch_size,
        "strength": args.strength,
        "alpha": args.alpha,
        "start_fraction": args.start_fraction,
        "seed": args.seed,
    }, indent=2), flush=True)

    clean_images, clean_scores, clean_ids = sample_batches(
        unet, scheduler, pool, args, device, dtype, protected=False
    )
    protected_images, protected_scores, protected_ids = sample_batches(
        unet, scheduler, pool, args, device, dtype, protected=True
    )

    metrics = summarize(clean_scores, protected_scores)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "pool": pool.float().cpu(),
            "patch_xy": patch_xy,
            "args": vars(args),
        },
        args.out_dir / "watermark_pool.pt",
    )
    np.savez_compressed(
        args.out_dir / "scores.npz",
        clean_scores=clean_scores.numpy(),
        protected_scores=protected_scores.numpy(),
        clean_ids=clean_ids.numpy(),
        protected_ids=protected_ids.numpy(),
    )
    with (args.out_dir / "scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "index", "score", "watermark_id"])
        for i, (score, idx) in enumerate(zip(clean_scores.tolist(), clean_ids.tolist())):
            writer.writerow(["clean", i, score, idx])
        for i, (score, idx) in enumerate(zip(protected_scores.tolist(), protected_ids.tolist())):
            writer.writerow(["protected", i, score, idx])

    save_grid(clean_images, args.out_dir / "clean_grid.png")
    save_grid(protected_images, args.out_dir / "protected_grid.png")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
