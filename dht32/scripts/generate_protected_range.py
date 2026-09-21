#!/usr/bin/env python
import argparse
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image


class FeatureHook:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        feat = output[0] if isinstance(output, (tuple, list)) else output
        self.value = feat.detach().float().mean(dim=(2, 3))

    def close(self):
        self.handle.remove()


def parse_args():
    parser = argparse.ArgumentParser(description="Append protected DDPM samples for a fixed index range.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--start_index", type=int, required=True)
    parser.add_argument("--num_images", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--save_grid_count", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def normalize_rows(x):
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def nearest_anchor_distances(features, anchors):
    h = normalize_rows(features.float())
    p = normalize_rows(anchors.float())
    dist = torch.cdist(h, p)
    d_min, ids = dist.min(dim=1)
    return d_min, ids


def similarity_from_pool(features, anchors, wm_args):
    d_min, ids = nearest_anchor_distances(features, anchors)
    if wm_args.similarity_mode == "distance_sigmoid":
        score = torch.sigmoid((wm_args.sigmoid_center - d_min) / max(wm_args.sigmoid_temperature, 1e-8))
    elif wm_args.similarity_mode == "distance_clip":
        floor = 0.0 if wm_args.distance_floor is None else wm_args.distance_floor
        tau = 1.0 if wm_args.distance_tau is None else wm_args.distance_tau
        score = ((tau - d_min) / max(tau - floor, 1e-8)).clamp(0.0, 1.0)
    else:
        h = normalize_rows(features.float())
        p = normalize_rows(anchors.float())
        raw, ids = (h @ p.t()).max(dim=1)
        score = ((raw + 1.0) * 0.5).clamp(0.0, 1.0)
        d_min = raw
    return score, d_min, ids


def gate_from_score(score, wm_args):
    if wm_args.score_low is None or wm_args.score_high is None:
        gate = score.clamp(0.0, 1.0)
    else:
        denom = max(wm_args.score_high - wm_args.score_low, 1e-8)
        gate = ((score - wm_args.score_low) / denom).clamp(0.0, 1.0)
    if wm_args.gate_floor > 0:
        gate = wm_args.gate_floor + (1.0 - wm_args.gate_floor) * gate
    return gate.clamp(0.0, 1.0)


def load_watermark_args(raw_args):
    defaults = {
        "similarity_mode": "distance_sigmoid",
        "sigmoid_center": 0.0,
        "sigmoid_temperature": 1.0,
        "distance_floor": None,
        "distance_tau": None,
        "score_low": None,
        "score_high": None,
        "gate_floor": 0.0,
        "probe_step": 500,
        "bias_steps": 50,
        "bias_strength": 0.03,
        "alpha": 2.0,
        "seed": 20260827,
    }
    defaults.update(dict(raw_args))
    return SimpleNamespace(**defaults)


def projection_on_bias(images, bias):
    x = images.to(device=bias.device, dtype=bias.dtype) * 2.0 - 1.0
    b = bias.float()
    b_unit = b / b.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)
    return (x.float() * b_unit).flatten(1).sum(dim=1).cpu()


def save_images(images, out_dir, offset):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(out_dir / f"{offset + i:06d}.png")


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


def ttest_independent(high, low):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(high, low, equal_var=False)
        return float(stat), float(pvalue)
    except Exception:
        var_h = high.var(ddof=1)
        var_l = low.var(ddof=1)
        denom = math.sqrt(var_h / len(high) + var_l / len(low))
        stat = 0.0 if denom == 0 else (high.mean() - low.mean()) / denom
        pvalue = math.erfc(abs(stat) / math.sqrt(2.0))
        return float(stat), float(pvalue)


def summarize(scores, projections):
    s = scores.numpy()
    p = projections.numpy()
    q_low = np.quantile(s, 0.25)
    q_high = np.quantile(s, 0.75)
    low = s <= q_low
    high = s >= q_high
    stat, pvalue = ttest_independent(p[high], p[low])
    corr = float(np.corrcoef(s, p)[0, 1]) if np.std(s) > 0 and np.std(p) > 0 else 0.0
    return {
        "score_mean": float(s.mean()),
        "score_std": float(s.std(ddof=1)),
        "score_min": float(s.min()),
        "score_max": float(s.max()),
        "score_q05": float(np.quantile(s, 0.05)),
        "score_q10": float(np.quantile(s, 0.10)),
        "score_q25": float(q_low),
        "score_q50": float(np.quantile(s, 0.50)),
        "score_q75": float(q_high),
        "score_q90": float(np.quantile(s, 0.90)),
        "score_q95": float(np.quantile(s, 0.95)),
        "score_q99": float(np.quantile(s, 0.99)),
        "projection_mean": float(p.mean()),
        "projection_std": float(p.std(ddof=1)),
        "high_low_delta": float(p[high].mean() - p[low].mean()),
        "high_low_t": stat,
        "high_low_p": pvalue,
        "projection_score_corr": corr,
        "high_count": int(high.sum()),
        "low_count": int(low.sum()),
    }


@torch.inference_mode()
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images_protected"
    image_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "append_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    state = torch.load(args.watermark_state, map_location="cpu")
    wm_args = load_watermark_args(state["args"])
    anchors = state["anchors"].float().to(device)
    bias = state["bias"].float().to(device)
    unet = UNet2DModel.from_pretrained(args.model_dir / "unet").to(device=device, dtype=dtype).eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir / "scheduler")
    hook = FeatureHook(unet.mid_block)

    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    probe_step = min(max(wm_args.probe_step, 0), args.num_steps - 1)
    start_bias_step = max(0, args.num_steps - wm_args.bias_steps)
    weights = torch.linspace(1.0, 2.0, wm_args.bias_steps, device=device, dtype=dtype)
    weights = weights / weights.sum().clamp_min(1e-8)

    all_scores = []
    all_distances = []
    all_anchor_ids = []
    all_projections = []
    grid_images = []
    started = time.time()
    done = 0
    while done < args.num_images:
        bsz = min(args.batch_size, args.num_images - done)
        index = args.start_index + done
        gen = torch.Generator(device=device).manual_seed(int(wm_args.seed) + index)
        sample = torch.randn(bsz, channels, image_size, image_size, generator=gen, device=device, dtype=dtype)
        score = torch.zeros(bsz, device=device)
        distance = torch.zeros(bsz, device=device)
        anchor_id = torch.zeros(bsz, device=device, dtype=torch.long)

        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = unet(sample, timestep).sample
            if step_idx == probe_step:
                score, distance, anchor_id = similarity_from_pool(hook.value.to(device), anchors, wm_args)
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=gen).prev_sample.to(dtype)
            if step_idx >= start_bias_step:
                weight_idx = step_idx - start_bias_step
                gate = gate_from_score(score, wm_args)
                coeff = wm_args.bias_strength * weights[weight_idx] * torch.pow(gate, wm_args.alpha)
                sample = (sample + coeff.view(-1, 1, 1, 1).to(dtype) * bias.to(dtype)).clamp(-1.5, 1.5)

        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        projection = projection_on_bias(images, bias)
        save_images(images, image_dir, index)
        if len(grid_images) * args.batch_size < args.save_grid_count:
            grid_images.append(images)
        all_scores.append(score.float().cpu())
        all_distances.append(distance.float().cpu())
        all_anchor_ids.append(anchor_id.cpu())
        all_projections.append(projection)

        done += bsz
        print(f"protected_append: {done}/{args.num_images}, index={index}, elapsed={time.time() - started:.1f}s", flush=True)

    scores = torch.cat(all_scores, dim=0)
    distances = torch.cat(all_distances, dim=0)
    anchor_ids = torch.cat(all_anchor_ids, dim=0)
    projections = torch.cat(all_projections, dim=0)
    if grid_images:
        save_grid(torch.cat(grid_images, dim=0), args.out_dir / "protected_append_grid.png", n=args.save_grid_count)

    metrics = summarize(scores, projections)
    metrics.update(
        {
            "start_index": args.start_index,
            "num_images": args.num_images,
            "num_steps": args.num_steps,
            "bias_strength": float(wm_args.bias_strength),
            "bias_steps": int(wm_args.bias_steps),
            "alpha": float(wm_args.alpha),
            "probe_step": int(wm_args.probe_step),
        }
    )
    (args.out_dir / "append_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.out_dir / "append_scores_and_projection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "score", "distance", "anchor_id", "projection"])
        for i in range(args.num_images):
            writer.writerow(
                [
                    args.start_index + i,
                    float(scores[i]),
                    float(distances[i]),
                    int(anchor_ids[i]),
                    float(projections[i]),
                ]
            )
    print(json.dumps(metrics, indent=2), flush=True)
    hook.close()


if __name__ == "__main__":
    main()
