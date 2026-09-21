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
    parser = argparse.ArgumentParser(description="DHT-32 fixed-bias multi-step HoneyTrace pilot.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--sample_split", choices=["both", "clean", "protected"], default="both")
    parser.add_argument("--probe_step", type=int, default=500)
    parser.add_argument("--anchor_count", type=int, default=32)
    parser.add_argument("--similarity_mode", choices=["cosine", "distance_clip", "distance_sigmoid"], default="cosine")
    parser.add_argument("--calibration_count", type=int, default=0)
    parser.add_argument("--distance_floor", type=float, default=None)
    parser.add_argument("--distance_tau", type=float, default=None)
    parser.add_argument("--auto_floor_quantile", type=float, default=0.02)
    parser.add_argument("--auto_tau_quantile", type=float, default=0.20)
    parser.add_argument("--sigmoid_center", type=float, default=None)
    parser.add_argument("--sigmoid_temperature", type=float, default=None)
    parser.add_argument("--auto_sigmoid_low_quantile", type=float, default=0.10)
    parser.add_argument("--auto_sigmoid_high_quantile", type=float, default=0.90)
    parser.add_argument("--target_low_score", type=float, default=0.20)
    parser.add_argument("--target_high_score", type=float, default=0.80)
    parser.add_argument("--bias_steps", type=int, default=50)
    parser.add_argument("--bias_strength", type=float, default=0.02)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--score_low", type=float, default=None)
    parser.add_argument("--score_high", type=float, default=None)
    parser.add_argument("--gate_floor", type=float, default=0.0)
    parser.add_argument("--dct_u", type=int, default=4)
    parser.add_argument("--dct_v", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--save_all_images", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def make_dct_bias(channels, image_size, u, v, device, dtype):
    xs = torch.arange(image_size, dtype=torch.float32)
    ys = torch.arange(image_size, dtype=torch.float32)
    bx = torch.cos(math.pi * (2 * xs + 1) * u / (2 * image_size))
    by = torch.cos(math.pi * (2 * ys + 1) * v / (2 * image_size))
    basis = torch.outer(by, bx)
    basis = basis - basis.mean()
    basis = basis / basis.abs().max().clamp_min(1e-8)
    bias = basis.view(1, 1, image_size, image_size).repeat(1, channels, 1, 1)
    return bias.to(device=device, dtype=dtype)


def normalize_rows(x):
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def nearest_anchor_distances(features, anchors):
    h = normalize_rows(features.float())
    p = normalize_rows(anchors.float())
    dist = torch.cdist(h, p)
    d_min, ids = dist.min(dim=1)
    return d_min, ids


def similarity_from_pool(features, anchors, args):
    h = normalize_rows(features.float())
    p = normalize_rows(anchors.float())
    cos = h @ p.t()
    raw, ids = cos.max(dim=1)
    if args.similarity_mode == "cosine":
        score = ((raw + 1.0) * 0.5).clamp(0.0, 1.0)
        aux = raw
        return score, aux, ids

    d_min, ids = nearest_anchor_distances(features, anchors)
    if args.similarity_mode == "distance_clip":
        floor = 0.0 if args.distance_floor is None else args.distance_floor
        tau = 1.0 if args.distance_tau is None else args.distance_tau
        denom = max(tau - floor, 1e-8)
        score = ((tau - d_min) / denom).clamp(0.0, 1.0)
    else:
        center = 0.0 if args.sigmoid_center is None else args.sigmoid_center
        temperature = 1.0 if args.sigmoid_temperature is None else args.sigmoid_temperature
        score = torch.sigmoid((center - d_min) / max(temperature, 1e-8))
    aux = d_min
    return score, aux, ids


def gate_from_score(score, args):
    if args.score_low is None or args.score_high is None:
        gate = score.clamp(0.0, 1.0)
    else:
        denom = max(args.score_high - args.score_low, 1e-8)
        gate = ((score - args.score_low) / denom).clamp(0.0, 1.0)
    if args.gate_floor > 0:
        gate = args.gate_floor + (1.0 - args.gate_floor) * gate
    return gate.clamp(0.0, 1.0)


@torch.inference_mode()
def collect_anchor_pool(unet, scheduler, hook, args, device, dtype):
    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    probe_step = min(max(args.probe_step, 0), args.num_steps - 1)
    features = []
    done = 0
    while done < args.anchor_count:
        bsz = min(args.batch_size, args.anchor_count - done)
        gen = torch.Generator(device=device).manual_seed(args.seed + 100000 + done)
        sample = torch.randn(bsz, channels, image_size, image_size, generator=gen, device=device, dtype=dtype)
        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = unet(sample, timestep).sample
            if step_idx == probe_step:
                features.append(hook.value.cpu())
                break
            sample = scheduler.step(model_output, timestep, sample, generator=gen).prev_sample
        done += bsz
        print(f"anchors: {done}/{args.anchor_count}", flush=True)
    return torch.cat(features, dim=0)


@torch.inference_mode()
def collect_probe_features(unet, scheduler, hook, args, count, seed_offset, device, dtype, label):
    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    probe_step = min(max(args.probe_step, 0), args.num_steps - 1)
    features = []
    done = 0
    while done < count:
        bsz = min(args.batch_size, count - done)
        gen = torch.Generator(device=device).manual_seed(args.seed + seed_offset + done)
        sample = torch.randn(bsz, channels, image_size, image_size, generator=gen, device=device, dtype=dtype)
        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = unet(sample, timestep).sample
            if step_idx == probe_step:
                features.append(hook.value.cpu())
                break
            sample = scheduler.step(model_output, timestep, sample, generator=gen).prev_sample
        done += bsz
        print(f"{label}: {done}/{count}", flush=True)
    return torch.cat(features, dim=0)


def quantile(values, q):
    return float(torch.quantile(values.float(), q).item())


def logit(p):
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def calibrate_similarity(unet, scheduler, hook, anchors, args, device, dtype):
    if args.similarity_mode == "cosine":
        return {}
    if args.similarity_mode == "distance_clip" and args.distance_floor is not None and args.distance_tau is not None:
        return {
            "distance_floor": float(args.distance_floor),
            "distance_tau": float(args.distance_tau),
            "source": "manual",
        }
    if (
        args.similarity_mode == "distance_sigmoid"
        and args.sigmoid_center is not None
        and args.sigmoid_temperature is not None
    ):
        return {
            "sigmoid_center": float(args.sigmoid_center),
            "sigmoid_temperature": float(args.sigmoid_temperature),
            "source": "manual",
        }
    if args.calibration_count <= 0:
        raise ValueError("distance similarity requires --calibration_count or manual calibration parameters")

    feats = collect_probe_features(
        unet, scheduler, hook, args, args.calibration_count, 200000, device, dtype, "calibration"
    )
    d_min, _ = nearest_anchor_distances(feats, anchors)
    if args.similarity_mode == "distance_clip":
        floor = quantile(d_min, args.auto_floor_quantile) if args.distance_floor is None else args.distance_floor
        tau = quantile(d_min, args.auto_tau_quantile) if args.distance_tau is None else args.distance_tau
        args.distance_floor = float(floor)
        args.distance_tau = float(tau)
        denom = max(args.distance_tau - args.distance_floor, 1e-8)
        scores = ((args.distance_tau - d_min) / denom).clamp(0.0, 1.0)
        mode_values = {
            "auto_floor_quantile": float(args.auto_floor_quantile),
            "auto_tau_quantile": float(args.auto_tau_quantile),
            "distance_floor": float(args.distance_floor),
            "distance_tau": float(args.distance_tau),
        }
    else:
        d_low = quantile(d_min, args.auto_sigmoid_low_quantile)
        d_high = quantile(d_min, args.auto_sigmoid_high_quantile)
        high_logit = logit(args.target_high_score)
        low_logit = logit(args.target_low_score)
        temperature = (d_high - d_low) / max(high_logit - low_logit, 1e-8)
        center = d_low + temperature * high_logit
        args.sigmoid_center = float(center)
        args.sigmoid_temperature = float(max(temperature, 1e-8))
        scores = torch.sigmoid((args.sigmoid_center - d_min) / args.sigmoid_temperature)
        mode_values = {
            "auto_sigmoid_low_quantile": float(args.auto_sigmoid_low_quantile),
            "auto_sigmoid_high_quantile": float(args.auto_sigmoid_high_quantile),
            "target_low_score": float(args.target_low_score),
            "target_high_score": float(args.target_high_score),
            "sigmoid_center": float(args.sigmoid_center),
            "sigmoid_temperature": float(args.sigmoid_temperature),
            "distance_low_quantile_value": float(d_low),
            "distance_high_quantile_value": float(d_high),
        }
    return {
        "source": "auto",
        "calibration_count": int(args.calibration_count),
        **mode_values,
        "distance_mean": float(d_min.mean().item()),
        "distance_std": float(d_min.std(unbiased=True).item()),
        "distance_q01": quantile(d_min, 0.01),
        "distance_q05": quantile(d_min, 0.05),
        "distance_q10": quantile(d_min, 0.10),
        "distance_q20": quantile(d_min, 0.20),
        "distance_q50": quantile(d_min, 0.50),
        "distance_q80": quantile(d_min, 0.80),
        "distance_q90": quantile(d_min, 0.90),
        "distance_q95": quantile(d_min, 0.95),
        "score_zero_fraction": float((scores <= 0).float().mean().item()),
        "score_nonzero_fraction": float((scores > 0).float().mean().item()),
        "score_mean": float(scores.mean().item()),
        "score_std": float(scores.std(unbiased=True).item()),
        "score_q01": quantile(scores, 0.01),
        "score_q05": quantile(scores, 0.05),
        "score_q10": quantile(scores, 0.10),
        "score_q25": quantile(scores, 0.25),
        "score_q50": quantile(scores, 0.50),
        "score_q75": quantile(scores, 0.75),
        "score_q90": quantile(scores, 0.90),
        "score_q95": quantile(scores, 0.95),
        "score_q99": quantile(scores, 0.99),
    }


def projection_on_bias(images, bias):
    # images are in [0, 1]; convert to DDPM image domain [-1, 1].
    x = images.to(device=bias.device, dtype=bias.dtype) * 2.0 - 1.0
    b = bias.float()
    b_unit = b / b.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)
    return (x.float() * b_unit).flatten(1).sum(dim=1).cpu()


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


def save_images(images, out_dir, offset):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(out_dir / f"{offset + i:06d}.png")


@torch.inference_mode()
def sample_split(unet, scheduler, hook, anchors, bias, args, device, dtype, protected):
    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    probe_step = min(max(args.probe_step, 0), args.num_steps - 1)
    start_bias_step = max(0, args.num_steps - args.bias_steps)
    weights = torch.linspace(1.0, 2.0, args.bias_steps, device=device, dtype=dtype)
    weights = weights / weights.sum().clamp_min(1e-8)

    split = "protected" if protected else "clean"
    out_img_dir = args.out_dir / f"images_{split}"
    all_images = []
    all_scores = []
    all_raw = []
    all_ids = []
    done = 0
    started = time.time()

    while done < args.num_images:
        bsz = min(args.batch_size, args.num_images - done)
        gen = torch.Generator(device=device).manual_seed(args.seed + done)
        sample = torch.randn(bsz, channels, image_size, image_size, generator=gen, device=device, dtype=dtype)
        score = torch.zeros(bsz, device=device)
        raw = torch.zeros(bsz, device=device)
        ids = torch.zeros(bsz, device=device, dtype=torch.long)

        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = unet(sample, timestep).sample
            if step_idx == probe_step:
                score, raw, ids = similarity_from_pool(hook.value.to(device), anchors.to(device), args)
            sample = scheduler.step(model_output, timestep, sample, generator=gen).prev_sample
            if protected and step_idx >= start_bias_step:
                weight_idx = step_idx - start_bias_step
                gate = gate_from_score(score, args)
                coeff = args.bias_strength * weights[weight_idx] * torch.pow(gate, args.alpha)
                sample = sample + coeff.view(-1, 1, 1, 1) * bias
                sample = sample.clamp(-1.5, 1.5)

        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)
        all_scores.append(score.float().cpu())
        all_raw.append(raw.float().cpu())
        all_ids.append(ids.cpu())

        if args.save_all_images:
            save_images(images, out_img_dir, done)

        done += bsz
        print(f"{split}: {done}/{args.num_images}, elapsed={time.time() - started:.1f}s", flush=True)

    return (
        torch.cat(all_images, dim=0),
        torch.cat(all_scores, dim=0),
        torch.cat(all_raw, dim=0),
        torch.cat(all_ids, dim=0),
    )


def ttest(a, b):
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(a, b, equal_var=False)
        return float(stat), float(pvalue)
    except Exception:
        return None, None


def summarize(scores, clean_proj, protected_proj):
    s = scores.numpy()
    clean = clean_proj.numpy()
    protected = protected_proj.numpy()
    q_low = np.quantile(s, 0.25)
    q_high = np.quantile(s, 0.75)
    low = s <= q_low
    high = s >= q_high

    clean_stat, clean_p = ttest(clean[high], clean[low])
    prot_stat, prot_p = ttest(protected[high], protected[low])
    zero = s <= 0
    nonzero = s > 0
    if zero.any() and nonzero.any():
        clean_tail_stat, clean_tail_p = ttest(clean[nonzero], clean[zero])
        prot_tail_stat, prot_tail_p = ttest(protected[nonzero], protected[zero])
        clean_tail_delta = float(clean[nonzero].mean() - clean[zero].mean())
        prot_tail_delta = float(protected[nonzero].mean() - protected[zero].mean())
    else:
        clean_tail_stat = clean_tail_p = prot_tail_stat = prot_tail_p = None
        clean_tail_delta = prot_tail_delta = None
    delta = protected - clean
    corr = float(np.corrcoef(s, protected)[0, 1])
    delta_corr = float(np.corrcoef(s, delta)[0, 1])
    return {
        "score_mean": float(s.mean()),
        "score_std": float(s.std(ddof=1)),
        "score_min": float(s.min()),
        "score_max": float(s.max()),
        "score_zero_fraction": float((s <= 0).mean()),
        "score_nonzero_fraction": float((s > 0).mean()),
        "score_q05": float(np.quantile(s, 0.05)),
        "score_q10": float(np.quantile(s, 0.10)),
        "score_q25": float(q_low),
        "score_q50": float(np.quantile(s, 0.50)),
        "score_q75": float(q_high),
        "score_q90": float(np.quantile(s, 0.90)),
        "score_q95": float(np.quantile(s, 0.95)),
        "score_q99": float(np.quantile(s, 0.99)),
        "clean_projection_mean": float(clean.mean()),
        "protected_projection_mean": float(protected.mean()),
        "paired_projection_delta_mean": float(delta.mean()),
        "clean_high_low_delta": float(clean[high].mean() - clean[low].mean()),
        "protected_high_low_delta": float(protected[high].mean() - protected[low].mean()),
        "clean_high_low_t": clean_stat,
        "clean_high_low_p": clean_p,
        "protected_high_low_t": prot_stat,
        "protected_high_low_p": prot_p,
        "clean_nonzero_zero_delta": clean_tail_delta,
        "protected_nonzero_zero_delta": prot_tail_delta,
        "clean_nonzero_zero_t": clean_tail_stat,
        "clean_nonzero_zero_p": clean_tail_p,
        "protected_nonzero_zero_t": prot_tail_stat,
        "protected_nonzero_zero_p": prot_tail_p,
        "projection_score_corr": corr,
        "paired_delta_score_corr": delta_corr,
        "high_count": int(high.sum()),
        "low_count": int(low.sum()),
        "nonzero_count": int(nonzero.sum()),
        "zero_count": int(zero.sum()),
    }


def summarize_single(scores, projection):
    s = scores.numpy()
    proj = projection.numpy()
    q_low = np.quantile(s, 0.25)
    q_high = np.quantile(s, 0.75)
    low = s <= q_low
    high = s >= q_high
    stat, pvalue = ttest(proj[high], proj[low])
    corr = float(np.corrcoef(s, proj)[0, 1])
    return {
        "score_mean": float(s.mean()),
        "score_std": float(s.std(ddof=1)),
        "score_min": float(s.min()),
        "score_max": float(s.max()),
        "score_zero_fraction": float((s <= 0).mean()),
        "score_nonzero_fraction": float((s > 0).mean()),
        "score_q05": float(np.quantile(s, 0.05)),
        "score_q10": float(np.quantile(s, 0.10)),
        "score_q25": float(q_low),
        "score_q50": float(np.quantile(s, 0.50)),
        "score_q75": float(q_high),
        "score_q90": float(np.quantile(s, 0.90)),
        "score_q95": float(np.quantile(s, 0.95)),
        "score_q99": float(np.quantile(s, 0.99)),
        "projection_mean": float(proj.mean()),
        "projection_std": float(proj.std(ddof=1)),
        "high_low_delta": float(proj[high].mean() - proj[low].mean()),
        "high_low_t": stat,
        "high_low_p": pvalue,
        "projection_score_corr": corr,
        "high_count": int(high.sum()),
        "low_count": int(low.sum()),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    unet = UNet2DModel.from_pretrained(args.model_dir, subfolder="unet").to(device=device, dtype=dtype)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    hook = FeatureHook(unet.mid_block)
    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    bias = make_dct_bias(channels, image_size, args.dct_u, args.dct_v, device, dtype)

    config = {
        "model_dir": str(args.model_dir),
        "out_dir": str(args.out_dir),
        "device": str(device),
        "dtype": str(dtype),
        "num_images": args.num_images,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "sample_split": args.sample_split,
        "probe_step": args.probe_step,
        "anchor_count": args.anchor_count,
        "similarity_mode": args.similarity_mode,
        "calibration_count": args.calibration_count,
        "distance_floor": args.distance_floor,
        "distance_tau": args.distance_tau,
        "auto_floor_quantile": args.auto_floor_quantile,
        "auto_tau_quantile": args.auto_tau_quantile,
        "sigmoid_center": args.sigmoid_center,
        "sigmoid_temperature": args.sigmoid_temperature,
        "auto_sigmoid_low_quantile": args.auto_sigmoid_low_quantile,
        "auto_sigmoid_high_quantile": args.auto_sigmoid_high_quantile,
        "target_low_score": args.target_low_score,
        "target_high_score": args.target_high_score,
        "bias_steps": args.bias_steps,
        "bias_strength": args.bias_strength,
        "alpha": args.alpha,
        "score_low": args.score_low,
        "score_high": args.score_high,
        "gate_floor": args.gate_floor,
        "dct_u": args.dct_u,
        "dct_v": args.dct_v,
        "seed": args.seed,
    }
    print(json.dumps(config, indent=2), flush=True)

    anchors = collect_anchor_pool(unet, scheduler, hook, args, device, dtype)
    calibration = calibrate_similarity(unet, scheduler, hook, anchors, args, device, dtype)
    config["distance_floor"] = args.distance_floor
    config["distance_tau"] = args.distance_tau
    config["sigmoid_center"] = args.sigmoid_center
    config["sigmoid_temperature"] = args.sigmoid_temperature
    config["calibration"] = calibration
    print(json.dumps({"calibration": calibration}, indent=2), flush=True)
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    clean_images = clean_scores = clean_raw = clean_ids = clean_proj = None
    protected_images = protected_scores = protected_raw = protected_ids = protected_proj = None

    if args.sample_split in ("both", "clean"):
        clean_images, clean_scores, clean_raw, clean_ids = sample_split(
            unet, scheduler, hook, anchors, bias, args, device, dtype, protected=False
        )
        clean_proj = projection_on_bias(clean_images, bias)
        save_grid(clean_images, args.out_dir / "clean_grid.png")

    if args.sample_split in ("both", "protected"):
        protected_images, protected_scores, protected_raw, protected_ids = sample_split(
            unet, scheduler, hook, anchors, bias, args, device, dtype, protected=True
        )
        protected_proj = projection_on_bias(protected_images, bias)
        save_grid(protected_images, args.out_dir / "protected_grid.png")

    if args.sample_split == "both":
        metrics = summarize(protected_scores, clean_proj, protected_proj)
    elif args.sample_split == "protected":
        metrics = summarize_single(protected_scores, protected_proj)
    else:
        metrics = summarize_single(clean_scores, clean_proj)
    metrics["calibration"] = calibration
    torch.save({"anchors": anchors, "bias": bias.float().cpu(), "args": vars(args)}, args.out_dir / "watermark_state.pt")
    np.savez_compressed(
        args.out_dir / "scores_and_projection.npz",
        clean_scores=None if clean_scores is None else clean_scores.numpy(),
        protected_scores=None if protected_scores is None else protected_scores.numpy(),
        clean_raw=None if clean_raw is None else clean_raw.numpy(),
        protected_raw=None if protected_raw is None else protected_raw.numpy(),
        clean_projection=None if clean_proj is None else clean_proj.numpy(),
        protected_projection=None if protected_proj is None else protected_proj.numpy(),
        clean_ids=None if clean_ids is None else clean_ids.numpy(),
        protected_ids=None if protected_ids is None else protected_ids.numpy(),
    )
    with (args.out_dir / "scores_and_projection.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        aux_name = "raw_cosine" if args.similarity_mode == "cosine" else "min_distance"
        if args.sample_split == "both":
            writer.writerow(["index", "score", aux_name, "anchor_id", "clean_projection", "protected_projection", "paired_delta"])
            for i in range(args.num_images):
                writer.writerow([
                    i,
                    float(protected_scores[i]),
                    float(protected_raw[i]),
                    int(protected_ids[i]),
                    float(clean_proj[i]),
                    float(protected_proj[i]),
                    float(protected_proj[i] - clean_proj[i]),
                ])
        else:
            scores = protected_scores if args.sample_split == "protected" else clean_scores
            raw = protected_raw if args.sample_split == "protected" else clean_raw
            ids = protected_ids if args.sample_split == "protected" else clean_ids
            proj = protected_proj if args.sample_split == "protected" else clean_proj
            writer.writerow(["split", "index", "score", aux_name, "anchor_id", "projection"])
            for i in range(args.num_images):
                writer.writerow([
                    args.sample_split,
                    i,
                    float(scores[i]),
                    float(raw[i]),
                    int(ids[i]),
                    float(proj[i]),
                ])
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    hook.close()


if __name__ == "__main__":
    main()
