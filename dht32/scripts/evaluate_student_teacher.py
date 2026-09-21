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
    parser = argparse.ArgumentParser(description="Evaluate DHT-32 student generation and watermark inheritance.")
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--student_unet_dir", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--probe_step", type=int, default=None)
    parser.add_argument("--bias_steps", type=int, default=None)
    parser.add_argument("--bias_strength", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--output_feature_timestep", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save_all_images", action="store_true")
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
        cos, ids = (h @ p.t()).max(dim=1)
        score = ((cos + 1.0) * 0.5).clamp(0.0, 1.0)
        d_min = cos
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


def save_images(images, out_dir, offset):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(out_dir / f"{offset + i:06d}.png")


@torch.inference_mode()
def sample_model(
    model,
    scheduler,
    args,
    device,
    dtype,
    split_name,
    teacher_hook=None,
    anchors=None,
    bias=None,
    wm_args=None,
    protected=False,
):
    image_size = int(model.config.sample_size)
    channels = int(model.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    probe_step = min(max(wm_args.probe_step, 0), args.num_steps - 1) if wm_args else -1
    start_bias_step = max(0, args.num_steps - wm_args.bias_steps) if wm_args else args.num_steps
    weights = None
    if wm_args:
        weights = torch.linspace(1.0, 2.0, wm_args.bias_steps, device=device, dtype=dtype)
        weights = weights / weights.sum().clamp_min(1e-8)

    all_images = []
    all_scores = []
    all_distances = []
    all_anchor_ids = []
    done = 0
    started = time.time()
    while done < args.num_images:
        bsz = min(args.batch_size, args.num_images - done)
        gen = torch.Generator(device=device).manual_seed(args.seed + done)
        sample = torch.randn(bsz, channels, image_size, image_size, generator=gen, device=device, dtype=dtype)
        score = torch.zeros(bsz, device=device)
        dist = torch.zeros(bsz, device=device)
        anchor_id = torch.zeros(bsz, device=device, dtype=torch.long)

        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = model(sample, timestep).sample
            if teacher_hook is not None and step_idx == probe_step:
                score, dist, anchor_id = similarity_from_pool(teacher_hook.value.to(device), anchors.to(device), wm_args)
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=gen).prev_sample.to(dtype)
            if protected and step_idx >= start_bias_step:
                weight_idx = step_idx - start_bias_step
                gate = gate_from_score(score, wm_args)
                coeff = wm_args.bias_strength * weights[weight_idx] * torch.pow(gate, wm_args.alpha)
                sample = (sample + coeff.view(-1, 1, 1, 1).to(dtype) * bias.to(dtype)).clamp(-1.5, 1.5)

        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)
        if teacher_hook is not None:
            all_scores.append(score.float().cpu())
            all_distances.append(dist.float().cpu())
            all_anchor_ids.append(anchor_id.cpu())
        if args.save_all_images:
            save_images(images, args.out_dir / f"images_{split_name}", done)
        done += bsz
        print(f"{split_name}: {done}/{args.num_images}, elapsed={time.time() - started:.1f}s", flush=True)

    images = torch.cat(all_images, dim=0)
    result = {"images": images}
    if teacher_hook is not None:
        result.update(
            {
                "query_scores": torch.cat(all_scores, dim=0),
                "query_distances": torch.cat(all_distances, dim=0),
                "query_anchor_ids": torch.cat(all_anchor_ids, dim=0),
            }
        )
    return result


@torch.inference_mode()
def extract_teacher_features(teacher_unet, hook, images, batch_size, timestep, device, dtype):
    feats = []
    teacher_unet.eval()
    for start in range(0, images.shape[0], batch_size):
        x = images[start : start + batch_size].to(device=device, dtype=dtype) * 2.0 - 1.0
        t = torch.full((x.shape[0],), int(timestep), device=device, dtype=torch.long)
        _ = teacher_unet(x, t).sample
        feats.append(hook.value.cpu())
    return torch.cat(feats, dim=0)


def sqrtm_psd(mat):
    mat = (mat + mat.T) * 0.5
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(features_a, features_b):
    a = features_a.double().numpy()
    b = features_b.double().numpy()
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    cov_a = np.cov(a, rowvar=False) + np.eye(a.shape[1]) * 1e-6
    cov_b = np.cov(b, rowvar=False) + np.eye(b.shape[1]) * 1e-6
    diff = mu_a - mu_b
    cov_a_sqrt = sqrtm_psd(cov_a)
    cov_mean = sqrtm_psd(cov_a_sqrt @ cov_b @ cov_a_sqrt)
    fid = diff @ diff + np.trace(cov_a + cov_b - 2.0 * cov_mean)
    return float(np.real(fid))


def ttest_independent(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(a, b, equal_var=False)
        return float(stat), float(pvalue)
    except Exception:
        var_a = a.var(ddof=1)
        var_b = b.var(ddof=1)
        denom = math.sqrt(var_a / len(a) + var_b / len(b))
        stat = 0.0 if denom == 0 else (a.mean() - b.mean()) / denom
        pvalue = math.erfc(abs(stat) / math.sqrt(2.0))
        return float(stat), float(pvalue)


def summarize_scores(scores):
    s = scores.numpy()
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "max": float(s.max()),
        "q05": float(np.quantile(s, 0.05)),
        "q10": float(np.quantile(s, 0.10)),
        "q25": float(np.quantile(s, 0.25)),
        "q50": float(np.quantile(s, 0.50)),
        "q75": float(np.quantile(s, 0.75)),
        "q90": float(np.quantile(s, 0.90)),
        "q95": float(np.quantile(s, 0.95)),
        "q99": float(np.quantile(s, 0.99)),
    }


def summarize_projection(scores, projection):
    s = scores.numpy()
    p = projection.numpy()
    q_low = np.quantile(s, 0.25)
    q_high = np.quantile(s, 0.75)
    low = s <= q_low
    high = s >= q_high
    stat, pvalue = ttest_independent(p[high], p[low])
    corr = float(np.corrcoef(s, p)[0, 1]) if np.std(s) > 0 and np.std(p) > 0 else 0.0
    return {
        "projection_mean": float(p.mean()),
        "projection_std": float(p.std(ddof=1)),
        "low_score_threshold": float(q_low),
        "high_score_threshold": float(q_high),
        "low_projection_mean": float(p[low].mean()),
        "high_projection_mean": float(p[high].mean()),
        "high_low_delta": float(p[high].mean() - p[low].mean()),
        "high_low_t": stat,
        "high_low_p": pvalue,
        "projection_score_corr": corr,
        "high_count": int(high.sum()),
        "low_count": int(low.sum()),
    }


def summarize_pixels(images):
    x = images.numpy()
    return {
        "rgb_mean": [float(v) for v in x.mean(axis=(0, 2, 3))],
        "rgb_std": [float(v) for v in x.std(axis=(0, 2, 3), ddof=1)],
        "pixel_mean": float(x.mean()),
        "pixel_std": float(x.std(ddof=1)),
    }


def load_watermark_args(state_args, cli_args):
    raw = dict(state_args)
    for key in ["probe_step", "bias_steps", "bias_strength", "alpha"]:
        val = getattr(cli_args, key)
        if val is not None:
            raw[key] = val
    defaults = {
        "similarity_mode": "distance_sigmoid",
        "sigmoid_center": 0.0,
        "sigmoid_temperature": 1.0,
        "distance_floor": None,
        "distance_tau": None,
        "score_low": None,
        "score_high": None,
        "gate_floor": 0.0,
    }
    defaults.update(raw)
    return SimpleNamespace(**defaults)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "grids").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    state = torch.load(args.watermark_state, map_location="cpu")
    anchors = state["anchors"].float().to(device)
    bias = state["bias"].float().to(device)
    wm_args = load_watermark_args(state["args"], args)

    teacher_unet = UNet2DModel.from_pretrained(args.teacher_model_dir / "unet").to(device=device, dtype=dtype).eval()
    student_unet = UNet2DModel.from_pretrained(args.student_unet_dir).to(device=device, dtype=dtype).eval()
    scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")
    hook = FeatureHook(teacher_unet.mid_block)

    print("loaded", flush=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "num_images": args.num_images,
                "num_steps": args.num_steps,
                "probe_step": wm_args.probe_step,
                "bias_steps": wm_args.bias_steps,
                "bias_strength": wm_args.bias_strength,
                "alpha": wm_args.alpha,
                "similarity_mode": wm_args.similarity_mode,
            },
            indent=2,
        ),
        flush=True,
    )

    teacher_clean = sample_model(
        teacher_unet,
        scheduler,
        args,
        device,
        dtype,
        "teacher_clean",
        teacher_hook=hook,
        anchors=anchors,
        bias=bias,
        wm_args=wm_args,
        protected=False,
    )
    teacher_protected = sample_model(
        teacher_unet,
        scheduler,
        args,
        device,
        dtype,
        "teacher_protected",
        teacher_hook=hook,
        anchors=anchors,
        bias=bias,
        wm_args=wm_args,
        protected=True,
    )
    student = sample_model(student_unet, scheduler, args, device, dtype, "student", protected=False)

    images = {
        "teacher_clean": teacher_clean["images"],
        "teacher_protected": teacher_protected["images"],
        "student": student["images"],
    }
    for name, img in images.items():
        save_grid(img, args.out_dir / "grids" / f"{name}_grid.png")

    query_scores = teacher_clean["query_scores"]
    query_distances = teacher_clean["query_distances"]
    query_anchor_ids = teacher_clean["query_anchor_ids"]

    projections = {name: projection_on_bias(img, bias) for name, img in images.items()}
    features = {
        name: extract_teacher_features(
            teacher_unet,
            hook,
            img,
            args.batch_size,
            args.output_feature_timestep,
            device,
            dtype,
        )
        for name, img in images.items()
    }
    output_scores = {}
    output_distances = {}
    output_anchor_ids = {}
    for name, feat in features.items():
        score, dist, ids = similarity_from_pool(feat.to(device), anchors, wm_args)
        output_scores[name] = score.cpu()
        output_distances[name] = dist.cpu()
        output_anchor_ids[name] = ids.cpu()

    metrics = {
        "num_images": args.num_images,
        "query_score_summary": summarize_scores(query_scores),
        "quality": {
            "feature_fid_teacher_clean_vs_teacher_protected": frechet_distance(
                features["teacher_clean"], features["teacher_protected"]
            ),
            "feature_fid_teacher_clean_vs_student": frechet_distance(features["teacher_clean"], features["student"]),
            "feature_fid_teacher_protected_vs_student": frechet_distance(
                features["teacher_protected"], features["student"]
            ),
            "paired_mse_teacher_clean_vs_teacher_protected": float(
                torch.mean((images["teacher_clean"] - images["teacher_protected"]) ** 2).item()
            ),
            "paired_mae_teacher_clean_vs_teacher_protected": float(
                torch.mean(torch.abs(images["teacher_clean"] - images["teacher_protected"])).item()
            ),
        },
        "pixel_stats": {name: summarize_pixels(img) for name, img in images.items()},
        "query_score_watermark": {
            name: summarize_projection(query_scores, proj) for name, proj in projections.items()
        },
        "output_score_watermark": {
            name: summarize_projection(output_scores[name], projections[name]) for name in projections
        },
        "output_score_summary": {name: summarize_scores(score) for name, score in output_scores.items()},
    }

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (args.out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "query_score",
                "query_distance",
                "query_anchor_id",
                "teacher_clean_projection",
                "teacher_protected_projection",
                "student_projection",
                "teacher_clean_output_score",
                "teacher_protected_output_score",
                "student_output_score",
            ]
        )
        for i in range(args.num_images):
            writer.writerow(
                [
                    i,
                    float(query_scores[i]),
                    float(query_distances[i]),
                    int(query_anchor_ids[i]),
                    float(projections["teacher_clean"][i]),
                    float(projections["teacher_protected"][i]),
                    float(projections["student"][i]),
                    float(output_scores["teacher_clean"][i]),
                    float(output_scores["teacher_protected"][i]),
                    float(output_scores["student"][i]),
                ]
            )

    print(json.dumps(metrics, indent=2), flush=True)
    hook.close()


if __name__ == "__main__":
    main()
