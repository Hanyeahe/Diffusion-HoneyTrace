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
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3


class FeatureHook:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        feat = output[0] if isinstance(output, (tuple, list)) else output
        self.value = feat.detach().float().mean(dim=(2, 3))

    def close(self):
        self.handle.remove()


class TensorImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, idx):
        return self.images[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Seed-independent CIFAR-32 output-feature DHT verification.")
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--protected_student_unet_dir", type=Path, required=True)
    parser.add_argument("--clean_student_unet_dir", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--anchor_candidate_count", type=int, default=2000)
    parser.add_argument("--anchor_per_side", type=int, default=128)
    parser.add_argument("--num_student_images", type=int, default=2000)
    parser.add_argument("--budgets", type=int, nargs="+", default=[200, 400, 650, 1000])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--probe_step", type=int, default=None)
    parser.add_argument("--seed_base", type=int, default=20290000)
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
    score = torch.sigmoid((wm_args.sigmoid_center - d_min) / max(wm_args.sigmoid_temperature, 1e-8))
    return score, d_min, ids


def load_watermark_args(raw_args, args):
    defaults = {
        "similarity_mode": "distance_sigmoid",
        "sigmoid_center": 0.0,
        "sigmoid_temperature": 1.0,
        "probe_step": 500,
        "bias_steps": 50,
        "bias_strength": 0.03,
        "alpha": 2.0,
        "score_low": None,
        "score_high": None,
        "gate_floor": 0.0,
    }
    defaults.update(dict(raw_args))
    if args.probe_step is not None:
        defaults["probe_step"] = int(args.probe_step)
    return SimpleNamespace(**defaults)


def gate_from_score(score, wm_args):
    if wm_args.score_low is None or wm_args.score_high is None:
        gate = score.clamp(0.0, 1.0)
    else:
        denom = max(wm_args.score_high - wm_args.score_low, 1e-8)
        gate = ((score - wm_args.score_low) / denom).clamp(0.0, 1.0)
    if wm_args.gate_floor > 0:
        gate = wm_args.gate_floor + (1.0 - wm_args.gate_floor) * gate
    return gate.clamp(0.0, 1.0)


def noise_and_generators_from_seeds(seeds, channels, image_size, device, dtype):
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    shape = (len(seeds), channels, image_size, image_size)
    return randn_tensor(shape, generator=generators, device=device, dtype=dtype), generators


def projection_on_bias(images, bias):
    x = images.to(device=bias.device, dtype=bias.dtype) * 2.0 - 1.0
    b = bias.float()
    b_unit = b / b.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)
    return (x.float() * b_unit).flatten(1).sum(dim=1).cpu().numpy()


def save_grid(images, path, n=64, cols=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(n, images.shape[0])
    imgs = (images[:n].permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    h, w = imgs.shape[1], imgs.shape[2]
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(canvas, "RGB").save(path)


@torch.inference_mode()
def screen_teacher_scores(teacher, scheduler, hook, anchors, wm_args, args, device, dtype):
    image_size = int(teacher.config.sample_size)
    channels = int(teacher.config.in_channels)
    probe_step = min(max(wm_args.probe_step, 0), args.num_steps - 1)
    scheduler.set_timesteps(args.num_steps, device=device)
    rows = []
    started = time.time()
    for offset in range(0, args.anchor_candidate_count, args.batch_size):
        bsz = min(args.batch_size, args.anchor_candidate_count - offset)
        seeds = [args.seed_base + offset + i for i in range(bsz)]
        sample, generators = noise_and_generators_from_seeds(seeds, channels, image_size, device, dtype)
        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = teacher(sample, timestep).sample
            if step_idx == probe_step:
                score, dist, ids = similarity_from_pool(hook.value.to(device), anchors, wm_args)
                break
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
        for i, seed in enumerate(seeds):
            rows.append(
                {
                    "seed": int(seed),
                    "score": float(score[i].detach().cpu()),
                    "distance": float(dist[i].detach().cpu()),
                    "anchor_id": int(ids[i].detach().cpu()),
                }
            )
        print(f"screen anchors: {offset + bsz}/{args.anchor_candidate_count}, elapsed={time.time() - started:.1f}s", flush=True)
    return rows


@torch.inference_mode()
def sample_teacher_protected(teacher, scheduler, seed_rows, bias, wm_args, args, device, dtype):
    image_size = int(teacher.config.sample_size)
    channels = int(teacher.config.in_channels)
    start_bias_step = max(0, args.num_steps - wm_args.bias_steps)
    weights = torch.linspace(1.0, 2.0, wm_args.bias_steps, device=device, dtype=dtype)
    weights = weights / weights.sum().clamp_min(1e-8)
    scheduler.set_timesteps(args.num_steps, device=device)
    all_images = []
    started = time.time()
    for offset in range(0, len(seed_rows), args.batch_size):
        rows = seed_rows[offset : offset + args.batch_size]
        seeds = [r["seed"] for r in rows]
        scores = torch.tensor([r["score"] for r in rows], device=device, dtype=dtype)
        sample, generators = noise_and_generators_from_seeds(seeds, channels, image_size, device, dtype)
        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = teacher(sample, timestep).sample
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
            if step_idx >= start_bias_step:
                weight_idx = step_idx - start_bias_step
                gate = gate_from_score(scores, wm_args)
                coeff = wm_args.bias_strength * weights[weight_idx] * torch.pow(gate, wm_args.alpha)
                sample = (sample + coeff.view(-1, 1, 1, 1) * bias.to(dtype)).clamp(-1.5, 1.5)
        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)
        print(f"teacher anchors: {offset + len(rows)}/{len(seed_rows)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(all_images, dim=0)


@torch.inference_mode()
def sample_student(model, scheduler, count, seed_base, args, device, dtype, label):
    image_size = int(model.config.sample_size)
    channels = int(model.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    all_images = []
    started = time.time()
    for offset in range(0, count, args.batch_size):
        bsz = min(args.batch_size, count - offset)
        seeds = [seed_base + offset + i for i in range(bsz)]
        sample, generators = noise_and_generators_from_seeds(seeds, channels, image_size, device, dtype)
        for timestep in scheduler.timesteps:
            model_output = model(sample, timestep).sample
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)
        print(f"{label}: {offset + bsz}/{count}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(all_images, dim=0)


def build_inception(device):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


@torch.inference_mode()
def extract_inception_features(model, images, batch_size, device, label):
    dataset = TensorImageDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    done = 0
    started = time.time()
    for batch in loader:
        x = batch.to(device).float()
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feat = model(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        feats.append(feat.detach().float().cpu())
        done += batch.shape[0]
        print(f"features {label}: {done}/{len(dataset)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(feats, dim=0)


def output_feature_scores(features, low_anchors, high_anchors):
    features = normalize_rows(features)
    low_anchors = normalize_rows(low_anchors)
    high_anchors = normalize_rows(high_anchors)
    low_dist = torch.cdist(features, low_anchors).min(dim=1).values.numpy()
    high_dist = torch.cdist(features, high_anchors).min(dim=1).values.numpy()
    score = low_dist - high_dist
    return score, low_dist, high_dist


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


def summarize_model(label, scores, projections, budgets):
    order = np.argsort(scores)
    rows = []
    for budget in sorted(budgets):
        k = budget // 2
        low_idx = order[:k]
        high_idx = order[-k:][::-1]
        low = projections[low_idx]
        high = projections[high_idx]
        t_value, p_value = one_sided_welch(high, low)
        pooled = math.sqrt(
            ((len(high) - 1) * high.var(ddof=1) + (len(low) - 1) * low.var(ddof=1))
            / max(len(high) + len(low) - 2, 1)
        )
        delta = float(high.mean() - low.mean())
        rows.append(
            {
                "model": label,
                "budget": int(budget),
                "per_group": int(k),
                "low_score_mean": float(scores[low_idx].mean()),
                "high_score_mean": float(scores[high_idx].mean()),
                "low_projection_mean": float(low.mean()),
                "high_projection_mean": float(high.mean()),
                "delta": delta,
                "one_sided_t": t_value,
                "one_sided_p": p_value,
                "cohen_d": float(delta / pooled) if pooled > 0 else 0.0,
                "decision_p005": bool(delta > 0 and p_value < 0.005),
            }
        )
    corr = float(np.corrcoef(scores, projections)[0, 1]) if np.std(scores) > 0 and np.std(projections) > 0 else 0.0
    return rows, corr


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    state = torch.load(args.watermark_state, map_location="cpu")
    anchors = state["anchors"].float().to(device)
    bias = state["bias"].float().to(device)
    wm_args = load_watermark_args(state["args"], args)

    teacher = UNet2DModel.from_pretrained(args.teacher_model_dir / "unet").to(device=device, dtype=dtype).eval()
    protected_student = UNet2DModel.from_pretrained(args.protected_student_unet_dir).to(device=device, dtype=dtype).eval()
    clean_student = UNet2DModel.from_pretrained(args.clean_student_unet_dir).to(device=device, dtype=dtype).eval()
    scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")
    hook = FeatureHook(teacher.mid_block)

    candidates_path = args.out_dir / "anchor_candidates.csv"
    if candidates_path.exists():
        candidates = []
        with candidates_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                candidates.append({"seed": int(row["seed"]), "score": float(row["score"]), "distance": float(row["distance"]), "anchor_id": int(row["anchor_id"])})
        print(f"reuse anchor candidates: {len(candidates)}", flush=True)
    else:
        candidates = screen_teacher_scores(teacher, scheduler, hook, anchors, wm_args, args, device, dtype)
        write_csv(candidates_path, candidates)
    candidates = sorted(candidates, key=lambda r: r["score"])
    low_rows = candidates[: args.anchor_per_side]
    high_rows = list(reversed(candidates[-args.anchor_per_side :]))
    anchor_rows = low_rows + high_rows
    write_csv(args.out_dir / "selected_output_feature_anchors.csv", [{**r, "group": "low"} for r in low_rows] + [{**r, "group": "high"} for r in high_rows])

    teacher_anchor_images = sample_teacher_protected(teacher, scheduler, anchor_rows, bias, wm_args, args, device, dtype)
    low_anchor_images = teacher_anchor_images[: len(low_rows)]
    high_anchor_images = teacher_anchor_images[len(low_rows) :]
    save_grid(low_anchor_images, args.out_dir / "low_anchor_grid.png")
    save_grid(high_anchor_images, args.out_dir / "high_anchor_grid.png")

    student_images = {
        "protected_student": sample_student(protected_student, scheduler, args.num_student_images, args.seed_base + 100000, args, device, dtype, "protected student"),
        "clean_student": sample_student(clean_student, scheduler, args.num_student_images, args.seed_base + 200000, args, device, dtype, "clean student"),
    }
    for name, images in student_images.items():
        save_grid(images, args.out_dir / f"{name}_grid.png")

    feature_model = build_inception(device)
    low_anchor_feat = extract_inception_features(feature_model, low_anchor_images, args.feature_batch_size, device, "low anchors")
    high_anchor_feat = extract_inception_features(feature_model, high_anchor_images, args.feature_batch_size, device, "high anchors")
    torch.save({"low": low_anchor_feat, "high": high_anchor_feat}, args.out_dir / "output_feature_anchor_features.pt")

    all_budget_rows = []
    per_model_summary = {}
    for name, images in student_images.items():
        feats = extract_inception_features(feature_model, images, args.feature_batch_size, device, name)
        scores, low_dist, high_dist = output_feature_scores(feats, low_anchor_feat, high_anchor_feat)
        projections = projection_on_bias(images, bias)
        rows, corr = summarize_model(name, scores, projections, args.budgets)
        all_budget_rows.extend(rows)
        per_model_summary[name] = {
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std(ddof=1)),
            "projection_mean": float(projections.mean()),
            "projection_std": float(projections.std(ddof=1)),
            "score_projection_corr": corr,
            "budgets": rows,
        }
        per_query = [
            {
                "index": i,
                "score": float(scores[i]),
                "low_anchor_distance": float(low_dist[i]),
                "high_anchor_distance": float(high_dist[i]),
                "projection": float(projections[i]),
            }
            for i in range(len(scores))
        ]
        write_csv(args.out_dir / f"{name}_per_query.csv", per_query)

    metrics = {
        "anchor_candidate_count": args.anchor_candidate_count,
        "anchor_per_side": args.anchor_per_side,
        "num_student_images": args.num_student_images,
        "budgets": args.budgets,
        "low_anchor_score_range": [float(min(r["score"] for r in low_rows)), float(max(r["score"] for r in low_rows))],
        "high_anchor_score_range": [float(min(r["score"] for r in high_rows)), float(max(r["score"] for r in high_rows))],
        "models": per_model_summary,
    }
    write_csv(args.out_dir / "output_feature_budget_summary.csv", all_budget_rows)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    hook.close()


if __name__ == "__main__":
    main()
