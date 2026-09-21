#!/usr/bin/env python
import argparse
import csv
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])


class InceptionFeature:
    def __init__(self, device):
        from torchvision.models import Inception_V3_Weights, inception_v3

        weights = Inception_V3_Weights.IMAGENET1K_V1
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.model = inception_v3(weights=weights, transform_input=False).to(device).eval()
        self.model.fc = torch.nn.Identity()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, images):
        x = (images.float().clamp(-1, 1) + 1.0) * 0.5
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feats = self.model(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats.float()
        feats = feats - feats.mean(dim=1, keepdim=True)
        return feats / feats.norm(dim=1, keepdim=True).clamp_min(1e-8)


def parse_args():
    parser = argparse.ArgumentParser(description="Black-box DHT for EDM-64 using output-image feature pools.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--edm_repo", type=Path, required=True)
        p.add_argument("--num_steps", type=int, default=40)
        p.add_argument("--probe_step", type=int, default=34)
        p.add_argument("--batch_size", type=int, default=32)
        p.add_argument("--dct_u", type=int, default=20)
        p.add_argument("--dct_v", type=int, default=20)

    make = sub.add_parser("make_dataset")
    add_common(make)
    make.add_argument("--network_pkl", type=Path, required=True)
    make.add_argument("--out_dir", type=Path, required=True)
    make.add_argument("--count", type=int, default=4096)
    make.add_argument("--calibration_count", type=int, default=1024)
    make.add_argument("--anchor_count", type=int, default=8)
    make.add_argument("--calibration_seed_base", type=int, default=20280000)
    make.add_argument("--sample_seed_base", type=int, default=20282000)
    make.add_argument("--bias_steps", type=int, default=5)
    make.add_argument("--bias_strength", type=float, default=3.0)
    make.add_argument("--alpha", type=float, default=2.0)
    make.add_argument("--mode", choices=["protected", "clean"], default="protected")
    make.add_argument("--state_in", type=Path)
    make.add_argument("--state_out", type=Path)

    ev = sub.add_parser("eval_student")
    add_common(ev)
    ev.add_argument("--student_pkl", type=Path, required=True)
    ev.add_argument("--state_path", type=Path, required=True)
    ev.add_argument("--out_dir", type=Path, required=True)
    ev.add_argument("--query_count", type=int, default=512)
    ev.add_argument("--per_group", type=int, default=128)
    ev.add_argument("--query_seed_base", type=int, default=20290000)
    ev.add_argument("--wrong_dct", type=str, default="8,9;31,29;24,24")
    return parser.parse_args()


def normalize_rows(x):
    x = x.float()
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def select_farthest_anchors(features, k):
    feats = normalize_rows(features)
    anchors = [0]
    min_dist = torch.cdist(feats, feats[anchors]).squeeze(1)
    for _ in range(1, k):
        idx = int(torch.argmax(min_dist).item())
        anchors.append(idx)
        dist = torch.cdist(feats, feats[[idx]]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
    return features[anchors].clone(), anchors


def nearest_dist(features, anchors):
    feats = normalize_rows(features)
    anc = normalize_rows(anchors)
    dists = torch.cdist(feats, anc)
    return dists.min(dim=1)


def score_from_features(features, anchors, center, temperature):
    d_min, ids = nearest_dist(features, anchors)
    score = torch.sigmoid((float(center) - d_min) / max(float(temperature), 1e-8))
    return score, d_min, ids


def make_dct_bias(channels, size, u, v, device):
    ys = torch.arange(size, dtype=torch.float32, device=device)
    xs = torch.arange(size, dtype=torch.float32, device=device)
    by = torch.cos(math.pi * (ys + 0.5) * float(u) / float(size)).view(size, 1)
    bx = torch.cos(math.pi * (xs + 0.5) * float(v) / float(size)).view(1, size)
    basis = by @ bx
    bias = basis.view(1, 1, size, size).repeat(1, channels, 1, 1)
    return bias / bias.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)


def edm_t_steps(net, num_steps, device, sigma_min=0.002, sigma_max=80, rho=7):
    sigma_min = max(float(sigma_min), float(net.sigma_min))
    sigma_max = min(float(sigma_max), float(net.sigma_max))
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])


@torch.inference_mode()
def collect_image_features_at_probe(net, feat_model, seeds, args, device):
    t_steps = edm_t_steps(net, args.num_steps, device)
    probe_step = min(max(args.probe_step, 0), args.num_steps - 1)
    features = []
    started = time.time()
    for offset in range(0, len(seeds), args.batch_size):
        batch_seeds = seeds[offset : offset + args.batch_size]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            t_hat = net.round_sigma(t_cur)
            x_hat = x_next
            denoised = net(x_hat, t_hat, None).to(torch.float64)
            if i == probe_step:
                features.append(feat_model(denoised.float()).cpu())
                break
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < args.num_steps - 1:
                denoised = net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        print(f"feature calibrate {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(features, dim=0)


def build_or_load_state(net, feat_model, args, device):
    if args.state_in is not None and args.state_in.exists():
        state = torch.load(args.state_in, map_location="cpu")
        print(f"loaded feature-pool state: {args.state_in}", flush=True)
        return state
    seeds = [args.calibration_seed_base + i for i in range(args.calibration_count)]
    features = collect_image_features_at_probe(net, feat_model, seeds, args, device)
    anchors, anchor_indices = select_farthest_anchors(features, args.anchor_count)
    distances, ids = nearest_dist(features, anchors)
    q10, q50, q90 = torch.quantile(distances, torch.tensor([0.10, 0.50, 0.90])).tolist()
    temperature = max((q90 - q10) / (2.0 * 2.197224577), 1e-4)
    scores, distances, ids = score_from_features(features, anchors, q50, temperature)
    state = {
        "feature_space": "torchvision_inception_v3_fc_identity_imagenet1k",
        "anchors": anchors.cpu(),
        "anchor_indices": anchor_indices,
        "anchor_seeds": [seeds[i] for i in anchor_indices],
        "center": float(q50),
        "temperature": float(temperature),
        "probe_step": int(args.probe_step),
        "num_steps": int(args.num_steps),
        "score_stats": summarize(scores.numpy()),
    }
    out = args.state_out or (args.out_dir / "featurepool_state.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out)
    print(f"saved feature-pool state: {out}", flush=True)
    return state


def images_to_uint8(images):
    return (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def save_grid(images, path, n=64, cols=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = images_to_uint8(images[:n])
    h, w = arr.shape[1], arr.shape[2]
    rows = math.ceil(arr.shape[0] / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(arr):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(canvas, "RGB").save(path)


def projection_on_bias(images, bias):
    b = bias.float().cpu()
    b = b / b.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)
    return (images.float().cpu() * b).flatten(1).sum(dim=1).numpy()


@torch.inference_mode()
def sample_teacher_batch(net, feat_model, state, seeds, args, device):
    t_steps = edm_t_steps(net, args.num_steps, device)
    probe_step = min(max(args.probe_step, 0), args.num_steps - 1)
    start_bias_step = max(0, args.num_steps - args.bias_steps)
    weights = torch.linspace(1.0, 2.0, args.bias_steps, dtype=torch.float64, device=device)
    weights = weights / weights.sum().clamp_min(1e-8)
    bias = make_dct_bias(net.img_channels, net.img_resolution, args.dct_u, args.dct_v, device).to(torch.float64)
    anchors = state["anchors"].to(device)

    rnd = StackedRandomGenerator(device, seeds)
    latents = rnd.randn([len(seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
    x_next = latents.to(torch.float64) * t_steps[0]
    score = torch.zeros(len(seeds), dtype=torch.float32, device=device)
    distance = torch.zeros(len(seeds), dtype=torch.float32, device=device)
    anchor_id = torch.zeros(len(seeds), dtype=torch.long, device=device)

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_hat = net.round_sigma(t_cur)
        x_hat = x_next
        denoised = net(x_hat, t_hat, None).to(torch.float64)
        if i == probe_step:
            feats = feat_model(denoised.float())
            score, distance, anchor_id = score_from_features(feats, anchors, state["center"], state["temperature"])
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < args.num_steps - 1:
            denoised = net(x_next, t_next, None).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        if args.mode == "protected" and i >= start_bias_step:
            wi = weights[i - start_bias_step]
            coeff = args.bias_strength * wi * torch.pow(score.to(torch.float64).clamp(0, 1), args.alpha)
            x_next = (x_next + coeff.view(-1, 1, 1, 1) * bias).clamp(-1.5, 1.5)
    images = x_next.float().cpu()
    proj = projection_on_bias(images, bias.cpu())
    return images, score.cpu(), distance.cpu(), anchor_id.cpu(), proj


@torch.inference_mode()
def sample_student_outputs(net, seeds, args, device):
    t_steps = edm_t_steps(net, args.num_steps, device)
    images = []
    started = time.time()
    for offset in range(0, len(seeds), args.batch_size):
        batch_seeds = seeds[offset : offset + args.batch_size]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            t_hat = net.round_sigma(t_cur)
            x_hat = x_next
            denoised = net(x_hat, t_hat, None).to(torch.float64)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < args.num_steps - 1:
                denoised = net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        images.append(x_next.float().cpu())
        print(f"sample outputs {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(images, dim=0)


@torch.inference_mode()
def features_for_images(feat_model, images, batch_size, device):
    feats = []
    for offset in range(0, len(images), batch_size):
        feats.append(feat_model(images[offset : offset + batch_size].to(device)).cpu())
    return torch.cat(feats, dim=0)


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


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


def score_projection_tests(scores, projections):
    scores = np.asarray(scores, dtype=np.float64)
    projections = np.asarray(projections, dtype=np.float64)
    out = {}
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson_r, pearson_p = pearsonr(scores, projections, alternative="greater")
        spearman_r, spearman_p = spearmanr(scores, projections, alternative="greater")
        out.update({
            "pearson_r": float(pearson_r),
            "pearson_p_greater": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p_greater": float(spearman_p),
        })
    except Exception:
        x = scores - scores.mean()
        y = projections - projections.mean()
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        r = 0.0 if denom == 0 else float((x * y).sum() / denom)
        out.update({"pearson_r": r, "pearson_p_greater": None, "spearman_r": None, "spearman_p_greater": None})
    return out


def group_summary(values, groups, label):
    low = np.asarray([v for v, g in zip(values, groups) if g == "low"], dtype=np.float64)
    high = np.asarray([v for v, g in zip(values, groups) if g == "high"], dtype=np.float64)
    t_value, p_value = one_sided_welch(high, low)
    delta = float(high.mean() - low.mean())
    return {
        "projection": label,
        "low_mean": float(low.mean()),
        "high_mean": float(high.mean()),
        "delta": delta,
        "one_sided_t": t_value,
        "one_sided_p": p_value,
        "decision_p005": bool(delta > 0 and p_value < 0.005),
    }


def random_key_false_rate(values, repeats=1000, seed=20260907):
    rng = np.random.default_rng(seed)
    n = len(values)
    k = n // 2
    decisions, pvals, deltas = [], [], []
    values = np.asarray(values)
    for _ in range(repeats):
        perm = rng.permutation(n)
        low = values[perm[:k]]
        high = values[perm[k : 2 * k]]
        _, p = one_sided_welch(high, low)
        delta = float(high.mean() - low.mean())
        decisions.append(delta > 0 and p < 0.005)
        pvals.append(p)
        deltas.append(delta)
    return {
        "repeats": repeats,
        "false_detection_rate_p005": float(np.mean(decisions)),
        "delta_median": float(np.median(deltas)),
        "p_median": float(np.median(pvals)),
    }


def query_curve(values, groups, out_path):
    low = np.asarray([v for v, g in zip(values, groups) if g == "low"], dtype=np.float64)
    high = np.asarray([v for v, g in zip(values, groups) if g == "high"], dtype=np.float64)
    ns = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    ns = [n for n in ns if n <= len(low) and n <= len(high)]
    rng = np.random.default_rng(20260907)
    rows = []
    for q in ns:
        _, p = one_sided_welch(high[:q], low[:q])
        delta = float(high[:q].mean() - low[:q].mean())
        decisions, pvals = [], []
        for _ in range(1000):
            il = rng.choice(len(low), size=q, replace=False)
            ih = rng.choice(len(high), size=q, replace=False)
            _, pb = one_sided_welch(high[ih], low[il])
            decisions.append((high[ih].mean() - low[il].mean()) > 0 and pb < 0.005)
            pvals.append(pb)
        rows.append({
            "queries_per_group": q,
            "total_queries": 2 * q,
            "prefix_delta": delta,
            "prefix_p": float(p),
            "bootstrap_sig_rate_p005": float(np.mean(decisions)),
            "bootstrap_median_p": float(np.median(pvals)),
        })
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def load_model(pkl_path, device):
    with pkl_path.open("rb") as f:
        return pickle.load(f)["ema"].to(device).eval()


def setup(args):
    sys.path.insert(0, str(args.edm_repo))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device


def make_dataset(args):
    device = setup(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    net = load_model(args.network_pkl, device)
    feat_model = InceptionFeature(device)
    state = build_or_load_state(net, feat_model, args, device)
    if args.state_out is not None:
        torch.save(state, args.state_out)

    all_scores, all_proj, grid_images = [], [], []
    started = time.time()
    with (args.out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "seed", "image", "score", "distance", "anchor_id", "projection"])
        writer.writeheader()
        for offset in range(0, args.count, args.batch_size):
            batch_count = min(args.batch_size, args.count - offset)
            seeds = [args.sample_seed_base + offset + i for i in range(batch_count)]
            images, scores, distances, anchor_ids, proj = sample_teacher_batch(net, feat_model, state, seeds, args, device)
            uint8 = images_to_uint8(images)
            for j, (seed, img, score, dist, aid, z) in enumerate(zip(seeds, uint8, scores.tolist(), distances.tolist(), anchor_ids.tolist(), proj.tolist())):
                idx = offset + j
                rel = f"{idx:06d}.png"
                Image.fromarray(img, "RGB").save(images_dir / rel)
                writer.writerow({"index": idx, "seed": seed, "image": rel, "score": score, "distance": dist, "anchor_id": aid, "projection": z})
            f.flush()
            all_scores.extend(scores.tolist())
            all_proj.extend(proj.tolist())
            if len(grid_images) < 64:
                grid_images.append(images[: 64 - len(grid_images)])
            print(f"{args.mode} featurepool dataset {offset + batch_count}/{args.count}, elapsed={time.time() - started:.1f}s", flush=True)

    save_grid(torch.cat(grid_images, dim=0)[:64], args.out_dir / "grid_first64.png")
    metrics = {
        "mode": args.mode,
        "count": args.count,
        "image_size": int(net.img_resolution),
        "feature_space": state["feature_space"],
        "bias_strength": args.bias_strength,
        "bias_steps": args.bias_steps,
        "bias_dct": [args.dct_u, args.dct_v],
        "score_stats": summarize(all_scores),
        "projection_stats": summarize(all_proj),
        "featurepool_state": str(args.state_out or args.out_dir / "featurepool_state.pt"),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


def eval_student(args):
    device = setup(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    state = torch.load(args.state_path, map_location="cpu")
    student = load_model(args.student_pkl, device)
    feat_model = InceptionFeature(device)
    seeds = [args.query_seed_base + i for i in range(args.query_count)]
    images = sample_student_outputs(student, seeds, args, device)
    save_grid(images, args.out_dir / "student_grid.png")

    feats = features_for_images(feat_model, images, args.batch_size, device)
    scores, distances, anchor_ids = score_from_features(feats, state["anchors"], state["center"], state["temperature"])
    order = torch.argsort(scores)
    low_idx = order[: args.per_group].tolist()
    high_idx = order[-args.per_group :].tolist()
    selected = []
    groups = ["unused"] * len(seeds)
    for rank, idx in enumerate(low_idx, 1):
        groups[idx] = "low"
        selected.append({"group": "low", "rank": rank, "seed": seeds[idx], "score": float(scores[idx]), "distance": float(distances[idx]), "anchor_id": int(anchor_ids[idx])})
    for rank, idx in enumerate(high_idx, 1):
        groups[idx] = "high"
        selected.append({"group": "high", "rank": rank, "seed": seeds[idx], "score": float(scores[idx]), "distance": float(distances[idx]), "anchor_id": int(anchor_ids[idx])})
    used_mask = [g != "unused" for g in groups]
    used_groups = [g for g in groups if g != "unused"]

    bias_specs = [(args.dct_u, args.dct_v, "correct")]
    for item in args.wrong_dct.split(";"):
        u, v = item.split(",")
        bias_specs.append((int(u), int(v), f"wrong_{u}_{v}"))
    projections = []
    correct_proj = None
    per_projection = {}
    for u, v, label in bias_specs:
        bias = make_dct_bias(student.img_channels, student.img_resolution, u, v, device).cpu()
        values_all = projection_on_bias(images, bias)
        values = np.asarray([v for v, keep in zip(values_all, used_mask) if keep], dtype=np.float64)
        projections.append({"image_set": "student", **group_summary(values, used_groups, label)})
        per_projection[label] = values_all
        if label == "correct":
            correct_proj = values

    curve_rows = query_curve(correct_proj, used_groups, args.out_dir / "query_curve.csv")
    random_key = random_key_false_rate(correct_proj)
    corr = score_projection_tests(scores.numpy(), per_projection["correct"])
    metrics = {
        "student_pkl": str(args.student_pkl),
        "state_path": str(args.state_path),
        "query_count": args.query_count,
        "per_group": args.per_group,
        "image_size": int(student.img_resolution),
        "feature_space": state["feature_space"],
        "bias_dct": [args.dct_u, args.dct_v],
        "score_stats_all_queries": summarize(scores.numpy()),
        "score_stats_selected": summarize([r["score"] for r in selected]),
        "projection_tests": projections,
        "score_projection_correlation": corr,
        "random_key": random_key,
        "query_curve": curve_rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out_dir / "selected_queries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "rank", "seed", "score", "distance", "anchor_id"])
        writer.writeheader()
        writer.writerows(selected)
    with (args.out_dir / "per_query.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "seed", "score", "distance", "anchor_id", "selected_group"]
        fieldnames += [f"projection_{label}" for label in per_projection]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, seed in enumerate(seeds):
            row = {
                "index": i,
                "seed": seed,
                "score": float(scores[i]),
                "distance": float(distances[i]),
                "anchor_id": int(anchor_ids[i]),
                "selected_group": groups[i],
            }
            for label, values in per_projection.items():
                row[f"projection_{label}"] = float(values[i])
            writer.writerow(row)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    args = parse_args()
    if args.command == "make_dataset":
        make_dataset(args)
    elif args.command == "eval_student":
        eval_student(args)
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
