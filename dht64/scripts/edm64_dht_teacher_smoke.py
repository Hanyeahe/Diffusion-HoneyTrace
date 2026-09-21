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
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="EDM-64 teacher-level DHT smoke test.")
    parser.add_argument("--edm_repo", type=Path, required=True)
    parser.add_argument("--network_pkl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--candidate_count", type=int, default=512)
    parser.add_argument("--per_group", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--probe_step", type=int, default=20)
    parser.add_argument("--bias_steps", type=int, default=5)
    parser.add_argument("--bias_strength", type=float, default=0.10)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--anchor_count", type=int, default=8)
    parser.add_argument("--seed_base", type=int, default=20266400)
    parser.add_argument("--hook_module", type=str, default="model.dec.8x8_in1")
    parser.add_argument("--dct_u", type=int, default=8)
    parser.add_argument("--dct_v", type=int, default=9)
    parser.add_argument("--wrong_dct", type=str, default="3,11;11,3;12,12")
    return parser.parse_args()


class FeatureHook:
    def __init__(self, module):
        self.value = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        feat = output[0] if isinstance(output, (tuple, list)) else output
        self.value = feat.detach().float().mean(dim=(2, 3))

    def close(self):
        self.handle.remove()


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, x):
        return self.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)


def get_submodule(root, dotted_name):
    cur = root
    for part in dotted_name.split("."):
        cur = getattr(cur, part)
    return cur


def normalize_rows(x):
    x = x - x.mean(dim=1, keepdim=True)
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def select_farthest_anchors(features, k):
    feats = normalize_rows(features.float())
    anchors = [0]
    min_dist = torch.cdist(feats, feats[anchors]).squeeze(1)
    for _ in range(1, k):
        idx = int(torch.argmax(min_dist).item())
        anchors.append(idx)
        dist = torch.cdist(feats, feats[[idx]]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
    return features[anchors].clone(), anchors


def nearest_dist(features, anchors):
    feats = normalize_rows(features.float())
    anc = normalize_rows(anchors.float())
    dists = torch.cdist(feats, anc)
    d_min, ids = dists.min(dim=1)
    return d_min, ids


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
def collect_probe_features(net, hook, seeds, args, device):
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
            x_cur = x_next
            t_hat = net.round_sigma(t_cur)
            x_hat = x_cur
            denoised = net(x_hat, t_hat, None).to(torch.float64)
            if i == probe_step:
                features.append(hook.value.cpu())
                break
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < args.num_steps - 1:
                denoised = net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        print(f"screen {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(features, dim=0)


def score_from_features(features, anchors, center, temperature):
    d_min, ids = nearest_dist(features, anchors)
    score = torch.sigmoid((center - d_min) / max(float(temperature), 1e-8))
    return score, d_min, ids


@torch.inference_mode()
def sample_edm(net, hook, anchors, center, temperature, selected, args, device, protected=False):
    t_steps = edm_t_steps(net, args.num_steps, device)
    probe_step = min(max(args.probe_step, 0), args.num_steps - 1)
    start_bias_step = max(0, args.num_steps - args.bias_steps)
    weights = torch.linspace(1.0, 2.0, args.bias_steps, dtype=torch.float64, device=device)
    weights = weights / weights.sum().clamp_min(1e-8)
    bias = make_dct_bias(net.img_channels, net.img_resolution, args.dct_u, args.dct_v, device).to(torch.float64)

    images = []
    scores = []
    distances = []
    anchor_ids = []
    started = time.time()
    seeds = [row["seed"] for row in selected]
    for offset in range(0, len(seeds), args.batch_size):
        batch_rows = selected[offset : offset + args.batch_size]
        batch_seeds = [row["seed"] for row in batch_rows]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
        x_next = latents.to(torch.float64) * t_steps[0]
        score = torch.zeros(len(batch_seeds), dtype=torch.float32, device=device)
        distance = torch.zeros(len(batch_seeds), dtype=torch.float32, device=device)
        anchor_id = torch.zeros(len(batch_seeds), dtype=torch.long, device=device)

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            t_hat = net.round_sigma(t_cur)
            x_hat = x_cur
            denoised = net(x_hat, t_hat, None).to(torch.float64)
            if i == probe_step:
                score, distance, anchor_id = score_from_features(hook.value.to(device), anchors.to(device), center, temperature)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            if i < args.num_steps - 1:
                denoised = net(x_next, t_next, None).to(torch.float64)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            if protected and i >= start_bias_step:
                wi = weights[i - start_bias_step]
                coeff = args.bias_strength * wi * torch.pow(score.to(torch.float64).clamp(0, 1), args.alpha)
                x_next = (x_next + coeff.view(-1, 1, 1, 1) * bias).clamp(-1.5, 1.5)

        images.append(x_next.float().cpu())
        scores.append(score.float().cpu())
        distances.append(distance.float().cpu())
        anchor_ids.append(anchor_id.cpu())
        print(f"{'protected' if protected else 'clean'} {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)

    return {
        "images": torch.cat(images, dim=0),
        "scores": torch.cat(scores, dim=0),
        "distances": torch.cat(distances, dim=0),
        "anchor_ids": torch.cat(anchor_ids, dim=0),
    }


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


def save_images(images, out_dir, seeds):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = images_to_uint8(images)
    for seed, img in zip(seeds, arr):
        Image.fromarray(img, "RGB").save(out_dir / f"{int(seed)}.png")


def projection_on_bias(images, bias):
    b = bias.float().cpu()
    b = b / b.flatten(1).norm(dim=1).view(1, 1, 1, 1).clamp_min(1e-8)
    return (images.float() * b).flatten(1).sum(dim=1).numpy()


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


def psnr_from_mae_mse(mse):
    return float(10.0 * math.log10(4.0 / max(mse, 1e-12)))


def random_key_false_rate(values, repeats=1000, seed=20260907):
    rng = np.random.default_rng(seed)
    n = len(values)
    k = n // 2
    decisions = []
    pvals = []
    deltas = []
    values = np.asarray(values)
    for _ in range(repeats):
        perm = rng.permutation(n)
        low = values[perm[:k]]
        high = values[perm[k : 2 * k]]
        _, p = one_sided_welch(high, low)
        delta = float(high.mean() - low.mean())
        pvals.append(p)
        deltas.append(delta)
        decisions.append(delta > 0 and p < 0.005)
    return {
        "repeats": repeats,
        "false_detection_rate_p005": float(np.mean(decisions)),
        "delta_median": float(np.median(deltas)),
        "p_median": float(np.median(pvals)),
    }


def main():
    args = parse_args()
    sys.path.insert(0, str(args.edm_repo))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    with args.network_pkl.open("rb") as f:
        net = pickle.load(f)["ema"].to(device).eval()

    hook = FeatureHook(get_submodule(net, args.hook_module))
    candidate_seeds = [args.seed_base + i for i in range(args.candidate_count)]
    features = collect_probe_features(net, hook, candidate_seeds, args, device)
    anchors, anchor_indices = select_farthest_anchors(features, args.anchor_count)
    distances, anchor_ids = nearest_dist(features, anchors)
    q10, q50, q90 = torch.quantile(distances, torch.tensor([0.10, 0.50, 0.90])).tolist()
    temperature = max((q90 - q10) / (2.0 * 2.197224577), 1e-4)
    center = q50
    scores, distances, anchor_ids = score_from_features(features, anchors, center, temperature)

    order = torch.argsort(scores)
    low_idx = order[: args.per_group].tolist()
    high_idx = order[-args.per_group :].tolist()
    selected = []
    for rank, idx in enumerate(low_idx, 1):
        selected.append({"group": "low", "rank": rank, "seed": candidate_seeds[idx], "score": float(scores[idx]), "distance": float(distances[idx]), "anchor_id": int(anchor_ids[idx])})
    for rank, idx in enumerate(high_idx, 1):
        selected.append({"group": "high", "rank": rank, "seed": candidate_seeds[idx], "score": float(scores[idx]), "distance": float(distances[idx]), "anchor_id": int(anchor_ids[idx])})
    groups = [row["group"] for row in selected]
    seeds = [row["seed"] for row in selected]

    clean = sample_edm(net, hook, anchors, center, temperature, selected, args, device, protected=False)
    protected = sample_edm(net, hook, anchors, center, temperature, selected, args, device, protected=True)

    clean_img = clean["images"]
    protected_img = protected["images"]
    save_grid(clean_img, args.out_dir / "clean_grid.png")
    save_grid(protected_img, args.out_dir / "protected_grid.png")
    save_images(clean_img, args.out_dir / "images_clean", seeds)
    save_images(protected_img, args.out_dir / "images_protected", seeds)

    diff = (protected_img - clean_img).float()
    mae = float(diff.abs().mean().item())
    mse = float(diff.square().mean().item())

    bias_specs = [(args.dct_u, args.dct_v, "correct")]
    for item in args.wrong_dct.split(";"):
        u, v = item.split(",")
        bias_specs.append((int(u), int(v), f"wrong_{u}_{v}"))

    projections = []
    for u, v, label in bias_specs:
        bias = make_dct_bias(net.img_channels, net.img_resolution, u, v, device).cpu()
        clean_proj = projection_on_bias(clean_img, bias)
        protected_proj = projection_on_bias(protected_img, bias)
        projections.append({"image_set": "clean_teacher", **group_summary(clean_proj, groups, label)})
        projections.append({"image_set": "protected_teacher", **group_summary(protected_proj, groups, label)})
        if label == "correct":
            random_key = random_key_false_rate(protected_proj)

    score_arr = scores.numpy()
    summary = {
        "model": "edm-ffhq-64x64-uncond-vp",
        "image_size": int(net.img_resolution),
        "num_steps": args.num_steps,
        "probe_step": args.probe_step,
        "hook_module": args.hook_module,
        "anchor_count": args.anchor_count,
        "anchor_candidate_indices": anchor_indices,
        "sigmoid_center": center,
        "sigmoid_temperature": temperature,
        "score_stats": {
            "mean": float(score_arr.mean()),
            "std": float(score_arr.std(ddof=1)),
            "q05": float(np.quantile(score_arr, 0.05)),
            "q10": float(np.quantile(score_arr, 0.10)),
            "q50": float(np.quantile(score_arr, 0.50)),
            "q90": float(np.quantile(score_arr, 0.90)),
            "q95": float(np.quantile(score_arr, 0.95)),
            "min": float(score_arr.min()),
            "max": float(score_arr.max()),
        },
        "paired_quality": {
            "mae_in_minus1_1": mae,
            "mse_in_minus1_1": mse,
            "psnr_db_minus1_1": psnr_from_mae_mse(mse),
        },
        "projection_tests": projections,
        "random_key": random_key,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (args.out_dir / "selected_key_queries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "rank", "seed", "score", "distance", "anchor_id"])
        writer.writeheader()
        writer.writerows(selected)

    with (args.out_dir / "candidate_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "score", "distance", "anchor_id"])
        writer.writeheader()
        for seed, score, dist, aid in zip(candidate_seeds, scores.tolist(), distances.tolist(), anchor_ids.tolist()):
            writer.writerow({"seed": seed, "score": score, "distance": dist, "anchor_id": aid})

    print(json.dumps(summary, indent=2), flush=True)
    hook.close()


if __name__ == "__main__":
    main()
