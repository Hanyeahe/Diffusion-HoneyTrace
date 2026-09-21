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
    parser = argparse.ArgumentParser(description="Generate a black-box query dataset from a protected EDM-64 teacher.")
    parser.add_argument("--edm_repo", type=Path, required=True)
    parser.add_argument("--network_pkl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--probe_step", type=int, default=20)
    parser.add_argument("--bias_steps", type=int, default=5)
    parser.add_argument("--bias_strength", type=float, default=0.30)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--anchor_count", type=int, default=8)
    parser.add_argument("--calibration_count", type=int, default=1024)
    parser.add_argument("--calibration_seed_base", type=int, default=20266400)
    parser.add_argument("--sample_seed_base", type=int, default=20270000)
    parser.add_argument("--hook_module", type=str, default="model.dec.8x8_in1")
    parser.add_argument("--dct_u", type=int, default=31)
    parser.add_argument("--dct_v", type=int, default=29)
    parser.add_argument("--mode", choices=["protected", "clean"], default="protected")
    parser.add_argument("--state_in", type=Path)
    parser.add_argument("--state_out", type=Path)
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
            t_hat = net.round_sigma(t_cur)
            x_hat = x_next
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
        print(f"calibrate {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(features, dim=0)


def build_or_load_state(net, hook, args, device):
    state_path = args.state_in
    if state_path is not None and state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        print(f"loaded watermark state: {state_path}", flush=True)
        return state

    seeds = [args.calibration_seed_base + i for i in range(args.calibration_count)]
    features = collect_probe_features(net, hook, seeds, args, device)
    anchors, anchor_indices = select_farthest_anchors(features, args.anchor_count)
    distances, anchor_ids = nearest_dist(features, anchors)
    q10, q50, q90 = torch.quantile(distances, torch.tensor([0.10, 0.50, 0.90])).tolist()
    temperature = max((q90 - q10) / (2.0 * 2.197224577), 1e-4)
    center = q50
    scores, distances, anchor_ids = score_from_features(features, anchors, center, temperature)
    state = {
        "anchors": anchors.cpu(),
        "anchor_candidate_indices": anchor_indices,
        "anchor_seeds": [seeds[i] for i in anchor_indices],
        "center": float(center),
        "temperature": float(temperature),
        "hook_module": args.hook_module,
        "probe_step": int(args.probe_step),
        "num_steps": int(args.num_steps),
        "score_stats": {
            "mean": float(scores.mean().item()),
            "std": float(scores.std(unbiased=True).item()),
            "q05": float(torch.quantile(scores, 0.05).item()),
            "q50": float(torch.quantile(scores, 0.50).item()),
            "q95": float(torch.quantile(scores, 0.95).item()),
            "min": float(scores.min().item()),
            "max": float(scores.max().item()),
        },
    }
    out_state = args.state_out or (args.out_dir / "watermark_state.pt")
    out_state.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_state)
    print(f"saved watermark state: {out_state}", flush=True)
    return state


def images_to_uint8(images):
    return (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def save_grid(images, path, cols=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = images_to_uint8(images)
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
def sample_batch(net, hook, state, seeds, args, device):
    t_steps = edm_t_steps(net, args.num_steps, device)
    probe_step = min(max(args.probe_step, 0), args.num_steps - 1)
    start_bias_step = max(0, args.num_steps - args.bias_steps)
    weights = torch.linspace(1.0, 2.0, args.bias_steps, dtype=torch.float64, device=device)
    weights = weights / weights.sum().clamp_min(1e-8)
    bias = make_dct_bias(net.img_channels, net.img_resolution, args.dct_u, args.dct_v, device).to(torch.float64)
    anchors = state["anchors"].to(device)
    center = float(state["center"])
    temperature = float(state["temperature"])

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
            score, distance, anchor_id = score_from_features(hook.value.to(device), anchors, center, temperature)
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


def main():
    args = parse_args()
    sys.path.insert(0, str(args.edm_repo))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    with args.network_pkl.open("rb") as f:
        net = pickle.load(f)["ema"].to(device).eval()

    hook = FeatureHook(get_submodule(net, args.hook_module))
    state = build_or_load_state(net, hook, args, device)
    if args.state_out and not args.state_out.exists():
        torch.save(state, args.state_out)

    all_scores = []
    all_proj = []
    grid_images = []
    csv_path = args.out_dir / "samples.csv"
    started = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "seed", "image", "score", "distance", "anchor_id", "projection"])
        writer.writeheader()
        for offset in range(0, args.count, args.batch_size):
            batch_count = min(args.batch_size, args.count - offset)
            seeds = [args.sample_seed_base + offset + i for i in range(batch_count)]
            images, scores, distances, anchor_ids, proj = sample_batch(net, hook, state, seeds, args, device)
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
                needed = 64 - len(grid_images)
                grid_images.append(images[:needed])
            print(f"{args.mode} dataset {offset + batch_count}/{args.count}, elapsed={time.time() - started:.1f}s", flush=True)

    grid_tensor = torch.cat(grid_images, dim=0)[:64] if grid_images else torch.empty(0)
    if len(grid_tensor):
        save_grid(grid_tensor, args.out_dir / "grid_first64.png")

    score_arr = np.asarray(all_scores, dtype=np.float64)
    proj_arr = np.asarray(all_proj, dtype=np.float64)
    metrics = {
        "mode": args.mode,
        "count": args.count,
        "image_size": int(net.img_resolution),
        "bias_strength": args.bias_strength,
        "bias_steps": args.bias_steps,
        "bias_dct": [args.dct_u, args.dct_v],
        "score_stats": {
            "mean": float(score_arr.mean()),
            "std": float(score_arr.std(ddof=1)),
            "q05": float(np.quantile(score_arr, 0.05)),
            "q50": float(np.quantile(score_arr, 0.50)),
            "q95": float(np.quantile(score_arr, 0.95)),
            "min": float(score_arr.min()),
            "max": float(score_arr.max()),
        },
        "projection_stats": {
            "mean": float(proj_arr.mean()),
            "std": float(proj_arr.std(ddof=1)),
            "q05": float(np.quantile(proj_arr, 0.05)),
            "q50": float(np.quantile(proj_arr, 0.50)),
            "q95": float(np.quantile(proj_arr, 0.95)),
        },
        "watermark_state": str(args.state_out or args.out_dir / "watermark_state.pt"),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    hook.close()


if __name__ == "__main__":
    main()
