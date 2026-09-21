#!/usr/bin/env python
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from edm64_featurepool_blackbox import (  # noqa: E402
    InceptionFeature,
    StackedRandomGenerator,
    build_or_load_state,
    edm_t_steps,
    images_to_uint8,
    load_model,
    make_dct_bias,
    projection_on_bias,
    save_grid,
    score_from_features,
    setup,
    summarize,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate an EDM-64 constant-amplitude DCT baseline dataset.")
    parser.add_argument("--edm_repo", type=Path, required=True)
    parser.add_argument("--network_pkl", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--probe_step", type=int, default=34)
    parser.add_argument("--bias_steps", type=int, default=5)
    parser.add_argument("--bias_strength", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--constant_power", type=float, required=True)
    parser.add_argument("--dct_u", type=int, default=20)
    parser.add_argument("--dct_v", type=int, default=20)
    parser.add_argument("--state_in", type=Path, required=True)
    parser.add_argument("--state_out", type=Path)
    parser.add_argument("--sample_seed_base", type=int, default=20282000)
    return parser.parse_args()


@torch.inference_mode()
def sample_constant_batch(net, feat_model, state, seeds, args, device):
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
    constant_power = torch.tensor(float(args.constant_power), dtype=torch.float64, device=device)

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
        if i >= start_bias_step:
            wi = weights[i - start_bias_step]
            coeff = args.bias_strength * wi * constant_power
            x_next = (x_next + coeff * bias).clamp(-1.5, 1.5)

    images = x_next.float().cpu()
    proj = projection_on_bias(images, bias.cpu())
    return images, score.cpu(), distance.cpu(), anchor_id.cpu(), proj


def main():
    args = parse_args()
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

    all_scores, all_proj, grid_chunks = [], [], []
    started = time.time()
    with (args.out_dir / "samples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "seed", "image", "score", "distance", "anchor_id", "projection"],
        )
        writer.writeheader()
        for offset in range(0, args.count, args.batch_size):
            batch_count = min(args.batch_size, args.count - offset)
            seeds = [args.sample_seed_base + offset + i for i in range(batch_count)]
            images, scores, distances, anchor_ids, proj = sample_constant_batch(net, feat_model, state, seeds, args, device)
            uint8 = images_to_uint8(images)
            for j, (seed, img, score, dist, aid, z) in enumerate(
                zip(seeds, uint8, scores.tolist(), distances.tolist(), anchor_ids.tolist(), proj.tolist())
            ):
                idx = offset + j
                rel = f"{idx:06d}.png"
                Image.fromarray(img, "RGB").save(images_dir / rel)
                writer.writerow(
                    {
                        "index": idx,
                        "seed": seed,
                        "image": rel,
                        "score": score,
                        "distance": dist,
                        "anchor_id": aid,
                        "projection": z,
                    }
                )
            f.flush()
            all_scores.extend(scores.tolist())
            all_proj.extend(proj.tolist())
            remaining_grid = 64 - sum(chunk.shape[0] for chunk in grid_chunks)
            if remaining_grid > 0:
                grid_chunks.append(images[:remaining_grid])
            print(f"constant featurepool dataset {offset + batch_count}/{args.count}, elapsed={time.time() - started:.1f}s", flush=True)

    save_grid(torch.cat(grid_chunks, dim=0)[:64], args.out_dir / "grid_first64.png")
    metrics = {
        "mode": "constant_dct",
        "count": args.count,
        "image_size": int(net.img_resolution),
        "feature_space": state["feature_space"],
        "bias_strength": args.bias_strength,
        "bias_steps": args.bias_steps,
        "bias_dct": [args.dct_u, args.dct_v],
        "alpha": args.alpha,
        "constant_power": args.constant_power,
        "equivalent_score": float(args.constant_power ** (1.0 / args.alpha)),
        "total_shift_coefficient": float(args.bias_strength * args.constant_power),
        "score_stats": summarize(all_scores),
        "projection_stats": summarize(all_proj),
        "featurepool_state": str(args.state_in),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
