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
from diffusers.utils.torch_utils import randn_tensor
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
    parser = argparse.ArgumentParser(description="Low-query DHT-32 key-query verification.")
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--student_unet_dir", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--candidates_csv", type=Path, default=None)
    parser.add_argument("--candidate_count", type=int, default=10000)
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 200, 300, 590])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--probe_step", type=int, default=None)
    parser.add_argument("--bias_steps", type=int, default=None)
    parser.add_argument("--bias_strength", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--seed_base", type=int, default=20261001)
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
        "probe_step": 500,
        "bias_steps": 50,
        "bias_strength": 0.03,
        "alpha": 2.0,
    }
    defaults.update(raw)
    return SimpleNamespace(**defaults)


def noise_and_generators_from_seeds(seeds, channels, image_size, device, dtype):
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    shape = (len(seeds), channels, image_size, image_size)
    return randn_tensor(shape, generator=generators, device=device, dtype=dtype), generators


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


def save_images(images, out_dir, seeds):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    for seed, img in zip(seeds, arr):
        Image.fromarray(img).save(out_dir / f"{int(seed)}.png")


@torch.inference_mode()
def screen_candidate_seeds_exact(teacher_unet, scheduler, hook, anchors, wm_args, args, device, dtype):
    image_size = int(teacher_unet.config.sample_size)
    channels = int(teacher_unet.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    probe_step = min(max(wm_args.probe_step, 0), args.num_steps - 1)

    rows = []
    started = time.time()
    for offset in range(0, args.candidate_count, args.batch_size):
        bsz = min(args.batch_size, args.candidate_count - offset)
        seeds = [args.seed_base + offset + i for i in range(bsz)]
        sample, generators = noise_and_generators_from_seeds(seeds, channels, image_size, device, dtype)
        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = teacher_unet(sample, timestep).sample
            if step_idx == probe_step:
                scores, distances, anchor_ids = similarity_from_pool(hook.value.to(device), anchors, wm_args)
                break
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)

        for i, seed in enumerate(seeds):
            rows.append(
                {
                    "seed": int(seed),
                    "score": float(scores[i].detach().cpu()),
                    "distance": float(distances[i].detach().cpu()),
                    "anchor_id": int(anchor_ids[i].detach().cpu()),
                }
            )
        print(f"screen: {offset + bsz}/{args.candidate_count}, elapsed={time.time() - started:.1f}s", flush=True)
    return rows


def load_candidate_rows(path):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "seed": int(row["seed"]),
                    "score": float(row["score"]),
                    "distance": float(row["distance"]),
                    "anchor_id": int(row["anchor_id"]),
                }
            )
    return rows


@torch.inference_mode()
def sample_selected(model, scheduler, args, device, dtype, seeds, split_name, wm_args=None, bias=None, hook=None, anchors=None, protected=False):
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
    started = time.time()
    for offset in range(0, len(seeds), args.batch_size):
        batch_seeds = seeds[offset : offset + args.batch_size]
        sample, generators = noise_and_generators_from_seeds(batch_seeds, channels, image_size, device, dtype)
        score = torch.zeros(len(batch_seeds), device=device)
        distance = torch.zeros(len(batch_seeds), device=device)
        anchor_id = torch.zeros(len(batch_seeds), device=device, dtype=torch.long)

        for step_idx, timestep in enumerate(scheduler.timesteps):
            model_output = model(sample, timestep).sample
            if hook is not None and step_idx == probe_step:
                score, distance, anchor_id = similarity_from_pool(hook.value.to(device), anchors, wm_args)
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
            if protected and step_idx >= start_bias_step:
                weight_idx = step_idx - start_bias_step
                gate = gate_from_score(score, wm_args)
                coeff = wm_args.bias_strength * weights[weight_idx] * torch.pow(gate, wm_args.alpha)
                sample = (sample + coeff.view(-1, 1, 1, 1).to(dtype) * bias.to(dtype)).clamp(-1.5, 1.5)

        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)
        if hook is not None:
            all_scores.append(score.float().cpu())
            all_distances.append(distance.float().cpu())
            all_anchor_ids.append(anchor_id.cpu())
        if args.save_all_images:
            save_images(images, args.out_dir / f"images_{split_name}", batch_seeds)
        print(f"{split_name}: {offset + len(batch_seeds)}/{len(seeds)}, elapsed={time.time() - started:.1f}s", flush=True)

    result = {"images": torch.cat(all_images, dim=0)}
    if hook is not None:
        result.update(
            {
                "scores": torch.cat(all_scores, dim=0),
                "distances": torch.cat(all_distances, dim=0),
                "anchor_ids": torch.cat(all_anchor_ids, dim=0),
            }
        )
    return result


def one_sided_welch(high, low):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(high, low, equal_var=False, alternative="greater")
        return float(stat), float(pvalue)
    except Exception:
        var_h = high.var(ddof=1)
        var_l = low.var(ddof=1)
        denom = math.sqrt(var_h / len(high) + var_l / len(low))
        stat = 0.0 if denom == 0 else (high.mean() - low.mean()) / denom
        pvalue = 0.5 * math.erfc(stat / math.sqrt(2.0))
        return float(stat), float(pvalue)


def two_sided_welch(high, low):
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(high, low, equal_var=False)
        return float(stat), float(pvalue)
    except Exception:
        stat, pvalue = one_sided_welch(high, low)
        return stat, float(min(1.0, 2.0 * min(pvalue, 1.0 - pvalue)))


def summarize_budget(selected_rows, projections, budget):
    k = budget // 2
    low_rows = selected_rows["low"][:k]
    high_rows = selected_rows["high"][:k]
    out = {"budget": int(budget), "per_group": int(k)}
    out["low_score_mean"] = float(np.mean([r["score"] for r in low_rows]))
    out["high_score_mean"] = float(np.mean([r["score"] for r in high_rows]))
    out["low_score_range"] = [float(min(r["score"] for r in low_rows)), float(max(r["score"] for r in low_rows))]
    out["high_score_range"] = [float(min(r["score"] for r in high_rows)), float(max(r["score"] for r in high_rows))]

    for model_name, model_proj in projections.items():
        low = np.asarray([model_proj[int(r["seed"])] for r in low_rows], dtype=np.float64)
        high = np.asarray([model_proj[int(r["seed"])] for r in high_rows], dtype=np.float64)
        t_one, p_one = one_sided_welch(high, low)
        t_two, p_two = two_sided_welch(high, low)
        pooled = math.sqrt(((len(high) - 1) * high.var(ddof=1) + (len(low) - 1) * low.var(ddof=1)) / max(len(high) + len(low) - 2, 1))
        out[model_name] = {
            "low_projection_mean": float(low.mean()),
            "high_projection_mean": float(high.mean()),
            "high_low_delta": float(high.mean() - low.mean()),
            "one_sided_t": t_one,
            "one_sided_p": p_one,
            "two_sided_t": t_two,
            "two_sided_p": p_two,
            "cohen_d": float((high.mean() - low.mean()) / pooled) if pooled > 0 else 0.0,
            "decision_p001": bool((high.mean() - low.mean()) > 0 and p_one < 0.001),
            "decision_p01": bool((high.mean() - low.mean()) > 0 and p_one < 0.01),
            "decision_p05": bool((high.mean() - low.mean()) > 0 and p_one < 0.05),
        }
    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "grids").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    max_budget = max(args.budgets)
    if max_budget % 2 != 0:
        raise ValueError("Budgets must be even so high/low query groups have equal size.")
    if args.candidate_count < max_budget:
        raise ValueError("candidate_count must be at least max(budgets).")

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
    student = UNet2DModel.from_pretrained(args.student_unet_dir).to(device=device, dtype=dtype).eval()
    scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")
    hook = FeatureHook(teacher.mid_block)

    print(
        json.dumps(
            {
                "device": str(device),
                "dtype": str(dtype),
                "candidate_count": args.candidate_count,
                "budgets": args.budgets,
                "max_budget": max_budget,
                "num_steps": args.num_steps,
                "probe_step": wm_args.probe_step,
                "bias_steps": wm_args.bias_steps,
                "bias_strength": wm_args.bias_strength,
                "alpha": wm_args.alpha,
            },
            indent=2,
        ),
        flush=True,
    )

    if args.candidates_csv is not None:
        candidates = load_candidate_rows(args.candidates_csv)
        print(f"loaded candidates: {len(candidates)} from {args.candidates_csv}", flush=True)
        if len(candidates) < max_budget:
            raise ValueError("candidates_csv must contain at least max(budgets) rows.")
    else:
        candidates = screen_candidate_seeds_exact(teacher, scheduler, hook, anchors, wm_args, args, device, dtype)
    candidates_sorted = sorted(candidates, key=lambda r: r["score"])
    each = max_budget // 2
    selected = {
        "low": candidates_sorted[:each],
        "high": list(reversed(candidates_sorted[-each:])),
    }
    selected_seeds = [r["seed"] for r in selected["low"]] + [r["seed"] for r in selected["high"]]

    with (args.out_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "score", "distance", "anchor_id"])
        writer.writeheader()
        writer.writerows(candidates_sorted)

    with (args.out_dir / "selected_key_queries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "rank", "seed", "score", "distance", "anchor_id"])
        writer.writeheader()
        for group, rows in selected.items():
            for rank, row in enumerate(rows, start=1):
                item = dict(row)
                item.update({"group": group, "rank": rank})
                writer.writerow(item)

    teacher_clean = sample_selected(
        teacher, scheduler, args, device, dtype, selected_seeds, "teacher_clean", wm_args=wm_args, bias=bias, hook=hook, anchors=anchors
    )
    teacher_protected = sample_selected(
        teacher,
        scheduler,
        args,
        device,
        dtype,
        selected_seeds,
        "teacher_protected",
        wm_args=wm_args,
        bias=bias,
        hook=hook,
        anchors=anchors,
        protected=True,
    )
    student_out = sample_selected(student, scheduler, args, device, dtype, selected_seeds, "student")

    images = {
        "teacher_clean": teacher_clean["images"],
        "teacher_protected": teacher_protected["images"],
        "student": student_out["images"],
    }
    for name, img in images.items():
        save_grid(img, args.out_dir / "grids" / f"{name}_key_grid.png")

    seed_to_index = {seed: idx for idx, seed in enumerate(selected_seeds)}
    projections = {}
    for name, img in images.items():
        proj = projection_on_bias(img, bias)
        projections[name] = {seed: float(proj[seed_to_index[seed]]) for seed in selected_seeds}

    budget_metrics = [summarize_budget(selected, projections, b) for b in sorted(args.budgets)]
    score_values = np.asarray([r["score"] for r in candidates_sorted], dtype=np.float64)
    metrics = {
        "candidate_count": args.candidate_count,
        "max_budget": max_budget,
        "budgets": budget_metrics,
        "candidate_score_summary": {
            "mean": float(score_values.mean()),
            "std": float(score_values.std(ddof=1)),
            "min": float(score_values.min()),
            "max": float(score_values.max()),
            "q001": float(np.quantile(score_values, 0.001)),
            "q01": float(np.quantile(score_values, 0.01)),
            "q05": float(np.quantile(score_values, 0.05)),
            "q10": float(np.quantile(score_values, 0.10)),
            "q50": float(np.quantile(score_values, 0.50)),
            "q90": float(np.quantile(score_values, 0.90)),
            "q95": float(np.quantile(score_values, 0.95)),
            "q99": float(np.quantile(score_values, 0.99)),
            "q999": float(np.quantile(score_values, 0.999)),
        },
        "selected_score_summary": {
            "low_min": float(min(r["score"] for r in selected["low"])),
            "low_max": float(max(r["score"] for r in selected["low"])),
            "low_mean": float(np.mean([r["score"] for r in selected["low"]])),
            "high_min": float(min(r["score"] for r in selected["high"])),
            "high_max": float(max(r["score"] for r in selected["high"])),
            "high_mean": float(np.mean([r["score"] for r in selected["high"]])),
        },
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (args.out_dir / "per_key_query.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "group",
            "rank",
            "seed",
            "score",
            "distance",
            "anchor_id",
            "teacher_clean_projection",
            "teacher_protected_projection",
            "student_projection",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for group, rows in selected.items():
            for rank, row in enumerate(rows, start=1):
                seed = row["seed"]
                writer.writerow(
                    {
                        "group": group,
                        "rank": rank,
                        "seed": seed,
                        "score": row["score"],
                        "distance": row["distance"],
                        "anchor_id": row["anchor_id"],
                        "teacher_clean_projection": projections["teacher_clean"][seed],
                        "teacher_protected_projection": projections["teacher_protected"][seed],
                        "student_projection": projections["student"][seed],
                    }
                )

    print(json.dumps(metrics, indent=2), flush=True)
    hook.close()


if __name__ == "__main__":
    main()
