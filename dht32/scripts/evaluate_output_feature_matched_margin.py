#!/usr/bin/env python
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Seed-independent matched output-feature verification for DHT-32 students."
    )
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--teacher_feature_dir", type=Path, required=True)
    parser.add_argument("--score_csv", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--protected_student_unet_dir", type=Path, required=True)
    parser.add_argument("--clean_student_unet_dir", type=Path, required=True)
    parser.add_argument("--constant_student_unet_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_student_images", type=int, default=3200)
    parser.add_argument("--split_budgets", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--train_fraction", type=float, default=0.75)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--force_resample", action="store_true")
    return parser.parse_args()


class TensorImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, idx):
        return self.images[idx]


def read_scores(path, n):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["index"]), float(row["score"])))
    rows.sort(key=lambda r: r[0])
    scores = np.asarray([score for _idx, score in rows[:n]], dtype=np.float64)
    if scores.shape[0] < n:
        raise ValueError(f"Need {n} scores from {path}, found {scores.shape[0]}")
    return scores


def standardize(train_x, *arrays):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True) + 1e-6
    return [(x - mean) / std for x in arrays], mean, std


def fit_ridge(x, y, alpha):
    y_mean = float(y.mean())
    yc = y - y_mean
    xtx = x.T @ x
    reg = alpha * np.eye(x.shape[1], dtype=np.float64)
    beta = np.linalg.solve(xtx + reg, x.T @ yc)
    return beta, y_mean


def predict_ridge(x, beta, y_mean):
    return x @ beta + y_mean


def corr(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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


def exact_sign_p(num_positive, num_trials):
    if num_trials <= 0:
        return 1.0
    tail = 0.0
    for k in range(num_positive, num_trials + 1):
        tail += math.comb(num_trials, k)
    return float(tail / (2.0**num_trials))


def noise_from_seeds(seeds, channels, image_size, device, dtype):
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    shape = (len(seeds), channels, image_size, image_size)
    return randn_tensor(shape, generator=generators, device=device, dtype=dtype), generators


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
def sample_student(model_dir, scheduler, count, seed_base, args, device, dtype, label):
    cache_path = args.out_dir / f"{label}_images.pt"
    existing_images = None
    if cache_path.exists() and not args.force_resample:
        existing_images = torch.load(cache_path, map_location="cpu")
        if int(existing_images.shape[0]) >= count:
            print(f"reuse {label} images: {tuple(existing_images.shape)}", flush=True)
            return existing_images[:count]
        print(f"extend {label} images: {existing_images.shape[0]} -> {count}", flush=True)

    model = UNet2DModel.from_pretrained(model_dir).to(device=device, dtype=dtype).eval()
    image_size = int(model.config.sample_size)
    channels = int(model.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    start = int(existing_images.shape[0]) if existing_images is not None else 0
    all_images = [existing_images] if existing_images is not None else []
    started = time.time()
    for offset in range(start, count, args.batch_size):
        bsz = min(args.batch_size, count - offset)
        seeds = [seed_base + offset + i for i in range(bsz)]
        sample, generators = noise_from_seeds(seeds, channels, image_size, device, dtype)
        for timestep in scheduler.timesteps:
            model_output = model(sample, timestep).sample
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)
        print(f"{label}: {offset + bsz}/{count}, elapsed={time.time() - started:.1f}s", flush=True)
    images = torch.cat(all_images, dim=0)
    torch.save(images, cache_path)
    save_grid(images, args.out_dir / f"{label}_grid.png")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return images


def build_inception(device):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    return model.eval().to(device)


@torch.inference_mode()
def extract_inception_features(model, images, batch_size, device, label, out_path):
    if out_path.exists():
        feats = np.load(out_path).astype(np.float64)
        if feats.shape[0] >= images.shape[0]:
            print(f"reuse features {label}: {feats.shape}", flush=True)
            return feats[: images.shape[0]]
    loader = DataLoader(
        TensorImageDataset(images),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats = []
    done = 0
    started = time.time()
    for batch in loader:
        x = batch.to(device, non_blocking=True).float()
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feat = model(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        feats.append(feat.detach().float().cpu())
        done += batch.shape[0]
        print(f"features {label}: {done}/{len(loader.dataset)}, elapsed={time.time() - started:.1f}s", flush=True)
    arr = torch.cat(feats, dim=0).numpy().astype(np.float64)
    np.save(out_path, arr)
    return arr


def projection_on_bias(images, bias):
    x = images.float() * 2.0 - 1.0
    b = bias.float().cpu()
    if b.ndim == 4:
        b = b.squeeze(0)
    b_unit = b / b.flatten().norm().clamp_min(1e-8)
    return (x * b_unit).flatten(1).sum(dim=1).numpy().astype(np.float64)


def summarize_delta(scores, projections, indices):
    scores = np.asarray(scores, dtype=np.float64)
    projections = np.asarray(projections, dtype=np.float64)
    local_order = indices[np.argsort(scores[indices])]
    k = len(indices) // 2
    low_idx = local_order[:k]
    high_idx = local_order[-k:]
    low = projections[low_idx]
    high = projections[high_idx]
    t_value, p_value = one_sided_welch(high, low)
    pooled = math.sqrt(
        ((len(high) - 1) * high.var(ddof=1) + (len(low) - 1) * low.var(ddof=1))
        / max(len(high) + len(low) - 2, 1)
    )
    delta = float(high.mean() - low.mean())
    return {
        "per_group": int(k),
        "low_score_mean": float(scores[low_idx].mean()),
        "high_score_mean": float(scores[high_idx].mean()),
        "low_projection_mean": float(low.mean()),
        "high_projection_mean": float(high.mean()),
        "delta": delta,
        "one_sided_t": float(t_value),
        "one_sided_p": float(p_value),
        "cohen_d": float(delta / pooled) if pooled > 0 else 0.0,
        "decision_p005": bool(delta > 0 and p_value < 0.005),
    }


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

    x_teacher = np.load(args.teacher_feature_dir / "features_teacher_protected.npy").astype(np.float64)
    y_teacher = read_scores(args.score_csv, x_teacher.shape[0])
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(x_teacher.shape[0])
    n_train = int(round(args.train_fraction * x_teacher.shape[0]))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]
    (x_train, x_val), _mean, _std = standardize(x_teacher[train_idx], x_teacher[train_idx], x_teacher[val_idx])
    y_train = y_teacher[train_idx]
    y_val = y_teacher[val_idx]

    alpha_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    val_rows = []
    best = None
    for alpha in alpha_grid:
        beta, y_mean = fit_ridge(x_train, y_train, alpha)
        pred_val = predict_ridge(x_val, beta, y_mean)
        row = {
            "alpha": float(alpha),
            "val_mse": float(np.mean((pred_val - y_val) ** 2)),
            "val_corr": corr(pred_val, y_val),
        }
        val_rows.append(row)
        if best is None or row["val_mse"] < best["val_mse"]:
            best = row

    (x_all,), train_mean, train_std = standardize(x_teacher, x_teacher)
    beta, y_mean = fit_ridge(x_all, y_teacher, float(best["alpha"]))
    pred_teacher = predict_ridge(x_all, beta, y_mean)
    np.savez(
        args.out_dir / "frozen_output_scorer.npz",
        beta=beta,
        y_mean=np.asarray([y_mean], dtype=np.float64),
        feature_mean=train_mean,
        feature_std=train_std,
        alpha=np.asarray([float(best["alpha"])], dtype=np.float64),
    )

    scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")
    state = torch.load(args.watermark_state, map_location="cpu")
    bias = state["bias"].float()

    model_specs = {
        "protected": (args.protected_student_unet_dir, args.seed + 1000000),
        "clean": (args.clean_student_unet_dir, args.seed + 2000000),
        "constant": (args.constant_student_unet_dir, args.seed + 3000000),
    }
    feature_model = None
    per_model = {}
    per_query_rows = {}
    for name, (model_dir, seed_base) in model_specs.items():
        images = sample_student(model_dir, scheduler, args.num_student_images, seed_base, args, device, dtype, name)
        if feature_model is None:
            feature_model = build_inception(device)
        feats = extract_inception_features(
            feature_model,
            images,
            args.feature_batch_size,
            device,
            name,
            args.out_dir / f"features_{name}.npy",
        )
        x = (feats.astype(np.float64) - train_mean) / train_std
        pred_score = predict_ridge(x, beta, y_mean)
        proj = projection_on_bias(images, bias)
        per_model[name] = {
            "scores": pred_score,
            "projections": proj,
            "score_mean": float(pred_score.mean()),
            "score_std": float(pred_score.std(ddof=1)),
            "projection_mean": float(proj.mean()),
            "projection_std": float(proj.std(ddof=1)),
            "score_projection_corr": corr(pred_score, proj),
        }
        per_query_rows[name] = [
            {"index": int(i), "score": float(pred_score[i]), "projection": float(proj[i])}
            for i in range(pred_score.shape[0])
        ]
        write_csv(args.out_dir / f"{name}_per_query.csv", per_query_rows[name])

    split_rows = []
    summary_by_budget = []
    for budget in sorted(args.split_budgets):
        if budget % 2 != 0:
            raise ValueError("Split budget must be even.")
        num_splits = args.num_student_images // budget
        split_indices = {
            name: rng.permutation(args.num_student_images)[: num_splits * budget].reshape(num_splits, budget)
            for name in per_model
        }
        margins = []
        protected_detects = 0
        control_detects = {"clean": 0, "constant": 0}
        for split_id in range(num_splits):
            row = {"budget": int(budget), "split": int(split_id)}
            deltas = {}
            pvals = {}
            for name in ["protected", "clean", "constant"]:
                stats = summarize_delta(
                    per_model[name]["scores"],
                    per_model[name]["projections"],
                    split_indices[name][split_id],
                )
                deltas[name] = stats["delta"]
                pvals[name] = stats["one_sided_p"]
                if name == "protected" and stats["decision_p005"]:
                    protected_detects += 1
                if name in control_detects and stats["decision_p005"]:
                    control_detects[name] += 1
                for key, value in stats.items():
                    row[f"{name}_{key}"] = value
            margin = float(deltas["protected"] - max(deltas["clean"], deltas["constant"]))
            row["matched_margin"] = margin
            row["matched_margin_positive"] = bool(margin > 0)
            row["matched_margin_vs"] = "clean" if deltas["clean"] >= deltas["constant"] else "constant"
            margins.append(margin)
            split_rows.append(row)

        positive = int(sum(m > 0 for m in margins))
        summary_by_budget.append(
            {
                "budget": int(budget),
                "num_splits": int(num_splits),
                "positive_margins": positive,
                "exact_one_sided_sign_p": exact_sign_p(positive, num_splits),
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "min_margin": float(np.min(margins)),
                "max_margin": float(np.max(margins)),
                "protected_decision_p005_splits": int(protected_detects),
                "clean_decision_p005_splits": int(control_detects["clean"]),
                "constant_decision_p005_splits": int(control_detects["constant"]),
            }
        )

    write_csv(args.out_dir / "ridge_validation.csv", val_rows)
    write_csv(args.out_dir / "matched_output_feature_splits.csv", split_rows)
    write_csv(args.out_dir / "matched_output_feature_summary.csv", summary_by_budget)

    metrics = {
        "teacher_feature_dir": str(args.teacher_feature_dir),
        "score_csv": str(args.score_csv),
        "num_teacher_for_scorer": int(x_teacher.shape[0]),
        "num_student_images_per_model": int(args.num_student_images),
        "split_budgets": [int(x) for x in args.split_budgets],
        "best_alpha": float(best["alpha"]),
        "validation": val_rows,
        "teacher_fit_corr": corr(pred_teacher, y_teacher),
        "models": {
            name: {k: v for k, v in values.items() if k not in {"scores", "projections"}}
            for name, values in per_model.items()
        },
        "matched_margin": summary_by_budget,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
