#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Output-feature score-regression verification for DHT-32.")
    parser.add_argument("--fid_feature_dir", type=Path, required=True)
    parser.add_argument("--score_csv", type=Path, required=True)
    parser.add_argument("--watermark_state", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[200, 400, 650, 1000])
    parser.add_argument("--train_fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def read_scores(path, n):
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["index"]), float(row["score"])))
    rows.sort(key=lambda r: r[0])
    scores = np.asarray([s for _i, s in rows[:n]], dtype=np.float64)
    if scores.shape[0] < n:
        raise ValueError(f"Need {n} scores, found {scores.shape[0]}")
    return scores


def load_images_projection(image_dir, bias):
    paths = sorted(p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    b = bias.float()
    b_unit = b / b.flatten().norm().clamp_min(1e-8)
    vals = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        vals.append(float((x * b_unit).flatten().sum().item()))
    return np.asarray(vals, dtype=np.float64)


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


def summarize(label, pred_score, projection, budgets):
    order = np.argsort(pred_score)
    rows = []
    for budget in sorted(budgets):
        k = budget // 2
        low_idx = order[:k]
        high_idx = order[-k:][::-1]
        low = projection[low_idx]
        high = projection[high_idx]
        stat, pvalue = one_sided_welch(high, low)
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
                "pred_low_mean": float(pred_score[low_idx].mean()),
                "pred_high_mean": float(pred_score[high_idx].mean()),
                "low_projection_mean": float(low.mean()),
                "high_projection_mean": float(high.mean()),
                "delta": delta,
                "one_sided_t": stat,
                "one_sided_p": pvalue,
                "cohen_d": float(delta / pooled) if pooled > 0 else 0.0,
                "decision_p005": bool(delta > 0 and pvalue < 0.005),
            }
        )
    return rows


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

    feature_dir = args.fid_feature_dir
    x_teacher = np.load(feature_dir / "features_teacher_protected.npy").astype(np.float64)
    y_teacher = read_scores(args.score_csv, x_teacher.shape[0])
    x_student_protected = np.load(feature_dir / "features_student_protected.npy").astype(np.float64)
    x_student_clean = np.load(feature_dir / "features_student_clean.npy").astype(np.float64)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(x_teacher.shape[0])
    n_train = int(round(args.train_fraction * x_teacher.shape[0]))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    (x_train, x_val), train_mean, train_std = standardize(x_teacher[train_idx], x_teacher[train_idx], x_teacher[val_idx])
    y_train = y_teacher[train_idx]
    y_val = y_teacher[val_idx]
    alpha_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    val_rows = []
    best = None
    for alpha in alpha_grid:
        beta, y_mean = fit_ridge(x_train, y_train, alpha)
        pred_val = predict_ridge(x_val, beta, y_mean)
        mse = float(np.mean((pred_val - y_val) ** 2))
        row = {"alpha": alpha, "val_mse": mse, "val_corr": corr(pred_val, y_val)}
        val_rows.append(row)
        if best is None or mse < best["val_mse"]:
            best = row

    (x_all, xsp, xsc), all_mean, all_std = standardize(x_teacher, x_teacher, x_student_protected, x_student_clean)
    beta, y_mean = fit_ridge(x_all, y_teacher, float(best["alpha"]))
    pred_teacher = predict_ridge(x_all, beta, y_mean)
    pred_sp = predict_ridge(xsp, beta, y_mean)
    pred_sc = predict_ridge(xsc, beta, y_mean)

    state = torch.load(args.watermark_state, map_location="cpu")
    bias = state["bias"].float().squeeze(0)
    proj_sp = load_images_projection(feature_dir / "images_student_protected", bias)
    proj_sc = load_images_projection(feature_dir / "images_student_clean", bias)
    if proj_sp.shape[0] != pred_sp.shape[0] or proj_sc.shape[0] != pred_sc.shape[0]:
        raise ValueError("Feature and image counts do not match")

    budget_rows = []
    budget_rows += summarize("student_protected", pred_sp, proj_sp, args.budgets)
    budget_rows += summarize("student_clean", pred_sc, proj_sc, args.budgets)
    write_csv(args.out_dir / "output_feature_regression_budget_summary.csv", budget_rows)
    write_csv(args.out_dir / "ridge_validation.csv", val_rows)

    metrics = {
        "feature_dir": str(feature_dir),
        "num_teacher": int(x_teacher.shape[0]),
        "num_student_protected": int(x_student_protected.shape[0]),
        "num_student_clean": int(x_student_clean.shape[0]),
        "train_fraction": args.train_fraction,
        "best_alpha": float(best["alpha"]),
        "validation": val_rows,
        "teacher_fit_corr": corr(pred_teacher, y_teacher),
        "student_protected_score_projection_corr": corr(pred_sp, proj_sp),
        "student_clean_score_projection_corr": corr(pred_sc, proj_sc),
        "budgets": budget_rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
