#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_mae(clean_dir, protected_dir):
    clean_paths = sorted(Path(clean_dir).glob("*.png"))
    protected_paths = sorted(Path(protected_dir).glob("*.png"))
    if len(clean_paths) != len(protected_paths):
        raise ValueError("clean/protected image counts differ")
    vals = []
    for clean_path, protected_path in zip(clean_paths, protected_paths):
        c = np.asarray(Image.open(clean_path).convert("RGB"), dtype=np.float32) / 255.0
        p = np.asarray(Image.open(protected_path).convert("RGB"), dtype=np.float32) / 255.0
        vals.append(float(np.mean(np.abs(c - p))))
    return float(np.mean(vals))


def positive_one_sided_from_two_sided(t_value, p_two_sided):
    if t_value >= 0:
        return float(p_two_sided / 2.0)
    return float(1.0 - p_two_sided / 2.0)


def main():
    parser = argparse.ArgumentParser(description="Summarize DHT-32 bias-strength ablation.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--s002_dir", type=Path, required=True)
    parser.add_argument("--s003_eval_metrics", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    s002_metrics = json.loads((args.s002_dir / "metrics.json").read_text())
    s003_metrics = json.loads(args.s003_eval_metrics.read_text())
    s003_teacher_clean = s003_metrics["query_score_watermark"]["teacher_clean"]
    s003_teacher_protected = s003_metrics["query_score_watermark"]["teacher_protected"]

    rows = [
        {
            "bias_strength": 0.0,
            "source": "teacher_clean_s003_eval",
            "high_low_gap": float(s003_teacher_clean["high_low_delta"]),
            "one_sided_p": positive_one_sided_from_two_sided(
                float(s003_teacher_clean["high_low_t"]),
                float(s003_teacher_clean["high_low_p"]),
            ),
            "paired_mae": 0.0,
            "paired_projection_shift": 0.0,
            "num_images": int(s003_metrics["num_images"]),
        },
        {
            "bias_strength": 0.02,
            "source": "exp_sigmoid_fixedbias_2000_K8_q1090_L50_1000step",
            "high_low_gap": float(s002_metrics["protected_high_low_delta"]),
            "one_sided_p": positive_one_sided_from_two_sided(
                float(s002_metrics["protected_high_low_t"]),
                float(s002_metrics["protected_high_low_p"]),
            ),
            "paired_mae": image_mae(args.s002_dir / "images_clean", args.s002_dir / "images_protected"),
            "paired_projection_shift": float(s002_metrics["paired_projection_delta_mean"]),
            "num_images": 2000,
        },
        {
            "bias_strength": 0.03,
            "source": "eval_student_teacher_2000_s003_1000step_ema_30k",
            "high_low_gap": float(s003_teacher_protected["high_low_delta"]),
            "one_sided_p": positive_one_sided_from_two_sided(
                float(s003_teacher_protected["high_low_t"]),
                float(s003_teacher_protected["high_low_p"]),
            ),
            "paired_mae": float(s003_metrics["quality"]["paired_mae_teacher_clean_vs_teacher_protected"]),
            "paired_projection_shift": float(s003_teacher_protected["projection_mean"] - s003_teacher_clean["projection_mean"]),
            "num_images": int(s003_metrics["num_images"]),
        },
    ]

    with (args.out_dir / "strength_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "strength_ablation_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        x = [row["bias_strength"] for row in rows]
        gaps = [row["high_low_gap"] for row in rows]
        maes = [row["paired_mae"] for row in rows]
        pvals = [max(row["one_sided_p"], 1e-8) for row in rows]

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=180)
        axes[0].plot(x, gaps, color="#dc2626", marker="o", linewidth=2.4, label="high-low gap")
        axes[0].axhline(0.0, color="#111827", linestyle="--", linewidth=1)
        axes[0].set_xlabel("Bias strength")
        axes[0].set_ylabel("Projection gap (high - low)")
        axes[0].set_title("Watermark signal")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(x, maes, color="#2563eb", marker="s", linewidth=2.4, label="paired MAE")
        axes[1].set_xlabel("Bias strength")
        axes[1].set_ylabel("Paired image MAE")
        axes[1].set_title("Image perturbation")
        axes[1].grid(True, alpha=0.25)
        for ax in axes:
            ax.set_xticks(x)
        fig.suptitle("DHT-32 bias-strength ablation", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.out_dir / "strength_ablation_curve.png", bbox_inches="tight")
        fig.savefig(args.out_dir / "strength_ablation_curve.pdf", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=180)
        ax.plot(x, pvals, color="#7c3aed", marker="o", linewidth=2.4)
        ax.axhline(0.005, color="#111827", linestyle="--", linewidth=1.2, label="p=0.005")
        ax.set_yscale("log")
        ax.set_xlabel("Bias strength")
        ax.set_ylabel("One-sided p-value")
        ax.set_title("Teacher-level detectability")
        ax.grid(True, which="both", alpha=0.25)
        ax.set_xticks(x)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(args.out_dir / "strength_ablation_pvalue.png", bbox_inches="tight")
        fig.savefig(args.out_dir / "strength_ablation_pvalue.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
