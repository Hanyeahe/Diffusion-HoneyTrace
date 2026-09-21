#!/usr/bin/env python
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap key-query detection stability.")
    parser.add_argument("--per_key_query", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model", choices=["teacher_clean", "teacher_protected", "student"], default="student")
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 590, 650, 700, 800, 900, 1000])
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def one_sided_welch(high, low):
    try:
        from scipy.stats import ttest_ind

        stat, pvalue = ttest_ind(high, low, equal_var=False, alternative="greater")
        return float(stat), float(pvalue)
    except Exception:
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        denom = math.sqrt(high.var(ddof=1) / len(high) + low.var(ddof=1) / len(low))
        stat = 0.0 if denom == 0 else (high.mean() - low.mean()) / denom
        pvalue = 0.5 * math.erfc(stat / math.sqrt(2.0))
        return float(stat), float(pvalue)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    proj_key = f"{args.model}_projection"
    low = []
    high = []
    with args.per_key_query.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["group"] == "low":
                low.append(float(row[proj_key]))
            elif row["group"] == "high":
                high.append(float(row[proj_key]))
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if len(low) != len(high):
        raise ValueError("Expected equal low/high group sizes.")

    rng = np.random.default_rng(args.seed)
    summary_rows = []
    repeat_rows = []
    for budget in args.budgets:
        k = budget // 2
        if k > len(low):
            continue
        deltas = []
        pvalues = []
        decisions = []
        for repeat in range(args.repeats):
            low_idx = rng.choice(len(low), size=k, replace=False)
            high_idx = rng.choice(len(high), size=k, replace=False)
            low_sample = low[low_idx]
            high_sample = high[high_idx]
            delta = float(high_sample.mean() - low_sample.mean())
            _stat, pvalue = one_sided_welch(high_sample, low_sample)
            decision = bool(delta > 0 and pvalue < args.alpha)
            deltas.append(delta)
            pvalues.append(pvalue)
            decisions.append(decision)
            repeat_rows.append(
                {
                    "budget": budget,
                    "per_group": k,
                    "repeat": repeat,
                    "delta": delta,
                    "p_value": pvalue,
                    "decision": decision,
                }
            )
        pvalues_arr = np.asarray(pvalues)
        deltas_arr = np.asarray(deltas)
        decisions_arr = np.asarray(decisions, dtype=np.float64)
        summary_rows.append(
            {
                "budget": budget,
                "per_group": k,
                "decision_rate": float(decisions_arr.mean()),
                "median_p": float(np.median(pvalues_arr)),
                "q10_p": float(np.quantile(pvalues_arr, 0.10)),
                "q90_p": float(np.quantile(pvalues_arr, 0.90)),
                "mean_delta": float(deltas_arr.mean()),
                "q10_delta": float(np.quantile(deltas_arr, 0.10)),
                "q90_delta": float(np.quantile(deltas_arr, 0.90)),
            }
        )

    with (args.out_dir / f"bootstrap_{args.model}_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.out_dir / f"bootstrap_{args.model}_repeats.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(repeat_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(repeat_rows)

    try:
        import matplotlib.pyplot as plt

        budgets = [row["budget"] for row in summary_rows]
        rates = [row["decision_rate"] for row in summary_rows]
        med_p = [row["median_p"] for row in summary_rows]
        q10_p = [row["q10_p"] for row in summary_rows]
        q90_p = [row["q90_p"] for row in summary_rows]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=180)
        axes[0].plot(budgets, rates, color="#dc2626", marker="o", linewidth=2.4)
        axes[0].axhline(0.95, color="#111827", linestyle="--", linewidth=1.2, label="95% repeats")
        axes[0].set_ylim(-0.02, 1.02)
        axes[0].set_xlabel("Total key-query samples")
        axes[0].set_ylabel(f"Decision rate at p<{args.alpha:g}")
        axes[0].set_title("Bootstrap detection stability")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(frameon=False)

        axes[1].plot(budgets, med_p, color="#2563eb", marker="o", linewidth=2.4, label="median p")
        axes[1].fill_between(budgets, q10_p, q90_p, color="#93c5fd", alpha=0.35, label="10-90% interval")
        axes[1].axhline(args.alpha, color="#111827", linestyle="--", linewidth=1.2, label=f"p={args.alpha:g}")
        axes[1].set_yscale("log")
        axes[1].set_ylim(1e-6, 1.0)
        axes[1].set_xlabel("Total key-query samples")
        axes[1].set_ylabel("One-sided Welch p-value")
        axes[1].set_title("Bootstrap p-value distribution")
        axes[1].grid(True, which="both", alpha=0.25)
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.set_xticks([20, 100, 200, 300, 400, 500, 590, 650, 800, 1000])
            ax.tick_params(axis="x", labelrotation=35)
        fig.suptitle(f"{args.model} key-query bootstrap ({args.repeats} repeats)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.out_dir / f"bootstrap_{args.model}_curve.png", bbox_inches="tight")
        fig.savefig(args.out_dir / f"bootstrap_{args.model}_curve.pdf", bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    first_95 = next((row for row in summary_rows if row["decision_rate"] >= 0.95), None)
    out = {
        "model": args.model,
        "alpha": args.alpha,
        "repeats": args.repeats,
        "group_size": len(low),
        "first_budget_decision_rate_ge_95": first_95["budget"] if first_95 else None,
        "summary": summary_rows,
    }
    (args.out_dir / f"bootstrap_{args.model}_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
