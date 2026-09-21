#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def main():
    parser = argparse.ArgumentParser(description="Plot a single student p-value curve at alpha=0.005.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.005)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(args.metrics.read_text())
    rows = []
    for budget_item in metrics["budgets"]:
        student = budget_item["student"]
        rows.append(
            {
                "budget": int(budget_item["budget"]),
                "per_group": int(budget_item["per_group"]),
                "p_value": float(student["one_sided_p"]),
                "delta": float(student["high_low_delta"]),
                "significant_p005": bool(student["high_low_delta"] > 0 and student["one_sided_p"] < args.alpha),
            }
        )

    with (args.out_dir / "student_p005_single_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["budget", "per_group", "p_value", "delta", "significant_p005"])
        writer.writeheader()
        writer.writerows(rows)

    first = next((row for row in rows if row["significant_p005"]), None)
    stable = None
    for idx, row in enumerate(rows):
        if all(later["significant_p005"] for later in rows[idx:]):
            stable = row
            break

    summary = {
        "alpha": args.alpha,
        "decision_rule": "student delta > 0 and one-sided Welch p < 0.005",
        "first_significant_budget": first["budget"] if first else None,
        "first_significant_per_group": first["per_group"] if first else None,
        "first_significant_p": first["p_value"] if first else None,
        "first_significant_delta": first["delta"] if first else None,
        "stable_from_budget": stable["budget"] if stable else None,
        "stable_from_per_group": stable["per_group"] if stable else None,
    }
    (args.out_dir / "student_p005_single_curve_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    budgets = [row["budget"] for row in rows]
    p_values = [max(row["p_value"], 1e-8) for row in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    ax.plot(
        budgets,
        p_values,
        marker="o",
        color="#dc2626",
        linewidth=2.5,
        markersize=5,
        label="Student 30k p-value",
    )
    ax.axhline(args.alpha, color="black", linestyle="--", linewidth=1.4, label="p=0.005")
    if first:
        ax.axvline(
            first["budget"],
            color="#16a34a",
            linestyle="--",
            linewidth=1.4,
            label=f"first pass: {first['budget']} queries",
        )
        ax.scatter(
            [first["budget"]],
            [first["p_value"]],
            s=75,
            color="#16a34a",
            edgecolor="white",
            linewidth=1.2,
            zorder=5,
        )
        ax.annotate(
            f"{first['budget']} queries\np={first['p_value']:.3g}",
            xy=(first["budget"], first["p_value"]),
            xytext=(first["budget"] + 45, first["p_value"] * 2.3),
            arrowprops={"arrowstyle": "->", "color": "#166534"},
            fontsize=10,
            color="#166534",
        )

    ax.set_yscale("log")
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlim(0, 1030)
    ax.set_xticks([20, 100, 200, 300, 400, 500, 590, 650, 800, 1000])
    ax.tick_params(axis="x", labelrotation=35)
    ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
    ax.set_xlabel("Total key-query samples")
    ax.set_ylabel("One-sided Welch p-value")
    ax.set_title("Minimum key-query samples for significant watermark inheritance")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out_dir / "student_p005_single_curve.png", bbox_inches="tight")
    fig.savefig(args.out_dir / "student_p005_single_curve.pdf", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
