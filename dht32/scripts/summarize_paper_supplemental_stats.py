#!/usr/bin/env python
import csv
import json
import math
from pathlib import Path


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_rows(rows, **conditions):
    out = []
    for row in rows:
        ok = True
        for key, value in conditions.items():
            if str(row.get(key)) != str(value):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def main():
    root = Path("/private/projects/DHT-32/outputs")
    out_dir = root / "paper_supplemental_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    random_rows = read_csv(root / "wrongkey_postprocess_protected_student_30k_1000q" / "random_key_summary.csv")
    post_rows = read_csv(root / "wrongkey_postprocess_protected_student_30k_1000q" / "wrongkey_postprocess_summary.csv")
    fid_rows = read_csv(root / "fid_kid_inception_2000_students_teacher_real" / "fid_kid_summary.csv")

    random_ci = []
    for row in random_rows:
        repeats = int(row["repeats"])
        fpr = float(row["false_detection_rate_p005"])
        false_count = int(round(fpr * repeats))
        lo, hi = wilson_ci(false_count, repeats)
        random_ci.append(
            {
                "budget": int(row["budget"]),
                "per_group": int(row["per_group"]),
                "false_count": false_count,
                "repeats": repeats,
                "fpr": fpr,
                "ci95_low": lo,
                "ci95_high": hi,
                "median_p": float(row["p_q50"]),
                "median_delta": float(row["delta_q50"]),
            }
        )

    post_keep = []
    for name in ["identity", "jpeg_q90", "jpeg_q70", "gaussian_0.005", "gaussian_0.01", "resize_28", "blur_r05"]:
        rows = select_rows(post_rows, postprocess=name, bias="correct_dct_4_5", grouping="correct_key", budget="400")
        if rows:
            row = rows[0]
            post_keep.append(
                {
                    "postprocess": name,
                    "budget": int(row["budget"]),
                    "delta": float(row["delta"]),
                    "p": float(row["one_sided_p"]),
                    "decision_p005": row["decision_p005"],
                }
            )

    wrong_keep = []
    for dct in ["wrong_dct_1_7", "wrong_dct_7_1", "wrong_dct_6_6"]:
        rows = select_rows(post_rows, postprocess="identity", bias=dct, grouping="correct_key", budget="650")
        if rows:
            row = rows[0]
            wrong_keep.append(
                {
                    "bias": dct,
                    "budget": int(row["budget"]),
                    "delta": float(row["delta"]),
                    "p": float(row["one_sided_p"]),
                    "decision_p005": row["decision_p005"],
                }
            )

    quality_keep = []
    for row in fid_rows:
        quality_keep.append(
            {
                "reference": row["reference"],
                "model": row["model"],
                "num_images": int(row["num_images"]),
                "fid": float(row["fid_inception"]),
                "kid_x1000_mean": float(row["kid_x1000_mean"]),
                "kid_x1000_std": float(row["kid_x1000_std"]),
            }
        )

    summary = {
        "random_key_fpr_ci": random_ci,
        "postprocess_400q": post_keep,
        "wrong_key_650q": wrong_keep,
        "fid_kid": quality_keep,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "random_key_fpr_ci.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(random_ci[0].keys()))
        writer.writeheader()
        writer.writerows(random_ci)
    with (out_dir / "postprocess_400q.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(post_keep[0].keys()))
        writer.writeheader()
        writer.writerows(post_keep)
    with (out_dir / "wrong_key_650q.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(wrong_keep[0].keys()))
        writer.writeheader()
        writer.writerows(wrong_keep)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
