#!/usr/bin/env python
import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DHT-64 detector after output post-processing.")
    parser.add_argument("--image_dir", type=Path, required=True)
    parser.add_argument("--state_path", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--query_count", type=int, default=1024)
    parser.add_argument("--per_group", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--dct_u", type=int, default=20)
    parser.add_argument("--dct_v", type=int, default=20)
    parser.add_argument("--wrong_dct", type=str, default="8,9;31,29;24,24")
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


def setup_imports():
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    from edm64_featurepool_blackbox import (
        InceptionFeature,
        features_for_images,
        group_summary,
        make_dct_bias,
        projection_on_bias,
        query_curve,
        random_key_false_rate,
        score_from_features,
        score_projection_tests,
        summarize,
    )

    return {
        "InceptionFeature": InceptionFeature,
        "features_for_images": features_for_images,
        "group_summary": group_summary,
        "make_dct_bias": make_dct_bias,
        "projection_on_bias": projection_on_bias,
        "query_curve": query_curve,
        "random_key_false_rate": random_key_false_rate,
        "score_from_features": score_from_features,
        "score_projection_tests": score_projection_tests,
        "summarize": summarize,
    }


def list_images(image_dir, limit):
    paths = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    paths = paths[:limit]
    if len(paths) < limit:
        raise ValueError(f"Expected {limit} images in {image_dir}, found {len(paths)}")
    return paths


def postprocess_image(img, variant, rng):
    if variant == "identity":
        return img
    if variant == "jpeg75":
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if variant == "resize48":
        return img.resize((48, 48), Image.Resampling.BICUBIC).resize((64, 64), Image.Resampling.BICUBIC)
    if variant == "blur06":
        return img.filter(ImageFilter.GaussianBlur(radius=0.6))
    if variant == "noise002":
        arr = np.asarray(img, dtype=np.float32)
        arr = arr + rng.normal(0.0, 0.02 * 255.0, size=arr.shape)
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")
    raise ValueError(f"Unknown variant: {variant}")


def load_variant(paths, variant, image_size, seed):
    rng = np.random.default_rng(seed)
    tensors = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        img = postprocess_image(img, variant, rng)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


def evaluate_variant(name, images, state, args, funcs, device):
    feat_model = funcs["InceptionFeature"](device)
    feats = funcs["features_for_images"](feat_model, images, args.batch_size, device)
    scores, distances, anchor_ids = funcs["score_from_features"](
        feats, state["anchors"], state["center"], state["temperature"]
    )
    order = torch.argsort(scores)
    low_idx = order[: args.per_group].tolist()
    high_idx = order[-args.per_group :].tolist()
    groups = ["unused"] * len(images)
    for idx in low_idx:
        groups[idx] = "low"
    for idx in high_idx:
        groups[idx] = "high"
    used_mask = [g != "unused" for g in groups]
    used_groups = [g for g in groups if g != "unused"]

    bias_specs = [(args.dct_u, args.dct_v, "correct")]
    for item in args.wrong_dct.split(";"):
        u, v = item.split(",")
        bias_specs.append((int(u), int(v), f"wrong_{u}_{v}"))

    projection_tests = []
    correct_proj_used = None
    correct_proj_all = None
    for u, v, label in bias_specs:
        bias = funcs["make_dct_bias"](3, args.image_size, u, v, device).cpu()
        values_all = funcs["projection_on_bias"](images, bias)
        values_used = np.asarray([v for v, keep in zip(values_all, used_mask) if keep], dtype=np.float64)
        projection_tests.append(funcs["group_summary"](values_used, used_groups, label))
        if label == "correct":
            correct_proj_used = values_used
            correct_proj_all = values_all

    curve_path = args.out_dir / f"query_curve_{name}.csv"
    curve_rows = funcs["query_curve"](correct_proj_used, used_groups, curve_path)
    return {
        "variant": name,
        "score_stats_all_queries": funcs["summarize"](scores.numpy()),
        "score_stats_selected": funcs["summarize"](scores.numpy()[used_mask]),
        "projection_tests": projection_tests,
        "score_projection_correlation": funcs["score_projection_tests"](scores.numpy(), correct_proj_all),
        "random_key": funcs["random_key_false_rate"](correct_proj_used),
        "query_curve": curve_rows,
    }


def main():
    args = parse_args()
    funcs = setup_imports()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    state = torch.load(args.state_path, map_location="cpu")
    paths = list_images(args.image_dir, args.query_count)
    variants = ["identity", "jpeg75", "resize48", "blur06", "noise002"]
    results = []
    for idx, variant in enumerate(variants):
        print(f"postprocess {variant}", flush=True)
        images = load_variant(paths, variant, args.image_size, args.seed + idx)
        results.append(evaluate_variant(variant, images, state, args, funcs, device))

    metrics = {
        "image_dir": str(args.image_dir),
        "state_path": str(args.state_path),
        "query_count": args.query_count,
        "per_group": args.per_group,
        "bias_dct": [args.dct_u, args.dct_v],
        "variants": results,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out_dir / "postprocess_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "variant",
            "delta",
            "p",
            "decision_p005",
            "pearson_r",
            "pearson_p",
            "spearman_r",
            "spearman_p",
            "random_key_fpr",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            correct = next(x for x in item["projection_tests"] if x["projection"] == "correct")
            corr = item["score_projection_correlation"]
            writer.writerow({
                "variant": item["variant"],
                "delta": correct["delta"],
                "p": correct["one_sided_p"],
                "decision_p005": correct["decision_p005"],
                "pearson_r": corr["pearson_r"],
                "pearson_p": corr["pearson_p_greater"],
                "spearman_r": corr["spearman_r"],
                "spearman_p": corr["spearman_p_greater"],
                "random_key_fpr": item["random_key"]["false_detection_rate_p005"],
            })
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
