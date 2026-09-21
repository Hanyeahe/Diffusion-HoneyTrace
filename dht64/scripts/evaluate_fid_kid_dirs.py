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
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_named_dir(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Directory name cannot be empty")
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Inception FID/KID between image directories.")
    parser.add_argument("--reference", type=parse_named_dir, required=True)
    parser.add_argument("--compare", type=parse_named_dir, action="append", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=4096)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--kid_subsets", type=int, default=20)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260907)
    return parser.parse_args()


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, image_size, limit=None):
        self.image_dir = Path(image_dir)
        self.image_size = int(image_size)
        self.paths = sorted(
            p for p in self.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def build_inception(device):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_features(model, dataset, batch_size, device, label):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    feats = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    started = time.time()
    done = 0
    for batch in loader:
        x = batch.to(device, non_blocking=True).float()
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        feat = model(x)
        if isinstance(feat, tuple):
            feat = feat[0]
        feats.append(feat.detach().float().cpu())
        done += batch.shape[0]
        print(f"features {label}: {done}/{len(dataset)}, elapsed={time.time() - started:.1f}s", flush=True)
    return torch.cat(feats, dim=0).numpy().astype(np.float64)


def fid_from_features_torch(real, fake, device):
    x = torch.from_numpy(real).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(fake).to(device=device, dtype=torch.float32)
    mu_x = x.mean(dim=0, keepdim=True)
    mu_y = y.mean(dim=0, keepdim=True)
    xc = x - mu_x
    yc = y - mu_y
    n = max(x.shape[0] - 1, 1)
    m = max(y.shape[0] - 1, 1)
    cross = xc @ yc.T
    trace_sqrt = torch.linalg.svdvals(cross).sum() / math.sqrt(float(n * m))
    trace_x = xc.square().sum() / float(n)
    trace_y = yc.square().sum() / float(m)
    diff = (mu_x - mu_y).flatten()
    fid = diff.dot(diff) + trace_x + trace_y - 2.0 * trace_sqrt
    return float(fid.detach().cpu())


def kid_from_features(real, fake, subsets=20, subset_size=1000, seed=20260907):
    rng = np.random.default_rng(seed)
    n = real.shape[0]
    m = fake.shape[0]
    d = real.shape[1]
    subset_size = min(subset_size, n, m)
    scores = []
    for _ in range(subsets):
        x = real[rng.choice(n, subset_size, replace=False)]
        y = fake[rng.choice(m, subset_size, replace=False)]
        k_xx = (x @ x.T / d + 1.0) ** 3
        k_yy = (y @ y.T / d + 1.0) ** 3
        k_xy = (x @ y.T / d + 1.0) ** 3
        sum_xx = (k_xx.sum() - np.trace(k_xx)) / (subset_size * (subset_size - 1))
        sum_yy = (k_yy.sum() - np.trace(k_yy)) / (subset_size * (subset_size - 1))
        sum_xy = k_xy.mean()
        scores.append(sum_xx + sum_yy - 2.0 * sum_xy)
    scores = np.asarray(scores, dtype=np.float64)
    return float(scores.mean()), float(scores.std(ddof=1))


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    named_dirs = [args.reference] + args.compare
    config = {
        "reference": [args.reference[0], str(args.reference[1])],
        "compare": [[name, str(path)] for name, path in args.compare],
        "num_images": args.num_images,
        "image_size": args.image_size,
        "feature_batch_size": args.feature_batch_size,
        "kid_subsets": args.kid_subsets,
        "kid_subset_size": args.kid_subset_size,
        "seed": args.seed,
        "device": str(device),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    feature_model = build_inception(device)
    features = {}
    for name, path in named_dirs:
        feature_path = args.out_dir / f"features_{name}.npy"
        if feature_path.exists():
            features[name] = np.load(feature_path)
            print(f"reuse features {name}: {features[name].shape}", flush=True)
            continue
        dataset = ImageFolderDataset(path, image_size=args.image_size, limit=args.num_images)
        features[name] = extract_features(feature_model, dataset, args.feature_batch_size, device, name)
        np.save(feature_path, features[name])

    ref_name, _ = args.reference
    rows = []
    for idx, (name, _) in enumerate(args.compare):
        print(f"compare {ref_name} -> {name}", flush=True)
        fid = fid_from_features_torch(features[ref_name], features[name], device)
        kid_mean, kid_std = kid_from_features(
            features[ref_name],
            features[name],
            subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            seed=args.seed + idx,
        )
        row = {
            "reference": ref_name,
            "model": name,
            "num_images": args.num_images,
            "fid_inception": fid,
            "kid_inception_mean": kid_mean,
            "kid_inception_std": kid_std,
            "kid_x1000_mean": kid_mean * 1000.0,
            "kid_x1000_std": kid_std * 1000.0,
        }
        rows.append(row)
        print(f"done {ref_name} -> {name}: FID={fid:.4f}, KIDx1000={kid_mean * 1000.0:.4f}", flush=True)

    with (args.out_dir / "fid_kid_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "reference",
            "model",
            "num_images",
            "fid_inception",
            "kid_inception_mean",
            "kid_inception_std",
            "kid_x1000_mean",
            "kid_x1000_std",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "backend": "torchvision_inception_v3_imagenet",
        "num_images": args.num_images,
        "rows": rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
