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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import ToTensor


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute standard Inception FID/KID for DHT-32 models.")
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--student_protected_unet_dir", type=Path, required=True)
    parser.add_argument("--student_clean_unet_dir", type=Path, required=True)
    parser.add_argument("--teacher_clean_image_dir", type=Path, required=True)
    parser.add_argument("--teacher_protected_image_dir", type=Path, required=True)
    parser.add_argument("--cifar_root", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--reference_mode", choices=["real", "teacher_clean", "teacher_both"], default="real")
    parser.add_argument("--num_images", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--feature_batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--kid_subsets", type=int, default=100)
    parser.add_argument("--kid_subset_size", type=int, default=1000)
    parser.add_argument("--force_resample_students", action="store_true")
    return parser.parse_args()


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, limit=None):
        self.image_dir = Path(image_dir)
        self.paths = sorted(
            p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if limit is not None:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.size != (32, 32):
            img = img.resize((32, 32), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def save_images(images, out_dir, offset):
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(out_dir / f"{offset + i:06d}.png")


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
    Image.fromarray(canvas).save(path)


def noise_from_seeds(seeds, channels, image_size, device, dtype):
    generators = [torch.Generator(device=device).manual_seed(int(seed)) for seed in seeds]
    shape = (len(seeds), channels, image_size, image_size)
    return randn_tensor(shape, generator=generators, device=device, dtype=dtype), generators


@torch.inference_mode()
def generate_student_images(model_dir, scheduler, image_dir, args, device, dtype, seed_offset):
    image_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p for p in image_dir.glob("*.png"))
    if len(existing) >= args.num_images and not args.force_resample_students:
        print(f"reuse {image_dir}: {len(existing)} images", flush=True)
        return

    for p in existing:
        p.unlink()

    model = UNet2DModel.from_pretrained(model_dir).to(device=device, dtype=dtype).eval()
    image_size = int(model.config.sample_size)
    channels = int(model.config.in_channels)
    scheduler.set_timesteps(args.num_steps, device=device)
    started = time.time()
    done = 0
    first = []
    while done < args.num_images:
        bsz = min(args.batch_size, args.num_images - done)
        seeds = [args.seed + seed_offset + done + i for i in range(bsz)]
        sample, generators = noise_from_seeds(seeds, channels, image_size, device, dtype)
        for timestep in scheduler.timesteps:
            model_output = model(sample, timestep).sample
            sample = scheduler.step(model_output.float(), timestep, sample.float(), generator=generators).prev_sample.to(dtype)
        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        if done == 0:
            first.append(images[:64])
        save_images(images, image_dir, done)
        done += bsz
        print(f"{image_dir.name}: generated {done}/{args.num_images}, elapsed={time.time() - started:.1f}s", flush=True)
    if first:
        save_grid(torch.cat(first, dim=0), image_dir.parent / f"{image_dir.name}_grid.png")


class CIFARImageDataset(Dataset):
    def __init__(self, root, train=True, limit=2000, seed=20260907):
        base = CIFAR10(root=str(root), train=train, download=True)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(base), size=min(limit, len(base)), replace=False)
        self.dataset = base
        self.indices = idx.tolist()
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _label = self.dataset[self.indices[idx]]
        return self.to_tensor(img.convert("RGB"))


def build_inception(device):
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.inference_mode()
def extract_features(model, dataset, batch_size, device, label):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device.type == "cuda"))
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

    # Non-zero eigenvalues of Cov(x)Cov(y) equal those of
    # (X_c Y_c^T)(X_c Y_c^T)^T / ((n-1)(m-1)). Thus the trace of the
    # covariance-product square root is the nuclear norm below.
    cross = xc @ yc.T
    trace_sqrt = torch.linalg.svdvals(cross).sum() / math.sqrt(float(n * m))
    trace_x = xc.square().sum() / float(n)
    trace_y = yc.square().sum() / float(m)
    diff = (mu_x - mu_y).flatten()
    fid = diff.dot(diff) + trace_x + trace_y - 2.0 * trace_sqrt
    return float(fid.detach().cpu())


def kid_from_features(real, fake, subsets=100, subset_size=1000, seed=20260907):
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
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")
    student_protected_dir = args.out_dir / "images_student_protected"
    student_clean_dir = args.out_dir / "images_student_clean"
    generate_student_images(args.student_protected_unet_dir, scheduler, student_protected_dir, args, device, dtype, seed_offset=1000000)
    generate_student_images(args.student_clean_unet_dir, scheduler, student_clean_dir, args, device, dtype, seed_offset=2000000)

    datasets = {
        "teacher_clean": ImageFolderDataset(args.teacher_clean_image_dir, limit=args.num_images),
        "teacher_protected": ImageFolderDataset(args.teacher_protected_image_dir, limit=args.num_images),
        "student_protected": ImageFolderDataset(student_protected_dir, limit=args.num_images),
        "student_clean": ImageFolderDataset(student_clean_dir, limit=args.num_images),
    }
    if args.reference_mode == "real":
        if args.cifar_root is None:
            raise ValueError("--cifar_root is required when --reference_mode real")
        datasets = {
            "cifar10_real_train": CIFARImageDataset(args.cifar_root, train=True, limit=args.num_images, seed=args.seed),
            **datasets,
        }

    features = {}
    feature_model = None
    for name, dataset in datasets.items():
        feature_path = args.out_dir / f"features_{name}.npy"
        if feature_path.exists():
            features[name] = np.load(feature_path)
            print(f"reuse features {name}: {features[name].shape}", flush=True)
        else:
            if feature_model is None:
                feature_model = build_inception(device)
            features[name] = extract_features(feature_model, dataset, args.feature_batch_size, device, name)
            np.save(feature_path, features[name])

    rows = []
    if args.reference_mode == "real":
        comparisons = [
            ("cifar10_real_train", "teacher_clean"),
            ("cifar10_real_train", "teacher_protected"),
            ("cifar10_real_train", "student_protected"),
            ("cifar10_real_train", "student_clean"),
        ]
    elif args.reference_mode == "teacher_clean":
        comparisons = [
            ("teacher_clean", "teacher_protected"),
            ("teacher_clean", "student_protected"),
            ("teacher_clean", "student_clean"),
        ]
    else:
        comparisons = [
            ("teacher_clean", "teacher_protected"),
            ("teacher_clean", "student_protected"),
            ("teacher_clean", "student_clean"),
            ("teacher_protected", "student_protected"),
            ("teacher_protected", "student_clean"),
        ]

    for reference, name in comparisons:
        print(f"compare {reference} -> {name}", flush=True)
        fid = fid_from_features_torch(features[reference], features[name], device)
        kid_mean, kid_std = kid_from_features(
            features[reference],
            features[name],
            subsets=args.kid_subsets,
            subset_size=args.kid_subset_size,
            seed=args.seed + len(rows),
        )
        rows.append(
            {
                "reference": reference,
                "model": name,
                "num_images": args.num_images,
                "fid_inception": fid,
                "kid_inception_mean": kid_mean,
                "kid_inception_std": kid_std,
                "kid_x1000_mean": kid_mean * 1000.0,
                "kid_x1000_std": kid_std * 1000.0,
            }
        )
        print(
            f"done {reference} -> {name}: FID={fid:.4f}, KIDx1000={kid_mean * 1000.0:.4f}",
            flush=True,
        )

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
        "reference_mode": args.reference_mode,
        "rows": rows,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
