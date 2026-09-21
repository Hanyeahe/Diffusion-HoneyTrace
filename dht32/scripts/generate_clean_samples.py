#!/usr/bin/env python
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Generate clean DDPM CIFAR-10 samples.")
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def save_grid(images, path, cols=8):
    n = images.shape[0]
    imgs = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
    h, w = imgs.shape[1], imgs.shape[2]
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img
    Image.fromarray(canvas).save(path)


@torch.inference_mode()
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.fp16 and device.type == "cuda" else torch.float32

    unet = UNet2DModel.from_pretrained(args.model_dir, subfolder="unet").to(device=device, dtype=dtype)
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(args.model_dir, subfolder="scheduler")
    scheduler.set_timesteps(args.num_steps, device=device)

    image_size = int(unet.config.sample_size)
    channels = int(unet.config.in_channels)
    all_images = []
    done = 0

    print(
        {
            "model_dir": str(args.model_dir),
            "out_dir": str(args.out_dir),
            "num_images": args.num_images,
            "batch_size": args.batch_size,
            "num_steps": args.num_steps,
            "device": str(device),
            "dtype": str(dtype),
        },
        flush=True,
    )

    while done < args.num_images:
        bsz = min(args.batch_size, args.num_images - done)
        gen = torch.Generator(device=device).manual_seed(args.seed + done)
        sample = torch.randn(bsz, channels, image_size, image_size, generator=gen, device=device, dtype=dtype)

        for timestep in scheduler.timesteps:
            model_output = unet(sample, timestep).sample
            sample = scheduler.step(model_output, timestep, sample, generator=gen).prev_sample

        images = (sample.float() / 2.0 + 0.5).clamp(0.0, 1.0).cpu()
        all_images.append(images)

        arr = (images.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
        for i, img in enumerate(arr):
            Image.fromarray(img).save(args.out_dir / f"{done + i:06d}.png")

        done += bsz
        print(f"generated {done}/{args.num_images}", flush=True)

    save_grid(torch.cat(all_images, dim=0), args.out_dir / "grid.png")
    print(f"saved {args.out_dir / 'grid.png'}", flush=True)


if __name__ == "__main__":
    main()
