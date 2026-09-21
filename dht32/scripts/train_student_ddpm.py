#!/usr/bin/env python
import argparse
import csv
import json
import math
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, image_size: int = 32):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.paths = sorted(
            p for p in self.image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.paths:
            raise ValueError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a student DDPM from protected teacher samples.")
    parser.add_argument("--teacher_model_dir", type=Path, required=True)
    parser.add_argument("--train_image_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--num_sample_images", type=int, default=64)
    parser.add_argument("--num_sample_steps", type=int, default=1000)
    parser.add_argument("--mixed_precision", choices=["no", "fp16"], default="fp16")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_checkpoints", type=int, default=3)
    return parser.parse_args()


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fresh_student(teacher_model_dir: Path):
    config_path = teacher_model_dir / "unet" / "config.json"
    if not config_path.exists():
        config_path = teacher_model_dir / "config.json"
    config = json.loads(config_path.read_text())
    return UNet2DModel.from_config(config)


@torch.no_grad()
def update_ema(ema_model, model, decay: float):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, buffer in model_buffers.items():
        ema_buffers[name].copy_(buffer)


def tensor_to_uint8(x):
    x = (x.detach().float().clamp(-1, 1) + 1.0) * 127.5
    return x.round().byte().permute(0, 2, 3, 1).cpu().numpy()


def save_grid(samples, path: Path, nrow: int = 8):
    imgs = tensor_to_uint8(samples)
    n, h, w, c = imgs.shape
    nrow = min(nrow, n)
    ncol = math.ceil(n / nrow)
    canvas = Image.new("RGB", (nrow * w, ncol * h))
    for i, arr in enumerate(imgs):
        y, x = divmod(i, nrow)
        canvas.paste(Image.fromarray(arr, "RGB"), (x * w, y * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


@torch.no_grad()
def sample_grid(model, scheduler, device, dtype, args, step: int):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed + step)
    shape = (args.num_sample_images, 3, args.image_size, args.image_size)
    sample = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    scheduler.set_timesteps(args.num_sample_steps)
    for t in scheduler.timesteps:
        timesteps = torch.full((sample.shape[0],), int(t), device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dtype == torch.float16)):
            noise_pred = model(sample, timesteps).sample
        sample = scheduler.step(noise_pred.float(), int(t), sample.float()).prev_sample.to(dtype)
    model.train()
    return sample.float()


def save_checkpoint(out_dir, model, ema_model, optimizer, scaler, global_step, epoch, args):
    latest = out_dir / "checkpoint-latest"
    latest.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(latest / "unet")
    ema_model.save_pretrained(latest / "unet_ema")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "global_step": global_step,
            "epoch": epoch,
        },
        latest / "training_state.pt",
    )

    if args.save_every > 0 and global_step % args.save_every == 0:
        snap = out_dir / f"checkpoint-{global_step:06d}"
        snap.mkdir(parents=True, exist_ok=True)
        ema_model.save_pretrained(snap / "unet_ema")
        torch.save({"global_step": global_step, "epoch": epoch}, snap / "training_state.pt")
        checkpoints = sorted(out_dir.glob("checkpoint-[0-9]*"))
        extra = len(checkpoints) - args.max_checkpoints
        for old in checkpoints[: max(0, extra)]:
            shutil.rmtree(old)


def load_checkpoint(out_dir, model, ema_model, optimizer, scaler, device):
    latest = out_dir / "checkpoint-latest"
    state_path = latest / "training_state.pt"
    if not state_path.exists():
        return 0, 0
    model_loaded = UNet2DModel.from_pretrained(latest / "unet").to(device)
    ema_loaded = UNet2DModel.from_pretrained(latest / "unet_ema").to(device)
    model.load_state_dict(model_loaded.state_dict())
    ema_model.load_state_dict(ema_loaded.state_dict())
    state = torch.load(state_path, map_location=device)
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("global_step", 0)), int(state.get("epoch", 0))


def append_loss(loss_path: Path, row):
    exists = loss_path.exists()
    with loss_path.open("a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "epoch", "loss", "lr", "images_seen", "elapsed_sec"],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    dtype = torch.float16 if device.type == "cuda" and args.mixed_precision == "fp16" else torch.float32

    dataset = ImageFolderDataset(args.train_image_dir, args.image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    model = load_fresh_student(args.teacher_model_dir).to(device)
    ema_model = deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    noise_scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")
    sample_scheduler = DDPMScheduler.from_pretrained(args.teacher_model_dir / "scheduler")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.mixed_precision == "fp16"))

    global_step = 0
    start_epoch = 0
    if args.resume:
        global_step, start_epoch = load_checkpoint(args.out_dir, model, ema_model, optimizer, scaler, device)

    metadata = {
        "dataset_size": len(dataset),
        "num_batches_per_epoch": len(dataloader),
        "device": str(device),
        "dtype": str(dtype),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "start_global_step": global_step,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)

    model.train()
    start_time = time.time()
    epoch = start_epoch
    loss_smooth = None

    while global_step < args.max_train_steps:
        epoch += 1
        for clean_images in dataloader:
            if global_step >= args.max_train_steps:
                break
            clean_images = clean_images.to(device, non_blocking=True)
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (clean_images.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda" and args.mixed_precision == "fp16"),
            ):
                pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(pred.float(), noise.float())

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            update_ema(ema_model, model, args.ema_decay)

            global_step += 1
            loss_value = float(loss.detach().item())
            loss_smooth = loss_value if loss_smooth is None else 0.98 * loss_smooth + 0.02 * loss_value

            if global_step % args.log_every == 0 or global_step == 1:
                elapsed = time.time() - start_time
                row = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": loss_value,
                    "lr": optimizer.param_groups[0]["lr"],
                    "images_seen": global_step * args.batch_size,
                    "elapsed_sec": elapsed,
                }
                append_loss(args.out_dir / "losses.csv", row)
                print(
                    f"step={global_step} epoch={epoch} loss={loss_value:.6f} "
                    f"smooth={loss_smooth:.6f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

            if args.sample_every > 0 and global_step % args.sample_every == 0:
                samples = sample_grid(ema_model, sample_scheduler, device, dtype, args, global_step)
                save_grid(samples, args.out_dir / "samples" / f"sample_grid_step_{global_step:06d}.png")

            if args.save_every > 0 and global_step % args.save_every == 0:
                save_checkpoint(args.out_dir, model, ema_model, optimizer, scaler, global_step, epoch, args)

    samples = sample_grid(ema_model, sample_scheduler, device, dtype, args, global_step)
    save_grid(samples, args.out_dir / "samples" / f"sample_grid_step_{global_step:06d}.png")
    save_checkpoint(args.out_dir, model, ema_model, optimizer, scaler, global_step, epoch, args)
    ema_model.save_pretrained(args.out_dir / "unet_ema_final")
    model.save_pretrained(args.out_dir / "unet_final")
    print(f"done step={global_step} epoch={epoch}", flush=True)


if __name__ == "__main__":
    main()
