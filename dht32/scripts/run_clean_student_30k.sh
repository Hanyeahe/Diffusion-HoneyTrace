#!/usr/bin/env bash
set -euo pipefail

cd /private/projects/DHT-32

PY=/opt/conda/envs/PyTorch-2.4.1/bin/python
TEACHER=/private/projects/DHT-32/ddpm-cifar-32
DATA=/private/projects/DHT-32/outputs/dataset_clean_30000_1000step_seed20263001
IMAGE_DIR="$DATA/images_clean"
STUDENT=/private/projects/DHT-32/outputs/student_clean_30000_fromscratch_100k

mkdir -p "$DATA" "$STUDENT"

COUNT=0
if [ -d "$IMAGE_DIR" ]; then
  COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -type f -name '*.png' | wc -l)
fi

echo "[clean-student-30k] existing clean images: $COUNT"
if [ "$COUNT" -lt 30000 ]; then
  "$PY" scripts/generate_clean_dataset.py \
    --model_dir "$TEACHER" \
    --out_dir "$DATA" \
    --num_images 30000 \
    --batch_size 128 \
    --num_steps 1000 \
    --seed 20263001
else
  echo "[clean-student-30k] skip clean generation"
fi

"$PY" scripts/train_student_ddpm.py \
  --teacher_model_dir "$TEACHER" \
  --train_image_dir "$IMAGE_DIR" \
  --out_dir "$STUDENT" \
  --image_size 32 \
  --batch_size 128 \
  --num_workers 8 \
  --max_train_steps 100000 \
  --learning_rate 1e-4 \
  --ema_decay 0.9999 \
  --save_every 10000 \
  --sample_every 10000 \
  --num_sample_images 64 \
  --num_sample_steps 1000 \
  --mixed_precision fp16 \
  --seed 20260904 \
  --resume

echo "[clean-student-30k] done"
