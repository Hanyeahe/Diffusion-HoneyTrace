#!/usr/bin/env bash
set -euo pipefail

cd /private/projects/DHT-64

DATASET="outputs/featurepool_dataset_constant_dct_4096_s300eq_d20_20"
STATE="outputs/featurepool_dataset_protected_4096_s300_d20_20/featurepool_state.pt"
MODEL="models/edm-ffhq-64x64-uncond-vp.pkl"
STUDENT50="outputs/student_edm64_featurepool_constant4096_transfer_50kimg_s300eq_d20_20_ema005"
STUDENT150="outputs/student_edm64_featurepool_constant4096_transfer_150kimg_s300eq_d20_20_ema005"
STUDENT300="outputs/student_edm64_featurepool_constant4096_transfer_300kimg_s300eq_d20_20_ema005"
EVAL="outputs/eval_student_edm64_featurepool_constant4096_transfer_300kimg_key1024_q256_s300eq_d20_20_corr"

echo "[$(date)] FFHQ constant-DCT baseline pipeline start"

if [ ! -f "$DATASET/metrics.json" ]; then
  python scripts/edm64_featurepool_constant_dataset.py \
    --edm_repo edm \
    --network_pkl "$MODEL" \
    --out_dir "$DATASET" \
    --count 4096 \
    --batch_size 32 \
    --num_steps 40 \
    --probe_step 34 \
    --bias_steps 5 \
    --bias_strength 3.0 \
    --alpha 2.0 \
    --constant_power 0.3711280526416022 \
    --dct_u 20 \
    --dct_v 20 \
    --state_in "$STATE" \
    --sample_seed_base 20282000
fi

if [ ! -f "$STUDENT50/network-snapshot-000050.pkl" ]; then
  python edm/train.py \
    --outdir "$STUDENT50" \
    --data "$DATASET/images" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=0.05 \
    --batch=64 \
    --batch-gpu=16 \
    --cbase=128 \
    --cres=1,2,2,2 \
    --lr=0.001 \
    --ema=0.005 \
    --dropout=0.13 \
    --augment=0 \
    --xflip=0 \
    --fp16=1 \
    --ls=1 \
    --bench=1 \
    --cache=1 \
    --workers=4 \
    --nosubdir \
    --tick=2 \
    --snap=5 \
    --dump=999 \
    --seed=20260907 \
    --transfer "$MODEL"
fi

if [ ! -f "$STUDENT150/network-snapshot-000150.pkl" ]; then
  python edm/train.py \
    --outdir "$STUDENT150" \
    --data "$DATASET/images" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=0.15 \
    --batch=64 \
    --batch-gpu=16 \
    --cbase=128 \
    --cres=1,2,2,2 \
    --lr=0.001 \
    --ema=0.005 \
    --dropout=0.13 \
    --augment=0 \
    --xflip=0 \
    --fp16=1 \
    --ls=1 \
    --bench=1 \
    --cache=1 \
    --workers=4 \
    --nosubdir \
    --tick=5 \
    --snap=5 \
    --dump=999 \
    --seed=20260907 \
    --resume "$STUDENT50/training-state-000050.pt"
fi

if [ ! -f "$STUDENT300/network-snapshot-000300.pkl" ]; then
  python edm/train.py \
    --outdir "$STUDENT300" \
    --data "$DATASET/images" \
    --cond=0 \
    --arch=ddpmpp \
    --precond=edm \
    --duration=0.30 \
    --batch=64 \
    --batch-gpu=16 \
    --cbase=128 \
    --cres=1,2,2,2 \
    --lr=0.001 \
    --ema=0.005 \
    --dropout=0.13 \
    --augment=0 \
    --xflip=0 \
    --fp16=1 \
    --ls=1 \
    --bench=1 \
    --cache=1 \
    --workers=4 \
    --nosubdir \
    --tick=5 \
    --snap=5 \
    --dump=15 \
    --seed=20260907 \
    --resume "$STUDENT150/training-state-000150.pt"
fi

if [ ! -f "$EVAL/metrics.json" ]; then
  python scripts/edm64_featurepool_blackbox.py eval_student \
    --edm_repo edm \
    --student_pkl "$STUDENT300/network-snapshot-000300.pkl" \
    --state_path "$STATE" \
    --out_dir "$EVAL" \
    --query_count 1024 \
    --per_group 256 \
    --query_seed_base 20290000 \
    --num_steps 40 \
    --probe_step 34 \
    --batch_size 32 \
    --dct_u 20 \
    --dct_v 20 \
    --wrong_dct "8,9;31,29;24,24"
fi

echo "[$(date)] FFHQ constant-DCT baseline pipeline complete"
