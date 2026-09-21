#!/usr/bin/env bash
set -euo pipefail

cd /private/projects/DHT-32

PY=/opt/conda/envs/PyTorch-2.4.1/bin/python
TEACHER=/private/projects/DHT-32/ddpm-cifar-32
STUDENT=/private/projects/DHT-32/outputs/student_clean_30000_fromscratch_100k
WM=/private/projects/DHT-32/outputs/dataset_protected_15000_sigmoid_K8_s003_L50_1000step/watermark_state.pt
SELECTED=/private/projects/DHT-32/outputs/eval_key_queries_10000c_curve_20_1000_ema_30k_reuse/selected_key_queries.csv
EVAL_RANDOM=/private/projects/DHT-32/outputs/eval_student_clean_30k_teacher_2000_s003_1000step_ema
EVAL_KEY=/private/projects/DHT-32/outputs/eval_key_queries_clean_student_30k_10000c_curve_20_1000_ema

echo "[watch-clean] waiting for $STUDENT/unet_ema_final/config.json"
while [ ! -f "$STUDENT/unet_ema_final/config.json" ]; do
  date
  if ! pgrep -af 'run_clean_student_30k.sh|generate_clean_dataset.py|train_student_ddpm.py' >/dev/null; then
    echo "[watch-clean] clean student training process is not running and final model is missing"
    exit 1
  fi
  sleep 300
done

echo "[watch-clean] clean student final found"

if [ ! -f "$EVAL_RANDOM/metrics.json" ]; then
  "$PY" scripts/evaluate_student_teacher.py \
    --teacher_model_dir "$TEACHER" \
    --student_unet_dir "$STUDENT/unet_ema_final" \
    --watermark_state "$WM" \
    --out_dir "$EVAL_RANDOM" \
    --num_images 2000 \
    --batch_size 128 \
    --num_steps 1000 \
    --seed 20260901
else
  echo "[watch-clean] skip random eval, metrics exist"
fi

if [ ! -f "$EVAL_KEY/metrics.json" ]; then
  "$PY" scripts/evaluate_student_only_key_queries.py \
    --teacher_model_dir "$TEACHER" \
    --student_unet_dir "$STUDENT/unet_ema_final" \
    --watermark_state "$WM" \
    --selected_key_queries "$SELECTED" \
    --out_dir "$EVAL_KEY" \
    --budgets 20 40 60 80 100 150 200 250 300 350 400 450 500 550 590 650 700 800 900 1000 \
    --batch_size 128 \
    --num_steps 1000
else
  echo "[watch-clean] skip key-query eval, metrics exist"
fi

echo "[watch-clean] done"
