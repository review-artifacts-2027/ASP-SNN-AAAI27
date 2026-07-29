#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs checkpoints

GPU_SHAPENET="${GPU_SHAPENET:-0}"
GPU_SCANOBJ="${GPU_SCANOBJ:-1}"
GPU_S3DIS="${GPU_S3DIS:-2}"
EPOCHS="${EPOCHS:-500}"

echo "Launching point-cloud full trainings in parallel"
echo "ShapeNetPart -> GPU ${GPU_SHAPENET}"
echo "ScanObjectNN -> GPU ${GPU_SCANOBJ}"
echo "S3DIS        -> GPU ${GPU_S3DIS}"

GPU="${GPU_SHAPENET}" EPOCHS="${EPOCHS}" \
  LOG_FILE="logs/shapenet_full_gpu${GPU_SHAPENET}.log" \
  bash scripts/train_shapenet_full.sh &
PID_SHAPENET=$!

GPU="${GPU_SCANOBJ}" EPOCHS="${EPOCHS}" \
  LOG_FILE="logs/scanobj_full_gpu${GPU_SCANOBJ}.log" \
  bash scripts/train_scanobj_full.sh &
PID_SCANOBJ=$!

GPU="${GPU_S3DIS}" EPOCHS="${EPOCHS}" \
  LOG_FILE="logs/s3dis_full_gpu${GPU_S3DIS}.log" \
  bash scripts/train_s3dis_full.sh &
PID_S3DIS=$!

wait "${PID_SHAPENET}"
wait "${PID_SCANOBJ}"
wait "${PID_S3DIS}"

echo "All point-cloud full trainings finished."
