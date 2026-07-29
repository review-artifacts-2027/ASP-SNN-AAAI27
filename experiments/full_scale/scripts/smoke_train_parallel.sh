#!/usr/bin/env bash
# Run quick smoke-training jobs in parallel on separate GPUs.
#
# Usage:
#   bash scripts/smoke_train_parallel.sh
#
# Optional GPU assignment:
#   GPU_SHAPENET=0 GPU_SCANOBJ=1 GPU_S3DIS=2 GPU_FOVEATER=3 bash scripts/smoke_train_parallel.sh

set -euo pipefail

GPU_SHAPENET="${GPU_SHAPENET:-0}"
GPU_SCANOBJ="${GPU_SCANOBJ:-1}"
GPU_S3DIS="${GPU_S3DIS:-2}"
GPU_FOVEATER="${GPU_FOVEATER:-3}"

mkdir -p logs checkpoints

echo "============================================"
echo "  ASP-SNN Parallel Smoke Training"
echo "============================================"
echo "ShapeNetPart GPU: ${GPU_SHAPENET}"
echo "ScanObjectNN GPU: ${GPU_SCANOBJ}"
echo "S3DIS GPU:        ${GPU_S3DIS}"
echo "FoveaTer GPU:     ${GPU_FOVEATER}"
echo "Logs:             logs/smoke_parallel_*.log"
echo "============================================"

CUDA_VISIBLE_DEVICES="${GPU_SHAPENET}" \
python train_shapenet.py --config configs/shapenet_seg.yaml \
  --set epochs=2 batch_size=4 num_workers=0 eval_interval=1 \
  > logs/smoke_parallel_shapenet.log 2>&1 &
PID_SHAPENET=$!

CUDA_VISIBLE_DEVICES="${GPU_SCANOBJ}" \
python train_scanobj.py --config configs/scanobj_cls.yaml \
  --set epochs=2 batch_size=4 num_workers=0 eval_interval=1 \
  > logs/smoke_parallel_scanobj.log 2>&1 &
PID_SCANOBJ=$!

CUDA_VISIBLE_DEVICES="${GPU_S3DIS}" \
python train_s3dis.py --config configs/s3dis_seg.yaml \
  --set epochs=2 batch_size=4 num_workers=0 eval_interval=1 \
  > logs/smoke_parallel_s3dis.log 2>&1 &
PID_S3DIS=$!

CUDA_VISIBLE_DEVICES="${GPU_FOVEATER}" \
python train_imagenet_foveater.py --config configs/imagenet_foveater.yaml \
  --set smoke=true epochs=1 batch_size=2 image_size=64 feature_grid=4 \
        embed_dim=48 depth=1 max_fixations=2 max_tokens=8 debug_steps=1 \
        num_workers=0 use_amp=false num_classes=10 \
  > logs/smoke_parallel_foveater.log 2>&1 &
PID_FOVEATER=$!

wait "${PID_SHAPENET}"
wait "${PID_SCANOBJ}"
wait "${PID_S3DIS}"
wait "${PID_FOVEATER}"

echo "============================================"
echo "  All parallel smoke jobs completed."
echo "============================================"
