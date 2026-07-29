#!/bin/bash
# scripts/train_shapenet_full.sh — Full ShapeNetPart training run.
# Sets num_workers=8 and batch=32 for H100. Pipes output to log file.

set -e
mkdir -p logs checkpoints

GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "Training ShapeNetPart on GPU ${GPU}"

CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 python -u train_shapenet.py \
    --config configs/shapenet_seg.yaml \
    --set num_workers=8 \
    2>&1 | tee logs/shapenet_full_gpu${GPU}.log

echo ""
echo "Training complete. Evaluating with per-category breakdown ..."
python eval_shapenet.py \
    --ckpt checkpoints/shapenet_best.pt \
    --config configs/shapenet_seg.yaml \
    --per_cat
