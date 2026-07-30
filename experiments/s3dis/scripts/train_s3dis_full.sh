#!/bin/bash
# scripts/train_s3dis_full.sh — Full S3DIS training (100 epochs, batch=64 on H100).

set -e
mkdir -p logs checkpoints

GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "Training S3DIS on GPU ${GPU}"

CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 python -u train_s3dis.py \
    --config configs/s3dis_seg.yaml \
    --set num_workers=8 \
    2>&1 | tee logs/s3dis_full_gpu${GPU}.log

echo ""
echo "Training complete. Evaluating with per-class IoU ..."
python eval_s3dis.py \
    --ckpt checkpoints/s3dis_best.pt \
    --config configs/s3dis_seg.yaml \
    --per_class
