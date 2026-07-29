#!/bin/bash
# scripts/train_scanobj_full.sh — Full ScanObjectNN training + 10-vote eval.

set -e
mkdir -p logs checkpoints

GPU=${CUDA_VISIBLE_DEVICES:-0}
echo "Training ScanObjectNN on GPU ${GPU}"

CUDA_VISIBLE_DEVICES=${GPU} PYTHONUNBUFFERED=1 python -u train_scanobj.py \
    --config configs/scanobj_cls.yaml \
    --set num_workers=8 \
    2>&1 | tee logs/scanobj_full_gpu${GPU}.log

echo ""
echo "Training complete. Evaluating with 10-vote TTA ..."
python eval_scanobj.py \
    --ckpt checkpoints/scanobj_best.pt \
    --config configs/scanobj_cls.yaml \
    --n_votes 10
