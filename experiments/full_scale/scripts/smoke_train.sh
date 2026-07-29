#!/bin/bash
# scripts/smoke_train.sh — Run 2-epoch training with tiny data to verify
# the full pipeline (data loading → model → loss → backward → checkpoint).
#
# Usage: bash scripts/smoke_train.sh
#
# This requires datasets to be downloaded. It overrides epochs and batch
# size to make each run finish in ~2-5 minutes.

set -e
echo "============================================"
echo "  ASP-SNN Smoke Training (2 epochs each)"
echo "============================================"

echo ""
echo "[1/3] ShapeNetPart (2 epochs, batch=4) ..."
python train_shapenet.py --config configs/shapenet_seg.yaml \
    --set epochs=2 batch_size=4 num_workers=0 eval_interval=1

echo ""
echo "[2/3] ScanObjectNN (2 epochs, batch=4) ..."
python train_scanobj.py --config configs/scanobj_cls.yaml \
    --set epochs=2 batch_size=4 num_workers=0 eval_interval=1

echo ""
echo "[3/3] S3DIS (2 epochs, batch=4) ..."
python train_s3dis.py --config configs/s3dis_seg.yaml \
    --set epochs=2 batch_size=4 num_workers=0 eval_interval=1

echo ""
echo "============================================"
echo "  All smoke training runs completed!"
echo "  Check checkpoints/ for saved models."
echo "============================================"
