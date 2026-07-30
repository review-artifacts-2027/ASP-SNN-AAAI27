#!/bin/bash
#SBATCH --job-name=asp-shapenet
#SBATCH --output=logs/shapenet_%j.log
#SBATCH --error=logs/shapenet_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# ── Activate environment ──────────────────────────────────────────────
source activate asp-snn 2>/dev/null || conda activate asp-snn

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"
echo "============================================"

# ── Create output directories ─────────────────────────────────────────
mkdir -p checkpoints logs

# ── Run training ──────────────────────────────────────────────────────
python train_shapenet.py --config configs/shapenet_seg.yaml

echo "============================================"
echo "End: $(date)"
