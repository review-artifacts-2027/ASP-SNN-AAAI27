#!/bin/bash
#SBATCH --job-name=asp-scanobj
#SBATCH --output=logs/scanobj_%j.log
#SBATCH --error=logs/scanobj_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

source activate asp-snn 2>/dev/null || conda activate asp-snn

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"
echo "============================================"

mkdir -p checkpoints logs

python train_scanobj.py --config configs/scanobj_cls.yaml

echo "============================================"
echo "End: $(date)"
