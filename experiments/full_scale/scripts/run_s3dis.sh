#!/bin/bash
#SBATCH --job-name=asp-s3dis
#SBATCH --output=logs/s3dis_%j.log
#SBATCH --error=logs/s3dis_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source activate asp-snn 2>/dev/null || conda activate asp-snn

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start:  $(date)"
echo "============================================"

mkdir -p checkpoints logs

python train_s3dis.py --config configs/s3dis_seg.yaml

echo "============================================"
echo "End: $(date)"
