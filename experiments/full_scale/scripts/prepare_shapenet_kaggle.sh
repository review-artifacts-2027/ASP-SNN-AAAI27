#!/bin/bash
# Download ShapeNetPart from Kaggle and convert it to the HDF5 format used by
# train_shapenet.py.
#
# Dataset:
#   https://www.kaggle.com/datasets/majdouline20/shapenetpart-dataset
#
# Prerequisites:
#   pip install kagglehub kaggle
#   # Kaggle credentials may be required:
#   # https://www.kaggle.com/docs/api
#
# Usage:
#   bash scripts/prepare_shapenet_kaggle.sh

set -euo pipefail

echo "========================================"
echo "  ShapeNetPart from Kaggle"
echo "========================================"
echo "Dataset: majdouline20/shapenetpart-dataset"
echo

python datasets/download.py --shapenet_kaggle
