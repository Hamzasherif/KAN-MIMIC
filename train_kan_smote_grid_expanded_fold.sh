#!/bin/bash
#SBATCH --job-name=kan-smotegrid
#SBATCH --partition=rtx3080,v100,a100,work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --time=23:00:00
#SBATCH --output=logs/kan_smotegrid_%j.out
#SBATCH --error=logs/kan_smotegrid_%j.err

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kan-mimic
cd ~/KAN-MIMIC

FOLD="$1"
if [[ ! "$FOLD" =~ ^[0-4]$ ]]; then
    echo "ERROR: FOLD must be between 0 and 4."
    exit 1
fi

echo "===== SMOTE GRID-SEARCH KAN: FOLD $FOLD ====="
nvidia-smi
python -u src/training/train_kan_smote_grid_expanded.py \
    --fold "$FOLD" \
    --epochs 100 \
    --patience 15 \
    --cv-epochs 30 \
    --cv-patience 7 \
    --inner-splits 3 \
    --seed 42

echo "===== SMOTE GRID-SEARCH KAN FOLD $FOLD FINISHED ====="
