#!/bin/bash
#SBATCH --job-name=swin_test_gt
#SBATCH --output=logs_eval/test_gt_%j.out
#SBATCH --error=logs_eval/test_gt_%j.err
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_init.sh"
mkdir -p logs_eval logs_train_fusion checkpoints
"$PYTHON" -u train_fusion.py --depth-source gt --epochs 1 --device "${DEVICE:-cuda}" --num-workers 0
"$PYTHON" -u eval.py --depth-source gt --device "${DEVICE:-cuda}" --batch-size 8 --num-workers 0
