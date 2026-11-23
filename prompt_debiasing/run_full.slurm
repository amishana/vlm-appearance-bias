#!/bin/bash
#SBATCH --job-name=prompt-run
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

# One model x one prompt on the full split. Submitted by submit_all.sh.
# Safe to resubmit: run_prompts.py skips images that are already done.

set -euo pipefail
module load conda
eval "$(conda shell.bash hook)"
conda activate vlm-bias
PY=$HOME/.conda/envs/vlm-bias/bin/python
export HF_HOME=/scratch1/$USER/hf_cache

SPLIT=${SPLIT:-val}
DATA=${DATA:-/scratch1/$USER/csci566/processed/socialcounterfactuals/exported}
TAG=${TAG:-}
OUT=/scratch1/$USER/csci566/results/${BACKEND}_${PROMPT}${TAG}/${SPLIT}_predictions.csv

nvidia-smi --query-gpu=name,memory.total --format=csv
echo "== $BACKEND $PROMPT $SPLIT  data=$DATA  $(date)"

$PY prompt_debiasing/run_prompts.py --backend "$BACKEND" --prompt "$PROMPT" \
    --split "$SPLIT" --data "$DATA" --output "$OUT"

echo "== done  $(date)"