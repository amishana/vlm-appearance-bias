#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|a40|l40s
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
module load conda
eval "$(conda shell.bash hook)"
conda activate vlm-bias
PY=$HOME/.conda/envs/vlm-bias/bin/python
export HF_HOME=/scratch1/$USER/hf_cache
SAMPLE=/scratch1/$USER/csci566/sc_sample
OUT=/scratch1/$USER/csci566/results_test

nvidia-smi --query-gpu=name,memory.total --format=csv

for B in smolvlm2 qwen2_5_vl internvl3; do
  echo "== $B P0  $(date)"
  $PY prompt_debiasing/run_prompts.py --backend $B --prompt P0 \
      --data $SAMPLE --output $OUT/${B}_P0/val_predictions.csv
done

echo "== match with Amisha's baseline"
$PY - <<PYEOF
import pandas as pd
for b in ["smolvlm2", "qwen2_5_vl", "internvl3"]:
    mine = pd.read_csv(f"$OUT/{b}_P0/val_predictions.csv")
    hers = pd.read_csv(f"baseline/results/{b}/val_predictions.csv")
    m = mine.merge(hers, on="image_id", suffixes=("_mine", "_hers"))
    same = (m.model_response_mine.str.strip().str.lower() ==
            m.model_response_hers.str.strip().str.lower()).mean()
    print(f"{b}: {len(m)} images, match = {same:.2f}")
PYEOF
echo "== done  $(date)"
