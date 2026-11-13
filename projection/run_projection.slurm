#!/bin/bash
# One job per model. Submit from the repo root (logs/ must exist):
#   sbatch --export=ALL,BACKEND=qwen2_5_vl,MODEL=/path/to/Qwen2.5-VL-3B-Instruct projection/run_projection.slurm
#   sbatch --export=ALL,BACKEND=smolvlm2,MODEL=/path/to/SmolVLM2-2.2B-Instruct  projection/run_projection.slurm
#   sbatch --export=ALL,BACKEND=internvl3,MODEL=/path/to/InternVL3-8B           projection/run_projection.slurm
# Use the same model paths as the baseline runs.
#SBATCH --job-name=leace
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -e
: "${BACKEND:?set BACKEND}" "${MODEL:?set MODEL}"

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate vlm-bias
export PYTHONNOUSERSITE=1
export HF_HOME="/scratch2/$USER/hf_cache"

cd /scratch2/$USER/vlm-appearance-bias
DATA=/scratch2/$USER/csci566/processed/socialcounterfactuals/exported
FEAT=/scratch2/$USER/csci566/features/$BACKEND
RES=projection/results

# 1. features + statistics (vision path only)
python -m projection.extract_features --backend $BACKEND --model $MODEL --data $DATA \
    --split train --max-images 20000 --stats --out $FEAT
python -m projection.extract_features --backend $BACKEND --model $MODEL --data $DATA \
    --split val --out $FEAT

# 2. erasers: all attributes jointly, plus one per attribute for the ablation
for T in "gender race physical" gender race physical; do
  python -m projection.fit_eraser --stats $FEAT/train_stats.pt --types $T \
      --out $FEAT/eraser_$(echo $T | tr ' ' _).pt
done

# 3. sanity check: alpha 0 must equal baseline/results/$BACKEND on these rows
python -m projection.run_erased --backend $BACKEND --model $MODEL --data $DATA \
    --eraser $FEAT/eraser_gender_race_physical.pt --alpha 0 --limit 200 \
    --output $RES/check_${BACKEND}_alpha0/val_predictions.csv
python - <<EOF
import pandas as pd
a = pd.read_csv("$RES/check_${BACKEND}_alpha0/val_predictions.csv")
b = pd.read_csv("baseline/results/$BACKEND/val_predictions.csv").set_index("image_id")
same = (a.set_index("image_id")["model_response"] == b.loc[a["image_id"], "model_response"]).mean()
print(f"alpha=0 matches baseline on {same:.1%} of {len(a)} rows")
assert same > 0.99, "hook or preprocessing differs from the baseline"
EOF

# 4. full erased run on val
python -m projection.run_erased --backend $BACKEND --model $MODEL --data $DATA \
    --eraser $FEAT/eraser_gender_race_physical.pt --alpha 1 \
    --output $RES/${BACKEND}_leace_all/val_predictions.csv
