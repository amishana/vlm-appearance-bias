#!/bin/bash
# Local (single-GPU desktop) version of projection/run_projection.slurm.
# One run per model. From the repo root:
#   BACKEND=qwen2_5_vl MODEL=~/Documents/models/Qwen2.5-VL-3B-Instruct bash projection/run_projection_local.sh
#   BACKEND=smolvlm2   MODEL=~/Documents/models/SmolVLM2-2.2B-Instruct  bash projection/run_projection_local.sh
#   BACKEND=internvl3  MODEL=~/Documents/models/InternVL3-8B           bash projection/run_projection_local.sh
# Use the same model paths as the baseline runs.
#
# Optional overrides (env vars):
#   REPO       repo root            (default: ~/Documents/vlm-appearance-bias)
#   DATA_ROOT  data root            (default: must be set, same as for export_images.py)
#   ENV_NAME   conda env to activate if no env is active (default: vlm-bias)
#   GPU        GPU index            (default: 0)
#   THREADS    CPU threads for torch/dataloading (default: 8)

#!/bin/bash
set -eo pipefail
: "${BACKEND:?set BACKEND}" "${MODEL:?set MODEL}" "${DATA_ROOT:?set DATA_ROOT}"

REPO="${REPO:-$HOME/Documents/vlm-appearance-bias}"
ENV_NAME="${ENV_NAME:-vlm-bias}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

cd "$REPO"
mkdir -p logs
LOG="logs/leace_${BACKEND}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

DATA="$DATA_ROOT/processed/socialcounterfactuals/exported"
FEAT="$DATA_ROOT/features/$BACKEND"
RES=projection/results
LOCAL_BASE="$RES/${BACKEND}_local/val_predictions.csv"
mkdir -p "$FEAT" "$RES"

# 0. baseline in THIS environment (resumable), the reference for all comparisons
python "baseline/run_${BACKEND}.py" --model "$MODEL" --data "$DATA" --split val \
    --output "$LOCAL_BASE"

# 1. features + statistics; one 40k pass also saves nested 5k/10k/20k statistics
python -m projection.extract_features --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
    --split train --max-images 40000 --stats --checkpoints 5000 10000 20000 --out "$FEAT"
python -m projection.extract_features --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
    --split val --out "$FEAT"

# 2a. main erasers (20k sample): all attributes jointly, plus one per attribute
for T in "gender race physical" gender race physical; do
  python -m projection.fit_eraser --stats "$FEAT/train_stats_n20000.pt" --types $T \
      --out "$FEAT/eraser_$(echo $T | tr ' ' _).pt"
done
# 2b. sample-size check erasers
for N in 5000 10000 20000; do
  python -m projection.fit_eraser --stats "$FEAT/train_stats_n$N.pt" \
      --types gender race physical --out "$FEAT/eraser_all_n$N.pt"
done
python -m projection.fit_eraser --stats "$FEAT/train_stats.pt" \
    --types gender race physical --out "$FEAT/eraser_all_n40000.pt"

# 3. sanity check: alpha 0 must equal the local baseline on these rows
python -m projection.run_erased --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
    --eraser "$FEAT/eraser_gender_race_physical.pt" --alpha 0 --limit 200 \
    --output "$RES/check_${BACKEND}_alpha0/val_predictions.csv"
python - <<EOF
import pandas as pd
a = pd.read_csv("$RES/check_${BACKEND}_alpha0/val_predictions.csv")
b = pd.read_csv("$LOCAL_BASE").set_index("image_id")
same = (a.set_index("image_id")["model_response"] == b.loc[a["image_id"], "model_response"]).mean()
print(f"alpha=0 matches local baseline on {same:.1%} of {len(a)} rows")
assert same > 0.99, "hook or preprocessing differs from the baseline"
EOF

# 4. full erased run on val
python -m projection.run_erased --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
    --eraser "$FEAT/eraser_gender_race_physical.pt" --alpha 1 \
    --output "$RES/${BACKEND}_leace_all/val_predictions.csv"

echo "Done: $BACKEND"