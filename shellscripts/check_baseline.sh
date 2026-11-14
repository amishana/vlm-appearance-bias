#!/bin/bash
# Local (single-GPU desktop) version of projection/run_projection.slurm.
# One run per model. From the repo root:
#   BACKEND=qwen2_5_vl MODEL=YOUR/PATH/TO/models/Qwen2.5-VL-3B-Instruct bash shellscripts/check_baseline.sh
#   BACKEND=smolvlm2   MODEL=YOUR/PATH/TO/models/SmolVLM2-2.2B-Instruct  bash shellscripts/check_baseline.sh
#   BACKEND=internvl3  MODEL=YOUR/PATH/TO/models/InternVL3-8B           bash shellscripts/check_baseline.sh
# Use the same model paths as the baseline runs.
#
# Optional overrides (env vars):
#   REPO       repo root            (default: YOUR/PATH/TO/vlm-appearance-bias)
#   DATA_ROOT  data root            (default: must be set, same as for export_images.py)
#   ENV_NAME   conda env to activate if no env is active (default: vlm-bias)
#   GPU        GPU index            (default: 0)
#   THREADS    CPU threads for torch/dataloading (default: 8)

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
# Guard against stray newlines / quoted ~ in paths
clean_path() { local p="${1//$'\n'/}"; echo "${p/#\~/$HOME}"; }
DATA_ROOT="$(clean_path "$DATA_ROOT")"
MODEL="$(clean_path "$MODEL")"
 
cd "$REPO"
mkdir -p logs
LOG="logs/leace_${BACKEND}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

DATA="$DATA_ROOT/processed/socialcounterfactuals/exported"
FEAT="$DATA_ROOT/features/$BACKEND"
RES=YOUR/RESULTS/PATH
LOCAL_BASE="$RES/${BACKEND}_local/val_predictions.csv"

mkdir -p "$FEAT" "$RES"

for f in "$DATA/val.csv" "$DATA/train.csv"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (check DATA_ROOT)" >&2; exit 1; }
done
[[ -d "$MODEL" ]] || { echo "ERROR: model folder not found: $MODEL" >&2; exit 1; }
 
# Number of data rows in a CSV (0 if the file does not exist)
n_rows() {
  python - "$1" <<'PY'
import csv, sys, os
p = sys.argv[1]
print(sum(1 for _ in csv.DictReader(open(p, newline=""))) if os.path.exists(p) else 0)
PY
}
 
# True if every file exists and is newer than the reference file
fresh() {
  local ref="$1"; shift
  for f in "$@"; do [[ -f "$f" && "$f" -nt "$ref" ]] || return 1; done
}
 
VAL_N=$(n_rows "$DATA/val.csv")
echo "val images: $VAL_N"

## 0. baseline in THIS environment, the reference for all comparisons
if [[ $(n_rows "$LOCAL_BASE") -eq $VAL_N ]]; then
  echo "[0] local baseline complete ($VAL_N rows), skipping"
else
  echo "[0] local baseline: $(n_rows "$LOCAL_BASE")/$VAL_N rows done, resuming"
  python "baseline/run_${BACKEND}.py" --model "$MODEL" --data "$DATA" --split val \
      --output "$LOCAL_BASE"
fi
