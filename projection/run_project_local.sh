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
RES=projection/results
LOCAL_BASE="$RES/${BACKEND}_local/val_predictions.csv"
CHECK="$RES/check_${BACKEND}_alpha0/val_predictions.csv"
ERASED="$RES/${BACKEND}_leace_all/val_predictions.csv"
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
 
# 1a. train features + statistics; one 40k pass also saves nested 5k/10k/20k statistics
TRAIN_OUT=("$FEAT/train_stats.pt" "$FEAT/train_pooled.npy" "$FEAT/train_meta.csv"
           "$FEAT/train_stats_n5000.pt" "$FEAT/train_stats_n10000.pt" "$FEAT/train_stats_n20000.pt")
if ls "${TRAIN_OUT[@]}" >/dev/null 2>&1; then
  echo "[1a] train features exist, skipping"
else
  echo "[1a] extracting train features"
  python -m projection.extract_features --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
      --split train --max-images 40000 --stats --checkpoints 5000 10000 20000 --out "$FEAT"
fi
 
# 1b. val features
if [[ -f "$FEAT/val_pooled.npy" && $(n_rows "$FEAT/val_meta.csv") -eq $VAL_N ]]; then
  echo "[1b] val features exist, skipping"
else
  echo "[1b] extracting val features"
  python -m projection.extract_features --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
      --split val --out "$FEAT"
fi
 
# 2. erasers (seconds each). Refit only if missing or older than their statistics.
fit() {  # fit <stats> <out> <types...>
  local stats="$1" out="$2"; shift 2
  if fresh "$stats" "$out"; then
    echo "[2] $(basename "$out") up to date, skipping"
  else
    python -m projection.fit_eraser --stats "$stats" --types "$@" --out "$out"
  fi
}
# 2a. main erasers (20k sample): all attributes jointly, plus one per attribute
for T in "gender race physical" gender race physical; do
  fit "$FEAT/train_stats_n20000.pt" "$FEAT/eraser_$(echo $T | tr ' ' _).pt" $T
done
# 2b. sample-size check erasers
for N in 5000 10000 20000; do
  fit "$FEAT/train_stats_n$N.pt" "$FEAT/eraser_all_n$N.pt" gender race physical
done
fit "$FEAT/train_stats.pt" "$FEAT/eraser_all_n40000.pt" gender race physical
 
MAIN_ERASER="$FEAT/eraser_gender_race_physical.pt"
 
# 3. sanity check: alpha 0 must equal the local baseline on these rows
if [[ $(n_rows "$CHECK") -eq 200 && "$CHECK" -nt "$MAIN_ERASER" ]]; then
  echo "[3] alpha=0 check predictions exist, skipping inference"
else
  echo "[3] alpha=0 check"
  rm -f "$CHECK" "${CHECK%.csv}.settings.json"   # eraser changed or partial run: redo
  python -m projection.run_erased --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
      --eraser "$MAIN_ERASER" --alpha 0 --limit 200 --output "$CHECK"
fi
python - <<EOF
import pandas as pd
a = pd.read_csv("$CHECK")
b = pd.read_csv("$LOCAL_BASE").set_index("image_id")
same = (a.set_index("image_id")["model_response"] == b.loc[a["image_id"], "model_response"]).mean()
print(f"[3] alpha=0 matches local baseline on {same:.1%} of {len(a)} rows")
assert same > 0.99, "hook or preprocessing differs from the baseline"
EOF
 
# 4. full erased run on val (resumable)
if [[ -f "$ERASED" && ! "$ERASED" -nt "$MAIN_ERASER" ]]; then
  echo "[4] eraser was refit after these predictions; starting the erased run over"
  rm -f "$ERASED" "${ERASED%.csv}.settings.json"
fi
if [[ $(n_rows "$ERASED") -eq $VAL_N ]]; then
  echo "[4] erased run complete ($VAL_N rows), skipping"
else
  echo "[4] erased run: $(n_rows "$ERASED")/$VAL_N rows done, resuming"
  python -m projection.run_erased --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
      --eraser "$MAIN_ERASER" --alpha 1 --output "$ERASED"
fi
 
echo "Done: $BACKEND"