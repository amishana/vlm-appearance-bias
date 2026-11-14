#!/bin/bash
# Counterfactual consistency LoRA (proposal 3.3) on the team's VLMs, single-GPU desktop.
# Every step is skipped if already complete; rerunning resumes interrupted steps.
#
# From the repo root:
#   BACKEND=qwen2_5_vl MODEL=$HOME/Documents/models/Qwen2.5-VL-3B-Instruct DATA_ROOT=$HOME/Documents/csci566 \
#       bash counterfactual/run_counterfactual_local.sh
#   BACKEND=smolvlm2   MODEL=$HOME/Documents/models/SmolVLM2-2.2B-Instruct  DATA_ROOT=$HOME/Documents/csci566 \
#       bash counterfactual/run_counterfactual_local.sh
#   (InternVL3-8B needs more than 16 GB to train; run it on the cluster.)
#
# Optional env vars:
#   SMOKE=1            quick end-to-end test: 20 training groups, 100 val images, *_smoke outputs
#   RUN_CONTROL=1      also train the CE-only control (consistency weight 0)  [default 1]
#   CW=0.1             consistency weight for the main run
#   EPOCHS=1  IMAGES_PER_GROUP=4  MAX_GROUPS=      (empty = all ~9.8k training groups)
#   REPO  ENV_NAME  GPU  THREADS  RES (prediction folder, default projection/results)

set -eo pipefail
: "${BACKEND:?set BACKEND}" "${MODEL:?set MODEL}" "${DATA_ROOT:?set DATA_ROOT}"

clean_path() { local p="${1//$'\n'/}"; echo "${p/#\~/$HOME}"; }
DATA_ROOT="$(clean_path "$DATA_ROOT")"
MODEL="$(clean_path "$MODEL")"
REPO="$(clean_path "${REPO:-$HOME/Documents/vlm-appearance-bias}")"
ENV_NAME="${ENV_NAME:-vlm-bias}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_CONTROL="${RUN_CONTROL:-1}"
CW="${CW:-0.1}"
EPOCHS="${EPOCHS:-1}"
IMAGES_PER_GROUP="${IMAGES_PER_GROUP:-4}"
MAX_GROUPS="${MAX_GROUPS:-}"
LIMIT=""
SUFFIX=""
if [[ "${SMOKE:-0}" == 1 ]]; then MAX_GROUPS=20; LIMIT=100; SUFFIX=_smoke; fi

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "$ENV_NAME"
fi

cd "$REPO"
mkdir -p logs
LOG="logs/cf_${BACKEND}${SUFFIX}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

DATA="$DATA_ROOT/processed/socialcounterfactuals/exported"
CF="$DATA_ROOT/cf_lora/$BACKEND"
RES="${RES:-projection/results}"
LOCAL_BASE="$RES/${BACKEND}_local/val_predictions.csv"
mkdir -p "$CF" "$RES"

for f in "$DATA/val.csv" "$DATA/train.csv"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f (check DATA_ROOT)" >&2; exit 1; }
done
[[ -d "$MODEL" ]] || { echo "ERROR: model folder not found: $MODEL" >&2; exit 1; }
python -c "import peft" 2>/dev/null || { echo "ERROR: pip install peft" >&2; exit 1; }

n_rows() {
  python - "$1" <<'PY'
import csv, sys, os
p = sys.argv[1]
print(sum(1 for _ in csv.DictReader(open(p, newline=""))) if os.path.exists(p) else 0)
PY
}
VAL_N=$(n_rows "$DATA/val.csv")
WANT_N=${LIMIT:-$VAL_N}

# 0. baseline in THIS environment (shared with the LEACE pipeline)
if [[ -z "$SUFFIX" ]]; then
  if [[ $(n_rows "$LOCAL_BASE") -eq $VAL_N ]]; then
    echo "[0] local baseline complete, skipping"
  else
    echo "[0] local baseline: $(n_rows "$LOCAL_BASE")/$VAL_N rows done, resuming"
    python "baseline/run_${BACKEND}.py" --model "$MODEL" --data "$DATA" --split val --output "$LOCAL_BASE"
  fi
fi

VARIANTS=("cf_lora:$CW")
[[ "$RUN_CONTROL" == 1 ]] && VARIANTS+=("cf_ce_only:0")

for V in "${VARIANTS[@]}"; do
  NAME="${V%%:*}${SUFFIX}"; WEIGHT="${V##*:}"
  OUT="$CF/$NAME"
  PRED="$RES/${BACKEND}_${NAME}/val_predictions.csv"

  # 1. train (resumes from its last checkpoint; skipped once DONE.json exists)
  if [[ -f "$OUT/DONE.json" ]]; then
    echo "[1] $NAME: adapter trained, skipping"
  else
    echo "[1] $NAME: training (consistency weight $WEIGHT)"
    python -m counterfactual.train_vlm_cf --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
        --out "$OUT" --consistency-weight "$WEIGHT" --epochs "$EPOCHS" \
        --images-per-group "$IMAGES_PER_GROUP" ${MAX_GROUPS:+--max-groups "$MAX_GROUPS"}
  fi

  # 2. predict on val with the adapter merged (resumable; redone if the adapter is newer)
  if [[ -f "$PRED" && ! "$PRED" -nt "$OUT/DONE.json" ]]; then
    echo "[2] $NAME: adapter retrained after these predictions; starting over"
    rm -f "$PRED" "${PRED%.csv}.settings.json"
  fi
  if [[ $(n_rows "$PRED") -eq $WANT_N ]]; then
    echo "[2] $NAME: predictions complete, skipping"
  else
    echo "[2] $NAME: predicting ($(n_rows "$PRED")/$WANT_N done)"
    python -m counterfactual.predict_vlm_cf --backend "$BACKEND" --model "$MODEL" --data "$DATA" \
        --adapter "$OUT/adapter" --output "$PRED" ${LIMIT:+--limit "$LIMIT"}
  fi
done

echo "Done: $BACKEND counterfactual LoRA${SUFFIX}"
