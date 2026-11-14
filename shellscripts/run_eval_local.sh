#!/bin/bash
# Evaluation pipeline for all local runs (CPU only; safe to run while the GPU is busy).
#
# From the repo root:
#   bash eval/run_eval_local.sh            (DATA_ROOT defaults to ~/Documents/csci566)
#
# For every backend with a local baseline (projection/results/<backend>_local/), it finds
# that backend's method runs (<backend>_leace_*, <backend>_cf_*), then:
#   1. scores baseline + methods            -> projection/scored/
#   2. per-model metrics (eval.metrics)     -> projection/eval_results/
#   3. paired comparison vs the baseline    -> projection/eval_results/compare/<backend>/
#   4. leakage probes per LEACE eraser      -> projection/eval_results/leakage.csv
#   5. sample-size check (stability + probes on 5k..40k erasers)
#   6. figures: eval.plots per backend      -> projection/eval_figures/<backend>/
#               eval.plot_compare (deltas)  -> projection/eval_figures/compare/
# Steps 1-3 and 6 take seconds and always rerun. Steps 4-5 are slow and skip work already in
# their CSVs; delete a row (or the CSV) to redo it.
#
# Optional env vars:
#   PROBES=0          skip steps 4-5
#   NO_MLP=1          linear probes only (much faster)
#   ALLOW_PARTIAL=1   include method runs that are not finished yet
#   REPO  THREADS  RES (prediction folder, default projection/results)

set -eo pipefail
DATA_ROOT="${DATA_ROOT:-$HOME/Documents/csci566}"
clean_path() { local p="${1//$'\n'/}"; echo "${p/#\~/$HOME}"; }
DATA_ROOT="$(clean_path "$DATA_ROOT")"
REPO="$(clean_path "${REPO:-$HOME/Documents/vlm-appearance-bias}")"
THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
export PYTHONNOUSERSITE=1
PROBES="${PROBES:-1}"
MLP_FLAG=""; [[ "${NO_MLP:-0}" == 1 ]] && MLP_FLAG="--no-mlp"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME:-vlm-bias}"
fi

cd "$REPO"
mkdir -p logs
LOG="logs/eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"

RES="${RES:-projection/results}"
SCORED=projection/scored
OUT=projection/eval_results
mkdir -p "$SCORED" "$OUT"

n_rows() {
  python - "$1" <<'PY'
import csv, sys, os
p = sys.argv[1]
print(sum(1 for _ in csv.DictReader(open(p, newline=""))) if os.path.exists(p) else 0)
PY
}
# true if a CSV already has a row whose first column equals $2
has_row() { [[ -f "$1" ]] && grep -q "^$2," "$1"; }

# ---- discover runs ----------------------------------------------------------

cd ~/Documents/vlm-appearance-bias
python -m eval.score \
    projection/results/qwen2_5_vl_local/val_predictions.csv \
    projection/results/qwen2_5_vl_leace_all/val_predictions.csv \
    --out projection/scored
python -m eval.metrics --scored projection/scored --out projection/eval_results

BACKENDS=()
for d in "$RES"/*_local; do
  [[ -f "$d/val_predictions.csv" ]] && BACKENDS+=("$(basename "$d" _local)")
done
[[ ${#BACKENDS[@]} -gt 0 ]] || { echo "ERROR: no $RES/<backend>_local/val_predictions.csv found" >&2; exit 1; }

declare -A METHODS
FILES=()
for B in "${BACKENDS[@]}"; do
  BASE="$RES/${B}_local/val_predictions.csv"
  N=$(n_rows "$BASE")
  FILES+=("$BASE")
  M=()
  for d in "$RES/${B}"_leace_* "$RES/${B}"_cf_*; do
    f="$d/val_predictions.csv"
    [[ -f "$f" && "$d" != *_smoke ]] || continue
    k=$(n_rows "$f")
    if [[ $k -lt $N && "${ALLOW_PARTIAL:-0}" != 1 ]]; then
      echo "skip $(basename "$d"): unfinished ($k/$N rows)"
      continue
    fi
    M+=("$(basename "$d")")
    FILES+=("$f")
  done
  METHODS[$B]="${M[*]}"
  echo "$B: baseline ${B}_local ($N rows); methods: ${M[*]:-none}"
done

# ---- 1-2. score + per-model metrics -----------------------------------------
echo; echo "[1] scoring ${#FILES[@]} prediction files"
python -m eval.score "${FILES[@]}" --out "$SCORED"
echo; echo "[2] per-model metrics"
python -m eval.metrics --scored "$SCORED" --out "$OUT"

# ---- 3. paired comparison per backend ---------------------------------------
for B in "${BACKENDS[@]}"; do
  if [[ -z "${METHODS[$B]}" ]]; then
    echo "[3] $B: no finished methods to compare yet"
    continue
  fi
  echo; echo "[3] $B: comparing ${METHODS[$B]} against ${B}_local"
  # shellcheck disable=SC2086
  python -m eval.compare --scored "$SCORED" --baseline "${B}_local" \
      --methods ${METHODS[$B]} --out "$OUT/compare/$B"
done

# ---- 4-5. probes and sample-size check (slow, skip what is done) -------------
if [[ "$PROBES" == 1 ]]; then
  for B in "${BACKENDS[@]}"; do
    FEAT="$DATA_ROOT/features/$B"
    if [[ ! -f "$FEAT/train_pooled.npy" || ! -f "$FEAT/val_pooled.npy" ]]; then
      echo "[4] $B: no extracted features in $FEAT, skipping probes"
      continue
    fi
    for E in gender_race_physical gender race physical; do
      NAME="${B}_leace_${E}"
      if [[ ! -f "$FEAT/eraser_$E.pt" ]]; then
        echo "[4] $NAME: eraser missing, skipping"
      elif has_row "$OUT/leakage.csv" "$NAME"; then
        echo "[4] $NAME: probes done, skipping"
      else
        echo "[4] $NAME: probes"
        python -m projection.probe --features "$FEAT" --eraser "$FEAT/eraser_$E.pt" \
            --alphas 0 1 --model-name "$NAME" --out "$OUT/leakage.csv" $MLP_FLAG
      fi
    done

    ERASERS=()
    for N in 5000 10000 20000 40000; do
      [[ -f "$FEAT/eraser_all_n$N.pt" ]] && ERASERS+=("$FEAT/eraser_all_n$N.pt")
    done
    if [[ ${#ERASERS[@]} -lt 2 ]]; then
      echo "[5] $B: sample-size erasers missing, skipping"
      continue
    fi
    if has_row "$OUT/stability.csv" "$B"; then
      echo "[5] $B: stability done, skipping"
    else
      echo "[5] $B: eraser stability"
      python -m projection.stability --model-name "$B" --erasers "${ERASERS[@]}" \
          --out "$OUT/stability.csv"
    fi
    for e in "${ERASERS[@]}"; do
      N=$(basename "$e" .pt); N=${N#eraser_all_n}
      if has_row "$OUT/leakage_by_n.csv" "${B}_n$N"; then
        echo "[5] ${B}_n$N: probes done, skipping"
      else
        echo "[5] ${B}_n$N: probes"
        python -m projection.probe --features "$FEAT" --eraser "$e" --alphas 1 \
            --model-name "${B}_n$N" --out "$OUT/leakage_by_n.csv" $MLP_FLAG
      fi
    done
  done
fi

# ---- 6. figures (seconds; always rerun) -------------------------------------
FIG=projection/eval_figures
if python -c "import plotly" 2>/dev/null; then
  for B in "${BACKENDS[@]}"; do
    echo; echo "[6] $B: figures"
    # shellcheck disable=SC2086
    python -m eval.plots --results "$OUT" --out "$FIG/$B" \
        --models "${B}_local" ${METHODS[$B]} --title "$B: baseline vs mitigation"
  done
  if ls "$OUT"/compare/*/compare.csv >/dev/null 2>&1; then
    python -m eval.plot_compare --compare "$OUT/compare" --out "$FIG/compare"
  fi
else
  echo "[6] plotly not installed (pip install plotly kaleido), skipping figures"
fi

echo
echo "Figures:"
for B in "${BACKENDS[@]}"; do echo "  $FIG/$B/index.html"; done
ls "$FIG"/compare/*_compare.html 2>/dev/null | sed 's/^/  /' || true
echo "Main tables:"
for B in "${BACKENDS[@]}"; do
  [[ -f "$OUT/compare/$B/compare.md" ]] && echo "  $OUT/compare/$B/compare.md"
done
[[ -f "$OUT/leakage.csv" ]] && echo "  $OUT/leakage.csv"
[[ -f "$OUT/stability.csv" ]] && echo "  $OUT/stability.csv"
