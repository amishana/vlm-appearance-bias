#!/bin/bash
set -euo pipefail
RUN=$1
MODE=$2
source "$RUN/config.sh"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONNOUSERSITE=1
export HF_HOME="$DATA_ROOT/huggingface" HF_HUB_OFFLINE=1
hostname
"$VLM_PYTHON" -m pip freeze > "$RUN/${MODE}_environment.txt"
case "$MODE" in
 smoke|ce|adv|evaluate)
  nvidia-smi
  ARGS=("$MODE" --root "$DATA_ROOT" --out "$RUN/$MODE" --steps "$STEPS" --accum 4 --train-groups "$TRAIN_GROUPS" --val-groups "$VAL_GROUPS" --seed 42)
  if [ "$MODE" = evaluate ]; then
   ARGS+=(--ce-checkpoint "$RUN/ce/train/checkpoint.pt" --adv-checkpoint "$RUN/adv/train/checkpoint.pt")
  fi
  "$VLM_PYTHON" -u "$RUN/code/adversarial/pilot.py" "${ARGS[@]}"
  ;;
 extract)
  nvidia-smi
  "$VLM_PYTHON" -u "$RUN/code/julia_followup/extract.py" --root "$DATA_ROOT" --extended "$EXTENDED" --out "$RUN/features"
  ;;
 probes)
  "$VLM_PYTHON" -u "$RUN/code/julia_followup/probes.py" --features "$RUN/features" --out "$RUN/probes"
  ;;
 metrics)
  "$VLM_PYTHON" -u "$RUN/code/julia_followup/analyze.py" --data "$DATA_ROOT/processed/socialcounterfactuals/exported" --extended "$EXTENDED" --out "$RUN/canonical"
  ;;
 *) echo "Unknown mode: $MODE" >&2; exit 1;;
esac
