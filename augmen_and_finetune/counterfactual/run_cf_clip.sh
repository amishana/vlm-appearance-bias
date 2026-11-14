#!/bin/bash
# The teammate's original CLIP counterfactual pipeline (counterfactual/clip/), step by
# step with skip-if-complete checks. Side result only: CLIP is a closed-set classifier
# evaluated on the TEST split with her own evaluator, so its numbers are not comparable
# to the VLM results scored with eval/*.
#
# From the repo root:
#   DATA_ROOT=$HOME/Documents/csci566 bash counterfactual/run_cf_clip.sh
#
# Optional env vars:
#   RUN_DIR     outputs (default $DATA_ROOT/cf_clip/run)
#   CLIP_VENV   separate venv with her pinned versions (default counterfactual/clip/.venv)
#   CAPABILITY=1  also run her CIFAR-10 capability check (downloads CIFAR-10)
#   REPO  GPU

set -eo pipefail
: "${DATA_ROOT:?set DATA_ROOT}"
clean_path() { local p="${1//$'\n'/}"; echo "${p/#\~/$HOME}"; }
DATA_ROOT="$(clean_path "$DATA_ROOT")"
REPO="$(clean_path "${REPO:-$HOME/Documents/vlm-appearance-bias}")"
CODE="$REPO/counterfactual/clip"
RUN_DIR="$(clean_path "${RUN_DIR:-$DATA_ROOT/cf_clip/run}")"
VENV="$(clean_path "${CLIP_VENV:-$CODE/.venv}")"
EXPORT="$DATA_ROOT/processed/socialcounterfactuals/exported"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export PYTHONNOUSERSITE=1

cd "$REPO"
mkdir -p logs "$RUN_DIR"
LOG="logs/cf_clip_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Logging to $LOG"
[[ -f "$CODE/train_lora.py" ]] || { echo "ERROR: her code not found in $CODE" >&2; exit 1; }
[[ -f "$EXPORT/val.csv" ]] || { echo "ERROR: missing $EXPORT/val.csv (check DATA_ROOT)" >&2; exit 1; }

# 0. her pinned environment (torch 2.5.1 / transformers 4.49 / peft 0.14), kept separate
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[0] creating $VENV"
  PYV=$(command -v python3.11 || command -v python3)
  "$PYV" -m venv "$VENV"
  # Her pin torch==2.5.1 has no kernels for RTX 50-series (Blackwell) GPUs, so torch
  # comes from the CUDA 12.8 index; everything else keeps her pinned versions.
  "$VENV/bin/python" -m pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128
  grep -v -E '^(torch|torchvision)==' "$CODE/requirements.txt" > "$VENV/requirements_no_torch.txt"
  "$VENV/bin/python" -m pip install -q -r "$VENV/requirements_no_torch.txt"
else
  echo "[0] venv exists, skipping"
fi
PY="$VENV/bin/python"
cd "$CODE"

# 1. import the team export (keeps the team's splits)
if [[ -f "$RUN_DIR/processed/summary.json" ]]; then
  echo "[1] import done, skipping"
else
  rm -rf "$RUN_DIR/processed"
  "$PY" import_benchmark.py --exported "$EXPORT" --output "$RUN_DIR/processed"
fi

# 2. run config (same as her run_benchmark.sh)
if [[ -f "$RUN_DIR/config.yaml" ]]; then
  echo "[2] config exists, skipping"
else
  "$PY" - "$CODE" "$RUN_DIR" "$EXPORT" <<'PY'
import sys
from pathlib import Path
import yaml
code, run, export = map(lambda s: Path(s).resolve(), sys.argv[1:4])
cfg = yaml.safe_load((code / "config.yaml").read_text())
cfg["experiment"]["status"] = "TEAM_BENCHMARK_SINGLE_SEED_EXPERIMENT"
cfg["data"] = {"source": "team_exported_csv", "processed_dir": str(run / "processed"),
               "image_dir": str(export / "images")}
cfg["training"]["output_dir"] = str(run / "training")
(run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
fi
CFG="$RUN_DIR/config.yaml"

# 3. baseline CLIP predictions (test split, her protocol)
if [[ -f "$RUN_DIR/baseline.metadata.json" ]]; then
  echo "[3] baseline predictions done, skipping"
else
  rm -f "$RUN_DIR/baseline.csv"
  "$PY" predict.py --config "$CFG" --output "$RUN_DIR/baseline.csv"
fi

# 4. LoRA training (her trainer resumes from training/resume.pt)
if grep -qs adapter_reload_check "$RUN_DIR/training/metadata.json"; then
  echo "[4] training done, skipping"
else
  if [[ -d "$RUN_DIR/training" && ! -f "$RUN_DIR/training/resume.pt" ]]; then
    echo "[4] partial training without a checkpoint; starting over"
    rm -rf "$RUN_DIR/training"
  fi
  "$PY" train_lora.py --config "$CFG" --output-dir "$RUN_DIR/training"
fi

# 5. adapted predictions
if [[ -f "$RUN_DIR/adapted.metadata.json" ]]; then
  echo "[5] adapted predictions done, skipping"
else
  rm -f "$RUN_DIR/adapted.csv"
  "$PY" predict.py --config "$CFG" --adapter "$RUN_DIR/training/best_adapter" --output "$RUN_DIR/adapted.csv"
fi

# 6. her paired evaluation
if [[ -f "$RUN_DIR/comparison.json" ]]; then
  echo "[6] evaluation done, skipping"
else
  "$PY" evaluate_predictions.py --baseline "$RUN_DIR/baseline.csv" --adapted "$RUN_DIR/adapted.csv" \
      --output "$RUN_DIR/comparison.json"
fi

# 7. optional CIFAR-10 capability check
if [[ "${CAPABILITY:-0}" == 1 ]]; then
  if [[ -n "$(ls -A "$RUN_DIR/capability" 2>/dev/null)" ]]; then
    echo "[7] capability check done, skipping"
  else
    "$PY" evaluate_capability.py --config "$CFG" --adapter "$RUN_DIR/training/best_adapter" \
        --output-dir "$RUN_DIR/capability"
  fi
fi

echo "Done: CLIP counterfactual run in $RUN_DIR"
