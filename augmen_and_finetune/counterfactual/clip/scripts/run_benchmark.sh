#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_benchmark.sh /path/to/team/exported /new/run/directory
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TEMPLATE_CONFIG="${CONFIG_PATH:-$ROOT_DIR/config.yaml}"
EXPORT_DIR="${1:?Supply team exported directory}"
RUN_DIR="${2:?Supply a NEW run directory}"
if [ -e "$RUN_DIR" ]; then echo 'Run directory already exists' >&2; exit 1; fi
# Import first; fail before downloading models if benchmark data is unavailable.
"$PYTHON_BIN" "$ROOT_DIR/import_benchmark.py" --exported "$EXPORT_DIR" --output "$RUN_DIR/processed"
"$PYTHON_BIN" - "$ROOT_DIR" "$RUN_DIR" "$EXPORT_DIR" "$TEMPLATE_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
root, run = map(Path, sys.argv[1:3])
run = run.resolve()
template = Path(sys.argv[4]).resolve()
cfg = yaml.safe_load(template.read_text())
local_model = (template.parent / cfg['model']['name']).resolve()
if local_model.is_dir():
    cfg['model']['name'] = str(local_model)
cfg['experiment']['status'] = 'TEAM_BENCHMARK_SINGLE_SEED_EXPERIMENT'
cfg['data'] = {'source':'team_exported_csv', 'processed_dir':str(run/'processed'), 'image_dir':str(Path(sys.argv[3]).resolve()/'images')}
cfg['training']['output_dir'] = str(run/'training')
(run/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
PY
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
"$PYTHON_BIN" "$ROOT_DIR/predict.py" --config "$RUN_DIR/config.yaml" --output "$RUN_DIR/baseline.csv"
"$PYTHON_BIN" "$ROOT_DIR/train_lora.py" --config "$RUN_DIR/config.yaml" --output-dir "$RUN_DIR/training"
"$PYTHON_BIN" "$ROOT_DIR/predict.py" --config "$RUN_DIR/config.yaml" --adapter "$RUN_DIR/training/best_adapter" --output "$RUN_DIR/adapted.csv"
"$PYTHON_BIN" "$ROOT_DIR/evaluate_predictions.py" --baseline "$RUN_DIR/baseline.csv" --adapted "$RUN_DIR/adapted.csv" --output "$RUN_DIR/comparison.json"
