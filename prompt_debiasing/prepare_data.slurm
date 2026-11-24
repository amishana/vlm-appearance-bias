#!/bin/bash
#SBATCH --job-name=sc-prepare
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

module load conda
eval "$(conda shell.bash hook)"
conda activate vlm-bias
PY=$HOME/.conda/envs/vlm-bias/bin/python
echo "Using $PY"; $PY -c "import datasets; print('datasets ok')"

export HF_HOME=/scratch1/$USER/hf_cache
DATA_ROOT=/scratch1/$USER/csci566
SC=$DATA_ROOT/processed/socialcounterfactuals
REV=182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1
mkdir -p "$SC"

echo "== 1/3 preprocess  $(date)"
if [ ! -f "$SC/metadata.csv" ]; then
  $PY data/preprocess_dataset.py --output "$SC" --revision $REV
fi

echo "== 2/3 splits  $(date)"
if [ ! -f "$SC/splits/val.csv" ]; then
  $PY data/create_splits.py --input "$SC/metadata.csv" \
      --output "$SC/splits" --seed 42
fi

echo "== 3/3 export images  $(date)"
$PY "data/export_images_muti-thread.py" \
    --metadata "$SC/metadata.csv" --splits "$SC/splits" \
    --output "$SC/exported" --revision $REV --workers 8

echo "== done  $(date)"
ls -lh "$SC/exported"
