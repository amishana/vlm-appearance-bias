#!/bin/bash
# Submit one GPU job per model x prompt (9 jobs by default).
#
# Usage (from the repo root on CARC):
#   bash prompt_debiasing/submit_all.sh               # full validation split
#   bash prompt_debiasing/submit_all.sh --groups 500  # 500-group subset (about 4x faster)
#
# Optional: PROMPTS="P1" or BACKENDS="qwen2_5_vl" to submit fewer jobs.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

EXPORTED=/scratch1/$USER/csci566/processed/socialcounterfactuals/exported
DATA=$EXPORTED
SPLIT=${SPLIT:-val}
BACKENDS=${BACKENDS:-"smolvlm2 qwen2_5_vl internvl3"}
PROMPTS=${PROMPTS:-"P1 P2 P4"}

if [ "${1:-}" = "--groups" ]; then
  N=$2
  DATA=/scratch1/$USER/csci566/processed/socialcounterfactuals/subset_${N}
  mkdir -p "$DATA"
  ln -sfn "$EXPORTED/images" "$DATA/images"
  OPENBLAS_NUM_THREADS=1 $HOME/.conda/envs/vlm-bias/bin/python - "$EXPORTED/$SPLIT.csv" "$DATA/$SPLIT.csv" "$N" <<'PY'
import sys, pandas as pd
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
df = pd.read_csv(src, keep_default_na=False)
# same random groups every time (seed 0), spread across all three segments
groups = (df.drop_duplicates("counterfactual_set")
            .groupby("dataset_segment", group_keys=False)
            .apply(lambda g: g.sample(frac=n / df.counterfactual_set.nunique(), random_state=0)))
sub = df[df.counterfactual_set.isin(groups.counterfactual_set)]
sub.to_csv(dst, index=False)
print(f"subset: {sub.counterfactual_set.nunique()} groups, {len(sub)} images -> {dst}")
PY
  SUFFIX="_sub${N}"
else
  SUFFIX=""
fi

for B in $BACKENDS; do
  for P in $PROMPTS; do
    sbatch --job-name="${B}_${P}${SUFFIX}" \
      --export=ALL,BACKEND=$B,PROMPT=$P,SPLIT=$SPLIT,DATA=$DATA,TAG=$SUFFIX \
      prompt_debiasing/run_full.slurm
  done
done
squeue -u "$USER"