#!/bin/bash
# Usage: bash adversarial/submit_handoff.sh pilot|extended|followup ABSOLUTE_NEW_RUN
set -euo pipefail
STAGE=${1:?Specify pilot, extended, or followup}
RUN=${2:?Specify a NEW absolute output directory}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${DATA_ROOT:?Set DATA_ROOT to the prepared csci566 directory}"
: "${VLM_PYTHON:?Set VLM_PYTHON to an absolute Python executable}"
ACCOUNT=${ACCOUNT:-yzhao010_1531}
[[ "$RUN" = /* && "$DATA_ROOT" = /* && "$VLM_PYTHON" = /* ]] || { echo 'Paths must be absolute' >&2; exit 1; }
[[ -x "$VLM_PYTHON" ]] || { echo 'Python executable not found' >&2; exit 1; }
case "$STAGE" in
 pilot) STEPS=100; TRAIN_GROUPS=256; VAL_GROUPS=64;;
 extended)
  : "${PILOT_EVAL_JOB:?Set PILOT_EVAL_JOB to the pilot evaluation job ID}"
  [[ "$PILOT_EVAL_JOB" =~ ^[0-9]+$ ]] || exit 1
  STEPS=1000; TRAIN_GROUPS=1024; VAL_GROUPS=256;;
 followup)
  : "${EXTENDED:?Set EXTENDED to the extended run directory}"
  : "${CE_JOB:?Set CE_JOB}"; : "${ADV_JOB:?Set ADV_JOB}"; : "${EVAL_JOB:?Set EVAL_JOB}"
  [[ "$EXTENDED" = /* && -d "$EXTENDED" && "$CE_JOB" =~ ^[0-9]+$ && "$ADV_JOB" =~ ^[0-9]+$ && "$EVAL_JOB" =~ ^[0-9]+$ ]] || exit 1
  for name in __init__ score occupation_map metrics; do test -f "$REPO/eval/$name.py"; done
  STEPS=1000; TRAIN_GROUPS=1024; VAL_GROUPS=256;;
 *) echo 'Use pilot, extended, or followup' >&2; exit 1;;
esac
for file in train.csv val.csv; do test -f "$DATA_ROOT/processed/socialcounterfactuals/exported/$file"; done
test -f "$DATA_ROOT/models/SmolVLM2-2.2B-Instruct/download_revision.txt"
# Atomic creation refuses duplicate runs. Never remove a partial record to retry blindly.
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
trap 'echo "Submission stopped. Inspect $RUN/jobs.txt before retrying; some jobs may already be queued." >&2' ERR
mkdir -p "$RUN/logs" "$RUN/code/adversarial" "$RUN/code/julia_followup"
cp "$REPO/adversarial/pilot.py" "$RUN/code/adversarial/"
cp "$REPO/julia_followup/"*.py "$RUN/code/julia_followup/"
cp "$REPO/adversarial/run_handoff.sbatch" "$RUN/run.sbatch"
if [ "$STAGE" = followup ]; then
 mkdir "$RUN/code/eval"
 for name in __init__ score occupation_map metrics; do cp "$REPO/eval/$name.py" "$RUN/code/eval/"; done
fi
for name in DATA_ROOT VLM_PYTHON STEPS TRAIN_GROUPS VAL_GROUPS EXTENDED; do
 printf '%s=%q\n' "$name" "${!name-}" >> "$RUN/config.sh"
done
find "$RUN/code" -name '*.py' -type f -exec sha256sum {} + > "$RUN/source_hashes.txt"
git -C "$REPO" rev-parse HEAD > "$RUN/git_commit.txt" 2>/dev/null || printf 'Not available\n' > "$RUN/git_commit.txt"
COMMON=(--parsable --account="$ACCOUNT" --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=48G --chdir="$REPO" --output="$RUN/logs/%x_%j.log" --kill-on-invalid-dep=yes)
submit() {
 local mode=$1 dep=$2 partition=$3 limit=$4 id
 local opts=("${COMMON[@]}" --job-name="vlm-$STAGE-$mode" --partition="$partition" --time="$limit")
 if [ "$partition" = gpu ]; then opts+=(--gres=gpu:1 '--constraint=a40|a100|l40s'); fi
 if [ -n "$dep" ]; then opts+=(--dependency="afterok:$dep"); fi
 id=$(sbatch "${opts[@]}" "$RUN/run.sbatch" "$RUN" "$mode")
 id=${id%%;*}
 [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $id" >&2; return 1; }
 printf '%s=%s\n' "$mode" "$id" >> "$RUN/jobs.txt"
 printf '%s\n' "$id"
}
printf 'RESULTS=%s\n' "$RUN" > "$RUN/jobs.txt"
if [ "$STAGE" = followup ]; then
 EXTRACT_JOB=$(submit extract "$CE_JOB:$ADV_JOB" gpu 02:00:00)
 submit probes "$EXTRACT_JOB" main 02:00:00
 submit metrics "$EVAL_JOB" main 01:00:00
else
 if [ "$STAGE" = pilot ]; then
  DEP=$(submit smoke '' gpu 01:00:00)
 else
  DEP=$PILOT_EVAL_JOB
 fi
 CE=$(submit ce "$DEP" gpu 04:00:00)
 ADV=$(submit adv "$DEP" gpu 04:00:00)
 submit evaluate "$CE:$ADV" gpu 04:00:00
fi
cat "$RUN/jobs.txt"
