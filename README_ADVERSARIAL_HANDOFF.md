# SmolVLM2 adversarial debiasing handoff

## Status as of November 23, 2026

Julia's original-model validation inference completed (job 14366766). The training
smoke test passed (14432662). CE and adversarial pilots each completed 100 optimizer
steps on L40S (14432663, 14432664). Pilot evaluation (14432665) was still pending
in the latest supplied queue output. This is implementation evidence, not evidence
of bias reduction. Extended training/evaluation and probes remain unverified on CARC.

This handoff keeps all four Python implementation files byte-for-byte identical
to the uploaded source. Only the portable submission/worker scripts and this README
are new. Existing queued jobs use their existing snapshots; installing these files
neither changes nor submits those jobs. Coordinate with Julia before launching duplicates.

## Method and limitations

SmolVLM2-2.2B-Instruct; train only model.connector (21,233,664 parameters).
Freeze the vision encoder and language model parameters, retaining the gradient
path from occupation loss to the connector. Compare original vs occupation-only
cross-entropy (CE) vs CE plus an adversarial gender head through gradient reversal.
Mean-pool connector tokens across image tiles, batch size one. Adversary:
2048 -> 256 -> ReLU -> 2. Female/male refer to dataset annotations, not inferred
identities. Missing gender annotations contribute occupation loss only.

Connector AdamW lr=1e-5, head lr=1e-4, weight decay=.01, gradient clipping=1,
GRL strength=.1, accumulation=4, seed=42. Frozen backbone BF16; connector FP32
with BF16 autocast. Occupation loss covers assistant answer tokens only.

Pilot: select 256 train groups, 100 updates, evaluate 64 complete validation groups.
Extended: select 1024 train groups, 1000 updates, evaluate 256 validation groups.
These are update-limited experiments, NOT full-dataset training or full-validation
comparisons. Four images per optimizer step means 400/4000 image presentations,
not necessarily every image in the selected groups. Extended training starts fresh;
it does not resume pilot training. Checkpoints omit optimizer state; interrupted
runs cannot automatically resume. All output directories refuse overwrite.

Evaluation uses greedy generation, max_new_tokens=32, and the occupation prompt
recorded in the manifest. Raw exact-match/consistency summaries are preliminary.
Canonical evaluation uses the team's eval module, exclusions, and synonyms.
Paired group-bootstrap intervals (1000 resamples) are conditional on one training
seed and not adjusted for multiple comparisons. Full-group consistency can be high
for consistently wrong predictions; always report accuracy alongside it.

Fresh linear and MLP gender probes use 512 training groups excluded from the
fine-tuning selection, divided into disjoint probe fit/dev groups. Probe epochs
are chosen on dev, then measured on annotated examples from the same 256-group
validation selection. Seeds 42 and 73 are PROBE seeds, not multiple VLM training
seeds. Extraction requires 1000-step checkpoints and assumes seed 42 / 256 val
groups. It is not a general-purpose follow-up for arbitrary runs. Probes assess
pooled gender recoverability, not complete erasure, race, physical attributes, or
all token-level information. Held-out test evaluation and capability checks are
not implemented in this handoff. Do not tune on the test set.

## Required repository files

Add this package inside the team's existing repository. It is not standalone:
canonical analysis needs eval/__init__.py, score.py, occupation_map.py, metrics.py
from the compatible projection implementation. The launcher snapshots those files
for follow-up jobs. It also snapshots this package's Python files and worker script,
writes source hashes, configuration, job IDs, and per-job environment inventories.
The original baseline runner/data-export scripts remain in the team's repository.

## Environment and prepared data

Use the established environment; don't upgrade packages in an environment used by
queued jobs. Observed training environment: Python 3.11, torch 2.8.0+cu126,
torchvision 0.23.0+cu126, transformers 4.57.1, accelerate 1.16.0,
huggingface_hub 0.36.2. Also requires num2words, Pillow, NumPy, pandas; dataset
preparation additionally uses datasets. Consult the completed jobs' pip-freeze
files for full versions. This list is NOT a complete environment lockfile.
Training/extraction require CUDA with BF16 support. Request one A40/A100/L40S.
CPU probe and canonical jobs use main. All submission defaults use yzhao010_1531.

DATA_ROOT must contain:
- models/SmolVLM2-2.2B-Instruct/ with downloaded model/processor files
  and download_revision.txt
- processed/socialcounterfactuals/exported/{train,val,test}.csv and images/
  (image_path in CSV is relative to the exported directory)

Model revision: be92141c8d0ceb98596a923c308ac2bbb4a7c466
Dataset revision: 182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1
Train: 120122 images / 9702 groups; val: 26116 / 2117; test: 24594 / 2005.
Julia verified no image-ID or group overlap across splits.

## Run commands (Tony: configure paths first)

These commands SUBMIT NEW JOBS. Do not run them to install the handoff or to monitor
Julia's existing jobs. Use your own prepared data and environment, or coordinate
read access with Julia. Do not change permissions recursively on her directories.

From the repository root:

```bash
export DATA_ROOT="/scratch2/$USER/csci566"
export VLM_PYTHON="/project2/yzhao010_1531/$USER/conda_envs/csci566-vlm/bin/python"
export ACCOUNT=yzhao010_1531
bash adversarial/submit_handoff.sh pilot "$DATA_ROOT/results/tony_pilot_seed42"
```

The resulting jobs.txt records smoke, ce, adv, evaluate IDs. To queue extended
training after that pilot evaluation, replace the example ID below with its actual
numeric ID (do not use Julia's IDs for an independent run):

```bash
export PILOT_EVAL_JOB=YOUR_PILOT_EVALUATION_JOB_ID
bash adversarial/submit_handoff.sh extended "$DATA_ROOT/results/tony_extended_seed42"
```

Then read extended jobs.txt and set its numeric IDs:

```bash
export EXTENDED="$DATA_ROOT/results/tony_extended_seed42"
export CE_JOB=YOUR_EXTENDED_CE_JOB_ID
export ADV_JOB=YOUR_EXTENDED_ADV_JOB_ID
export EVAL_JOB=YOUR_EXTENDED_EVALUATION_JOB_ID
bash adversarial/submit_handoff.sh followup "$EXTENDED/followup"
```

Replace placeholders before running; validation rejects nonnumeric IDs. Existing
run directories prevent duplicates. If submission fails halfway, inspect jobs.txt;
do not delete it and resubmit blindly. Dependencies require successful upstream
completion and cancel downstream jobs if dependencies become invalid. They do
not guarantee prompt scheduling or scientifically good results.

## Outputs

- ce/train/checkpoint.pt and adv/train/checkpoint.pt: trained connectors and heads
- ce/train/training.jsonl, adv/train/training.jsonl: step diagnostics
- evaluate/evaluation/{original,ce,adv}_predictions.csv: matched validation outputs
- evaluate/evaluation/summary.json: raw metrics
- followup/features/: frozen embeddings and split metadata
- followup/probes/summary.csv: linear/MLP probe scores
- followup/canonical/metrics/: team metrics
- followup/canonical/paired_comparisons.json and comparison.svg: comparisons

## Julia's existing jobs: coordinate rather than duplicate

Pilot root: /scratch2/jlchun/csci566/results/adversarial_pilot_20261120_235322_497732
Extended root: /scratch2/jlchun/csci566/results/adversarial_extended_seed42
Follow-up root: extended root + /followup

| Job | ID | Latest supplied status |
| --- | --- | --- |
| Full original validation inference | 14366766 | Completed |
| Training smoke | 14432662 | Completed |
| CE pilot | 14432663 | Completed |
| Adversarial pilot | 14432664 | Completed |
| Pilot evaluation | 14432665 | Pending, priority |
| Extended CE | 14460572 | Pending, dependency on pilot evaluation |
| Extended adversarial | 14460573 | Pending, dependency on pilot evaluation |
| Extended evaluation | 14460574 | Pending, dependency on extended CE + ADV |
| Feature extraction | 14461319 | Pending, dependency on extended CE + ADV |
| Fresh probes | 14461320 | Pending, dependency on extraction |
| Canonical metrics | 14461321 | Pending, dependency on extended evaluation |

Portable launcher verification: Python parsing, Bash syntax, mocked Slurm job
submission/dependency construction, duplicate-run rejection, and worker argument
routing. These are local checks; the new launcher has not executed GPU jobs on CARC.
