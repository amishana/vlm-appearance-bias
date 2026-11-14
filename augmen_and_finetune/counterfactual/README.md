# Counterfactual consistency fine-tuning (proposal 3.3)

## What the method is

No new images are generated. The "augmentation" is the SocialCounterfactuals
groups themselves: each training step uses one group (same occupation and
scene, different gender / race / physical attribute) and minimizes

    mean_i CE(gold occupation | image_i)  +  λ · JS(answer distributions of the group's variants)

so the model is pushed to give the same answer for every variant. λ = 0 is the
CE-only control: it learns the task without the consistency term, which
separates "consistency because of the JS term" from "consistency because accuracy went up".

## Two implementations

| | `counterfactual/` (VLM port, main result) | `counterfactual/clip/` (teammate's original, side result) |
|---|---|---|
| Model | Qwen2.5-VL-3B, SmolVLM2-2.2B (InternVL3-8B on the cluster) | CLIP ViT-B/32, closed-set over 156 labels |
| LoRA | vision encoder attention + connector; LM frozen | vision q/v |
| Output | free-text answer, same prompt as baseline | argmax over label prompts |
| Trained on / evaluated on | train / **val**, scored with `eval/score.py` + `eval/metrics.py` | train (val for model selection) / **test**, her evaluator |
| Comparable to baseline and LEACE? | yes | no |

## Run (desktop, 16 GB GPU)

```bash
pip install peft
# 1. ~10-minute end-to-end check (20 groups, 100 val images, *_smoke outputs)
SMOKE=1 BACKEND=qwen2_5_vl MODEL=$HOME/Documents/models/Qwen2.5-VL-3B-Instruct \
  DATA_ROOT=$HOME/Documents/csci566 bash counterfactual/run_counterfactual_local.sh
# 2. full run: main (λ=0.1) + CE-only control, each trained then predicted on val
BACKEND=qwen2_5_vl MODEL=$HOME/Documents/models/Qwen2.5-VL-3B-Instruct \
  DATA_ROOT=$HOME/Documents/csci566 bash counterfactual/run_counterfactual_local.sh
# 3. teammate's CLIP pipeline (separate venv, her pinned versions except torch)
DATA_ROOT=$HOME/Documents/csci566 bash counterfactual/run_cf_clip.sh
```

Every step skips if complete and resumes if interrupted (training checkpoints
every 250 steps). Adapters go to `$DATA_ROOT/cf_lora/<backend>/{cf_lora,cf_ce_only}/`,
predictions to `projection/results/<backend>_cf_lora/` and `<backend>_cf_ce_only/`.

Defaults: 1 epoch over all ~9.8k training groups, 4 variants per group. The
trainer prints the estimated hours left; to shorten, set `MAX_GROUPS=3000`
(a different `MAX_GROUPS` needs a new output folder). If training runs out of
GPU memory, use `IMAGES_PER_GROUP=2`.

## Scoring with everything else

```bash
python -m eval.score projection/results/*_local/val_predictions.csv \
    projection/results/*_leace_*/val_predictions.csv \
    projection/results/*_cf_*/val_predictions.csv --out projection/scored
python -m eval.metrics --scored projection/scored --out projection/eval_results
```

Compare `<backend>_cf_lora` against both `<backend>_local` (baseline) and
`<backend>_cf_ce_only` (control). Leave the `*_smoke` folders out.

## Caveats for the writeup

* Fine-tuning on the occupation labels also improves the task itself, so report
  the CE-only control next to the main run.
* Her CLIP run lost 10.7 points on CIFAR-10 (88.3 → 77.6); capability cost is
  part of the tradeoff, and a VLM capability check is still to be added.
* One seed. Training uses only the train split; val is used for evaluation only
  (no checkpoint selection on val).
