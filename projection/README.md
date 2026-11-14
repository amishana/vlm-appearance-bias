# Connector-level concept erasure (LEACE)

Models (backend name = baseline results folder):

| backend | checkpoint | connector hooked | dim |
|---|---|---|---|
| `qwen2_5_vl` | Qwen2.5-VL-3B-Instruct | `visual.merger` | 2048 |
| `smolvlm2` | SmolVLM2-2.2B-Instruct | `connector` | 2048 |
| `internvl3` | InternVL3-8B | `mlp1` | 3584 |

The eraser is fitted on the **train** split and applied to the connector
output at inference. `projection/models.py` loads and preprocesses each model
exactly like its baseline runner, and predictions keep the baseline columns,
so `eval/score.py` and `eval/metrics.py` run on them unchanged.

## Pipeline

GPU steps, one job per model (edit the model paths):

```bash
mkdir -p logs
sbatch --export=ALL,BACKEND=qwen2_5_vl,MODEL=/path/to/Qwen2.5-VL-3B-Instruct projection/run_projection.slurm
sbatch --export=ALL,BACKEND=smolvlm2,MODEL=/path/to/SmolVLM2-2.2B-Instruct  projection/run_projection.slurm
sbatch --export=ALL,BACKEND=internvl3,MODEL=/path/to/InternVL3-8B           projection/run_projection.slurm
```

Each job extracts features and statistics, fits the four main erasers (all
attributes plus one per attribute, from a 20k-image sample) and the
sample-size erasers, checks that `--alpha 0` reproduces the baseline, and runs
the erased model on val.

CPU steps, after the jobs finish (`pip install scikit-learn` once):

```bash
# leakage / capability probes, one block per model and eraser
for B in qwen2_5_vl smolvlm2 internvl3; do
  FEAT=/scratch2/$USER/csci566/features/$B
  for E in gender_race_physical gender race physical; do
    python -m projection.probe --features $FEAT --eraser $FEAT/eraser_$E.pt \
        --alphas 0 1 --model-name ${B}_leace_$E --out projection/eval_results/leakage.csv
  done
done

# existing evaluation, baselines and erased models side by side
python -m eval.score baseline/results/*/val_predictions.csv \
    projection/results/*_leace_*/val_predictions.csv --out projection/scored
python -m eval.metrics --scored projection/scored --out projection/eval_results
```

## Sample-size check (is 20k enough?)

The Slurm job extracts 40k train images once and saves statistics after the
first ~5k, 10k and 20k (nested samples: each is a prefix of the next, and
identical to a separate run with that `--max-images`). It then fits
`eraser_all_n{5000,10000,20000,40000}.pt`. Afterwards, on CPU:

```bash
for B in qwen2_5_vl smolvlm2 internvl3; do
  FEAT=/scratch2/$USER/csci566/features/$B
  # how similar is each eraser to the 40k one?
  python -m projection.stability --model-name $B \
      --erasers $FEAT/eraser_all_n{5000,10000,20000,40000}.pt \
      --out projection/eval_results/stability.csv
  # does held-out leakage / occupation decodability change with the eraser's sample?
  # (probes always trained on all 40k extracted train images, so only the eraser varies)
  for N in 5000 10000 20000 40000; do
    python -m projection.probe --features $FEAT --eraser $FEAT/eraser_all_n$N.pt \
        --alphas 1 --model-name ${B}_n$N --out projection/eval_results/leakage_by_n.csv
  done
done
```

Report it as converged if `subspace_overlap` is near 1 and the probe numbers
change by less than your threshold (e.g. 0.01 balanced accuracy) from 20k to
40k. `probe.py --train-max N` restricts probe training to the same nested
sample, if you also want probes trained on N images.

## Outputs

| File | Answers |
|---|---|
| `projection/eval_results/*.csv` (eval.metrics) | Did accuracy gaps, flip rates and set consistency shrink? Did overall accuracy drop? |
| `projection/eval_results/leakage.csv` | Is the attribute still decodable (linear, MLP)? Is occupation still decodable? How far did features move? |

## Notes

* Images carry two of three attribute types, so statistics are kept per type
  and erasers are composed. Each eraser is affine, so later steps keep earlier
  types erased; `fit_eraser` prints `residual_xcov` (should be ~1e-12 or less).
* LEACE guarantees **linear** erasure only. MLP probe accuracy above chance is
  the evidence for nonlinear leakage (Expected Outcome 3).
* Pooled erasure equals token-level erasure when every image yields the same
  number of connector tokens; `extract_features` prints the token counts.
* `alpha` between 0 and 1 weakens the edit the LM sees, but a retrained probe
  still decodes the attribute until alpha = 1. For the tradeoff plot, compare
  the per-attribute erasers instead.
* `eval/plots.py` only draws models listed in `MODEL_COLORS`; add the
  `*_leace_*` names there to include them in figures.
