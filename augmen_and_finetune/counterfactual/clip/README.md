# Counterfactual Augmentation + LoRA Team Code

This package handles the complete SocialCounterfactuals benchmark workflow, including data integration, CLIP vision LoRA training, prediction, and paired evaluation.

Place this code package at the same directory level as the separate `counterfactual-team-model` model package.

## Installation

**Tested environment:** Python 3.11, PyTorch 2.5.1, Transformers 4.49.0, and PEFT 0.14.0.

The pipeline has been executed on macOS using MPS and supports automatic device selection among CUDA, MPS, and CPU.

CARC and Google Colab have not been directly tested. Long-running CARC jobs should be executed within an allocated compute resource session.

```bash
cd counterfactual-team-code

python3.11 -m venv .venv

source .venv/bin/activate

python -m pip install -r requirements.txt

python -m pytest -q
```

## Running the Benchmark Using an Existing Team Export

The input directory must be `exported/`, containing:

* `train.csv`
* `val.csv`
* `test.csv`
* `images/`

The input is **not** a CSV-only `splits/` directory.

```bash
PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/run_benchmark.sh /path/to/exported "$PWD/outputs/new_run"
```

The output directory must not already exist.

The script automatically performs the following operations:

1. Imports the benchmark while preserving the team's original data splits.
2. Generates predictions using the baseline model.
3. Trains the LoRA-adapted model for 3 epochs.
4. Generates predictions using the fine-tuned model.
5. Performs paired evaluation.

A custom model or training configuration can be supplied using:

```bash
CONFIG_PATH=/path/to/custom.yaml
```

This overrides the default model and training configuration template.

**Generated artifacts include:**

* `processed/`
* `config.yaml`
* `baseline.csv`
* `adapted.csv`
* Two metadata files
* `training/best_adapter/`
* Training logs
* `comparison.json`

**Important:** Results obtained using different models or candidate vocabularies must not be treated as directly comparable results for the same task.

## Downloading and Running the Complete Benchmark from Scratch

To download and execute the full benchmark pipeline:

```bash
python -u scripts/full_benchmark_pipeline.py
```

The pipeline automatically:

1. Downloads all 61 dataset shards (27.22 GB) from a fixed revision.
2. Verifies each shard against its official LFS SHA-256 checksum.
3. Runs the team's original preprocessing pipeline, preserving the original file and row order.
4. Creates the dataset splits using random seed 42.
5. Exports all images as PNG files.
6. Runs model training and evaluation.
7. Performs an additional CIFAR-10 evaluation as a proxy for general capability retention.

**Disk space requirement:** At least 110 GB of available storage is recommended.

### Data and Output Directories

Downloaded and processed data:

```text
data/benchmark_parquet_20261027/
data/benchmark_full_20261027/
```

Experiment outputs:

```text
outputs/benchmark_full_20261027/
```

The dates in these directory names are fixed experimental protocol identifiers. They do not restrict execution to those dates, and the pipeline can run on other machines.

### Monitoring and Resuming

Pipeline progress can be monitored using:

```text
PIPELINE_STATUS.json
```

Individual stage logs are also available.

The download process supports resuming interrupted downloads.

During training, the pipeline saves a recovery checkpoint every 250 steps:

```text
resume.pt
```

If execution is interrupted, rerun the same pipeline entry point with the same configuration to resume.

If the configuration changes, use a new experiment output directory or a separate code package.

On macOS, the pipeline can automatically prevent the system from sleeping during execution. On Linux, it runs directly without this feature.

## Running Inference with a Pretrained Adapter

To perform inference directly on a local image using an already trained LoRA adapter:

```bash
python infer_images.py --adapter ../counterfactual-team-model/checkpoints/full_benchmark_lora --image /path/to/image.jpg --output outputs/image_predictions.json
```

### Base Model

The fixed base model is:

```text
openai/clip-vit-base-patch32
```

Model revision:

```text
3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
```

The base model weights are downloaded from Hugging Face during the first execution.

No private API key is required.

The model package includes:

* The image processor.
* A fixed vocabulary of 156 occupations.
* The trained LoRA adapter.

**Model limitations:**

This is a closed-set classification model, not a generative conversational vision-language model (VLM).

It predicts scores over a fixed set of occupation candidates.

The softmax outputs should not be interpreted as calibrated probabilities of real-world occupations.

### Benchmark Prediction with Metadata

To generate benchmark predictions while preserving sample-level and attribute metadata, use:

```bash
python predict.py --config outputs/new_run/config.yaml --adapter ../counterfactual-team-model/checkpoints/full_benchmark_lora --split test --output outputs/reexported_predictions.csv
```

The following must match the configuration used during adapter training:

* Candidate vocabulary and its ordering.
* Base model.
* Prompt configuration.

If the benchmark manifest and the adapter's training provenance do not match, the paired evaluation will reject the results as belonging to the same held-out experiment.

## File Responsibilities

| File                      | Responsibility                                                                                                                                                                    |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `import_benchmark.py`     | Imports CSV data while preserving sample IDs and dataset splits; validates complete counterfactual groups and checks for image/pixel leakage.                                     |
| `cf_model.py`             | Implements CLIP, image-text feature extraction, group-balanced cross-entropy (CE), and Jensen–Shannon (JS) divergence.                                                            |
| `train_lora.py`           | Implements vision q/v LoRA training, validation-based model selection, checkpoint saving and reloading, and optimizer/RNG state restoration.                                      |
| `predict.py`              | Generates per-image occupation scores, attributes, and provenance metadata.                                                                                                       |
| `evaluate_predictions.py` | Computes accuracy, group consistency, attribute-specific performance gaps, and paired whole-group bootstrap evaluations.                                                          |
| `evaluate_capability.py`  | Evaluates general capability retention using the CIFAR-10 ten-class test set as a proxy.                                                                                          |
| `infer_images.py`         | Runs inference on local images without requiring a benchmark manifest.                                                                                                            |
| `prepare_data.py`         | Provides shared metadata validation, complete-group checks, and the earlier data preparation interface. Use the full pipeline entry point above for complete benchmark execution. |
| `benchmark_source/data/`  | Contains the team's original download, preprocessing, splitting, and export code from the source ZIP archive.                                                                     |
| `tests/`                  | Contains implementation tests using synthetic pixels and a randomly initialized tiny CLIP model. These tests do not represent experimental results.                               |

## Experimental Results and Limitations

### Dataset Statistics

The complete dataset contains:

| Metric                                       |   Value |
| -------------------------------------------- | ------: |
| Total images                                 | 170,832 |
| Counterfactual groups                        |  13,824 |
| Candidate occupations                        |     156 |
| Training images                              | 120,122 |
| Validation images                            |  26,116 |
| Test images                                  |  24,594 |
| Occupations represented in test ground truth |     114 |

Although the test ground truth contains 114 occupations, the classification candidate set remains fixed at 156 classes throughout evaluation.

### Training Configuration

| Parameter                      |   Value |
| ------------------------------ | ------: |
| Training epochs                |       3 |
| Total optimization steps       |  29,106 |
| Best checkpoint                | Epoch 3 |
| Checkpoint reload verification |  Passed |

### Occupation Prediction Performance

| Metric                     | Baseline CLIP | LoRA-Adapted CLIP |
| -------------------------- | ------------: | ----------------: |
| Occupation accuracy        |        60.68% |            84.16% |
| Complete-group consistency |        39.10% |            75.61% |

The LoRA-adapted model improved occupation classification accuracy and complete-group consistency on the benchmark.

### CIFAR-10 Capability Retention Evaluation

A correction was made after discovering that a trailing empty line in the CIFAR-10 class list had incorrectly been interpreted as an additional category.

The corrected ten-class evaluation was rerun on **October 28, 2026**.

| Metric            | Baseline CLIP | LoRA-Adapted CLIP |
| ----------------- | ------------: | ----------------: |
| CIFAR-10 accuracy |        88.30% |            77.64% |

**Performance change: −10.66 percentage points.**

These corrected results show a measurable decrease in performance on the CIFAR-10 capability proxy.

Therefore, the experiment **does not support a claim of lossless general capability retention**.

The previous incorrect results of 87.13% and 77.00% should not be cited or reused.

### Experimental Limitations

The experiment has several important limitations:

* Only one base model was evaluated.
* Only one random seed was used.
* The occupation benchmark uses synthetic data.
* The results do not establish causal fairness for real-world populations.
* Improvements in benchmark group consistency should not be interpreted as proof of fairness in real-world deployment.
* CIFAR-10 is an external capability proxy, not a comprehensive measurement of general model capabilities.

For the complete experimental summary, refer to:

```text
counterfactual-team-model/results/RESULTS_CN.md
```

## Testing and Verification

All **33 implementation tests passed**.

The test suite includes a complete execution of the shell entry point, covering the workflow from benchmark import through model training and evaluation using a randomly initialized test model.

Additional verification records generated after relocating the code are available in:

```text
verification/
```

These tests verify implementation behavior and pipeline functionality. They should not be interpreted as evidence of experimental model performance.

## Package Contents and Exclusions

The code package does **not** include:

* Benchmark images.
* Downloaded dataset shards.
* Python virtual environments.
* Base CLIP model weights.
* Private credentials or API keys.

The required benchmark data and base model weights must be provided or downloaded separately using the procedures described above.