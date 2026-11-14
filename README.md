# VLM Appearance Bias

CSCI 566 group project studying how race, gender, and physical characteristics affect occupation predictions in Vision-Language Models.

## Dataset Setup (USC CARC)

Go to USC CARC OnDemand and select JupyterLab from the left sidebar. 
```https://ondemand.carc.usc.edu/pun/sys/dashboard/batch_connect/sessions```
Fill out the required information to request a CARC allocation. All steps in this README can be completed using CPUs, so a GPU allocation is not necessary. CPU allocations have near-zero queue times.

I recommend requesting a 12-hour allocation and completing the setup in one session, since exporting all the images takes approximately 6 hours.
Read through the entire README before starting, especially the important note about file paths at the bottom.

Note: You must be connected to USC Wi-Fi (in person or through the VPN) to access CARC.

### 1. Clone the repository

```bash
git clone git@github.com:amishana/vlm-appearance-bias.git
cd vlm-appearance-bias
```

### 2. Set up the environment

```bash
module load conda
eval "$(conda shell.bash hook)"
conda create -n vlm-bias python=3.11 -y
conda activate vlm-bias
pip install -r requirements.txt
```

### 3. Set up scratch2

```bash
export DATA_ROOT="/scratch2/$USER/csci566"
mkdir -p "$DATA_ROOT/processed"
```

### 4. Preprocess the dataset

The scripts load SocialCounterfactuals from Hugging Face automatically.

```bash
python data/preprocess_dataset.py \
  --output "$DATA_ROOT/processed/socialcounterfactuals" \
  --revision 182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1
```

This creates `metadata.csv` with occupation labels, captions, demographic attributes, and counterfactual group IDs.

### 5. Create train/validation/test splits

```bash
python data/create_splits.py \
  --input "$DATA_ROOT/processed/socialcounterfactuals/metadata.csv" \
  --output "$DATA_ROOT/processed/socialcounterfactuals/splits" \
  --seed 42
```

The splits use approximately 70% training, 15% validation, and 15% testing, keeping complete counterfactual groups together.

### 6. Export images

```bash
python data/export_images.py \
  --metadata "$DATA_ROOT/processed/socialcounterfactuals/metadata.csv" \
  --splits "$DATA_ROOT/processed/socialcounterfactuals/splits" \
  --output "$DATA_ROOT/processed/socialcounterfactuals/exported" \
  --revision 182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1
```

This saves the images and generates updated CSV files containing image paths.

Run it in a CARC compute allocation with sufficient time. On CPUs only, this took me about 6 hours.

### 7. Final dataset location

```text
/scratch2/<username>/csci566/processed/socialcounterfactuals/
├── metadata.csv
├── preprocessing_report.json
├── splits/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── group_assignments.csv
└── exported/
    ├── images/
    ├── metadata_with_images.csv
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── export_report.json
```

Use the CSV files inside `exported/` for experiments after image export completes. Image paths are relative to the `exported/` directory.

**Note:** Large datasets and model weights stay on CARC scratch2, not GitHub. The dataset scripts should be run on a CARC compute node.


### IMPORTANT NOTE ON FILE PATHS

`data/inspect_dataset.py` and `data/validate_occupations.py`
currently have `/scratch2/amishana/` as their default output path.
If running either script, pass `--output` with your own scratch2 path.

The main preprocessing, splitting, and image export commands above
already use `$DATA_ROOT`, so no path changes are needed.

## Running Experiments on a Local GPU

Once the dataset is exported, experiments can run on any single-GPU machine (a desktop or a CARC GPU node). Every run is launched from one file, `run.sh`, at the repo root. Each collaborator adds their own lines to it, so in the end the whole pipeline runs with:

```bash
bash run.sh
```

### Scripts

| Script | What it does |
|---|---|
| `shellscripts/check_baseline.sh` | Runs the baseline model on the validation split and writes `val_predictions.csv`. |
| `projection/run_project_local.sh` | The full projection (LEACE) pipeline for one model: (0) local baseline, (1) train and val feature extraction, (2) fit the erasers, (3) alpha=0 sanity check against the baseline, (4) full erased run on val. |

Each call runs **one model**. Both scripts are **resumable**: finished steps are skipped and partial prediction files pick up where they stopped, so you can rerun `run.sh` after a crash or interruption without redoing work.

### Setup on your machine

1. **Clone the repo and create the environment** (same as steps 1 and 2 above, skipping `module load conda` if you're not on CARC).

2. **Copy the exported dataset** to your machine, keeping the `processed/socialcounterfactuals/exported/` layout from step 7. Point `DATA_ROOT` at the folder that contains `processed/`:
   ```bash
   export DATA_ROOT=YOUR/PATH/TO/csci566     # add to ~/.bashrc so run.sh always sees it
   ```

3. **Download the model weights** to a local folder:
   ```bash
   hf download Qwen/Qwen2.5-VL-3B-Instruct       --local-dir YOUR/PATH/TO/models/Qwen2.5-VL-3B-Instruct
   hf download HuggingFaceTB/SmolVLM2-2.2B-Instruct --local-dir YOUR/PATH/TO/models/SmolVLM2-2.2B-Instruct
   hf download OpenGVLab/InternVL3-8B             --local-dir YOUR/PATH/TO/models/InternVL3-8B
   ```

4. **Test one model by itself** before adding it to `run.sh`:
   ```bash
   BACKEND=smolvlm2 MODEL=YOUR/PATH/TO/models/SmolVLM2-2.2B-Instruct \
     bash projection/run_project_local.sh
   ```

### Settings

Set these on the same line as the command, or `export` them beforehand.

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `BACKEND` | yes | | `qwen2_5_vl`, `smolvlm2` or `internvl3` |
| `MODEL` | yes | | Path to the downloaded model folder |
| `DATA_ROOT` | yes | | Folder containing `processed/socialcounterfactuals/exported/` |
| `REPO` | no | `~/Documents/vlm-appearance-bias` | Repo root. **Set this if you cloned somewhere else.** |
| `ENV_NAME` | no | `vlm-bias` | Conda env, activated only if no env is active |
| `GPU` | no | `0` | GPU index |
| `THREADS` | no | `8` | CPU threads for torch and data loading |

### Adding your runs to `run.sh`

`run.sh` is a list of commands grouped by method. To add yours:

1. Put each run on **its own line** under the matching section (`# baseline set up`, `#Projection method`, or a new `# <method name>` section for a new method).
2. Use the **exact command you tested** in step 4, with your real paths, not `YOUR/PATH/TO/...` placeholders.
3. Add a comment with **your name** above your lines so others know whose they are.
4. Keep **one model per line**. Don't loop inside `run.sh`.
5. If you add a new method script, follow the same pattern as the existing ones (read `BACKEND`, `MODEL`, `DATA_ROOT` from env vars, skip finished steps), put it in `shellscripts/` or your method's folder, and document it in the Scripts table above.

Example:

```bash
#!/bin/bash
# baseline set up
BACKEND=qwen2_5_vl MODEL=~/Documents/models/Qwen2.5-VL-3B-Instruct bash shellscripts/check_baseline.sh

#Projection method
BACKEND=qwen2_5_vl MODEL=~/Documents/models/Qwen2.5-VL-3B-Instruct bash projection/run_project_local.sh
BACKEND=smolvlm2   MODEL=~/Documents/models/SmolVLM2-2.2B-Instruct  bash projection/run_project_local.sh

# <your method> (<your name>)
BACKEND=internvl3  MODEL=~/Documents/models/InternVL3-8B           bash <your_folder>/run_<your_method>_local.sh
```

Always run `run.sh` **from the repo root**, since its paths are relative to it. A failing line doesn't stop the lines after it; check the logs.

### Outputs and logs

- **Logs:** `logs/leace_<backend>_<timestamp>.log`, one per run.
- **Features and erasers:** `$DATA_ROOT/features/<backend>/`.
- **Predictions:** `projection/results/`:
  - `<backend>_local/val_predictions.csv` (baseline)
  - `check_<backend>_alpha0/` (sanity check)
  - `<backend>_leace_all/val_predictions.csv` (erased run)

Don't commit features, model weights or large result files. Share them through scratch2 or a shared drive instead.