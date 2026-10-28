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

### Chosen Models
The models chosen for comparison in this experiment are as follows: 
  SmolVLM2-2.2B-Instruct
  Qwen2.5-VL-3B
  InternVL3-8B
Download them onto your CARC scratch2 space through HuggingFace.
