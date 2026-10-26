# VLM Appearance Bias

CSCI 566 group project studying how race, gender, and physical characteristics affect occupation predictions in Vision-Language Models.

## Dataset Setup (USC CARC)

We use [SocialCounterfactuals](https://huggingface.co/datasets/Intel/SocialCounterfactuals), containing 170,832 images across 13,824 counterfactual groups.

### 1. Clone the repository

```bash
git clone git@github.com:amishana/vlm-appearance-bias.git
cd vlm-appearance-bias
git switch baseline
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
  --output "$DATA_ROOT/processed/socialcounterfactuals"
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
