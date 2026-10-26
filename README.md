# VLM Appearance Bias
CSCI 566 group project studying how race, gender, and physical characteristics affect VLM occupation predictions.

## Before starting
You must have memory and CPUs/GPUs allocated on USC's CARC service. In a browser, visit ```ondemand.carc.usc.edu/pun/sys/dashboard/batch_connect/sessions``` and click "JupyterLab" on the left tab. Fill out the information as needed, and wait to be allocated resources. 48GB of memory should be sufficient. You must be on USC Wi-Fi to visit this website. 

## 1. Clone the repo

```bash
git clone git@github.com:amishana/vlm-appearance-bias.git
cd vlm-appearance-bias
git switch baseline
```

## 2. Set up Python on CARC

```bash
module load conda
eval "$(conda shell.bash hook)"

conda create -n vlm-bias python=3.11 -y
conda activate vlm-bias
pip install -r requirements.txt
```

## 3. Set up scratch2

```bash
export DATA_ROOT="/scratch2/$USER/csci566"
mkdir -p "$DATA_ROOT/processed"
```

## 4. Preprocess SocialCounterfactuals

The scripts download/stream the dataset from Hugging Face automatically.

First, validate occupation extraction:

```bash
python baseline/validate_occupations.py \
  --output "$DATA_ROOT/processed/occupation_validation"
```

Then preprocess the metadata:

```bash
python baseline/preprocess_dataset.py \
  --output "$DATA_ROOT/processed/socialcounterfactuals"
```

Run these on a CARC compute node.

## 5. Output

The processed data will be saved in:

```text
/scratch2/<username>/csci566/processed/socialcounterfactuals/
├── metadata.csv
└── preprocessing_report.json
```

`metadata.csv` contains occupation labels, captions, demographic/physical attributes, and counterfactual group IDs.

The dataset has 170,832 images, 156 extracted occupation labels, and 13,824 counterfactual groups.

**Note:** This step only preprocesses metadata. Images have not been exported yet. Large datasets stay on scratch2, not GitHub.
