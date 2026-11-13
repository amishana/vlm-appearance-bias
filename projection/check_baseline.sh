

DATA=$DATA_ROOT/processed/socialcounterfactuals/exported
python baseline/run_smolvlm2.py --model "$MODEL" --data "$DATA" --split val --limit 200 \
    --output projection/results/check_smolvlm2_baseline_local/val_predictions.csv

python - <<'EOF'
import pandas as pd
r = "projection/results/"
local = pd.read_csv(r + "check_smolvlm2_baseline_local/val_predictions.csv").set_index("image_id")
ours  = pd.read_csv(r + "check_smolvlm2_alpha0/val_predictions.csv").set_index("image_id")
orig  = pd.read_csv("baseline/results/smolvlm2/val_predictions.csv").set_index("image_id")
ids = ours.index
print("alpha0 vs local baseline:", (ours.model_response == local.loc[ids, "model_response"]).mean())
print("local vs teammate baseline:", (local.model_response == orig.loc[ids, "model_response"]).mean())
diff = ours.model_response != orig.loc[ids, "model_response"]
print(pd.DataFrame({"ours": ours.model_response[diff], "teammate": orig.loc[ids][diff].model_response}).head(15))
EOF