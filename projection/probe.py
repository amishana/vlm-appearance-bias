"""Leakage and capability probes on pooled connector features (CPU only).

For each erasure strength alpha, trains probes on erased TRAIN features and
evaluates on erased VAL features (probes are always retrained after erasure).

    attribute probes   gender / race / physical, linear and MLP,
                       balanced accuracy vs chance (1 / n_classes)
    occupation probe   linear, accuracy and macro F1 (what should survive)
    distortion         relative L2 change and cosine to the original features

Pooled erasure is exact: the eraser is affine, so erasing the token mean
equals the mean of erased tokens.

Usage (from the repo root):
    python -m projection.probe --features $FEAT --eraser $FEAT/eraser_all.pt \
        --alphas 0 0.25 0.5 0.75 1 --model-name qwen2_5_vl_leace_all \
        --out projection/eval_results/leakage.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from eval.occupation_map import EXCLUDED, GOLD_TO_CANONICAL
from eval.score import normalize as label_normalize
from projection.common import ATTR_ORDER, AffineEraser


def load(features, split):
    X = np.load(features / f"{split}_pooled.npy")
    meta = pd.read_csv(features / f"{split}_meta.csv", keep_default_na=False)
    assert len(X) == len(meta)
    return X, meta


def attr_labels(meta, atype):
    lab = np.where(meta["a1_type"] == atype, meta["a1"],
                   np.where(meta["a2_type"] == atype, meta["a2"], ""))
    return lab


def occupation_labels(meta):
    norm = meta["occupation_raw"].map(label_normalize)
    return norm.map(lambda g: "" if g in EXCLUDED else GOLD_TO_CANONICAL.get(g, "")).to_numpy()


def fit_eval(Xtr, ytr, Xva, yva, kind, seed):
    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva = scaler.transform(Xtr), scaler.transform(Xva)
    if kind == "linear":
        clf = LogisticRegression(max_iter=2000, C=1.0)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(512,), early_stopping=True,
                            max_iter=200, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf.predict(Xva)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--eraser", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0])
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-mlp", action="store_true")
    p.add_argument("--seed", type=int, default=566)
    args = p.parse_args()

    Xtr0, mtr = load(args.features, "train")
    Xva0, mva = load(args.features, "val")
    probes = ["linear"] if args.no_mlp else ["linear", "mlp"]
    rows = []

    for alpha in args.alphas:
        er = AffineEraser.load(args.eraser, alpha=alpha)
        Xtr = er(torch.from_numpy(Xtr0)).numpy()
        Xva = er(torch.from_numpy(Xva0)).numpy()
        base = {"model": args.model_name, "eraser": args.eraser.name,
                "erased_types": " ".join(er.meta.get("types", [])), "alpha": alpha}

        diff = np.linalg.norm(Xva - Xva0, axis=1) / np.linalg.norm(Xva0, axis=1)
        cos = (Xva * Xva0).sum(1) / (np.linalg.norm(Xva, axis=1) * np.linalg.norm(Xva0, axis=1))
        rows.append({**base, "target": "distortion", "probe": "",
                     "rel_l2": float(diff.mean()), "cosine": float(cos.mean())})

        for atype, values in ATTR_ORDER.items():
            ytr, yva = attr_labels(mtr, atype), attr_labels(mva, atype)
            itr, iva = ytr != "", yva != ""
            for kind in probes:
                pred = fit_eval(Xtr[itr], ytr[itr], Xva[iva], yva[iva], kind, args.seed)
                rows.append({**base, "target": atype, "probe": kind,
                             "n_train": int(itr.sum()), "n_val": int(iva.sum()),
                             "n_classes": len(values), "chance": 1 / len(values),
                             "balanced_acc": balanced_accuracy_score(yva[iva], pred)})
            print(f"alpha={alpha} {atype} done", flush=True)

        ytr, yva = occupation_labels(mtr), occupation_labels(mva)
        itr, iva = ytr != "", yva != ""
        pred = fit_eval(Xtr[itr], ytr[itr], Xva[iva], yva[iva], "linear", args.seed)
        rows.append({**base, "target": "occupation", "probe": "linear",
                     "n_train": int(itr.sum()), "n_val": int(iva.sum()),
                     "n_classes": len(set(ytr[itr])),
                     "accuracy": accuracy_score(yva[iva], pred),
                     "macro_f1": f1_score(yva[iva], pred, average="macro")})

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():  # append runs for other erasers / models
        out = pd.concat([pd.read_csv(args.out), out], ignore_index=True)
    out.to_csv(args.out, index=False)
    print(out[out["alpha"].isin(args.alphas)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
