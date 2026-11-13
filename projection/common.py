"""Shared pieces for connector-level concept erasure.

The prompt, output columns and `correct` rule are imported from the baseline
runner (identical across the three baselines) so erased predictions are
scored exactly like the baseline ones.
"""

import csv
import random
from pathlib import Path

import torch

from baseline.run_qwen2_5_vl import FIELDS, PROMPT, normalize  # noqa: F401
from eval.metrics import ATTR_ORDER

ATTR_TYPES = list(ATTR_ORDER)  # gender, race, physical


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def read_rows(data_dir, split):
    with (Path(data_dir) / f"{split}.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def sample_by_set(rows, max_images, seed):
    """Random whole counterfactual sets until ~max_images rows are taken."""
    if max_images is None or max_images >= len(rows):
        return rows
    by_set = {}
    for r in rows:
        by_set.setdefault(r["counterfactual_set"], []).append(r)
    sets = sorted(by_set)
    random.Random(seed).shuffle(sets)
    out = []
    for s in sets:
        if len(out) >= max_images:
            break
        out.extend(by_set[s])
    return out


def image_attrs(row):
    """{'gender': 'male', 'race': 'Asian'} for one image (two of three types)."""
    return {row["a1_type"]: row["a1"], row["a2_type"]: row["a2"]}


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------

def _tensor_of(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_tensor(output, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    if isinstance(output, list):
        return [new] + list(output[1:])
    return new


# ---------------------------------------------------------------------------
# eraser: x -> x + alpha * (x @ D.T + b), with D = A - I
# ---------------------------------------------------------------------------

class AffineEraser:
    def __init__(self, D, b, alpha=1.0, meta=None):
        self.D = D.float()
        self.b = b.float()
        self.alpha = float(alpha)
        self.meta = meta or {}

    @classmethod
    def load(cls, path, alpha=1.0, device="cpu"):
        obj = torch.load(path, map_location=device)
        return cls(obj["D"].to(device), obj["b"].to(device), alpha, obj.get("meta"))

    def to(self, device):
        self.D, self.b = self.D.to(device), self.b.to(device)
        return self

    def __call__(self, x):
        shape, dtype = x.shape, x.dtype
        h = x.reshape(-1, shape[-1]).float()
        h = h + self.alpha * (h @ self.D.T + self.b)
        return h.reshape(shape).to(dtype)

    def hook(self):
        """Forward hook that rewrites the connector output in place."""
        def fn(module, inputs, output):
            return _replace_tensor(output, self(_tensor_of(output)))
        return fn


class StopForward(Exception):
    pass


def capture_hook(store):
    """Forward hook that stores the connector output and aborts the forward
    pass, so feature extraction never runs the language model."""
    def fn(module, inputs, output):
        x = _tensor_of(output)
        store["x"] = x.reshape(-1, x.shape[-1]).detach()
        raise StopForward
    return fn
