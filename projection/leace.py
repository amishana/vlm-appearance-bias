"""Closed-form LEACE (Belrose et al., 2023) from streamed sufficient statistics.

Each image has labels for only two of the three attribute types, so statistics
are kept per type over the images that carry that type. Erasers are fitted one
type at a time on the output of the previous ones. Because every eraser is
affine, a later eraser keeps the earlier cross-covariances at zero, so the
composed map erases every requested type.
"""

import torch

from projection.common import ATTR_ORDER


class StatsAccumulator:
    """Token-level sums per attribute type, in float64."""

    def __init__(self, d, device):
        self.d = d
        self.stats = {}
        for t, values in ATTR_ORDER.items():
            k = len(values)
            self.stats[t] = {
                "n": 0.0,
                "sx": torch.zeros(d, dtype=torch.float64, device=device),
                "sxx": torch.zeros(d, d, dtype=torch.float64, device=device),
                "sz": torch.zeros(k, dtype=torch.float64, device=device),
                "sxz": torch.zeros(d, k, dtype=torch.float64, device=device),
            }

    def update(self, X, attrs):
        """X: [T, d] connector tokens of one image; attrs: {type: value}."""
        X = X.to(torch.float64)
        T = X.shape[0]
        s = X.sum(0)
        xx = X.T @ X
        for t, v in attrs.items():
            c = ATTR_ORDER[t].index(v)
            st = self.stats[t]
            st["n"] += T
            st["sx"] += s
            st["sxx"] += xx
            st["sz"][c] += T
            st["sxz"][:, c] += s

    def state_dict(self):
        return {t: {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in st.items()}
                for t, st in self.stats.items()}


def moments(st):
    n = st["n"]
    mu = st["sx"] / n
    Sxx = st["sxx"] / n - torch.outer(mu, mu)
    pz = st["sz"] / n
    Sxz = st["sxz"] / n - torch.outer(mu, pz)
    return mu, (Sxx + Sxx.T) / 2, Sxz


def leace_matrix(Sxx, Sxz, tol=1e-7):
    """M such that x -> x - M (x - mu) is the LEACE eraser."""
    lam, V = torch.linalg.eigh((Sxx + Sxx.T) / 2)
    keep = lam > tol * lam.max()
    lam, V = lam[keep], V[:, keep]
    W = (V * lam.rsqrt()) @ V.T
    W_pinv = (V * lam.sqrt()) @ V.T
    U, s, _ = torch.linalg.svd(W @ Sxz, full_matrices=False)
    if s.numel() == 0 or s.max() == 0:
        return torch.zeros_like(Sxx), 0
    U = U[:, s > 1e-6 * s.max()]
    return W_pinv @ U @ U.T @ W, U.shape[1]


def fit_sequential(stats, types, tol=1e-7):
    """Compose erasers for `types` in order. Returns D = A - I, b, report."""
    d = stats[types[0]]["sx"].shape[0]
    I = torch.eye(d, dtype=torch.float64)
    A, b = I.clone(), torch.zeros(d, dtype=torch.float64)
    report = []
    for t in types:
        mu, Sxx, Sxz = moments(stats[t])
        mu_e, Sxx_e, Sxz_e = A @ mu + b, A @ Sxx @ A.T, A @ Sxz
        M, rank = leace_matrix(Sxx_e, Sxz_e, tol)
        A = (I - M) @ A
        b = (I - M) @ b + M @ mu_e
        report.append({"type": t, "rank_removed": rank})
    # check: cross-covariance left for every type, relative to before
    for row in report:
        mu, Sxx, Sxz = moments(stats[row["type"]])
        row["residual_xcov"] = float((A @ Sxz).norm() / Sxz.norm())
    return A - I, b, report
