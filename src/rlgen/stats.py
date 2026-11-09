"""Seed-level statistics. The seed is the unit of replication."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean and percentile-bootstrap (1 - alpha) CI over seeds."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(v))}
