"""Threshold selection helpers."""

import numpy as np


def percentile_threshold(scores: np.ndarray, q: float = 95.0) -> float:
    return float(np.percentile(scores, q))


def pot_threshold(train_scores: np.ndarray, init_q: float = 0.98) -> float:
    """
    Simplified Peak-Over-Threshold (OmniAnomaly-style heuristic).
    Fits on 'normal' training reconstruction scores.
    """
    scores = np.asarray(train_scores, dtype=np.float64)
    if len(scores) < 20:
        return float(np.median(scores))
    u = np.quantile(scores, init_q)
    tail = scores[scores > u]
    if len(tail) < 5:
        return float(u)
    excess = tail - u
    # GPD-ish: threshold slightly above high quantile of tail
    return float(u + np.mean(excess) + 2.0 * np.std(excess))
