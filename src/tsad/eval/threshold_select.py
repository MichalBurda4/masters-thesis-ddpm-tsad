"""Unified threshold selection for evaluation."""

import numpy as np

from tsad.eval.metrics import best_f1_threshold
from tsad.eval.threshold import percentile_threshold, pot_threshold


def select_threshold(
    method: str,
    scores_for_fit: np.ndarray,
    labels_for_fit: np.ndarray | None = None,
    pot_init_q: float = 0.98,
    percentile_q: float = 95.0,
) -> tuple[float, str]:
    """
    Returns (threshold, method_used).
    - f1: maximize F1 on scores_for_fit vs labels_for_fit (needs labels)
    - pot: peak-over-threshold on normal reconstruction scores (no labels)
    - percentile: high quantile of scores_for_fit
    """
    method = method.lower()
    if method == "f1":
        if labels_for_fit is None:
            raise ValueError("f1 threshold requires labels")
        t, _ = best_f1_threshold(scores_for_fit, labels_for_fit)
        return t, "f1"
    if method == "pot":
        return pot_threshold(scores_for_fit, init_q=pot_init_q), "pot"
    if method == "percentile":
        return percentile_threshold(scores_for_fit, q=percentile_q), "percentile"
    raise ValueError(f"Unknown threshold method: {method}")
