"""Point-level and window-level metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Find threshold maximizing F1 on given scores/labels."""
    thresholds = np.unique(np.percentile(scores, np.linspace(5, 95, 40)))
    best_t, best_f1 = thresholds[0], 0.0
    for t in thresholds:
        pred = (scores >= t).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 >= best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t), float(best_f1)


def point_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    y = labels.astype(int)
    out = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) > 1:
        out["auc_roc"] = float(roc_auc_score(y, scores))
        out["auc_pr"] = float(average_precision_score(y, scores))
    else:
        out["auc_roc"] = None
        out["auc_pr"] = None
    return out


def attribution_overlap(point_scores: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """Fraction of true anomaly points with score above threshold."""
    mask = mask > 0.5
    if mask.sum() == 0:
        return 0.0
    pred = point_scores >= threshold
    return float((pred & mask).sum() / mask.sum())


def evaluate_windows(
    point_scores_list: list[np.ndarray],
    point_labels_list: list[np.ndarray],
    window_labels: np.ndarray,
    series_scores: np.ndarray,
    threshold: float | None = None,
) -> dict:
    all_scores = np.concatenate(point_scores_list)
    all_labels = np.concatenate(point_labels_list)

    if threshold is None:
        threshold, val_f1 = best_f1_threshold(all_scores, all_labels)
    else:
        val_f1 = point_metrics(all_scores, all_labels, threshold)["f1"]

    pm = point_metrics(all_scores, all_labels, threshold)

    w_pred = (series_scores >= threshold).astype(int)
    wl = window_labels.astype(int)
    wm = {
        "window_precision": float(precision_score(wl, w_pred, zero_division=0)),
        "window_recall": float(recall_score(wl, w_pred, zero_division=0)),
        "window_f1": float(f1_score(wl, w_pred, zero_division=0)),
    }

    attr_hits = [
        attribution_overlap(ps, pl, threshold)
        for ps, pl in zip(point_scores_list, point_labels_list)
        if pl.max() > 0.5
    ]
    attr_mean = float(np.mean(attr_hits)) if attr_hits else 0.0

    return {
        "threshold": threshold,
        "val_f1_at_threshold": val_f1,
        "point": pm,
        "window": wm,
        "attribution_recall_on_anomaly_windows": attr_mean,
    }
