"""Range-based (segment) metrics for time-series anomaly detection."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def find_segments(binary: np.ndarray) -> list[tuple[int, int]]:
    segs = []
    i, n = 0, len(binary)
    while i < n:
        if binary[i]:
            j = i
            while j < n and binary[j]:
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def _segments_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def range_f1(pred: np.ndarray, labels: np.ndarray) -> dict:
    """
    Segment-level F1: GT segment is hit if any overlap with a predicted segment.
    """
    pred = pred.astype(bool)
    labels = labels.astype(bool)
    gt = find_segments(labels)
    pr = find_segments(pred)

    if not gt and not pr:
        return {"range_precision": 1.0, "range_recall": 1.0, "range_f1": 1.0, "n_gt": 0, "n_pred": 0}
    if not gt:
        return {"range_precision": 0.0, "range_recall": 1.0, "range_f1": 0.0, "n_gt": 0, "n_pred": len(pr)}
    if not pr:
        return {"range_precision": 1.0, "range_recall": 0.0, "range_f1": 0.0, "n_gt": len(gt), "n_pred": 0}

    tp = 0
    matched_pred = set()
    for g in gt:
        for j, p in enumerate(pr):
            if j not in matched_pred and _segments_overlap(g, p):
                tp += 1
                matched_pred.add(j)
                break

    fp = len(pr) - len(matched_pred)
    fn = len(gt) - tp
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return {
        "range_precision": float(prec),
        "range_recall": float(rec),
        "range_f1": float(f1),
        "n_gt": len(gt),
        "n_pred": len(pr),
    }


def dilate_labels(labels: np.ndarray, bias: int) -> np.ndarray:
    if bias <= 0:
        return labels.astype(bool)
    out = labels.astype(bool).copy()
    n = len(out)
    for i in range(n):
        if labels[i] > 0.5:
            lo, hi = max(0, i - bias), min(n, i + bias + 1)
            out[lo:hi] = True
    return out


def vus_pr(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    max_bias: int | None = None,
) -> dict:
    """
    Simplified VUS-PR: average PR-F1 over temporal tolerances (bias 0..max_bias).
    Common in TSAD benchmarks (TSB-AD style tolerance sweep).
    """
    labels = labels.astype(np.float32)
    if max_bias is None:
        max_bias = min(20, max(1, len(labels) // 50))

    f1s, precs, recs = [], [], []
    for b in range(max_bias + 1):
        dilated = dilate_labels(labels, b)
        if dilated.sum() == 0:
            continue
        pred = (scores >= threshold).astype(int)
        precs.append(float(precision_score(dilated, pred, zero_division=0)))
        recs.append(float(recall_score(dilated, pred, zero_division=0)))
        f1s.append(float(f1_score(dilated, pred, zero_division=0)))

    if not f1s:
        return {"vus_pr": None, "vus_f1_mean": None, "max_bias": max_bias}

    return {
        "vus_pr": float(np.mean(precs) * np.mean(recs)),  # simplified surface proxy
        "vus_f1_mean": float(np.mean(f1s)),
        "vus_precision_mean": float(np.mean(precs)),
        "vus_recall_mean": float(np.mean(recs)),
        "max_bias": max_bias,
    }
