"""Metrics for segmentation, element quantities and retrieval."""

from __future__ import annotations

import numpy as np

from .segment import CLASS_NAMES


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    idx = y_true.astype(np.int64).ravel() * n + y_pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def segmentation_scores(cm: np.ndarray) -> dict:
    """Per class IoU and F1 from a pixel confusion matrix (rows = truth)."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), np.nan)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    support = cm.sum(axis=1)
    weighted_f1 = float(np.nansum(f1 * support) / support.sum())
    return {
        "iou": {c: round(float(v), 4) for c, v in zip(CLASS_NAMES, iou)},
        "f1": {c: round(float(v), 4) for c, v in zip(CLASS_NAMES, f1)},
        "miou": round(float(np.nanmean(iou)), 4),
        "macro_f1": round(float(np.nanmean(f1)), 4),
        "weighted_f1": round(weighted_f1, 4),
        "pixel_accuracy": round(float(tp.sum() / cm.sum()), 4),
    }


def worst_state_scores(true_states: list[int], pred_states: list[int]) -> dict:
    """Agreement on the worst condition state per photo, the number that drives escalation."""
    t, p = np.asarray(true_states), np.asarray(pred_states)
    cm = confusion(t - 1, p - 1, 4)
    exact = float((t == p).mean())
    within_one = float((np.abs(t - p) <= 1).mean())
    # escalation = CS3 or CS4 present. Missing one is the expensive error.
    esc_t, esc_p = t >= 3, p >= 3
    recall = float((esc_t & esc_p).sum() / max(1, esc_t.sum()))
    precision = float((esc_t & esc_p).sum() / max(1, esc_p.sum()))
    return {
        "exact": round(exact, 4),
        "within_one": round(within_one, 4),
        "escalation_recall": round(recall, 4),
        "escalation_precision": round(precision, 4),
        "escalation_missed": int((esc_t & ~esc_p).sum()),
        "escalation_true": int(esc_t.sum()),
        "confusion": cm.tolist(),
    }


def recall_at_k(ranked: list[list[str]], relevant: list[set[str]], k: int) -> float:
    hits = [bool(set(r[:k]) & rel) for r, rel in zip(ranked, relevant)]
    return round(float(np.mean(hits)), 4)


def mrr(ranked: list[list[str]], relevant: list[set[str]], k: int = 10) -> float:
    out = []
    for r, rel in zip(ranked, relevant):
        rr = 0.0
        for i, cid in enumerate(r[:k]):
            if cid in rel:
                rr = 1.0 / (i + 1)
                break
        out.append(rr)
    return round(float(np.mean(out)), 4)
