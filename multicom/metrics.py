from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

from .schema import INT_TO_LABEL


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    out: dict[str, float | int] = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
    }
    recalls = []
    for label_id, label in INT_TO_LABEL.items():
        mask = y_true == label_id
        recall = float((y_pred[mask] == label_id).mean()) if mask.any() else math.nan
        out[f"recall_{label.lower()}"] = recall
        out[f"n_{label.lower()}"] = int(mask.sum())
        recalls.append(recall)
    out["min_recall"] = float(np.nanmin(recalls))
    out["h_to_nh"] = int(((y_true == 2) & (y_pred == 0)).sum())
    out["nh_to_h"] = int(((y_true == 0) & (y_pred == 2)).sum())
    out["cross_error"] = int(out["h_to_nh"] + out["nh_to_h"])
    return out


def binomial_se_pct(accuracy: float, n: int) -> float:
    if n <= 0:
        return math.nan
    return 100.0 * math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / n)

