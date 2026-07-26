from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix_labels": [-1, 0, 1],
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=[-1, 0, 1],
        ).tolist(),
    }


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if not len(y_true):
        raise ValueError("No finite target/prediction pairs.")

    rank_ic = spearmanr(y_true, y_pred).statistic
    directional = np.mean(np.sign(y_true) == np.sign(y_pred))

    return {
        "mae_bps": float(mean_absolute_error(y_true, y_pred)),
        "rmse_bps": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "rank_ic": float(rank_ic) if np.isfinite(rank_ic) else 0.0,
        "directional_accuracy": float(directional),
    }
