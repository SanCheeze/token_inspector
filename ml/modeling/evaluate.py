from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    true_rank = pd.Series(y_true).rank().to_numpy()
    pred_rank = pd.Series(y_pred).rank().to_numpy()
    if true_rank.size < 2:
        return 0.0
    if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        return 0.0
    corr = np.corrcoef(true_rank, pred_rank)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(spearman_corr(y_true, y_pred)),
    }
