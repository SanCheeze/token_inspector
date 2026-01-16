from __future__ import annotations

import numpy as np


def gini(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    if np.allclose(array, 0):
        return 0.0
    sorted_values = np.sort(array)
    cumulative = np.cumsum(sorted_values)
    total = cumulative[-1]
    if total == 0:
        return 0.0
    index = np.arange(1, array.size + 1)
    gini_value = (np.sum((2 * index - array.size - 1) * sorted_values) / (array.size * total))
    return float(gini_value)
