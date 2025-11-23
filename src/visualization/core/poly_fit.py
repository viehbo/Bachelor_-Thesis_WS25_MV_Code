# src/visualization/utilities/poly_fit.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Tuple, List

def poly_fit_datetime(
    t: Iterable,
    y: Iterable,
    degree: int,
    points: int | str = "original"
) -> Tuple[List, List]:
    """
    Fit a polynomial of given degree to (t, y) where t are datetimes.
    Returns (t_fit_list, y_fit_list).
    """
    t = pd.to_datetime(pd.Series(list(t)))
    y = pd.Series(list(y)).astype(float)

    x = t.astype("int64") / 1e9  # seconds

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < degree + 1:
        return t.tolist(), [np.nan] * len(t)

    x_m = x[mask].to_numpy()
    y_m = y[mask].to_numpy()

    x_mean = x_m.mean()
    x_scale = x_m.std() or 1.0
    x_n = (x_m - x_mean) / x_scale

    coeffs = np.polyfit(x_n, y_m, degree)
    p = np.poly1d(coeffs)

    x_eval = x.to_numpy() if points == "original" else np.linspace(x_m.min(), x_m.max(), int(points))
    y_fit = p((x_eval - x_mean) / x_scale)

    t_fit = pd.to_datetime(x_eval, unit="s").to_pydatetime().tolist()
    return t_fit, y_fit.tolist()
