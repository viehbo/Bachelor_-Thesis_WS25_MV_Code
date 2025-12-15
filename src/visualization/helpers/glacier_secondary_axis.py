# src/visualization/helpers/glacier_secondary_axis.py
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_glacier_series_for_secondary_axis(
    s: pd.Series,
    *,
    multiplier: float = 10.0,
    offset: float = 0.0,
) -> pd.Series:
    """Normalize a glacier mass-balance series into a compact range for a secondary y-axis.

    - Guards against empty / all-NaN series to avoid RuntimeWarning: All-NaN slice encountered
    - Handles constant-valued series without division by zero
    """
    arr = np.asarray(s.values, dtype=float)

    # Guard: empty or all-NaN -> keep NaNs (so plot stays empty) without warnings
    if arr.size == 0 or not np.isfinite(arr).any():
        return pd.Series(np.full_like(arr, np.nan, dtype=float), index=s.index, name=s.name)

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        y = (arr - vmin) / (vmax - vmin) * float(multiplier) + float(offset)
    else:
        # Degenerate/constant series: still produce something finite
        y = np.full_like(arr, float(offset), dtype=float)

    return pd.Series(y, index=s.index, name=s.name)


def update_glacier_secondary_axis(
    *,
    s_norm: pd.Series,
    ts_glacier_source,
    ts_fig,
) -> None:
    """Push normalized glacier series into a ColumnDataSource and autoscale the 'glacier' extra y-range."""
    if ts_glacier_source is None:
        return
    if not (hasattr(ts_fig, "extra_y_ranges") and "glacier" in ts_fig.extra_y_ranges):
        return

    t_vals = pd.to_datetime(s_norm.index).to_pydatetime().tolist()
    y_vals = np.asarray(s_norm.values, dtype=float).tolist()
    ts_glacier_source.data = dict(t=t_vals, y=y_vals)

    if not y_vals:
        return

    arr = np.asarray(y_vals, dtype=float)

    # Guard: all-NaN -> do not autoscale; keep existing range (and avoid warnings)
    if not np.isfinite(arr).any():
        return

    ymin = float(np.nanmin(arr))
    ymax = float(np.nanmax(arr))

    if not (np.isfinite(ymin) and np.isfinite(ymax)):
        return

    pad = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) or 1.0))
    rng = ts_fig.extra_y_ranges["glacier"]
    rng.start = ymin - pad
    rng.end = ymax + pad
