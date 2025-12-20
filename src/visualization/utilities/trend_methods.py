# src/visualization/utilities/trend_methods.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd

from src.visualization.core.poly_fit import poly_fit_datetime


@dataclass(frozen=True)
class TrendConfig:
    method: str  # "polyfit" | "rolling_mean" | "ewma" | "annual_linear"
    # shared optional pre-smoothing
    pre_smooth_enabled: bool = False
    pre_smooth_window_days: int = 30

    # method parameters (only some used depending on method)
    poly_degree: int = 3
    rolling_window_days: int = 30
    ewma_span_days: int = 30
    annual_min_years: int = 5


def _to_series(t: Iterable, y: Iterable, name: str = "y") -> pd.Series:
    idx = pd.to_datetime(pd.Series(list(t)))
    s = pd.Series(list(y), index=idx, name=name).astype(float)
    s = s.sort_index()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def _apply_pre_smooth(s: pd.Series, enabled: bool, window_days: int) -> pd.Series:
    if not enabled:
        return s
    wd = max(int(window_days), 1)
    # time-based rolling window
    return s.rolling(f"{wd}D", min_periods=1).mean()


def _trend_polyfit(s: pd.Series, degree: int) -> Tuple[List, List]:
    t_fit, y_fit = poly_fit_datetime(s.index, s.values, degree=int(degree), points="original")
    return t_fit, y_fit


def _trend_rolling_mean(s: pd.Series, window_days: int) -> Tuple[List, List]:
    wd = max(int(window_days), 1)
    y = s.rolling(f"{wd}D", min_periods=1).mean()
    return s.index.to_pydatetime().tolist(), y.tolist()


def _trend_ewma(s: pd.Series, span_days: int) -> Tuple[List, List]:
    sd = max(int(span_days), 1)
    daily = s.resample("1D").mean()
    y_daily = daily.ewm(span=sd, adjust=False).mean()

    # align to original timestamps for overlay plotting
    y = y_daily.reindex(s.index, method="ffill")
    return s.index.to_pydatetime().tolist(), y.tolist()



def _trend_annual_linear(s: pd.Series, min_years: int) -> Tuple[List, List]:
    # annual mean values
    annual = s.resample("YS").mean().dropna()
    if annual.shape[0] < max(int(min_years), 2):
        # fallback: just return the original timestamps with NaNs (caller can hide)
        return s.index.to_pydatetime().tolist(), [np.nan] * len(s)

    # regress annual means vs year as float
    years = annual.index.year.astype(float)
    y = annual.values.astype(float)

    # simple least squares
    A = np.vstack([years, np.ones_like(years)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    # Evaluate trend on the ORIGINAL index (for overlay in same plot)
    yrs_full = s.index.year.astype(float) + (s.index.dayofyear.astype(float) / 366.0)
    y_fit = m * yrs_full + b
    return s.index.to_pydatetime().tolist(), y_fit.tolist()


def compute_trend_line(
    t: Iterable,
    y: Iterable,
    cfg: TrendConfig,
) -> Tuple[List, List]:
    """
    Returns (t_fit_list, y_fit_list) aligned to original timestamps where possible.
    """
    s = _to_series(t, y)
    if s.empty:
        return [], []

    s = _apply_pre_smooth(s, cfg.pre_smooth_enabled, cfg.pre_smooth_window_days)

    method = (cfg.method or "").lower().strip()
    if method == "polyfit":
        return _trend_polyfit(s, cfg.poly_degree)
    if method == "rolling_mean":
        return _trend_rolling_mean(s, cfg.rolling_window_days)
    if method == "ewma":
        return _trend_ewma(s, cfg.ewma_span_days)
    if method == "annual_linear":
        return _trend_annual_linear(s, cfg.annual_min_years)

    raise ValueError(f"Unknown trend method: {cfg.method!r}")
