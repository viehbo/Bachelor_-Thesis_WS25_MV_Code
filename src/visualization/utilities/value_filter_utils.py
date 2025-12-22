from __future__ import annotations
import numpy as np
import pandas as pd


def filter_series_by_int_range(s: pd.Series, vmin: int | None, vmax: int | None) -> pd.Series:
    """Inclusive integer range filter. None means unbounded."""
    if s is None or len(s) == 0:
        return s
    s2 = s.copy()
    if vmin is not None:
        s2 = s2[s2 >= int(vmin)]
    if vmax is not None:
        s2 = s2[s2 <= int(vmax)]
    return s2


def filter_df_by_mask(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if mask is None:
        return df
    return df.loc[mask]
