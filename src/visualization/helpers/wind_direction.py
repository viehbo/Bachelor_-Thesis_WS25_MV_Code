# src/visualization/helpers/wind_direction.py
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_wind_direction_deg_from_uv(
    uv_df: pd.DataFrame,
    *,
    u_col: str = "u",
    v_col: str = "v",
    name: str = "dir",
) -> pd.Series:
    """
    Compute wind direction (meteorological, 'from') in degrees [0, 360).

    Convention used in your codebase:
      dir = (270 - atan2(v, u) in degrees) mod 360
    """
    u = uv_df[u_col].to_numpy()
    v = uv_df[v_col].to_numpy()
    dir_deg = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return pd.Series(dir_deg, index=uv_df.index, name=name)
