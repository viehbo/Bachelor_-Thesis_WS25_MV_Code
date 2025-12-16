"""
Data validation utilities for common guard patterns.
Reduces repetitive validation code across modules.
"""
import numpy as np
import pandas as pd
import math
from typing import Any, Union


def is_empty_or_nan(value: Any) -> bool:
    """
    Check if a value is empty, None, NaN, or an empty string variant.

    Parameters
    ----------
    value : any
        Value to check

    Returns
    -------
    bool
        True if value is considered empty
    """
    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    if isinstance(value, str):
        s = value.strip().lower()
        if s in ('', 'nan', 'none'):
            return True

    return False


def has_finite_data(arr: Union[np.ndarray, pd.Series]) -> bool:
    """
    Check if array/series has at least one finite value.

    Parameters
    ----------
    arr : array-like
        Data to check

    Returns
    -------
    bool
        True if at least one finite value exists
    """
    arr = np.asarray(arr, dtype=float)
    return arr.size > 0 and np.isfinite(arr).any()


def is_all_nan(arr: Union[np.ndarray, pd.Series]) -> bool:
    """
    Check if all values in array are NaN or infinite.

    Parameters
    ----------
    arr : array-like
        Data to check

    Returns
    -------
    bool
        True if all values are NaN/inf or array is empty
    """
    return not has_finite_data(arr)


def is_constant(arr: Union[np.ndarray, pd.Series]) -> bool:
    """
    Check if all finite values in array are the same.

    Parameters
    ----------
    arr : array-like
        Data to check

    Returns
    -------
    bool
        True if all finite values are equal
    """
    arr = np.asarray(arr, dtype=float)

    if not has_finite_data(arr):
        return True

    finite = arr[np.isfinite(arr)]
    return finite.min() == finite.max()


def safe_minmax(arr: Union[np.ndarray, pd.Series], default: tuple = (0.0, 1.0)) -> tuple:
    """
    Get min/max of array with safe defaults.

    Parameters
    ----------
    arr : array-like
        Data to analyze
    default : tuple
        Default (min, max) if no finite data

    Returns
    -------
    tuple
        (min, max) values
    """
    arr = np.asarray(arr, dtype=float)

    if not has_finite_data(arr):
        return default

    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))

    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        return default

    if vmin == vmax:
        # Return a small range around the constant value
        if vmin == 0:
            return (0.0, 1.0)
        return (vmin * 0.95, vmin * 1.05)

    return (vmin, vmax)


def validate_series_for_fit(series: pd.Series, min_points: int) -> bool:
    """
    Check if series has enough data points for polynomial fitting.

    Parameters
    ----------
    series : pd.Series
        Time series data
    min_points : int
        Minimum required points (typically degree + 1)

    Returns
    -------
    bool
        True if series has sufficient valid data
    """
    if series is None or len(series) < min_points:
        return False

    finite_count = np.isfinite(series.values).sum()
    return finite_count >= min_points


def parse_year_value(year_value: Any) -> Union[int, None]:
    """
    Parse a year value from various formats to integer or None.

    Parameters
    ----------
    year_value : any
        Value that might represent a year

    Returns
    -------
    int or None
        Parsed year as integer, or None if invalid
    """
    if is_empty_or_nan(year_value):
        return None

    try:
        return int(year_value)
    except (ValueError, TypeError):
        return None


def filter_valid_years(year_list: list) -> list:
    """
    Filter a list of year values to only valid integers.

    Parameters
    ----------
    year_list : list
        List of year values (may contain None, NaN, strings, etc.)

    Returns
    -------
    list
        List of valid year integers
    """
    valid = []
    for y in year_list:
        parsed = parse_year_value(y)
        if parsed is not None:
            valid.append(parsed)
    return valid


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max.

    Parameters
    ----------
    value : float
        Value to clamp
    min_val : float
        Minimum allowed value
    max_val : float
        Maximum allowed value

    Returns
    -------
    float
        Clamped value
    """
    return max(min_val, min(max_val, value))