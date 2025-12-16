"""
Datetime utility functions for consistent handling across the application.
Consolidates timezone handling, conversion, and normalization logic.
"""
from datetime import datetime, timezone, timedelta
from typing import Union, Tuple, Any
import pandas as pd
import numpy as np


def to_naive_datetime(dt_any: Any) -> datetime:
    """
    Convert various datetime-like objects to timezone-naive Python datetime.

    Handles:
    - pandas Timestamp
    - numpy datetime64
    - Python datetime (with or without timezone)
    - cftime objects

    Parameters
    ----------
    dt_any : various
        Any datetime-like object

    Returns
    -------
    datetime
        Timezone-naive Python datetime

    Raises
    ------
    ValueError
        If the input cannot be converted
    """
    # Already a naive Python datetime
    if isinstance(dt_any, datetime):
        return dt_any.replace(tzinfo=None)

    # Pandas Timestamp or numpy datetime64
    if isinstance(dt_any, (np.datetime64, pd.Timestamp)):
        ts = pd.to_datetime(dt_any)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.to_pydatetime().replace(tzinfo=None)

    # cftime object (no tz, convert by components)
    mod = type(dt_any).__module__
    if 'cftime' in mod:
        return datetime(
            dt_any.year, dt_any.month, dt_any.day,
            getattr(dt_any, 'hour', 0),
            getattr(dt_any, 'minute', 0),
            getattr(dt_any, 'second', 0),
            getattr(dt_any, 'microsecond', 0),
        )

    # Last resort: try pandas conversion
    try:
        ts = pd.to_datetime(dt_any, errors='coerce')
        if pd.isna(ts):
            raise ValueError(f"Cannot convert {dt_any!r} to datetime")
        return ts.to_pydatetime().replace(tzinfo=None)
    except Exception as e:
        raise ValueError(f"Cannot convert {dt_any!r} to datetime") from e


def convert_ms_epoch_to_datetime(ms_value: Union[int, float]) -> datetime:
    """
    Convert JavaScript-style milliseconds since epoch to naive datetime.

    Handles negative offsets (dates before 1970) safely.

    Parameters
    ----------
    ms_value : int or float
        Milliseconds since Unix epoch

    Returns
    -------
    datetime
        Timezone-naive datetime
    """
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    dt = epoch + timedelta(milliseconds=ms_value)
    return dt.replace(tzinfo=None)


def safe_value_as_datetime(value: Any) -> Union[datetime, Tuple[datetime, datetime]]:
    """
    Convert Panel slider values to naive datetime(s).

    Handles both single values and tuples (for range sliders).
    This is used to patch Panel's datetime conversion issues.

    Parameters
    ----------
    value : various
        Single value or tuple of values

    Returns
    -------
    datetime or tuple of datetime
        Converted naive datetime(s)
    """

    def convert_single(v):
        # JavaScript milliseconds timestamp
        if isinstance(v, (int, float)):
            return convert_ms_epoch_to_datetime(v)

        # Already a datetime-like object
        return to_naive_datetime(v)

    # Handle tuple (range slider)
    if isinstance(value, tuple):
        return convert_single(value[0]), convert_single(value[1])

    # Single value
    return convert_single(value)


def ensure_naive_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Ensure a DatetimeIndex is timezone-naive.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Input index (may have timezone)

    Returns
    -------
    pd.DatetimeIndex
        Timezone-naive index
    """
    index = pd.to_datetime(index)
    if getattr(index, 'tz', None) is not None:
        index = index.tz_localize(None)
    return index


def ensure_naive_timestamp(ts: pd.Timestamp) -> pd.Timestamp:
    """
    Ensure a Timestamp is timezone-naive.

    Parameters
    ----------
    ts : pd.Timestamp
        Input timestamp (may have timezone)

    Returns
    -------
    pd.Timestamp
        Timezone-naive timestamp
    """
    ts = pd.to_datetime(ts)
    if getattr(ts, 'tz', None) is not None:
        ts = ts.tz_localize(None)
    return ts


def datetime_to_python_list(dt_iterable) -> list:
    """
    Convert datetime iterable to list of Python datetime objects.

    Parameters
    ----------
    dt_iterable : iterable
        Any iterable of datetime-like objects

    Returns
    -------
    list
        List of Python datetime objects
    """
    return pd.to_datetime(dt_iterable).to_pydatetime().tolist()