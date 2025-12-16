"""
Time-based filtering utilities for time series data.
Consolidates hour-of-day and other temporal filtering logic.
"""
from typing import Union, Optional, Iterable, Set
import pandas as pd
import xarray as xr


def filter_by_hours(
        data: Union[pd.Series, pd.DataFrame],
        hours: Optional[Iterable[int]]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Filter time series data to include only specified hours of the day.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time series with DatetimeIndex
    hours : iterable of int, optional
        Hours to keep (0-23). If None or empty, returns data unchanged.

    Returns
    -------
    pd.Series or pd.DataFrame
        Filtered data
    """
    if not hours:
        return data

    hour_set = set(int(h) for h in hours)
    idx = pd.to_datetime(data.index)
    mask = idx.hour.astype(int).isin(hour_set)

    return data.loc[mask]


def filter_dataset_by_hours(
        ds: xr.Dataset,
        time_dim: str,
        hours: Optional[Iterable[int]]
) -> xr.Dataset:
    """
    Filter an xarray Dataset to include only specified hours.

    Uses coordinate labels for selection to avoid alignment issues.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to filter
    time_dim : str
        Name of the time dimension/coordinate
    hours : iterable of int, optional
        Hours to keep (0-23). If None or empty, returns ds unchanged.

    Returns
    -------
    xr.Dataset
        Filtered dataset
    """
    if not hours:
        return ds

    hour_set = set(int(h) for h in hours)

    # Build DatetimeIndex from coordinate
    idx = pd.to_datetime(ds[time_dim].to_index())

    # Select labels where hour matches
    keep = idx[idx.hour.astype(int).isin(hour_set)]

    if len(keep) == 0:
        # No matching timestamps; return empty selection
        return ds.isel({time_dim: slice(0, 0)})

    return ds.sel({time_dim: keep})


def extract_unique_hours(data: Union[pd.Series, pd.DataFrame, pd.DatetimeIndex]) -> list:
    """
    Extract unique hours present in time series data.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or pd.DatetimeIndex
        Time series or index with datetime values

    Returns
    -------
    list
        Sorted list of unique hours (0-23)
    """
    if isinstance(data, pd.DatetimeIndex):
        idx = data
    else:
        idx = pd.to_datetime(data.index)

    hours = pd.Index(idx.hour).unique().sort_values().tolist()

    # Map 24 -> 0 if 1..24 convention appears (rare)
    if (24 in hours) and (0 not in hours):
        hours = sorted({0 if h == 24 else h for h in hours})

    return hours


def format_hours_as_strings(hours: Iterable[int]) -> list:
    """
    Format hours as zero-padded strings.

    Parameters
    ----------
    hours : iterable of int
        Hours (0-23)

    Returns
    -------
    list of str
        Formatted hours like ["00", "01", ..., "23"]
    """
    return [f"{h:02d}" for h in sorted(hours)]


def parse_hour_strings(hour_strings: Iterable[str]) -> Set[int]:
    """
    Parse hour strings back to integer set.

    Parameters
    ----------
    hour_strings : iterable of str
        Hour strings like ["00", "12", "18"]

    Returns
    -------
    set of int
        Parsed hours as integers
    """
    return {int(h) for h in hour_strings}