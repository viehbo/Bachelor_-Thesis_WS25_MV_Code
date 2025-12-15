import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from typing import Iterable, Optional

from src.visualization.helpers.find_var import _find_var
from src.visualization.helpers.lon_lat_names import _lon_lat_names
from src.visualization.helpers.time_name import _time_name


def _slice_time(ds, tname: str, t_range):
    """Safe time slicing by (start, end)."""
    if not t_range:
        return ds
    t0, t1 = t_range
    t0 = np.datetime64(pd.Timestamp(t0))
    t1 = np.datetime64(pd.Timestamp(t1))
    return ds.sel({tname: slice(t0, t1)})


def _filter_hours(ds, tname: str, hours: Optional[Iterable[int]]):
    """Keep only the given hours of day using coordinate labels (no exact-match alignment)."""
    if not hours:
        return ds
    # Build a pandas DatetimeIndex from the coord, select labels whose hour is in hours,
    # and use .sel with those labels to avoid boolean alignment surprises.
    idx = pd.to_datetime(ds[tname].to_index())
    keep = idx[idx.hour.astype(int).isin(set(int(h) for h in hours))]
    if len(keep) == 0:
        # No matching timestamps; return an empty selection on time
        return ds.isel({tname: slice(0, 0)})
    return ds.sel({tname: keep})


def compute_average(DATASETS, selected_files, ds_key, t_range, hours=None):
    """
    Open all files with a tolerant join and compute the map average for the
    selected dataset, optionally subsetting by time range and hours-of-day.
    """
    if not selected_files:
        raise ValueError("No files selected.")
    cfg = DATASETS[ds_key]

    # Accept Path or str
    files = [str(p) if isinstance(p, (str, Path)) else str(p) for p in selected_files]

    # Robust open across non-identical coordinates
    with xr.open_mfdataset(
        files,
        combine="by_coords",
        join="outer",
        parallel=False,
    ) as ds:
        lon_name, lat_name = _lon_lat_names(ds)
        extent = [
            float(ds[lon_name].min().values),
            float(ds[lon_name].max().values),
            float(ds[lat_name].min().values),
            float(ds[lat_name].max().values),
        ]
        lon = ds[lon_name].values
        lat = ds[lat_name].values

        tname = _time_name(ds)
        ds = _slice_time(ds, tname, t_range)
        ds = _filter_hours(ds, tname, hours)

        if cfg["mode"] == "wind":
            u_name = _find_var(ds, cfg["candidates"]["u"])
            v_name = _find_var(ds, cfg["candidates"]["v"])

            mean_u = ds[u_name].mean(dim=tname)
            mean_v = ds[v_name].mean(dim=tname)
            speed  = np.hypot(mean_u, mean_v)

            # Ensure (lat, lon) are last dims
            mean_u_2d = mean_u.transpose(..., lat_name, lon_name).values
            mean_v_2d = mean_v.transpose(..., lat_name, lon_name).values
            speed_2d  = speed.transpose(...,  lat_name, lon_name).values
            units = getattr(ds[u_name], "attrs", {}).get("units", "")

            return speed_2d, extent, units, lon, lat, mean_u_2d, mean_v_2d
        else:
            x_name = _find_var(ds, cfg["candidates"]["x"])
            avg = ds[x_name].mean(dim=tname)
            avg2d = avg.transpose(..., lat_name, lon_name).values
            units = getattr(ds[x_name], "attrs", {}).get("units", "")
            return avg2d, extent, units, lon, lat, None, None
