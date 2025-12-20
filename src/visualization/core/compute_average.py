import numpy as np
import pandas as pd
import xarray as xr

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from src.visualization.helpers.find_var import _find_var
from src.visualization.helpers.lon_lat_names import _lon_lat_names
from src.visualization.helpers.time_name import _time_name


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _slice_time(ds: xr.Dataset, tname: str, t_range):
    if not t_range:
        return ds
    t0, t1 = t_range
    t0 = np.datetime64(pd.Timestamp(t0))
    t1 = np.datetime64(pd.Timestamp(t1))
    return ds.sel({tname: slice(t0, t1)})


def _filter_hours(ds: xr.Dataset, tname: str, hours: Optional[Iterable[int]]):
    if not hours:
        return ds

    idx = pd.to_datetime(ds[tname].to_index())
    keep = idx[idx.hour.astype(int).isin(set(int(h) for h in hours))]

    if len(keep) == 0:
        return ds.isel({tname: slice(0, 0)})

    return ds.sel({tname: keep})


def _spatial_dims_from_coords(ds: xr.Dataset, lon_name: str, lat_name: str) -> Tuple[str, ...]:
    lon_dims = tuple(ds[lon_name].dims)
    lat_dims = tuple(ds[lat_name].dims)

    if lon_dims and lat_dims and lon_dims == lat_dims:
        return lon_dims

    if len(lat_dims) == 1 and len(lon_dims) == 1:
        return (lat_dims[0], lon_dims[0])

    dims = []
    for d in lat_dims + lon_dims:
        if d not in dims:
            dims.append(d)

    if not dims:
        raise ValueError("Could not determine spatial dimensions from lon/lat.")

    return tuple(dims)


def _ensure_spatial_array(da: xr.DataArray, spatial_dims: Sequence[str], *, name: str) -> np.ndarray:
    if da.ndim == 0:
        raise ValueError(f"{name} became scalar (dims=()).")

    missing = [d for d in spatial_dims if d not in da.dims]
    if missing:
        raise ValueError(
            f"{name} missing spatial dims {missing}. Available dims: {list(da.dims)}"
        )

    other_dims = [d for d in da.dims if d not in spatial_dims]
    if other_dims:
        da = da.mean(dim=other_dims)

    return da.transpose(*spatial_dims).values


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def compute_average(DATASETS, selected_files, ds_key, mode, t_range, hours=None):
    """
    Compute averaged data for wind, temperature, or glacier datasets.

    mode MUST be one of: "wind", "temperature", "glacier"
    """

    if not selected_files:
        raise ValueError("No files selected.")

    if mode not in {"wind", "temperature", "glacier"}:
        raise ValueError(f"Unsupported mode: {mode}")

    cfg = DATASETS[ds_key]
    files = [str(p) for p in selected_files]

    with xr.open_mfdataset(files, combine="by_coords", join="outer") as ds:

        # --------------------------------------------------
        # lon / lat + extent
        # --------------------------------------------------
        lon_name, lat_name = _lon_lat_names(ds)

        lon_vals = ds[lon_name].values
        lat_vals = ds[lat_name].values

        extent = [
            float(np.nanmin(lon_vals)),
            float(np.nanmax(lon_vals)),
            float(np.nanmin(lat_vals)),
            float(np.nanmax(lat_vals)),
        ]

        lon = lon_vals
        lat = lat_vals

        # --------------------------------------------------
        # time handling
        # --------------------------------------------------
        tname = _time_name(ds)
        if tname in ds:
            ds = _slice_time(ds, tname, t_range)
            ds = _filter_hours(ds, tname, hours)

        # --------------------------------------------------
        # GLACIER (point data)
        # --------------------------------------------------
        if mode == "glacier":
            if "annual_mass_balance" not in ds.data_vars:
                raise ValueError("Glacier mode selected but 'annual_mass_balance' not found.")

            mb = ds["annual_mass_balance"]
            if tname in mb.dims:
                mb = mb.mean(dim=tname)

            units = mb.attrs.get("units", "")
            return mb.values, extent, units, lon, lat, None, None

        # --------------------------------------------------
        # GRID DATA (wind / temperature)
        # --------------------------------------------------
        spatial_dims = _spatial_dims_from_coords(ds, lon_name, lat_name)

        # ---------------- WIND ----------------
        if mode == "wind":
            u_name = _find_var(ds, cfg["candidates"]["u"])
            v_name = _find_var(ds, cfg["candidates"]["v"])

            mean_u = ds[u_name].mean(dim=tname)
            mean_v = ds[v_name].mean(dim=tname)

            u2d = _ensure_spatial_array(mean_u, spatial_dims, name=f"mean_u ({u_name})")
            v2d = _ensure_spatial_array(mean_v, spatial_dims, name=f"mean_v ({v_name})")

            avg2d = np.hypot(u2d, v2d)
            units = ds[u_name].attrs.get("units", "")

            return avg2d, extent, units, lon, lat, u2d, v2d

        # ------------- TEMPERATURE -------------
        # Prefer dataset config if available, otherwise auto-detect
        if "candidates" in cfg and "x" in cfg["candidates"]:
            x_name = _find_var(ds, cfg["candidates"]["x"])
        else:
            # robust fallback for real temperature files
            for cand in ("temperature_2m", "t2m", "tas", "temperature"):
                if cand in ds.data_vars:
                    x_name = cand
                    break
            else:
                raise ValueError(
                    "Temperature mode selected, but no temperature variable found. "
                    f"Available data variables: {list(ds.data_vars)}"
                )

        avg = ds[x_name].mean(dim=tname)
        avg2d = _ensure_spatial_array(avg, spatial_dims, name=f"avg ({x_name})")

        units = ds[x_name].attrs.get("units", "")
        return avg2d, extent, units, lon, lat, None, None

