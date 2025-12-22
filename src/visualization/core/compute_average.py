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

def _full_circle_selected(dmin: int, dmax: int) -> bool:
    # treat 0..360 (or 0..>=360) as full circle
    return (int(dmin) % 360) == 0 and (int(dmax) % 360) == 0 and int(dmax) >= 360


def _dir_mask_circular(direction_deg: xr.DataArray, dmin: int, dmax: int) -> xr.DataArray:
    """
    Inclusive circular mask for direction on [0, 360).
    Wrap-around supported (e.g., 300..30).
    Full-circle (0..360) returns all True.
    """
    if _full_circle_selected(dmin, dmax):
        return xr.ones_like(direction_deg, dtype=bool)

    d = direction_deg % 360.0
    s = float(int(dmin) % 360)
    e = float(int(dmax) % 360)

    if s <= e:
        return (d >= s) & (d <= e)
    return (d >= s) | (d <= e)



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

def compute_average(
    DATASETS,
    selected_files,
    ds_key,
    mode,
    t_range,
    hours=None,
    *,
    value_filter_enabled: bool = False,
    temp_range=(None, None),
    speed_range=(None, None),
    dir_range=(0, 360),
):

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

            u = ds[u_name]
            v = ds[v_name]

            # Stage-2 value filter (wind) applied BEFORE averaging
            if value_filter_enabled:
                smin, smax = speed_range
                dmin, dmax = dir_range

                smin = -np.inf if smin is None else float(smin)
                smax = np.inf if smax is None else float(smax)

                speed = xr.apply_ufunc(np.hypot, u, v)

                # Direction in degrees in [0, 360)
                # Convention: meteorological direction (from which wind blows) often uses atan2(-u, -v)
                direction = (xr.apply_ufunc(np.degrees, xr.apply_ufunc(np.arctan2, -u, -v)) + 360.0) % 360.0

                mask_speed = (speed >= smin) & (speed <= smax)
                mask_dir = _dir_mask_circular(direction, int(dmin), int(dmax))
                mask = mask_speed & mask_dir

                u = u.where(mask)
                v = v.where(mask)

            mean_u = u.mean(dim=tname, skipna=True)
            mean_v = v.mean(dim=tname, skipna=True)

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

        avg_da = ds[x_name]

        # Stage-2 value filter (temperature) applied BEFORE averaging
        if value_filter_enabled:
            vmin, vmax = temp_range
            if vmin is not None:
                avg_da = avg_da.where(avg_da >= float(vmin))
            if vmax is not None:
                avg_da = avg_da.where(avg_da <= float(vmax))

        avg = avg_da.mean(dim=tname, skipna=True)
        avg2d = _ensure_spatial_array(avg, spatial_dims, name=f"avg ({x_name})")

        units = ds[x_name].attrs.get("units", "")
        return avg2d, extent, units, lon, lat, None, None


