"""
Map tap event handler for time series visualization.
Handles clicks on both grid points and glacier locations.
"""
from __future__ import annotations

from math import pi
from typing import Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.visualization.helpers.mercator_transformer import y_to_lat, x_to_lon
from src.visualization.helpers.set_timeseries import set_timeseries
from src.visualization.plots.yearly_overlays import set_yearly_overlays
from src.visualization.core.dataset import GLACIERS
from src.visualization.helpers.extract_timeseries_grid import _extract_timeseries_grid
from src.visualization.core.poly_fit import poly_fit_datetime
from src.visualization.helpers.glacier_secondary_axis import (
    normalize_glacier_series_for_secondary_axis,
    update_glacier_secondary_axis,
)
from src.visualization.helpers.wind_direction import compute_wind_direction_deg_from_uv

# Utilities
from src.visualization.utilities.bokeh_utils import (
    clear_and_hide_pairs,
    update_timeseries_source,
    set_figure_axes_labels,
    set_figure_title,
)
from src.visualization.utilities.time_filter_utils import filter_by_hours
from src.visualization.utilities.validation_utils import has_finite_data

from src.visualization.utilities.value_filter_utils import filter_series_by_int_range


R = 6378137.0  # WebMercator Earth radius (meters)
_DUMMY_START = _dt(2000, 1, 1)
_DUMMY_END = _dt(2000, 12, 31, 23, 59, 59)


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------


def _distance_to_nearest_grid(lon_vals, lat_vals, lon_click, lat_click) -> float:
    """
    Return approximate distance to the nearest grid point.
    Supports:
      - 1D lon + 1D lat (regular grid)  -> uses nearest lon + nearest lat
      - 2D lon + 2D lat (curvilinear)   -> brute force nearest point
    Distance is returned in "degrees" (lon/lat units), good enough for gating clicks.
    """
    lon_arr = np.asarray(lon_vals)
    lat_arr = np.asarray(lat_vals)

    # Regular grid: lon and lat are separate 1D axes
    if lon_arr.ndim == 1 and lat_arr.ndim == 1:
        lon_1d = lon_arr[np.isfinite(lon_arr)]
        lat_1d = lat_arr[np.isfinite(lat_arr)]
        if lon_1d.size == 0 or lat_1d.size == 0:
            return np.inf

        dlon = float(np.min(np.abs(lon_1d - lon_click)))
        dlat = float(np.min(np.abs(lat_1d - lat_click)))
        return float(np.hypot(dlon, dlat))

    # Curvilinear grid: lon/lat are 2D arrays of same shape
    if lon_arr.ndim == 2 and lat_arr.ndim == 2 and lon_arr.shape == lat_arr.shape:
        m = np.isfinite(lon_arr) & np.isfinite(lat_arr)
        if not m.any():
            return np.inf
        d2 = (lon_arr[m] - lon_click) ** 2 + (lat_arr[m] - lat_click) ** 2
        return float(np.sqrt(np.min(d2)))

    # Fallback: attempt to flatten only if shapes match
    lon_flat = lon_arr.ravel()
    lat_flat = lat_arr.ravel()
    if lon_flat.shape == lat_flat.shape:
        m = np.isfinite(lon_flat) & np.isfinite(lat_flat)
        if not m.any():
            return np.inf
        d2 = (lon_flat[m] - lon_click) ** 2 + (lat_flat[m] - lat_click) ** 2
        return float(np.sqrt(np.min(d2)))

    # Unknown / unsupported coordinate structure
    return np.inf





def _show_stats(values: Union[np.ndarray, Iterable[float]], unit: str, panes: Dict[str, object]) -> None:
    """Update statistics panes (mean, max, min)."""
    mean_p, max_p, min_p = panes.get("mean"), panes.get("max"), panes.get("min")
    if mean_p is None or max_p is None or min_p is None:
        return

    if not has_finite_data(values):
        mean_p.object = "**Mean:** —"
        max_p.object = "**Max:** —"
        min_p.object = "**Min:** —"
        return

    arr = np.asarray(list(values), dtype=float)
    mean_p.object = f"**Mean:** {np.nanmean(arr):.3f} {unit}"
    max_p.object = f"**Max:** {np.nanmax(arr):.3f} {unit}"
    min_p.object = f"**Min:** {np.nanmin(arr):.3f} {unit}"


def _nearest_glacier_name(glaciers: dict, x: float, y: float, max_dist_m: float = 10_000) -> Optional[str]:
    """Return nearest glacier name within max_dist_m, or None."""
    if not glaciers:
        return None
    lon, lat, names = glaciers.get("lon"), glaciers.get("lat"), glaciers.get("name")
    if lon is None or lat is None or getattr(lon, "size", 0) == 0:
        return None

    Gxm = (lon * (pi / 180.0) * R)
    Gym = R * np.log(np.tan((pi / 4.0) + (lat * (pi / 180.0) / 2.0)))
    di = np.argmin((Gxm - x) ** 2 + (Gym - y) ** 2)
    d_m = float(np.sqrt((Gxm[di] - x) ** 2 + (Gym[di] - y) ** 2))

    if d_m < max_dist_m:
        name = names[di]
        return name.item() if hasattr(name, "item") else str(name)
    return None


def _polyfit_and_update_source(ts_index: Iterable, y_values: Iterable[float], fit_degree: int, target_source) -> None:
    """Compute polynomial fit and update ColumnDataSource."""
    t_vals = list(pd.to_datetime(ts_index).to_pydatetime())
    y_vals = list(map(float, y_values))

    if len(t_vals) >= fit_degree + 1:
        t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=int(fit_degree), points=400)
        target_source.data = dict(t=t_fit, y=y_fit)
    else:
        target_source.data = dict(t=[], y=[])


def _get_dummy_range(yearly_window_widget) -> Optional[Tuple[_dt, _dt]]:
    """Extract (start, end) from yearly dummy-year slider."""
    if yearly_window_widget is None:
        return None
    try:
        v = yearly_window_widget.value
        if v and all(v):
            return v
    except Exception:
        pass
    return None


def _get_selected_years(year_fields) -> List[int]:
    """Extract selected years from widget(s)."""
    if hasattr(year_fields, "value"):
        return list(year_fields.value or [])
    return [w.value for w in (year_fields or [])]


def _lock_x_range_to_dummy_year(ts_fig, ts_fig_dir=None):
    """Lock x-ranges to dummy year bounds."""
    try:
        ts_fig.x_range.start = _DUMMY_START
        ts_fig.x_range.end = _DUMMY_END
        if ts_fig_dir is not None:
            ts_fig_dir.x_range.start = _DUMMY_START
            ts_fig_dir.x_range.end = _DUMMY_END
            ts_fig_dir.y_range.start = 0
            ts_fig_dir.y_range.end = 360
    except Exception:
        pass

def _in_circular_range_deg(angle_deg: np.ndarray, start_deg: int, end_deg: int) -> np.ndarray:
    """
    Inclusive circular range on [0, 360).
    Supports wrap-around: e.g. start=300, end=30 selects [300..360) U [0..30].

    IMPORTANT:
    - Interpret 0..360 as "full circle" (no filtering).
    - Treat 360° as equivalent to 0° for angle values, but NOT for the range endpoint.
    """
    a = np.asarray(angle_deg, dtype=float) % 360.0

    s_in = int(start_deg)
    e_in = int(end_deg)

    # Full-circle selection: do not filter
    if (s_in % 360) == 0 and (e_in % 360) == 0 and e_in >= 360:
        return np.ones_like(a, dtype=bool)

    s = float(s_in % 360)
    e = float(e_in % 360)

    # Non-wrapping interval
    if s <= e:
        return (a >= s) & (a <= e)

    # Wrap-around interval
    return (a >= s) | (a <= e)



# -----------------------------------------------------------------------------
# robust time series extraction (fallback-safe)
# -----------------------------------------------------------------------------

_WIND_U_CANDS = ("u", "u10", "10u", "u_component_of_wind", "eastward_wind")
_WIND_V_CANDS = ("v", "v10", "10v", "v_component_of_wind", "northward_wind")
_TEMP_CANDS = ("temperature_2m", "t2m", "2t", "tas", "temperature", "air_temperature", "temp")


def _pick_first_data_var(ds: xr.Dataset, candidates: Tuple[str, ...]) -> Optional[str]:
    dv = {k.lower(): k for k in ds.data_vars.keys()}
    for c in candidates:
        if c.lower() in dv:
            return dv[c.lower()]
    return None


def _guess_lon_lat_names(ds: xr.Dataset) -> Tuple[str, str]:
    """
    Try to find lon/lat variable/coord names in a dataset.
    Prefers common conventions.
    """
    # Prefer coordinates if present
    all_names = set(ds.coords.keys()) | set(ds.variables.keys())

    lon_priority = ("longitude", "lon", "x")
    lat_priority = ("latitude", "lat", "y")

    lon_name = next((n for n in lon_priority if n in all_names), None)
    lat_name = next((n for n in lat_priority if n in all_names), None)

    if lon_name is None or lat_name is None:
        raise ValueError(f"Could not find lon/lat names. Available: {sorted(all_names)}")

    return lon_name, lat_name


def _guess_time_name(ds: xr.Dataset) -> str:
    """
    Find a time-like coordinate name.
    """
    candidates = ("time", "valid_time")
    for c in candidates:
        if c in ds.coords or c in ds.variables:
            return c
    # last resort: any coordinate with datetime dtype
    for n in ds.coords:
        try:
            if np.issubdtype(ds[n].dtype, np.datetime64):
                return n
        except Exception:
            pass
    raise ValueError("Could not find a time coordinate (e.g. 'time' or 'valid_time').")


def _select_nearest_gridpoint(da: xr.DataArray, lon_name: str, lat_name: str, lon_click: float, lat_click: float) -> xr.DataArray:
    """
    Robust nearest selection for 1D or 2D lon/lat coordinate grids.
    - If lon/lat are 1D coords: use .sel(method="nearest")
    - If lon/lat are 2D coords: compute nearest index and .isel(...)
    """
    lon_obj = da[lon_name] if lon_name in da.coords or lon_name in da.data_vars else None
    lat_obj = da[lat_name] if lat_name in da.coords or lat_name in da.data_vars else None

    if lon_obj is None or lat_obj is None:
        # If lon/lat not attached to the DataArray, try from parent dataset via da._to_dataset?
        raise ValueError(f"lon/lat coordinates '{lon_name}/{lat_name}' are not available on the DataArray.")

    lon_vals = lon_obj.values
    lat_vals = lat_obj.values

    # 1D grid (regular lon/lat)
    if np.ndim(lon_vals) == 1 and np.ndim(lat_vals) == 1:
        return da.sel({lon_name: lon_click, lat_name: lat_click}, method="nearest")

    # 2D grid (curvilinear): nearest by brute force
    if np.ndim(lon_vals) == 2 and np.ndim(lat_vals) == 2:
        dist2 = (lon_vals - lon_click) ** 2 + (lat_vals - lat_click) ** 2
        flat_i = int(np.nanargmin(dist2))
        iy, ix = np.unravel_index(flat_i, dist2.shape)

        # dims for lon/lat grid; typically ('y','x')
        ydim, xdim = lon_obj.dims
        return da.isel({ydim: iy, xdim: ix})

    raise ValueError(f"Unsupported lon/lat grid dimensionality: lon.ndim={np.ndim(lon_vals)} lat.ndim={np.ndim(lat_vals)}")


def _extract_timeseries_fallback(files: List[Path], mode: str, t_range, lon_click: float, lat_click: float):
    """
    Fallback extractor that does not depend on DATASETS config.
    Returns:
      - (pd.Series, meta) for scalar
      - (pd.DataFrame(u,v), meta) for wind uv
    """
    if not files:
        raise ValueError("No files loaded in _last['files'].")

    with xr.open_mfdataset([str(p) for p in files], combine="by_coords", join="outer", parallel=False) as ds:
        tname = _guess_time_name(ds)
        lon_name, lat_name = _guess_lon_lat_names(ds)

        # Optional time slicing
        if t_range and all(t_range):
            t0, t1 = t_range
            ds = ds.sel({tname: slice(np.datetime64(pd.Timestamp(t0)), np.datetime64(pd.Timestamp(t1)))})

        if mode == "wind":
            u_name = _pick_first_data_var(ds, _WIND_U_CANDS)
            v_name = _pick_first_data_var(ds, _WIND_V_CANDS)
            if not u_name or not v_name:
                raise ValueError(f"Wind mode but could not find u/v in data_vars={list(ds.data_vars)}")

            da_u = ds[u_name]
            da_v = ds[v_name]

            # select point if spatial dims exist; otherwise accept as-is
            if (lon_name in da_u.coords or lon_name in da_u.data_vars) and (lat_name in da_u.coords or lat_name in da_u.data_vars) and da_u.ndim > 1:
                da_u = _select_nearest_gridpoint(da_u, lon_name, lat_name, lon_click, lat_click)
                da_v = _select_nearest_gridpoint(da_v, lon_name, lat_name, lon_click, lat_click)

            # Ensure we have a time series (time dimension present)
            if tname not in da_u.dims:
                raise ValueError(f"Selected wind u does not contain time dim '{tname}'. dims={da_u.dims}")
            if tname not in da_v.dims:
                raise ValueError(f"Selected wind v does not contain time dim '{tname}'. dims={da_v.dims}")

            idx = pd.to_datetime(da_u[tname].values)
            df = pd.DataFrame({"u": np.asarray(da_u.values).astype(float), "v": np.asarray(da_v.values).astype(float)}, index=idx)
            meta = {"kind": "uv", "units": da_u.attrs.get("units", "m/s")}
            return df, meta

        if mode == "temperature":
            x_name = _pick_first_data_var(ds, _TEMP_CANDS)
            if not x_name:
                raise ValueError(f"Temperature mode but could not find temp var in data_vars={list(ds.data_vars)}")

            da = ds[x_name]

            if (lon_name in da.coords or lon_name in da.data_vars) and (lat_name in da.coords or lat_name in da.data_vars) and da.ndim > 1:
                da = _select_nearest_gridpoint(da, lon_name, lat_name, lon_click, lat_click)

            if tname not in da.dims:
                raise ValueError(f"Selected temperature variable does not contain time dim '{tname}'. dims={da.dims}")

            idx = pd.to_datetime(da[tname].values)
            s = pd.Series(np.asarray(da.values).astype(float), index=idx, name=x_name)
            meta = {"kind": "scalar", "units": da.attrs.get("units", "")}
            return s, meta

        if mode == "glacier":
            raise ValueError("Fallback extractor does not handle glacier time series here.")

        raise ValueError(f"Unsupported mode for fallback extraction: {mode}")


def _extract_timeseries_safe(files, ds_key, t_range, lon_click, lat_click, mode: str):
    """
    Try the existing extractor first; if it fails (common when switching datasets),
    fall back to a robust xarray-based extractor.
    """
    try:
        ts, meta = _extract_timeseries_grid(files, ds_key, t_range, lon_click, lat_click)
        # sanity: ensure meta has kind
        if meta is None or "kind" not in meta:
            raise ValueError("Extractor returned invalid meta.")
        return ts, meta
    except Exception:
        # fallback based on detected mode from files
        if mode in ("wind", "temperature"):
            return _extract_timeseries_fallback([Path(p) for p in files], mode, t_range, lon_click, lat_click)
        # glacier handled elsewhere
        raise


# -----------------------------------------------------------------------------
# plotting branches
# -----------------------------------------------------------------------------

def _update_yearly_overlays(*, yearly_enabled_widget, yearly_window_widget, year_fields, alpha_widget, fit_degree: int,
                            ts_fig, base_series_or_df, kind: str, units_label: str, title_prefix: str,
                            ts_year_sources, ts_year_renderers, ts_year_fit_sources, ts_year_fit_renderers,
                            ts_fig_dir=None, ts_dir_year_sources=None, ts_dir_year_renderers=None,
                            ts_dir_year_fit_sources=None, ts_dir_year_fit_renderers=None,
                            trend_method: str = "polyfit",
                            trend_param: int = 3,
                            pre_smooth_enabled: bool = False,
                            pre_smooth_window_days: int = 30) -> None:

    """Draw or clear yearly overlays on main (and optional direction) plots."""
    yearly_on = bool(getattr(yearly_enabled_widget, "value", False))

    if not yearly_on:
        clear_and_hide_pairs(ts_year_sources, ts_year_renderers)
        clear_and_hide_pairs(ts_year_fit_sources, ts_year_fit_renderers)
        clear_and_hide_pairs(ts_dir_year_sources, ts_dir_year_renderers)
        clear_and_hide_pairs(ts_dir_year_fit_sources, ts_dir_year_fit_renderers)
        return

    years_selected = _get_selected_years(year_fields)
    dummy_range = _get_dummy_range(yearly_window_widget)
    alpha = float(getattr(alpha_widget, "value", 0.35) or 0.35)

    set_yearly_overlays(
        ts_fig,
        base_series_or_df,
        kind=kind,
        years=years_selected,
        yearly_sources=ts_year_sources,
        yearly_renderers=ts_year_renderers,
        yearly_fit_sources=ts_year_fit_sources,
        yearly_fit_renderers=ts_year_fit_renderers,
        fit_degree=int(fit_degree),
        alpha=alpha,
        units=units_label,
        title_prefix=title_prefix,
        dummy_range=dummy_range,
        trend_method=trend_method,
        trend_param=trend_param,
        pre_smooth_enabled=pre_smooth_enabled,
        pre_smooth_window_days=pre_smooth_window_days,


    )

    if ts_fig_dir is not None and ts_dir_year_sources and ts_dir_year_renderers:
        dir_series = compute_wind_direction_deg_from_uv(base_series_or_df) if kind == "uv" else base_series_or_df
        set_yearly_overlays(
            ts_fig_dir,
            dir_series,
            kind="scalar",
            years=years_selected,
            yearly_sources=ts_dir_year_sources,
            yearly_renderers=ts_dir_year_renderers,
            yearly_fit_sources=ts_dir_year_fit_sources,
            yearly_fit_renderers=ts_dir_year_fit_renderers,
            fit_degree=int(fit_degree),
            alpha=alpha,
            units="°",
            title_prefix="Yearly (direction)",
            dummy_range=dummy_range,
            trend_method=trend_method,
            trend_param=trend_param,
            pre_smooth_enabled=pre_smooth_enabled,
            pre_smooth_window_days=pre_smooth_window_days
        )

    _lock_x_range_to_dummy_year(ts_fig, ts_fig_dir)


def _plot_scalar_branch(*, ts, units, fit_degree, hours, ts_source, ts_source_fit=None, ts_fig=None,
                        stat_panes, yearly_enabled_widget=None, yearly_window_widget=None, year_fields=None,
                        alpha_widget=None, ts_year_sources=None, ts_year_renderers=None, ts_year_fit_sources=None,
                        ts_year_fit_renderers=None, ts_fig_dir=None, ts_dir_year_sources=None,
                        ts_dir_year_renderers=None, ts_dir_year_fit_sources=None,
                        ts_dir_year_fit_renderers=None,
                        trend_method: str = "polyfit",
                        trend_param: int = 3,
                        pre_smooth_enabled: bool = False,
                        pre_smooth_window_days: int = 30,
                        # NEW: stage-2 value filter config (temp)
                        value_filter_enabled: bool = False,
                        temp_range: tuple = (None, None)) -> Tuple[pd.Series, str]:


    """Render scalar time series with polynomial fit and yearly overlays."""
    ts_filtered = filter_by_hours(ts, hours)

    # Stage-2 value filter (temperature)
    if bool(value_filter_enabled):
        vmin, vmax = temp_range
        ts_filtered = filter_series_by_int_range(ts_filtered, vmin, vmax)

    if ts_source_fit is not None:
        _polyfit_and_update_source(ts_filtered.index, ts_filtered.values, fit_degree, ts_source_fit)

    if ts_fig is not None:
        ylab = f"Value ({units})" if units else "Value"
        set_figure_axes_labels(ts_fig, x_label="Time", y_label=ylab)
        set_figure_title(ts_fig, ylab)

    _update_yearly_overlays(
        yearly_enabled_widget=yearly_enabled_widget,
        yearly_window_widget=yearly_window_widget,
        year_fields=year_fields,
        alpha_widget=alpha_widget,
        fit_degree=fit_degree,
        ts_fig=ts_fig,
        base_series_or_df=ts_filtered,
        kind="scalar",
        units_label=units,
        title_prefix="Yearly",
        ts_year_sources=ts_year_sources,
        ts_year_renderers=ts_year_renderers,
        ts_year_fit_sources=ts_year_fit_sources,
        ts_year_fit_renderers=ts_year_fit_renderers,
        ts_fig_dir=ts_fig_dir,
        ts_dir_year_sources=ts_dir_year_sources,
        ts_dir_year_renderers=ts_dir_year_renderers,
        ts_dir_year_fit_sources=ts_dir_year_fit_sources,
        ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
        trend_method=trend_method,
        trend_param=trend_param,
        pre_smooth_enabled=pre_smooth_enabled,
        pre_smooth_window_days=pre_smooth_window_days,
    )

    return ts_filtered, "scalar"


def _plot_uv_branch(*, ts_uv, units, fit_degree, hours, ts_source, ts_source_fit=None, ts_source_dir=None,
                    ts_source_dir_fit=None, ts_fig=None, ts_fig_dir=None, stat_panes=None,
                    yearly_enabled_widget=None, yearly_window_widget=None, year_fields=None, alpha_widget=None,
                    ts_year_sources=None, ts_year_renderers=None, ts_year_fit_sources=None, ts_year_fit_renderers=None,
                    ts_dir_year_sources=None, ts_dir_year_renderers=None, ts_dir_year_fit_sources=None,
                    ts_dir_year_fit_renderers=None,
                    trend_method: str = "polyfit",
                    trend_param: int = 3,
                    pre_smooth_enabled: bool = False,
                    pre_smooth_window_days: int = 30,
                    # NEW: stage-2 value filter config (wind)
                    value_filter_enabled: bool = False,
                    speed_range: tuple = (None, None),
                    dir_range: tuple = (None, None)) -> Tuple[pd.DataFrame, str]:



    """Render UV wind data: speed on main plot, direction on direction plot."""
    df_filtered = filter_by_hours(ts_uv, hours)
    if df_filtered is None or df_filtered.empty:
        update_timeseries_source(ts_source, [], [])
        if ts_source_dir is not None:
            update_timeseries_source(ts_source_dir, [], [])
        return df_filtered, "uv"

    # Compute speed + direction first (direction needed for filtering)
    speed = np.hypot(df_filtered["u"].to_numpy(), df_filtered["v"].to_numpy())
    dir_series = compute_wind_direction_deg_from_uv(df_filtered)  # pd.Series aligned to df_filtered.index
    direction = dir_series.to_numpy()

    # Stage-2 value filters (speed + circular direction)
    if bool(value_filter_enabled):
        smin, smax = speed_range
        dmin, dmax = dir_range

        smin = -np.inf if smin is None else float(smin)
        smax = np.inf if smax is None else float(smax)

        mask_speed = (speed >= smin) & (speed <= smax)
        mask_dir = _in_circular_range_deg(direction, int(dmin), int(dmax))
        mask = mask_speed & mask_dir

        df_filtered = df_filtered.loc[mask]
        speed = speed[mask]
        dir_series = dir_series.loc[df_filtered.index]  # keep aligned


    update_timeseries_source(ts_source, df_filtered.index, speed)
    _show_stats(speed, units, stat_panes or {})

    if ts_source_fit is not None:
        _polyfit_and_update_source(df_filtered.index, speed, fit_degree, ts_source_fit)

    if ts_source_dir is not None:
        update_timeseries_source(ts_source_dir, dir_series.index, dir_series.values)
        if ts_source_dir_fit is not None:
            _polyfit_and_update_source(dir_series.index, dir_series.values, fit_degree, ts_source_dir_fit)


    if ts_fig is not None:
        set_figure_axes_labels(ts_fig, x_label="Time", y_label=f"Wind speed ({units})")
        set_figure_title(ts_fig, f"Wind speed ({units})")

    if ts_fig_dir is not None:
        set_figure_axes_labels(ts_fig_dir, x_label="Time", y_label="Wind direction (°)")
        set_figure_title(ts_fig_dir, "Wind direction (°)")

    _update_yearly_overlays(
        yearly_enabled_widget=yearly_enabled_widget,
        yearly_window_widget=yearly_window_widget,
        year_fields=year_fields,
        alpha_widget=alpha_widget,
        fit_degree=fit_degree,
        ts_fig=ts_fig,
        base_series_or_df=df_filtered,
        kind="uv",
        units_label=units,
        title_prefix="Yearly",
        ts_year_sources=ts_year_sources,
        ts_year_renderers=ts_year_renderers,
        ts_year_fit_sources=ts_year_fit_sources,
        ts_year_fit_renderers=ts_year_fit_renderers,
        ts_fig_dir=ts_fig_dir,
        ts_dir_year_sources=ts_dir_year_sources,
        ts_dir_year_renderers=ts_dir_year_renderers,
        ts_dir_year_fit_sources=ts_dir_year_fit_sources,
        ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
        trend_method=trend_method,
        trend_param=trend_param,
        pre_smooth_enabled=pre_smooth_enabled,
        pre_smooth_window_days=pre_smooth_window_days,
    )

    return df_filtered, "uv"


# -----------------------------------------------------------------------------
# click handlers
# -----------------------------------------------------------------------------

def _handle_glacier_click(*, glacier_name, lon_click, lat_click, _last, w_status, w_stat_ndatapoints,
                          w_glacier_multiplier, w_glacier_offset, ts_glacier_source, ts_fig, fit_degree,
                          ts_source, ts_source_fit, stat_panes, ts_source_dir=None, ts_source_dir_fit=None,
                          ts_fig_dir=None, yearly_enabled_widget=None, yearly_window_widget=None, year_fields=None,
                          alpha_widget=None, ts_year_sources=None, ts_year_renderers=None, ts_year_fit_sources=None,
                          ts_year_fit_renderers=None, ts_dir_year_sources=None, ts_dir_year_renderers=None,
                          ts_dir_year_fit_sources=None, ts_dir_year_fit_renderers=None) -> None:
    """Handle click on a glacier location."""
    from src.visualization.plots.glacier_overlay import glacier_series_by_name

    s_glacier = glacier_series_by_name(GLACIERS["dir"], str(glacier_name))
    mult = float(getattr(w_glacier_multiplier, "value", 10.0) or 10.0)
    off = float(getattr(w_glacier_offset, "value", 0.0) or 0.0)

    s_norm = normalize_glacier_series_for_secondary_axis(s_glacier, multiplier=mult, offset=off)
    update_glacier_secondary_axis(s_norm=s_norm, ts_glacier_source=ts_glacier_source, ts_fig=ts_fig)

    if w_glacier_multiplier is not None:
        w_glacier_multiplier.visible = True
    if w_glacier_offset is not None:
        w_glacier_offset.visible = True

    _last["selected_glacier_name"] = str(glacier_name)
    _last["glacier_series_raw"] = s_glacier

    _last["trend_method_climate"] = _last.get("trend_method_climate", "polyfit")
    _last["trend_param_climate"] = _last.get("trend_param_climate", 3)
    _last["pre_smooth_enabled_climate"] = _last.get("pre_smooth_enabled_climate", False)
    _last["pre_smooth_window_days_climate"] = _last.get("pre_smooth_window_days_climate", 30)




def _handle_grid_click(*, lon_click, lat_click, _last, w_status, w_stat_ndatapoints, fit_degree,
                       ts_source, ts_source_fit, stat_panes, ts_source_dir=None, ts_source_dir_fit=None,
                       ts_fig=None, ts_fig_dir=None, yearly_enabled_widget=None, yearly_window_widget=None,
                       year_fields=None, alpha_widget=None, ts_year_sources=None, ts_year_renderers=None,
                       ts_year_fit_sources=None, ts_year_fit_renderers=None, ts_dir_year_sources=None,
                       ts_dir_year_renderers=None, ts_dir_year_fit_sources=None,
                       ts_dir_year_fit_renderers=None) -> None:
    """Handle click on a regular grid point."""
    mode = _last.get("data_kind") or "temperature"

    # Reject clicks far away from grid points
    lon_vals = _last.get("lon")
    lat_vals = _last.get("lat")

    if lon_vals is not None and lat_vals is not None:
        d = _distance_to_nearest_grid(lon_vals, lat_vals, lon_click, lat_click)

        # threshold ~ half a grid cell (tune if needed)
        if d > 0.25:
            w_status.object = "Clicked outside data grid."
            return

    try:
        ts, meta = _extract_timeseries_safe(_last["files"], _last["ds_key"], _last["time_range"], lon_click, lat_click, mode=mode)
    except Exception as e:
        w_status.object = f"Tap failed: {e}"
        return

    if ts is None or meta is None or len(ts) == 0:
        w_status.object = "No data at clicked location."
        return

    hours = _last.get("hours")
    if w_stat_ndatapoints is not None:
        try:
            w_stat_ndatapoints.object = f"**Number of datapoints in the range:** {ts.shape[0]}"
        except Exception:
            pass

    if meta["kind"] == "scalar":
        ts_used, kind_used = _plot_scalar_branch(
            ts=ts, units=meta.get("units", ""), fit_degree=fit_degree, hours=hours,
            ts_source=ts_source, ts_source_fit=ts_source_fit, ts_fig=ts_fig, stat_panes=stat_panes,
            yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
            year_fields=year_fields, alpha_widget=alpha_widget, ts_year_sources=ts_year_sources,
            ts_year_renderers=ts_year_renderers, ts_year_fit_sources=ts_year_fit_sources,
            ts_year_fit_renderers=ts_year_fit_renderers, ts_fig_dir=ts_fig_dir,
            ts_dir_year_sources=ts_dir_year_sources, ts_dir_year_renderers=ts_dir_year_renderers,
            ts_dir_year_fit_sources=ts_dir_year_fit_sources, ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
            value_filter_enabled=_last.get("value_filter_enabled", False),
            temp_range=_last.get("temp_range", (None, None)),
        )
    else:
        ts_used, kind_used = _plot_uv_branch(
            ts_uv=ts, units=meta.get("units", "m/s"), fit_degree=fit_degree, hours=hours,
            ts_source=ts_source, ts_source_fit=ts_source_fit, ts_source_dir=ts_source_dir,
            ts_source_dir_fit=ts_source_dir_fit, ts_fig=ts_fig, ts_fig_dir=ts_fig_dir, stat_panes=stat_panes,
            yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
            year_fields=year_fields, alpha_widget=alpha_widget, ts_year_sources=ts_year_sources,
            ts_year_renderers=ts_year_renderers, ts_year_fit_sources=ts_year_fit_sources,
            ts_year_fit_renderers=ts_year_fit_renderers, ts_dir_year_sources=ts_dir_year_sources,
            ts_dir_year_renderers=ts_dir_year_renderers, ts_dir_year_fit_sources=ts_dir_year_fit_sources,
            ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
            value_filter_enabled=_last.get("value_filter_enabled", False),
            speed_range=_last.get("speed_range", (None, None)),
            dir_range=_last.get("dir_range", (0, 360)),

        )

    _last["picked_series"] = ts_used
    _last["picked_kind"] = kind_used
    _last["picked_units"] = meta.get("units", "" if kind_used == "scalar" else "m/s")

    _last["trend_method_climate"] = _last.get("trend_method_climate", "polyfit")
    _last["trend_param_climate"] = _last.get("trend_param_climate", 3)
    _last["pre_smooth_enabled_climate"] = _last.get("pre_smooth_enabled_climate", False)
    _last["pre_smooth_window_days_climate"] = _last.get("pre_smooth_window_days_climate", 30)

    w_status.object = f"Selected grid point ({lon_click:.3f}, {lat_click:.3f})"

    # IMPORTANT: set_timeseries requires keyword-only 'kind'
    set_timeseries(ts_source, ts_used, kind=kind_used)


def _on_tap(evt, w_status, _last, ts_source, ts_fig, _stat_panes, ts_source_fit=None, fit_degree=3,
            ts_source_dir=None, ts_source_dir_fit=None, ts_fig_dir=None, yearly_enabled_widget=None,
            year_fields=None, alpha_widget=None, yearly_window_widget=None, ts_year_sources=None,
            ts_year_renderers=None, ts_dir_year_sources=None, ts_dir_year_renderers=None, w_stat_ndatapoints=None,
            ts_year_fit_sources=None, ts_year_fit_renderers=None, ts_dir_year_fit_sources=None,
            ts_dir_year_fit_renderers=None, ts_glacier_source=None, w_glacier_multiplier=None, w_glacier_offset=None):
    """Handle map tap events. Hook: bkplot.on_event(Tap, lambda evt: _on_tap(...))"""

    # Defensive reset when switching datasets / kinds
    if _last.get("picked_kind") not in (None, "scalar", "uv"):
        _last["picked_kind"] = None

    try:
        lon_click = float(x_to_lon(evt.x))
        lat_click = float(y_to_lat(evt.y))
        w_status.object = f"Tap at x={evt.x:.1f}, y={evt.y:.1f}"

        glacier_name = _nearest_glacier_name(_last.get("glaciers"), evt.x, evt.y)

        if glacier_name:
            _handle_glacier_click(
                glacier_name=glacier_name, lon_click=lon_click, lat_click=lat_click, _last=_last,
                w_status=w_status, w_stat_ndatapoints=w_stat_ndatapoints, w_glacier_multiplier=w_glacier_multiplier,
                w_glacier_offset=w_glacier_offset, ts_glacier_source=ts_glacier_source, ts_fig=ts_fig,
                fit_degree=fit_degree, ts_source=ts_source, ts_source_fit=ts_source_fit, stat_panes=_stat_panes,
                ts_source_dir=ts_source_dir, ts_source_dir_fit=ts_source_dir_fit, ts_fig_dir=ts_fig_dir,
                yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
                year_fields=year_fields, alpha_widget=alpha_widget, ts_year_sources=ts_year_sources,
                ts_year_renderers=ts_year_renderers, ts_year_fit_sources=ts_year_fit_sources,
                ts_year_fit_renderers=ts_year_fit_renderers, ts_dir_year_sources=ts_dir_year_sources,
                ts_dir_year_renderers=ts_dir_year_renderers, ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                ts_dir_year_fit_renderers=ts_dir_year_fit_renderers
            )
            return
        else:
            _handle_grid_click(
                lon_click=lon_click, lat_click=lat_click, _last=_last, w_status=w_status,
                w_stat_ndatapoints=w_stat_ndatapoints, fit_degree=fit_degree, ts_source=ts_source,
                ts_source_fit=ts_source_fit, stat_panes=_stat_panes, ts_source_dir=ts_source_dir,
                ts_source_dir_fit=ts_source_dir_fit, ts_fig=ts_fig, ts_fig_dir=ts_fig_dir,
                yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
                year_fields=year_fields, alpha_widget=alpha_widget, ts_year_sources=ts_year_sources,
                ts_year_renderers=ts_year_renderers, ts_year_fit_sources=ts_year_fit_sources,
                ts_year_fit_renderers=ts_year_fit_renderers, ts_dir_year_sources=ts_dir_year_sources,
                ts_dir_year_renderers=ts_dir_year_renderers, ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                ts_dir_year_fit_renderers=ts_dir_year_fit_renderers
            )
    except Exception as e:
        w_status.object = f"Tap handler error: {e}"
