# src/visualization/actions/on_tap.py
from __future__ import annotations

from math import pi
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.visualization.helpers.mercator_transformer import y_to_lat, x_to_lon
from src.visualization.helpers.set_timeseries import set_timeseries
from src.visualization.plots.yearly_overlays import set_yearly_overlays

from src.visualization.core.dataset import GLACIERS
from src.visualization.helpers.extract_timeseries_grid import _extract_timeseries_grid
from src.visualization.core.poly_fit import poly_fit_datetime


# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #

R = 6378137.0  # WebMercator Earth radius (meters)

# Dummy-year bounds used in Yearly mode
from datetime import datetime as _dt
_DUMMY_START = _dt(2000, 1, 1)
_DUMMY_END = _dt(2000, 12, 31, 23, 59, 59)


# ----------------------------------------------------------------------------- #
# Small utilities
# ----------------------------------------------------------------------------- #

def _apply_hour_filter(
    ts: Union[pd.Series, pd.DataFrame],
    hours: Optional[Iterable[int]],
) -> Union[pd.Series, pd.DataFrame]:
    """Filter a time series (Series or DataFrame) to the given set of hours."""
    if not hours:
        return ts
    idx = pd.to_datetime(ts.index)
    mask = idx.hour.astype(int).isin(set(int(h) for h in hours))
    return ts.loc[mask]


def _show_stats(values: Union[np.ndarray, Iterable[float]], unit: str, panes: Dict[str, object]) -> None:
    """Update the side stats panes."""
    mean_p, max_p, min_p = panes.get("mean"), panes.get("max"), panes.get("min")
    if mean_p is None or max_p is None or min_p is None:
        return

    a = np.asarray(list(values), dtype=float)
    if a.size == 0 or not np.isfinite(a).any():
        mean_p.object = "**Mean:** –"
        max_p.object = "**Max:** –"
        min_p.object = "**Min:** –"
        return

    mean_p.object = f"**Mean:** {np.nanmean(a):.3f} {unit}"
    max_p.object = f"**Max:** {np.nanmax(a):.3f} {unit}"
    min_p.object = f"**Min:** {np.nanmin(a):.3f} {unit}"


def _nearest_glacier_name(glaciers: dict, x: float, y: float, max_dist_m: float = 10_000) -> Optional[str]:
    """Return nearest glacier name within max_dist_m, or None."""
    if not glaciers:
        return None
    lon = glaciers.get("lon")
    lat = glaciers.get("lat")
    names = glaciers.get("name")
    if lon is None or lat is None or getattr(lon, "size", 0) == 0:
        return None

    # Project glacier lon/lat to WebMercator meters to compare with evt.x/evt.y
    Gxm = (lon * (pi / 180.0) * R)
    Gym = R * np.log(np.tan((pi / 4.0) + (lat * (pi / 180.0) / 2.0)))
    di = np.argmin((Gxm - x) ** 2 + (Gym - y) ** 2)
    d_m = float(np.sqrt((Gxm[di] - x) ** 2 + (Gym[di] - y) ** 2))
    if d_m < max_dist_m:
        name = names[di]
        return name.item() if hasattr(name, "item") else str(name)
    return None


def _polyfit_series_to_source(
    ts_index: Iterable,
    y_values: Iterable[float],
    fit_degree: int,
    target_source,
) -> None:
    """Compute a polynomial fit and push into a Bokeh ColumnDataSource."""
    t_vals = list(pd.to_datetime(ts_index).to_pydatetime())
    y_vals = list(map(float, y_values))
    if len(t_vals) >= fit_degree + 1:
        t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=int(fit_degree), points=400)
        target_source.data = dict(t=t_fit, y=y_fit)
    else:
        target_source.data = dict(t=[], y=[])


def _compute_direction_series(uv_df: pd.DataFrame) -> pd.Series:
    """Return wind direction (from) in degrees [0, 360) from a UV dataframe."""
    # dir = 270 - arctan2(v, u) in degrees, modulo 360
    dir_deg = (270.0 - np.degrees(np.arctan2(uv_df["v"].to_numpy(), uv_df["u"].to_numpy()))) % 360.0
    return pd.Series(dir_deg, index=uv_df.index, name="dir")


def _current_dummy_range(yearly_window_widget) -> Optional[Tuple[_dt, _dt]]:
    """Read (start, end) from the yearly dummy-year slider, if available."""
    if yearly_window_widget is None:
        return None
    try:
        v = yearly_window_widget.value
        if v and all(v):
            return v  # (start, end) in dummy year
    except Exception:
        pass
    return None


def _selected_years(year_fields) -> List[int]:
    """
    Return the list of selected years from either:
    - a MultiChoice (via .value), or
    - a list of Select widgets (legacy).
    """
    if hasattr(year_fields, "value"):
        return list(year_fields.value or [])
    return [w.value for w in (year_fields or [])]


def _update_yearly_overlays(
    *,
    yearly_enabled_widget,
    yearly_window_widget,
    year_fields,
    alpha_widget,
    fit_degree: int,
    # main figure
    ts_fig,
    base_series_or_df: Union[pd.Series, pd.DataFrame],
    kind: str,               # "scalar" | "uv"
    units_label: str,
    title_prefix: str,
    ts_year_sources,
    ts_year_renderers,
    ts_year_fit_sources,
    ts_year_fit_renderers,
    # direction figure (optional)
    ts_fig_dir=None,
    ts_dir_year_sources=None,
    ts_dir_year_renderers=None,
    ts_dir_year_fit_sources=None,
    ts_dir_year_fit_renderers=None,
) -> None:
    """
    Draw or clear yearly overlays on the main (and optional direction) plot.
    Also snaps x-ranges to the dummy year when yearly mode is ON.
    """
    YEARLY_ON = bool(getattr(yearly_enabled_widget, "value", False))

    def _clear_pair(src_list, rnd_list):
        if src_list and rnd_list:
            for s, r in zip(src_list, rnd_list):
                s.data = dict(t=[], y=[])
                r.visible = False

    if not YEARLY_ON:
        _clear_pair(ts_year_sources, ts_year_renderers)
        _clear_pair(ts_year_fit_sources, ts_year_fit_renderers)
        _clear_pair(ts_dir_year_sources, ts_dir_year_renderers)
        _clear_pair(ts_dir_year_fit_sources, ts_dir_year_fit_renderers)
        return

    years_selected = _selected_years(year_fields)
    dummy_range = _current_dummy_range(yearly_window_widget)
    alpha = float(getattr(alpha_widget, "value", 0.35) or 0.35)

    # Main overlays
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
    )

    # Direction overlays (if a direction fig exists)
    if ts_fig_dir is not None and ts_dir_year_sources and ts_dir_year_renderers:
        if kind == "uv":
            dir_series = _compute_direction_series(base_series_or_df)
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
            )
        else:
            # Optional: reuse scalar series on direction fig if desired
            set_yearly_overlays(
                ts_fig_dir,
                base_series_or_df,
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
            )

    # Lock x-ranges to dummy year
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


# ----------------------------------------------------------------------------- #
# Glacier helpers
# ----------------------------------------------------------------------------- #

def _normalize_glacier_series_for_secondary_axis(s: pd.Series) -> pd.Series:
    """Normalize glacier series to 0..10 for a secondary axis line."""
    arr = s.values.astype(float)
    vmin = float(np.nanmin(arr))
    vmax = float(np.nanmax(arr))
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        s_norm = (arr - vmin) / (vmax - vmin) * 10.0
    else:
        s_norm = np.full_like(arr, 5.0)
    return pd.Series(s_norm, index=s.index, name=s.name)


def _update_glacier_secondary_axis(
    *,
    s_norm: pd.Series,
    ts_glacier_source,
    ts_fig,
) -> None:
    """Push glacier series to the secondary y-axis and autoscale it."""
    if ts_glacier_source is None:
        return
    if not (hasattr(ts_fig, "extra_y_ranges") and "glacier" in ts_fig.extra_y_ranges):
        return

    t_vals = s_norm.index.to_pydatetime().tolist()
    y_vals = s_norm.values.astype(float).tolist()
    ts_glacier_source.data = dict(t=t_vals, y=y_vals)

    if y_vals:
        arr = np.asarray(y_vals, dtype=float)
        ymin = float(np.nanmin(arr))
        ymax = float(np.nanmax(arr))
        if not (np.isfinite(ymin) and np.isfinite(ymax)):
            ymin, ymax = -1.0, 1.0
        pad = 0.05 * (ymax - ymin if ymax > ymin else (abs(ymax) or 1.0))
        rng = ts_fig.extra_y_ranges["glacier"]
        rng.start = ymin - pad
        rng.end = ymax + pad


# ----------------------------------------------------------------------------- #
# Plotting branches (scalar / uv)
# ----------------------------------------------------------------------------- #

def _plot_scalar_branch(
    *,
    ts: pd.Series,
    units: str,
    fit_degree: int,
    hours: Optional[Iterable[int]],
    # bokeh sources/figs
    ts_source,
    ts_source_fit=None,
    ts_fig=None,
    # stats
    stat_panes: Dict[str, object],
    # yearly
    yearly_enabled_widget=None,
    yearly_window_widget=None,
    year_fields=None,
    alpha_widget=None,
    ts_year_sources=None,
    ts_year_renderers=None,
    ts_year_fit_sources=None,
    ts_year_fit_renderers=None,
    # dir plot (optional)
    ts_fig_dir=None,
    ts_dir_year_sources=None,
    ts_dir_year_renderers=None,
    ts_dir_year_fit_sources=None,
    ts_dir_year_fit_renderers=None,
) -> Tuple[pd.Series, str]:
    """Render scalar time series + fits + yearly overlays; return (series, kind)."""
    ts_f = _apply_hour_filter(ts, hours)
    _show_stats(ts_f.values, units, stat_panes)

    set_timeseries(ts_source, ts_f, kind="scalar", units=units, title="Point time series", fig=ts_fig)

    if ts_source_fit is not None:
        _polyfit_series_to_source(ts_f.index, ts_f.values, fit_degree, ts_source_fit)

    _update_yearly_overlays(
        yearly_enabled_widget=yearly_enabled_widget,
        yearly_window_widget=yearly_window_widget,
        year_fields=year_fields,
        alpha_widget=alpha_widget,
        fit_degree=fit_degree,
        ts_fig=ts_fig,
        base_series_or_df=ts_f,
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
    )

    return ts_f, "scalar"


def _plot_uv_branch(
    *,
    ts_uv: pd.DataFrame,
    units: str,
    fit_degree: int,
    hours: Optional[Iterable[int]],
    # bokeh sources/figs
    ts_source,
    ts_source_fit=None,
    ts_source_dir=None,
    ts_source_dir_fit=None,
    ts_fig=None,
    ts_fig_dir=None,
    # stats
    stat_panes: Dict[str, object],
    # yearly
    yearly_enabled_widget=None,
    yearly_window_widget=None,
    year_fields=None,
    alpha_widget=None,
    ts_year_sources=None,
    ts_year_renderers=None,
    ts_year_fit_sources=None,
    ts_year_fit_renderers=None,
    ts_dir_year_sources=None,
    ts_dir_year_renderers=None,
    ts_dir_year_fit_sources=None,
    ts_dir_year_fit_renderers=None,
) -> Tuple[pd.DataFrame, str]:
    """Render UV time series + direction + fits + yearly overlays; return (df, kind)."""
    ts_f = _apply_hour_filter(ts_uv, hours)

    speed = np.hypot(ts_f["u"].to_numpy(), ts_f["v"].to_numpy())
    _show_stats(speed, units, stat_panes)

    # Primary: UV as magnitude plot (Observed)
    set_timeseries(ts_source, ts_f, kind="uv", units=units, title="Wind speed (|u,v|)", fig=ts_fig)

    # Direction series and raw direction plot
    dir_series = _compute_direction_series(ts_f)
    if ts_source_dir is not None:
        set_timeseries(ts_source_dir, dir_series, kind="scalar", units="°", title="Wind direction (from)", fig=ts_fig_dir)

    # Fits
    if ts_source_dir_fit is not None:
        _polyfit_series_to_source(dir_series.index, dir_series.values, fit_degree, ts_source_dir_fit)

    if ts_source_fit is not None:
        _polyfit_series_to_source(ts_f.index, speed.tolist(), fit_degree, ts_source_fit)

    # Yearly overlays (main + direction)
    _update_yearly_overlays(
        yearly_enabled_widget=yearly_enabled_widget,
        yearly_window_widget=yearly_window_widget,
        year_fields=year_fields,
        alpha_widget=alpha_widget,
        fit_degree=fit_degree,
        ts_fig=ts_fig,
        base_series_or_df=ts_f,
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
    )

    return ts_f, "uv"


# ----------------------------------------------------------------------------- #
# External entry point
# ----------------------------------------------------------------------------- #

def _on_tap(
    evt,
    w_status,
    _last: dict,
    ts_source,
    ts_fig,
    _stat_panes,
    ts_source_fit=None,
    fit_degree: int = 3,
    ts_source_dir=None,
    ts_source_dir_fit=None,
    ts_fig_dir=None,
    # Yearly UI / state
    yearly_enabled_widget=None,
    year_fields=None,
    alpha_widget=None,
    yearly_window_widget=None,
    ts_year_sources=None,
    ts_year_renderers=None,
    ts_dir_year_sources=None,
    ts_dir_year_renderers=None,
    w_stat_ndatapoints=None,
    # Yearly FIT overlays
    ts_year_fit_sources=None,
    ts_year_fit_renderers=None,
    ts_dir_year_fit_sources=None,
    ts_dir_year_fit_renderers=None,
    # Glacier overlay (secondary axis)
    ts_glacier_source=None,
):
    """
    External tap handler. Hook from app:
        bkplot.on_event(Tap, lambda evt: _on_tap(...))
    """
    try:
        # --- Click prelude: feedback + click location ---
        w_status.object = f"Tap at x={evt.x:.1f}, y={evt.y:.1f}"
        lon_click = float(x_to_lon(evt.x))
        lat_click = float(y_to_lat(evt.y))

        # --- If near a glacier, handle that branch first ----------------------
        name_used = _nearest_glacier_name(_last.get("glaciers"), evt.x, evt.y)
        if name_used:
            from src.visualization.plots.glacier_overlay import glacier_series_by_name

            # Glacier series (always 'scalar'); normalize for the secondary axis
            s_glacier = glacier_series_by_name(GLACIERS["dir"], str(name_used))
            s_norm = _normalize_glacier_series_for_secondary_axis(s_glacier)
            # Push to secondary axis (if wired)
            _update_glacier_secondary_axis(s_norm=s_norm, ts_glacier_source=ts_glacier_source, ts_fig=ts_fig)

            # Additionally try to load the grid series at the clicked point,
            # so both can be seen together on primary axis.
            try:
                ts_grid, meta_grid = _extract_timeseries_grid(
                    _last["files"], _last["ds_key"], _last["time_range"], lon_click, lat_click
                )
            except Exception:
                ts_grid, meta_grid = None, None

            # Apply hours if we were able to extract a grid series
            if ts_grid is not None and meta_grid is not None and len(ts_grid) > 0:
                hours = _last.get("hours")
                if w_stat_ndatapoints is not None:
                    w_stat_ndatapoints.object = f"**Number of datapoints in the range:** {ts_grid.shape[0]}"

                if meta_grid["kind"] == "scalar":
                    ts_used, kind_used = _plot_scalar_branch(
                        ts=ts_grid,
                        units=meta_grid.get("units", ""),
                        fit_degree=fit_degree,
                        hours=hours,
                        ts_source=ts_source,
                        ts_source_fit=ts_source_fit,
                        ts_fig=ts_fig,
                        stat_panes=_stat_panes,
                        yearly_enabled_widget=yearly_enabled_widget,
                        yearly_window_widget=yearly_window_widget,
                        year_fields=year_fields,
                        alpha_widget=alpha_widget,
                        ts_year_sources=ts_year_sources,
                        ts_year_renderers=ts_year_renderers,
                        ts_year_fit_sources=ts_year_fit_sources,
                        ts_year_fit_renderers=ts_year_fit_renderers,
                        ts_fig_dir=ts_fig_dir,
                        ts_dir_year_sources=ts_dir_year_sources,
                        ts_dir_year_renderers=ts_dir_year_renderers,
                        ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                        ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
                    )
                else:
                    ts_used, kind_used = _plot_uv_branch(
                        ts_uv=ts_grid,
                        units=meta_grid.get("units", "m/s"),
                        fit_degree=fit_degree,
                        hours=hours,
                        ts_source=ts_source,
                        ts_source_fit=ts_source_fit,
                        ts_source_dir=ts_source_dir,
                        ts_source_dir_fit=ts_source_dir_fit,
                        ts_fig=ts_fig,
                        ts_fig_dir=ts_fig_dir,
                        stat_panes=_stat_panes,
                        yearly_enabled_widget=yearly_enabled_widget,
                        yearly_window_widget=yearly_window_widget,
                        year_fields=year_fields,
                        alpha_widget=alpha_widget,
                        ts_year_sources=ts_year_sources,
                        ts_year_renderers=ts_year_renderers,
                        ts_year_fit_sources=ts_year_fit_sources,
                        ts_year_fit_renderers=ts_year_fit_renderers,
                        ts_dir_year_sources=ts_dir_year_sources,
                        ts_dir_year_renderers=ts_dir_year_renderers,
                        ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                        ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
                    )

                _last["picked_series"] = ts_used
                _last["picked_kind"] = kind_used
                _last["picked_units"] = meta_grid.get("units", "" if kind_used == "scalar" else "m/s")

            # Status
            w_status.object = f"Selected glacier: **{name_used}** ({lon_click:.3f}, {lat_click:.3f})"
            return

        # --- Normal grid branch ------------------------------------------------
        ts, meta = _extract_timeseries_grid(
            _last["files"], _last["ds_key"], _last["time_range"], lon_click, lat_click
        )

        hours = _last.get("hours")
        if w_stat_ndatapoints is not None:
            w_stat_ndatapoints.object = f"**Number of datapoints in the range:** {ts.shape[0]}"

        if meta["kind"] == "scalar":
            ts_used, kind_used = _plot_scalar_branch(
                ts=ts,
                units=meta.get("units", ""),
                fit_degree=fit_degree,
                hours=hours,
                ts_source=ts_source,
                ts_source_fit=ts_source_fit,
                ts_fig=ts_fig,
                stat_panes=_stat_panes,
                yearly_enabled_widget=yearly_enabled_widget,
                yearly_window_widget=yearly_window_widget,
                year_fields=year_fields,
                alpha_widget=alpha_widget,
                ts_year_sources=ts_year_sources,
                ts_year_renderers=ts_year_renderers,
                ts_year_fit_sources=ts_year_fit_sources,
                ts_year_fit_renderers=ts_year_fit_renderers,
                ts_fig_dir=ts_fig_dir,
                ts_dir_year_sources=ts_dir_year_sources,
                ts_dir_year_renderers=ts_dir_year_renderers,
                ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
            )
        else:
            ts_used, kind_used = _plot_uv_branch(
                ts_uv=ts,
                units=meta.get("units", "m/s"),
                fit_degree=fit_degree,
                hours=hours,
                ts_source=ts_source,
                ts_source_fit=ts_source_fit,
                ts_source_dir=ts_source_dir,
                ts_source_dir_fit=ts_source_dir_fit,
                ts_fig=ts_fig,
                ts_fig_dir=ts_fig_dir,
                stat_panes=_stat_panes,
                yearly_enabled_widget=yearly_enabled_widget,
                yearly_window_widget=yearly_window_widget,
                year_fields=year_fields,
                alpha_widget=alpha_widget,
                ts_year_sources=ts_year_sources,
                ts_year_renderers=ts_year_renderers,
                ts_year_fit_sources=ts_year_fit_sources,
                ts_year_fit_renderers=ts_year_fit_renderers,
                ts_dir_year_sources=ts_dir_year_sources,
                ts_dir_year_renderers=ts_dir_year_renderers,
                ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
            )

        # Cache last pick for slider-driven updates
        _last["picked_series"] = ts_used
        _last["picked_kind"] = kind_used
        _last["picked_units"] = meta.get("units", "" if kind_used == "scalar" else "m/s")

        # Final status
        w_status.object = f"Picked grid point at ({lon_click:.3f}, {lat_click:.3f})."

    except Exception as ex:
        w_status.object = f"**Click error:** {ex}"
