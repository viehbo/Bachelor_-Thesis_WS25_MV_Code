"""
Map tap event handler for time series visualization.
Handles clicks on both grid points and glacier locations.
"""
from __future__ import annotations

from math import pi
from typing import Dict, Iterable, List, Optional, Tuple, Union
from datetime import datetime as _dt

import numpy as np
import pandas as pd

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

# Import new utilities
from src.visualization.utilities.bokeh_utils import (
    clear_and_hide_pairs,
    update_timeseries_source,
    set_figure_axes_labels,
    set_figure_title,
)
from src.visualization.utilities.time_filter_utils import filter_by_hours
from src.visualization.utilities.validation_utils import has_finite_data

R = 6378137.0  # WebMercator Earth radius (meters)
_DUMMY_START = _dt(2000, 1, 1)
_DUMMY_END = _dt(2000, 12, 31, 23, 59, 59)


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


def _update_yearly_overlays(*, yearly_enabled_widget, yearly_window_widget, year_fields, alpha_widget, fit_degree: int,
                            ts_fig, base_series_or_df, kind: str, units_label: str, title_prefix: str,
                            ts_year_sources, ts_year_renderers, ts_year_fit_sources, ts_year_fit_renderers,
                            ts_fig_dir=None, ts_dir_year_sources=None, ts_dir_year_renderers=None,
                            ts_dir_year_fit_sources=None, ts_dir_year_fit_renderers=None) -> None:
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

    set_yearly_overlays(ts_fig, base_series_or_df, kind=kind, years=years_selected,
                        yearly_sources=ts_year_sources, yearly_renderers=ts_year_renderers,
                        yearly_fit_sources=ts_year_fit_sources, yearly_fit_renderers=ts_year_fit_renderers,
                        fit_degree=int(fit_degree), alpha=alpha, units=units_label,
                        title_prefix=title_prefix, dummy_range=dummy_range)

    if ts_fig_dir is not None and ts_dir_year_sources and ts_dir_year_renderers:
        dir_series = compute_wind_direction_deg_from_uv(base_series_or_df) if kind == "uv" else base_series_or_df
        set_yearly_overlays(ts_fig_dir, dir_series, kind="scalar", years=years_selected,
                            yearly_sources=ts_dir_year_sources, yearly_renderers=ts_dir_year_renderers,
                            yearly_fit_sources=ts_dir_year_fit_sources, yearly_fit_renderers=ts_dir_year_fit_renderers,
                            fit_degree=int(fit_degree), alpha=alpha, units="°",
                            title_prefix="Yearly (direction)", dummy_range=dummy_range)

    _lock_x_range_to_dummy_year(ts_fig, ts_fig_dir)


def _plot_scalar_branch(*, ts, units, fit_degree, hours, ts_source, ts_source_fit=None, ts_fig=None,
                        stat_panes, yearly_enabled_widget=None, yearly_window_widget=None, year_fields=None,
                        alpha_widget=None, ts_year_sources=None, ts_year_renderers=None, ts_year_fit_sources=None,
                        ts_year_fit_renderers=None, ts_fig_dir=None, ts_dir_year_sources=None,
                        ts_dir_year_renderers=None, ts_dir_year_fit_sources=None,
                        ts_dir_year_fit_renderers=None) -> Tuple[pd.Series, str]:
    """Render scalar time series with polynomial fit and yearly overlays."""
    ts_filtered = filter_by_hours(ts, hours)
    update_timeseries_source(ts_source, ts_filtered.index, ts_filtered.values)
    _show_stats(ts_filtered.values, units, stat_panes)

    if ts_source_fit is not None:
        _polyfit_and_update_source(ts_filtered.index, ts_filtered.values, fit_degree, ts_source_fit)

    if ts_fig is not None:
        set_figure_axes_labels(ts_fig, y_label=units)
        set_figure_title(ts_fig, f"Timeseries ({units})")

    _update_yearly_overlays(yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
                            year_fields=year_fields, alpha_widget=alpha_widget, fit_degree=fit_degree,
                            ts_fig=ts_fig, base_series_or_df=ts_filtered, kind="scalar", units_label=units,
                            title_prefix="Yearly", ts_year_sources=ts_year_sources, ts_year_renderers=ts_year_renderers,
                            ts_year_fit_sources=ts_year_fit_sources, ts_year_fit_renderers=ts_year_fit_renderers,
                            ts_fig_dir=ts_fig_dir, ts_dir_year_sources=ts_dir_year_sources,
                            ts_dir_year_renderers=ts_dir_year_renderers,
                            ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                            ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)

    return ts_filtered, "scalar"


def _plot_uv_branch(*, ts_uv, units, fit_degree, hours, ts_source, ts_source_fit=None, ts_source_dir=None,
                    ts_source_dir_fit=None, ts_fig=None, ts_fig_dir=None, stat_panes, yearly_enabled_widget=None,
                    yearly_window_widget=None, year_fields=None, alpha_widget=None, ts_year_sources=None,
                    ts_year_renderers=None, ts_year_fit_sources=None, ts_year_fit_renderers=None,
                    ts_dir_year_sources=None, ts_dir_year_renderers=None, ts_dir_year_fit_sources=None,
                    ts_dir_year_fit_renderers=None) -> Tuple[pd.DataFrame, str]:
    """Render UV wind data: speed on main plot, direction on direction plot."""
    df_filtered = filter_by_hours(ts_uv, hours)
    speed = np.hypot(df_filtered["u"].to_numpy(), df_filtered["v"].to_numpy())

    update_timeseries_source(ts_source, df_filtered.index, speed)
    _show_stats(speed, units, stat_panes)

    if ts_source_fit is not None:
        _polyfit_and_update_source(df_filtered.index, speed, fit_degree, ts_source_fit)

    if ts_source_dir is not None:
        dir_series = compute_wind_direction_deg_from_uv(df_filtered)
        update_timeseries_source(ts_source_dir, dir_series.index, dir_series.values)
        if ts_source_dir_fit is not None:
            _polyfit_and_update_source(dir_series.index, dir_series.values, fit_degree, ts_source_dir_fit)

    if ts_fig is not None:
        set_figure_axes_labels(ts_fig, y_label=units)
        set_figure_title(ts_fig, f"Speed ({units})")
    if ts_fig_dir is not None:
        set_figure_axes_labels(ts_fig_dir, y_label="Direction (°)")
        set_figure_title(ts_fig_dir, "Direction (°)")

    _update_yearly_overlays(yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
                            year_fields=year_fields, alpha_widget=alpha_widget, fit_degree=fit_degree,
                            ts_fig=ts_fig, base_series_or_df=df_filtered, kind="uv", units_label=units,
                            title_prefix="Yearly", ts_year_sources=ts_year_sources, ts_year_renderers=ts_year_renderers,
                            ts_year_fit_sources=ts_year_fit_sources, ts_year_fit_renderers=ts_year_fit_renderers,
                            ts_fig_dir=ts_fig_dir, ts_dir_year_sources=ts_dir_year_sources,
                            ts_dir_year_renderers=ts_dir_year_renderers,
                            ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                            ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)

    return df_filtered, "uv"


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

    try:
        ts_grid, meta_grid = _extract_timeseries_grid(_last["files"], _last["ds_key"], _last["time_range"], lon_click,
                                                      lat_click)
    except Exception:
        ts_grid, meta_grid = None, None

    if ts_grid is not None and meta_grid is not None and len(ts_grid) > 0:
        hours = _last.get("hours")
        if w_stat_ndatapoints is not None:
            w_stat_ndatapoints.object = f"**Number of datapoints in the range:** {ts_grid.shape[0]}"

        if meta_grid["kind"] == "scalar":
            ts_used, kind_used = _plot_scalar_branch(
                ts=ts_grid, units=meta_grid.get("units", ""), fit_degree=fit_degree, hours=hours,
                ts_source=ts_source, ts_source_fit=ts_source_fit, ts_fig=ts_fig, stat_panes=stat_panes,
                yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
                year_fields=year_fields, alpha_widget=alpha_widget, ts_year_sources=ts_year_sources,
                ts_year_renderers=ts_year_renderers, ts_year_fit_sources=ts_year_fit_sources,
                ts_year_fit_renderers=ts_year_fit_renderers, ts_fig_dir=ts_fig_dir,
                ts_dir_year_sources=ts_dir_year_sources, ts_dir_year_renderers=ts_dir_year_renderers,
                ts_dir_year_fit_sources=ts_dir_year_fit_sources, ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)
        else:
            ts_used, kind_used = _plot_uv_branch(
                ts_uv=ts_grid, units=meta_grid.get("units", "m/s"), fit_degree=fit_degree, hours=hours,
                ts_source=ts_source, ts_source_fit=ts_source_fit, ts_source_dir=ts_source_dir,
                ts_source_dir_fit=ts_source_dir_fit, ts_fig=ts_fig, ts_fig_dir=ts_fig_dir, stat_panes=stat_panes,
                yearly_enabled_widget=yearly_enabled_widget, yearly_window_widget=yearly_window_widget,
                year_fields=year_fields, alpha_widget=alpha_widget, ts_year_sources=ts_year_sources,
                ts_year_renderers=ts_year_renderers, ts_year_fit_sources=ts_year_fit_sources,
                ts_year_fit_renderers=ts_year_fit_renderers, ts_dir_year_sources=ts_dir_year_sources,
                ts_dir_year_renderers=ts_dir_year_renderers, ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)

        _last["picked_series"] = ts_used
        _last["picked_kind"] = kind_used
        _last["picked_units"] = meta_grid.get("units", "" if kind_used == "scalar" else "m/s")

    w_status.object = f"Selected glacier: **{glacier_name}** ({lon_click:.3f}, {lat_click:.3f})"


def _handle_grid_click(*, lon_click, lat_click, _last, w_status, w_stat_ndatapoints, fit_degree,
                       ts_source, ts_source_fit, stat_panes, ts_source_dir=None, ts_source_dir_fit=None,
                       ts_fig=None, ts_fig_dir=None, yearly_enabled_widget=None, yearly_window_widget=None,
                       year_fields=None, alpha_widget=None, ts_year_sources=None, ts_year_renderers=None,
                       ts_year_fit_sources=None, ts_year_fit_renderers=None, ts_dir_year_sources=None,
                       ts_dir_year_renderers=None, ts_dir_year_fit_sources=None,
                       ts_dir_year_fit_renderers=None) -> None:
    """Handle click on a regular grid point."""
    try:
        ts, meta = _extract_timeseries_grid(_last["files"], _last["ds_key"], _last["time_range"], lon_click, lat_click)
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
            ts_dir_year_fit_sources=ts_dir_year_fit_sources, ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)
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
            ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)

    _last["picked_series"] = ts_used
    _last["picked_kind"] = kind_used
    _last["picked_units"] = meta.get("units", "" if kind_used == "scalar" else "m/s")

    w_status.object = f"Selected grid point ({lon_click:.3f}, {lat_click:.3f})"
    set_timeseries(ts_source, ts_used)


def _on_tap(evt, w_status, _last, ts_source, ts_fig, _stat_panes, ts_source_fit=None, fit_degree=3,
            ts_source_dir=None, ts_source_dir_fit=None, ts_fig_dir=None, yearly_enabled_widget=None,
            year_fields=None, alpha_widget=None, yearly_window_widget=None, ts_year_sources=None,
            ts_year_renderers=None, ts_dir_year_sources=None, ts_dir_year_renderers=None, w_stat_ndatapoints=None,
            ts_year_fit_sources=None, ts_year_fit_renderers=None, ts_dir_year_fit_sources=None,
            ts_dir_year_fit_renderers=None, ts_glacier_source=None, w_glacier_multiplier=None, w_glacier_offset=None):
    """Handle map tap events. Hook: bkplot.on_event(Tap, lambda evt: _on_tap(...))"""
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
                ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)
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
                ts_dir_year_fit_renderers=ts_dir_year_fit_renderers)
    except Exception as e:
        w_status.object = f"Tap handler error: {e}"