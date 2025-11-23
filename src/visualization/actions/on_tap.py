# src/visualization/actions/on_tap.py
from math import pi
import numpy as np
import pandas as pd

from src.visualization.helpers.mercator_transformer import y_to_lat, x_to_lon
from src.visualization.helpers.set_timeseries import set_timeseries

from src.visualization.plots.yearly_overlays import set_yearly_overlays


from src.visualization.core.dataset import GLACIERS

from src.visualization.helpers.extract_timeseries_grid import _extract_timeseries_grid
# poly fit helper
from src.visualization.core.poly_fit import poly_fit_datetime



R = 6378137.0


def _show_stats(values, unit, panes):
    """Update the three stats panes passed in from the app."""
    mean_p, max_p, min_p = panes["mean"], panes["max"], panes["min"]
    if len(values) == 0 or not np.isfinite(values).any():
        mean_p.object = "**Mean:** –"
        max_p.object = "**Max:** –"
        min_p.object = "**Min:** –"
        return
    v = np.asarray(values, dtype=float)
    mean_p.object = f"**Mean:** {np.nanmean(v):.3f} {unit}"
    max_p.object = f"**Max:** {np.nanmax(v):.3f} {unit}"
    min_p.object = f"**Min:** {np.nanmin(v):.3f} {unit}"


def _on_tap(
    evt,
    w_status,
    _last,
    ts_source,
    ts_fig,
    _stat_panes,
    ts_source_fit=None,
    fit_degree=3,
    ts_source_dir=None,
    ts_source_dir_fit=None,
    ts_fig_dir=None,
    # Yearly UI/state
    yearly_enabled_widget=None,
    year_fields=None,
    alpha_widget=None,
    yearly_window_widget=None,
    ts_year_sources=None,
    ts_year_renderers=None,
    ts_dir_year_sources=None,
    ts_dir_year_renderers=None,
    # Yearly FIT overlays (main & direction)
    ts_year_fit_sources=None,
    ts_year_fit_renderers=None,
    ts_dir_year_fit_sources=None,
    ts_dir_year_fit_renderers=None,
    # Glacier overlay on secondary y-axis
    ts_glacier_source=None,
):
    """
    External tap handler. Hook from app like:
    bkplot.on_event(Tap, lambda evt: _on_tap(...))
    """
    try:
        # --- helper to maintain yearly overlays ---
        def _maybe_update_yearly_overlays(kind, base_series_or_df, units_label, title_prefix):
            """
            Draw/clear yearly overlays based on the Yearly mode checkbox.
            Also snaps x-range to a single dummy-year window when Yearly is ON.
            """
            YEARLY_ON = bool(getattr(yearly_enabled_widget, "value", False))

            # Current dummy-year sub-window (if any)
            yearly_window = None
            if yearly_window_widget is not None:
                try:
                    v = yearly_window_widget.value
                    if v and all(v):
                        yearly_window = v  # (start, end) in dummy year 2000
                except Exception:
                    yearly_window = None


            # Clear function for any pair of lists
            def _clear_pair(src_list, rnd_list):
                if src_list and rnd_list:
                    for s, r in zip(src_list, rnd_list):
                        s.data = dict(t=[], y=[])
                        r.visible = False

            if not YEARLY_ON:
                # Clear overlays; DO NOT return from the whole tap handler
                _clear_pair(ts_year_sources, ts_year_renderers)
                _clear_pair(ts_year_fit_sources, ts_year_fit_renderers)
                _clear_pair(ts_dir_year_sources, ts_dir_year_renderers)
                _clear_pair(ts_dir_year_fit_sources, ts_dir_year_fit_renderers)
                return

            # Build selected years
            years_selected = [w.value for w in (year_fields or [])]

            # Main plot overlays
            set_yearly_overlays(
                ts_fig,
                base_series_or_df,
                kind=kind,  # "scalar" or "uv"
                years=years_selected,
                yearly_sources=ts_year_sources,
                yearly_renderers=ts_year_renderers,
                yearly_fit_sources=ts_year_fit_sources,
                yearly_fit_renderers=ts_year_fit_renderers,
                fit_degree=fit_degree,
                alpha=float(getattr(alpha_widget, "value", 0.35) or 0.35),
                units=units_label,
                title_prefix=title_prefix,
                dummy_range=yearly_window,
            )

            # Direction overlays (if we have a direction figure)
            if ts_fig_dir is not None and ts_dir_year_sources and ts_dir_year_renderers:
                if kind == "uv":
                    # Direction (0..360) from u,v
                    dir_deg = (
                        270.0 - np.degrees(np.arctan2(base_series_or_df["v"].to_numpy(),
                                                      base_series_or_df["u"].to_numpy()))
                    ) % 360.0
                    dir_series = pd.Series(dir_deg, index=base_series_or_df.index, name="dir")
                    set_yearly_overlays(
                        ts_fig_dir,
                        dir_series,
                        kind="scalar",
                        years=years_selected,
                        yearly_sources=ts_dir_year_sources,
                        yearly_renderers=ts_dir_year_renderers,
                        yearly_fit_sources=ts_dir_year_fit_sources,
                        yearly_fit_renderers=ts_dir_year_fit_renderers,
                        fit_degree=fit_degree,
                        alpha=float(getattr(alpha_widget, "value", 0.35) or 0.35),
                        units="°",
                        title_prefix="Yearly (direction)",
                        dummy_range=yearly_window,
                    )
                else:
                    # If you also want direction overlays for scalar, reuse scalar series
                    set_yearly_overlays(
                        ts_fig_dir,
                        base_series_or_df,
                        kind="scalar",
                        years=years_selected,
                        yearly_sources=ts_dir_year_sources,
                        yearly_renderers=ts_dir_year_renderers,
                        yearly_fit_sources=ts_dir_year_fit_sources,
                        yearly_fit_renderers=ts_dir_year_fit_renderers,
                        fit_degree=fit_degree,
                        alpha=float(getattr(alpha_widget, "value", 0.35) or 0.35),
                        units="°",
                        title_prefix="Yearly (direction)",
                        dummy_range=yearly_window,
                    )


            # Snap both x-ranges to 1-year dummy window
            from datetime import datetime

            DUMMY_START = datetime(2000, 1, 1)
            DUMMY_END = datetime(2000, 12, 31, 23, 59, 59)
            try:
                ts_fig.x_range.start = DUMMY_START
                ts_fig.x_range.end = DUMMY_END
                if ts_fig_dir is not None:
                    ts_fig_dir.x_range.start = DUMMY_START
                    ts_fig_dir.x_range.end = DUMMY_END
                    ts_fig_dir.y_range.start = 0
                    ts_fig_dir.y_range.end = 360
            except Exception:
                pass

        # ------------------ click prelude ------------------
        w_status.object = f"Tap at x={evt.x:.1f}, y={evt.y:.1f}"
        # WebMercator -> lon/lat
        lon_click = float(x_to_lon(evt.x))
        lat_click = float(y_to_lat(evt.y))

        # Near-glacier detection (optional)
        name_used = None
        gl = _last.get("glaciers")
        if gl and gl.get("lon") is not None and getattr(gl["lon"], "size", 0):
            Gxm = (gl["lon"] * (pi / 180.0) * R)
            Gym = R * np.log(np.tan((pi / 4.0) + (gl["lat"] * (pi / 180.0) / 2.0)))
            di = np.argmin((Gxm - evt.x) ** 2 + (Gym - evt.y) ** 2)
            d_m = np.sqrt((Gxm[di] - evt.x) ** 2 + (Gym[di] - evt.y) ** 2)
            if d_m < 10_000:
                name = gl["name"][di]
                name_used = name.item() if hasattr(name, "item") else name

        # ------------------ glacier branch ------------------
        if name_used:
            from src.visualization.plots.glacier_overlay import glacier_series_by_name

            # 1) Glacier time series (always kept on secondary axis)
            s = glacier_series_by_name(GLACIERS["dir"], str(name_used))

            # Show glacier stats in the side panes
            _show_stats(s.values, "mm w.e.", _stat_panes)

            # Update / draw glacier line on secondary y-axis if wired
            if ts_glacier_source is not None and hasattr(ts_fig, "extra_y_ranges") and "glacier" in ts_fig.extra_y_ranges:
                t_vals = s.index.to_pydatetime().tolist()
                y_vals = s.values.astype(float).tolist()
                ts_glacier_source.data = dict(t=t_vals, y=y_vals)

                # Rescale the secondary y-axis to glacier data
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
            else:
                # Fallback: if secondary axis is not set up, behave as before:
                set_timeseries(
                    ts_source,
                    s,
                    kind="scalar",
                    units="mm w.e.",
                    title=f"Glacier {name_used} — time series",
                    fig=ts_fig,
                )
                _last["picked_series"] = s
                _last["picked_kind"] = "scalar"
                _last["picked_units"] = "mm w.e."

                # single poly-fit (normal mode)
                if ts_source_fit is not None:
                    t_vals = s.index.to_pydatetime().tolist()
                    y_vals = s.values.astype(float).tolist()
                    if len(t_vals) >= fit_degree + 1:
                        t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                        ts_source_fit.data = dict(t=t_fit, y=y_fit)
                    else:
                        ts_source_fit.data = dict(t=[], y=[])

                # yearly overlays / fits (if Yearly ON)
                _maybe_update_yearly_overlays(
                    kind="scalar",
                    base_series_or_df=s,
                    units_label="mm w.e.",
                    title_prefix="Yearly",
                )

                w_status.object = f"Selected glacier: **{name_used}** ({lon_click:.3f}, {lat_click:.3f})"
                return

            # 2) In the "normal" case (secondary axis available):
            #    ALSO plot the underlying grid/dataset time series on the primary y-axis,
            #    so both series are visible together.

            try:
                ts, meta = _extract_timeseries_grid(
                    _last["files"], _last["ds_key"], _last["time_range"], lon_click, lat_click
                )
            except Exception:
                ts, meta = None, None

            if ts is not None and len(ts) > 0 and meta is not None:
                # --- apply daily hour filter from _last, if present (same logic as grid branch) ---
                hs = _last.get("hour_start")
                he = _last.get("hour_end")
                if hs is not None and he is not None:
                    if hasattr(hs, "time"):
                        hs = hs.time()
                    if hasattr(he, "time"):
                        he = he.time()
                    if hs > he:
                        hs, he = he, hs

                    idx = pd.to_datetime(ts.index)
                    mask = (idx.time >= hs) & (idx.time <= he)
                    ts = ts.loc[mask]

                # Now behave like the normal grid branch for the primary axis,
                # but keep glacier stats on the side.
                if meta["kind"] == "scalar":
                    set_timeseries(
                        ts_source,
                        ts,
                        kind="scalar",
                        units=meta.get("units", ""),
                        title="Point time series",
                        fig=ts_fig,
                    )

                    _last["picked_series"] = ts
                    _last["picked_kind"] = "scalar"
                    _last["picked_units"] = meta.get("units", "")

                    # Poly-fit of the main (grid) series
                    if ts_source_fit is not None:
                        t_vals = ts.index.to_pydatetime().tolist()
                        y_vals = ts.values.astype(float).tolist()
                        if len(t_vals) >= fit_degree + 1:
                            t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                            ts_source_fit.data = dict(t=t_fit, y=y_fit)
                        else:
                            ts_source_fit.data = dict(t=[], y=[])

                    # Yearly overlays use the grid series
                    _maybe_update_yearly_overlays(
                        kind="scalar",
                        base_series_or_df=ts,
                        units_label=meta.get("units", ""),
                        title_prefix="Yearly",
                    )

                else:
                    # uv / wind case
                    spd = np.hypot(ts["u"].to_numpy(), ts["v"].to_numpy())
                    # Keep glacier stats in the side panes, so do NOT call _show_stats(spd, ...)

                    set_timeseries(
                        ts_source,
                        ts,
                        kind="uv",
                        units=meta.get("units", "m/s"),
                        title="Wind speed (|u,v|)",
                        fig=ts_fig,
                    )

                    _last["picked_series"] = ts
                    _last["picked_kind"] = "uv"
                    _last["picked_units"] = meta.get("units", "m/s")

                    # Direction series and direction plot
                    dir_deg = (270.0 - np.degrees(np.arctan2(ts["v"].to_numpy(), ts["u"].to_numpy()))) % 360.0
                    dir_series = pd.Series(dir_deg, index=ts.index, name="dir")
                    if ts_source_dir is not None:
                        set_timeseries(
                            ts_source_dir,
                            dir_series,
                            kind="scalar",
                            units="°",
                            title="Wind direction (from)",
                            fig=ts_fig_dir,
                        )

                    # Poly-fit for direction
                    if ts_source_dir_fit is not None:
                        t_vals = dir_series.index.to_pydatetime().tolist()
                        y_vals = dir_series.values.astype(float).tolist()
                        if len(t_vals) >= fit_degree + 1:
                            t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                            ts_source_dir_fit.data = dict(t=t_fit, y=y_fit)
                        else:
                            ts_source_dir_fit.data = dict(t=[], y=[])

                    # Poly-fit for speed magnitude
                    if ts_source_fit is not None:
                        t_vals = ts.index.to_pydatetime().tolist()
                        y_vals = spd.astype(float).tolist()
                        if len(t_vals) >= fit_degree + 1:
                            t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                            ts_source_fit.data = dict(t=t_fit, y=y_fit)
                        else:
                            ts_source_fit.data = dict(t=[], y=[])

                    # Yearly overlays based on the grid series (uv)
                    _maybe_update_yearly_overlays(
                        kind="uv",
                        base_series_or_df=ts,
                        units_label=meta.get("units", "m/s"),
                        title_prefix="Yearly",
                    )

            else:
                # If grid data can't be extracted, fall back to using the glacier series
                # on the primary axis as before.
                set_timeseries(
                    ts_source,
                    s,
                    kind="scalar",
                    units="mm w.e.",
                    title=f"Glacier {name_used} — time series",
                    fig=ts_fig,
                )
                _last["picked_series"] = s
                _last["picked_kind"] = "scalar"
                _last["picked_units"] = "mm w.e."

                if ts_source_fit is not None:
                    t_vals = s.index.to_pydatetime().tolist()
                    y_vals = s.values.astype(float).tolist()
                    if len(t_vals) >= fit_degree + 1:
                        t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                        ts_source_fit.data = dict(t=t_fit, y=y_fit)
                    else:
                        ts_source_fit.data = dict(t=[], y=[])

                _maybe_update_yearly_overlays(
                    kind="scalar",
                    base_series_or_df=s,
                    units_label="mm w.e.",
                    title_prefix="Yearly",
                )

            w_status.object = f"Selected glacier: **{name_used}** ({lon_click:.3f}, {lat_click:.3f})"
            return


        # ------------------ grid branch (dataset) ------------------
        ts, meta = _extract_timeseries_grid(
            _last["files"], _last["ds_key"], _last["time_range"], lon_click, lat_click
        )

        # --- apply daily hour filter from _last, if present ---
        hs = _last.get("hour_start")
        he = _last.get("hour_end")

        if hs is not None and he is not None and len(ts) > 0:
            # normalize to plain datetime.time
            if hasattr(hs, "time"):
                hs = hs.time()
            if hasattr(he, "time"):
                he = he.time()
            if hs > he:
                hs, he = he, hs

            idx = pd.to_datetime(ts.index)
            mask = (idx.time >= hs) & (idx.time <= he)

            # ts is a Series for scalar, DataFrame for uv; boolean mask works for both
            ts = ts.loc[mask]


        if meta["kind"] == "scalar":
            _show_stats(ts.values, meta.get("units", ""), _stat_panes)
            set_timeseries(
                ts_source,
                ts,
                kind="scalar",
                units=meta.get("units", ""),
                title="Point time series",
                fig=ts_fig,
            )

            # single poly-fit (normal mode)
            if ts_source_fit is not None:
                t_vals = ts.index.to_pydatetime().tolist()
                y_vals = ts.values.astype(float).tolist()
                if len(t_vals) >= fit_degree + 1:
                    t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                    ts_source_fit.data = dict(t=t_fit, y=y_fit)
                else:
                    ts_source_fit.data = dict(t=[], y=[])

            # yearly overlays / fits (if Yearly ON)
            _maybe_update_yearly_overlays(
                kind="scalar",
                base_series_or_df=ts,
                units_label=meta.get("units", ""),
                title_prefix="Yearly",
            )

            _last["picked_series"] = ts
            _last["picked_kind"] = "scalar"
            _last["picked_units"] = meta.get("units", "")


        else:
            # uv/wind
            spd = np.hypot(ts["u"].to_numpy(), ts["v"].to_numpy())
            _show_stats(spd, meta.get("units", "m/s"), _stat_panes)
            set_timeseries(
                ts_source,
                ts,
                kind="uv",
                units=meta.get("units", "m/s"),
                title="Wind speed (|u,v|)",
                fig=ts_fig,
            )

            _last["picked_series"] = ts
            _last["picked_kind"] = "uv"
            _last["picked_units"] = meta.get("units", "m/s")


            # Direction series (0..360) and raw direction plot
            dir_deg = (270.0 - np.degrees(np.arctan2(ts["v"].to_numpy(), ts["u"].to_numpy()))) % 360.0
            dir_series = pd.Series(dir_deg, index=ts.index, name="dir")
            if ts_source_dir is not None:
                set_timeseries(
                    ts_source_dir,
                    dir_series,
                    kind="scalar",
                    units="°",
                    title="Wind direction (from)",
                    fig=ts_fig_dir,
                )

            # poly-fit for direction
            if ts_source_dir_fit is not None:
                t_vals = dir_series.index.to_pydatetime().tolist()
                y_vals = dir_series.values.astype(float).tolist()
                if len(t_vals) >= fit_degree + 1:
                    t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                    ts_source_dir_fit.data = dict(t=t_fit, y=y_fit)
                else:
                    ts_source_dir_fit.data = dict(t=[], y=[])

            # poly-fit for speed magnitude (Observed fit in normal mode)
            if ts_source_fit is not None:
                t_vals = ts.index.to_pydatetime().tolist()
                y_vals = spd.astype(float).tolist()
                if len(t_vals) >= fit_degree + 1:
                    t_fit, y_fit = poly_fit_datetime(t_vals, y_vals, degree=fit_degree, points=400)
                    ts_source_fit.data = dict(t=t_fit, y=y_fit)
                else:
                    ts_source_fit.data = dict(t=[], y=[])

            # yearly overlays / fits (if Yearly ON)
            _maybe_update_yearly_overlays(
                kind="uv",
                base_series_or_df=ts,
                units_label=meta.get("units", "m/s"),
                title_prefix="Yearly",
            )

        w_status.object = f"Picked grid point at ({lon_click:.3f}, {lat_click:.3f})."

    except Exception as ex:
        w_status.object = f"**Click error:** {ex}"
