"""
Yearly overlay visualization for time series data.
Refactored to use common utilities for better maintainability.
"""
import pandas as pd
import numpy as np
from bokeh.palettes import Category10
from bokeh.models import Legend, LegendItem

from src.visualization.helpers.normalize_to_dummy_year import _normalize_to_dummy_year
from src.visualization.helpers.slice_series_by_year import slice_series_by_year
from src.visualization.utilities.trend_methods import TrendConfig, compute_trend_line

# Import new utilities
from src.visualization.utilities.bokeh_utils import (
    clear_and_hide_renderer_pair,
    update_timeseries_source,
    show_renderer,
    set_figure_axes_labels,
    set_figure_title,
)
from src.visualization.utilities.datetime_utils import (
    ensure_naive_timestamp,
    ensure_naive_index,
    datetime_to_python_list,
)
from src.visualization.utilities.validation_utils import (
    parse_year_value,
    filter_valid_years,
    validate_series_for_fit,
)

PALETTE10 = list(Category10[10])


def _update_yearly_legend(fig, items):
    """
    Update the SINGLE existing legend on the figure.
    We must NOT add new Legend() models, otherwise Bokeh will refuse legend_label=...
    """
    # Preferred: the stored reference from make_lineplot.py
    lg = getattr(fig, "_main_legend", None)

    # Fallback: use first existing legend if present
    if lg is None:
        try:
            if fig.legend and len(fig.legend) > 0:
                lg = fig.legend[0]
        except Exception:
            lg = None

    if lg is None:
        # If there is truly no legend yet, we cannot safely create a new one here
        # because the rest of the codebase uses legend_label=... convenience args.
        return

    lg.items = items
    lg.click_policy = "hide"
    lg.location = "top_left"



def _convert_to_speed_series(base_series_or_df, kind: str) -> pd.Series:
    """
    Convert input data to a speed series based on kind.

    Parameters
    ----------
    base_series_or_df : pd.Series or pd.DataFrame
        Input data
    kind : str
        'scalar' or 'uv'

    Returns
    -------
    pd.Series
        Sorted time series
    """
    if kind == "scalar":
        return pd.Series(base_series_or_df).sort_index()
    elif kind == "uv":
        df = pd.DataFrame(base_series_or_df).sort_index()
        return pd.Series(
            np.hypot(df["u"].to_numpy(), df["v"].to_numpy()),
            index=df.index,
            name="speed",
        )
    else:
        raise ValueError(f"Unsupported kind={kind!r} for yearly overlays.")


def _apply_dummy_year_window(series: pd.Series, year: int, dummy_range: tuple) -> pd.Series:
    """
    Apply a dummy-year time window to a real-year series.

    Parameters
    ----------
    series : pd.Series
        Time series for a specific year
    year : int
        The real year
    dummy_range : tuple
        (start, end) datetimes in dummy year 2000

    Returns
    -------
    pd.Series
        Windowed series, or original if window fails
    """
    if dummy_range is None:
        return series

    try:
        dr_start, dr_end = dummy_range

        # Ensure tz-naive timestamps
        dr_start = ensure_naive_timestamp(pd.to_datetime(dr_start))
        dr_end = ensure_naive_timestamp(pd.to_datetime(dr_end))

        # Map dummy window to real year
        real_start = dr_start.replace(year=year)
        real_end = dr_end.replace(year=year)

        # Filter series
        idx_real = ensure_naive_index(pd.to_datetime(series.index))
        mask = (idx_real >= real_start) & (idx_real <= real_end)

        return series.loc[mask]

    except Exception:
        # On failure, return original series
        return series


def _clear_year_slot(src, rnd, fit_src, fit_rnd):
    """Clear and hide a year slot (both raw and fit renderers)."""
    clear_and_hide_renderer_pair(src, rnd)
    clear_and_hide_renderer_pair(fit_src, fit_rnd)


def _setup_raw_overlay(
        src,
        rnd,
        series: pd.Series,
        color: str,
        label: str,
        alpha: float
):
    """
    Setup a raw yearly overlay (normalized to dummy year).

    Parameters
    ----------
    src : ColumnDataSource
        Data source to update
    rnd : Renderer
        Renderer to configure
    series : pd.Series
        Windowed series for this year
    color : str
        Line color
    label : str
        Legend label (kept for API compatibility; legend is built explicitly elsewhere)
    alpha : float
        Line alpha
    """
    # Normalize to dummy year
    t_norm = _normalize_to_dummy_year(series.index)
    t_norm = ensure_naive_index(pd.to_datetime(t_norm))

    # Update source
    update_timeseries_source(
        src,
        t_norm,
        series.values,
        convert_datetime=True
    )

    # Configure renderer
    rnd.glyph.line_color = color

    show_renderer(rnd, alpha=max(alpha, 0.75))


def _setup_fit_overlay(
        fit_src,
        fit_rnd,
        series: pd.Series,
        fit_degree: int,
        color: str,
        label: str,
        *,
        trend_method: str,
        trend_param: int,
        pre_smooth_enabled: bool,
        pre_smooth_window_days: int = 30,
):
    """
    Setup a trend overlay for a yearly series (method selectable).

    trend_param is interpreted depending on trend_method:
      - polyfit: degree
      - rolling_mean: window_days
      - ewma: span_days
      - annual_linear: min_years
    """
    # For polyfit we still use the series-length check based on degree;
    # for other methods we just require >= 2 points.
    if str(trend_method).lower() == "polyfit":
        if not validate_series_for_fit(series, int(trend_param) + 1):
            clear_and_hide_renderer_pair(fit_src, fit_rnd)
            return
    else:
        if series is None or len(series) < 2:
            clear_and_hide_renderer_pair(fit_src, fit_rnd)
            return

    cfg = TrendConfig(
        method=str(trend_method),
        pre_smooth_enabled=bool(pre_smooth_enabled),
        pre_smooth_window_days=int(pre_smooth_window_days),
        poly_degree=int(trend_param),
        rolling_window_days=int(trend_param),
        ewma_span_days=int(trend_param),
        annual_min_years=int(trend_param),
    )

    # Compute trend line on the real timestamps
    t_fit, y_fit = compute_trend_line(series.index, series.values, cfg)

    # Normalize to dummy year for overlay comparison
    t_fit_norm = _normalize_to_dummy_year(t_fit)
    t_fit_norm = ensure_naive_index(pd.to_datetime(t_fit_norm))

    update_timeseries_source(
        fit_src,
        t_fit_norm,
        y_fit,
        convert_datetime=True
    )

    fit_rnd.glyph.line_color = color
    fit_rnd.glyph.line_dash = "solid"
    fit_rnd.glyph.line_alpha = 1.0
    fit_rnd.glyph.line_width = 3

    show_renderer(fit_rnd, alpha=1.0)


def set_yearly_overlays(
        fig,
        base_series_or_df,
        *,
        kind: str,
        years: list,
        yearly_sources: list,
        yearly_renderers: list,
        yearly_fit_sources: list,
        yearly_fit_renderers: list,
        fit_degree: int,
        alpha: float = 0.35,
        units: str = "",
        title_prefix: str = "Yearly",
        dummy_range=None,
        # NEW (backward compatible defaults):
        trend_method: str = "polyfit",
        trend_param: int = 3,
        pre_smooth_enabled: bool = False,
        pre_smooth_window_days: int = 30,
):
    """
    Draw yearly overlays (and per-year trend fits) of a scalar or UV time series.

    Each selected year is:
    1. Extracted from the full series
    2. Optionally windowed to a specific time range (dummy_range)
    3. Normalized to dummy year 2000 for overlay comparison
    4. Trend is computed and overlaid

    Parameters
    ----------
    fig : bokeh Figure
        Figure to update
    base_series_or_df : pd.Series or pd.DataFrame
        Base time series data
    kind : str
        'scalar' for single-variable or 'uv' for wind vectors
    years : list
        List of years to overlay (up to 10)
    yearly_sources : list
        List of 10 ColumnDataSource objects for raw data
    yearly_renderers : list
        List of 10 renderers for raw lines
    yearly_fit_sources : list
        List of 10 ColumnDataSource objects for fit data
    yearly_fit_renderers : list
        List of 10 renderers for fit lines
    fit_degree : int
        Polynomial degree for fitting (legacy; still used by callers, not necessarily used by trend_method)
    alpha : float
        Line alpha for raw overlays (default: 0.35)
    units : str
        Y-axis units label
    title_prefix : str
        Prefix for figure title
    dummy_range : tuple, optional
        (start, end) datetimes in dummy year 2000.
        The same month/day window is applied to every selected year.
    """
    # Validation
    if (
            yearly_sources is None
            or yearly_renderers is None
            or yearly_fit_sources is None
            or yearly_fit_renderers is None
    ):
        return

    # Convert to speed series
    s_all = _convert_to_speed_series(base_series_or_df, kind)

    # Filter to valid years and create color map
    valid_years = filter_valid_years(years)
    color_map = {yr: PALETTE10[i % len(PALETTE10)] for i, yr in enumerate(valid_years)}

    # Process each of the 10 overlay slots
    for i in range(10):
        src = yearly_sources[i]
        rnd = yearly_renderers[i]
        fit_src = yearly_fit_sources[i]
        fit_rnd = yearly_fit_renderers[i]

        # Get year for this slot
        ysel = years[i] if i < len(years) else None
        year_int = parse_year_value(ysel)

        # If no valid year, clear this slot
        if year_int is None:
            _clear_year_slot(src, rnd, fit_src, fit_rnd)
            continue

        # Extract this year's data
        s_one = slice_series_by_year(s_all, year_int)
        if len(s_one) == 0:
            _clear_year_slot(src, rnd, fit_src, fit_rnd)
            continue

        # Apply dummy-year window
        s_window = _apply_dummy_year_window(s_one, year_int, dummy_range)
        if len(s_window) == 0:
            _clear_year_slot(src, rnd, fit_src, fit_rnd)
            continue

        # Setup raw overlay
        color = color_map.get(year_int, PALETTE10[i % len(PALETTE10)])
        label = str(year_int)

        _setup_raw_overlay(src, rnd, s_window, color, label, alpha)

        # Setup fit overlay
        _setup_fit_overlay(
            fit_src, fit_rnd, s_window, fit_degree, color, label,
            trend_method=trend_method,
            trend_param=trend_param,
            pre_smooth_enabled=pre_smooth_enabled,
            pre_smooth_window_days=pre_smooth_window_days,
        )

    # Update figure cosmetics
    xlab = "Day of year (dummy year)"
    ylab = units
    if ylab:
        set_figure_axes_labels(fig, x_label=xlab, y_label=ylab)
    else:
        set_figure_axes_labels(fig, x_label=xlab)

    if title_prefix:
        selected_count = len(valid_years)
        set_figure_title(fig, f"{title_prefix} overlays ({selected_count} selected)")

    # Configure base legend(s) if they exist
    try:
        if fig.legend:
            for lg in fig.legend:
                lg.click_policy = "hide"
                lg.location = "top_left"
    except Exception:
        pass

    # ------------------------------------------------------------
    # Build / refresh the dedicated yearly legend items (explicit)
    # ------------------------------------------------------------
    yearly_items = []
    for i in range(10):
        if i >= len(valid_years):
            continue

        yr = valid_years[i]
        raw_r = yearly_renderers[i]
        fit_r = yearly_fit_renderers[i]

        # Skip slots that ended up hidden/empty
        raw_vis = (raw_r is not None and getattr(raw_r, "visible", False))
        fit_vis = (fit_r is not None and getattr(fit_r, "visible", False))
        if not raw_vis and not fit_vis:
            continue

        renderers = []
        if raw_r is not None:
            renderers.append(raw_r)
        if fit_r is not None:
            renderers.append(fit_r)

        yearly_items.append(LegendItem(label=str(yr), renderers=renderers))

    _update_yearly_legend(fig, yearly_items)
