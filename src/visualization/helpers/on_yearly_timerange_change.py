from src.visualization.helpers.yearly_mode import (
    DUMMY_START,
    DUMMY_END,
    get_selected_years,
    get_dummy_year_window,
)
from src.visualization.helpers.wind_direction import compute_wind_direction_deg_from_uv


def _on_yearly_timerange_change(
    w_yearly_mode,
    _last,
    w_years,
    w_yearly_timerange,
    w_alpha_value,
    set_yearly_overlays,
    ts_fig,
    ts_fig_dir,
    ts_year_sources,
    ts_year_renderers,
    ts_year_fit_sources,
    ts_year_fit_renderers,
    ts_dir_year_sources,
    ts_dir_year_renderers,
    ts_dir_year_fit_sources,
    ts_dir_year_fit_renderers,
    w_fit_degree,
    # NEW:
    w_trend_method_climate,
    w_trend_param_climate,
    w_pre_smooth_enabled_climate,
    w_pre_smooth_window_days_climate,
):

    """
    Recompute yearly overlays when the yearly timeframe slider changes,
    using the last picked series cached in `_last`.
    """
    if not w_yearly_mode.value:
        return

    s = _last.get("picked_series")
    kind = _last.get("picked_kind")
    units = _last.get("picked_units", "")
    if s is None or kind not in ("scalar", "uv"):
        return

    years_selected = get_selected_years(w_years)

    yearly_window = get_dummy_year_window(w_yearly_timerange)
    if yearly_window is None:
        return

    alpha = float(w_alpha_value.value or 0.35)
    fit_degree = int(w_fit_degree.value)

    trend_method = str(w_trend_method_climate.value)
    trend_param = int(w_trend_param_climate.value)
    pre_smooth_enabled = bool(w_pre_smooth_enabled_climate.value)
    pre_smooth_window_days = int(w_pre_smooth_window_days_climate.value)


    # --- Main overlays ---
    set_yearly_overlays(
        ts_fig,
        s,
        kind=kind,
        years=years_selected,
        yearly_sources=ts_year_sources,
        yearly_renderers=ts_year_renderers,
        yearly_fit_sources=ts_year_fit_sources,
        yearly_fit_renderers=ts_year_fit_renderers,
        fit_degree=fit_degree,
        alpha=alpha,
        units=units,
        title_prefix="Yearly overlays",
        dummy_range=yearly_window,
        trend_method=trend_method,
        trend_param=trend_param,
        pre_smooth_enabled=pre_smooth_enabled,
        pre_smooth_window_days=pre_smooth_window_days,

    )

    # --- Direction overlays for UV ---
    if kind == "uv" and ts_fig_dir is not None:
        dir_series = compute_wind_direction_deg_from_uv(s)

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
            alpha=alpha,
            units="°",
            title_prefix="Yearly (direction)",
            dummy_range=yearly_window,
            trend_method=trend_method,
            trend_param=trend_param,
            pre_smooth_enabled=pre_smooth_enabled,
            pre_smooth_window_days=pre_smooth_window_days,

        )

    # Keep x-ranges locked to dummy year
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
