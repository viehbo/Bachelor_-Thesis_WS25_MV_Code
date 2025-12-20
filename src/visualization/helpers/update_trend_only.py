from __future__ import annotations

from typing import Optional

from src.visualization.utilities.trend_methods import TrendConfig, compute_trend_line


def update_trend_only(
    *,
    # climate (wind/temp) widgets
    w_trend_method_climate,
    w_trend_param_climate,
    w_pre_smooth_enabled_climate,
    w_pre_smooth_window_days_climate,
    # glacier widgets
    w_trend_method_glacier,
    w_trend_param_glacier,
    w_pre_smooth_enabled_glacier,
    w_pre_smooth_window_days_glacier,
    # bokeh sources
    ts_source,
    ts_source_fit,
    ts_source_dir=None,
    ts_source_dir_fit=None,
    ts_glacier_source=None,
    ts_glacier_source_fit=None,
):
    """
    Update ONLY the trend/fit sources for:
      - climate main series (ts_source -> ts_source_fit)
      - optional climate direction series (ts_source_dir -> ts_source_dir_fit)
      - optional glacier series overlay (ts_glacier_source -> ts_glacier_source_fit)

    This is called e.g. from a "Set trend" button, similar to the old polyfit-only behavior.
    """

    def cfg_from_widgets(method_w, param_w, pre_on_w, pre_win_w) -> TrendConfig:
        m = str(method_w.value)
        p = int(param_w.value)
        pre_on = bool(pre_on_w.value)
        pre_win = int(pre_win_w.value)

        # Map the single "param" to the right field depending on method
        return TrendConfig(
            method=m,
            pre_smooth_enabled=pre_on,
            pre_smooth_window_days=pre_win,
            poly_degree=p,
            rolling_window_days=p,
            ewma_span_days=p,
            annual_min_years=p,
        )

    cfg_climate = cfg_from_widgets(
        w_trend_method_climate,
        w_trend_param_climate,
        w_pre_smooth_enabled_climate,
        w_pre_smooth_window_days_climate,
    )
    cfg_glacier = cfg_from_widgets(
        w_trend_method_glacier,
        w_trend_param_glacier,
        w_pre_smooth_enabled_glacier,
        w_pre_smooth_window_days_glacier,
    )

    def update_one(src_raw, src_fit, cfg: TrendConfig):
        if src_raw is None or src_fit is None:
            return
        t = src_raw.data.get("t", []) or []
        y = src_raw.data.get("y", []) or []
        if len(t) < 2:
            src_fit.data = dict(t=[], y=[])
            return

        t_fit, y_fit = compute_trend_line(t, y, cfg)
        src_fit.data = dict(t=t_fit, y=y_fit)

    # climate main trend
    update_one(ts_source, ts_source_fit, cfg_climate)

    # optional: direction trend (still climate config)
    update_one(ts_source_dir, ts_source_dir_fit, cfg_climate)

    # optional: glacier trend (glacier config)
    update_one(ts_glacier_source, ts_glacier_source_fit, cfg_glacier)


# Backwards compatibility: keep the old name so existing imports won't break.
# If you already imported update_poly_fit_only elsewhere, it will still work.
def update_poly_fit_only(*args, **kwargs):
    raise RuntimeError(
        "update_poly_fit_only() was replaced by update_trend_only(). "
        "Update your imports and calls accordingly."
    )
