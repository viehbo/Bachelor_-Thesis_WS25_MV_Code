import pandas as pd
from src.visualization.helpers.populate_year_options_from_timerange import populate_year_options_from_timerange

from src.visualization.helpers.yearly_mode import (
    set_yearly_widgets_visibility,
    ensure_dummy_year_domain,
    sync_x_ranges_for_yearly_mode,
)

from src.visualization.helpers.renderer_visibility import set_main_series_visibility

NORMAL_PLOT_HEIGHT = 260
YEARLY_PLOT_HEIGHT = 520


def toggle_yearly_visibility(
    w_yearly_mode,
    w_years,
    w_set_alpha,
    w_alpha_value,
    w_timerange,
    ts_fig,
    ts_source,
    ts_source_fit,
    ts_fig_dir,
    ts_source_dir,
    ts_source_dir_fit,
    w_yearly_timerange,
):
    """
    Toggle Yearly mode:
      - Switch widget visibility
      - Hide raw series (and glacier secondary axis) in yearly mode
      - Lock x-ranges to dummy year in yearly mode
      - Increase plot height only in yearly mode (double height)
    """
    vis = bool(w_yearly_mode.value)

    # Widget visibility
    set_yearly_widgets_visibility(
        yearly_enabled=vis,
        w_years=w_years,
        w_yearly_timerange=w_yearly_timerange,
        w_timerange=w_timerange,
    )

    # Ensure dummy-year slider domain when enabled
    if vis:
        ensure_dummy_year_domain(w_yearly_timerange)

    # If yearly mode ON and options not filled yet, populate available years
    # from the main timerange
    if vis and (not w_years.options):
        populate_year_options_from_timerange(w_timerange, w_years)

    # Hide/show renderers (raw + fit) and glacier line in yearly mode
    set_main_series_visibility(
        yearly_enabled=vis,
        ts_fig=ts_fig,
        ts_source=ts_source,
        ts_source_fit=ts_source_fit,
        ts_fig_dir=ts_fig_dir,
        ts_source_dir=ts_source_dir,
        ts_source_dir_fit=ts_source_dir_fit,
        hide_glacier_in_yearly=True,
    )

    # X-range sync (dummy year vs main timerange)
    sync_x_ranges_for_yearly_mode(
        yearly_enabled=vis,
        w_timerange=w_timerange,
        ts_fig=ts_fig,
        ts_fig_dir=ts_fig_dir,
    )

    # Plot height: double height only in yearly mode
    if vis:
        ts_fig.height = YEARLY_PLOT_HEIGHT
        ts_fig_dir.height = YEARLY_PLOT_HEIGHT
    else:
        ts_fig.height = NORMAL_PLOT_HEIGHT
        ts_fig_dir.height = NORMAL_PLOT_HEIGHT
