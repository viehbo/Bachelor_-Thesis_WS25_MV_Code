# src/visualization/helpers/yearly_mode.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence, Tuple

import pandas as pd


DUMMY_START = datetime(2000, 1, 1)
DUMMY_END = datetime(2000, 12, 31, 23, 59, 59)


def set_yearly_widgets_visibility(
    *,
    yearly_enabled: bool,
    w_years,
    w_yearly_timerange,
    w_timerange,
) -> None:
    """Show/hide Yearly-only widgets and the main timerange widget."""
    vis = bool(yearly_enabled)

    # main timeframe slider only when NOT in yearly mode
    try:
        w_timerange.visible = not vis
    except Exception:
        pass

    # years selector and yearly dummy-year range only when in yearly mode
    try:
        w_years.visible = vis
    except Exception:
        pass

    try:
        w_yearly_timerange.visible = vis
    except Exception:
        pass


def ensure_dummy_year_domain(w_yearly_timerange) -> None:
    """Ensure the yearly timerange slider is configured for the dummy year."""
    try:
        w_yearly_timerange.start = DUMMY_START
        w_yearly_timerange.end = DUMMY_END
        if not getattr(w_yearly_timerange, "value", None) or not all(w_yearly_timerange.value):
            w_yearly_timerange.value = (DUMMY_START, DUMMY_END)
    except Exception:
        pass


def sync_x_ranges_for_yearly_mode(
    *,
    yearly_enabled: bool,
    w_timerange,
    ts_fig,
    ts_fig_dir=None,
) -> None:
    """When Yearly mode is ON, lock x-range to dummy year. When OFF, follow main timerange."""
    try:
        if yearly_enabled:
            ts_fig.x_range.start = DUMMY_START
            ts_fig.x_range.end = DUMMY_END
            if ts_fig_dir is not None:
                ts_fig_dir.x_range.start = DUMMY_START
                ts_fig_dir.x_range.end = DUMMY_END
        else:
            start, end = w_timerange.value
            ts_fig.x_range.start = pd.to_datetime(start)
            ts_fig.x_range.end = pd.to_datetime(end)
            if ts_fig_dir is not None:
                ts_fig_dir.x_range.start = pd.to_datetime(start)
                ts_fig_dir.x_range.end = pd.to_datetime(end)
    except Exception:
        pass


def clear_yearly_renderers(
    *,
    yearly_sources: Optional[Sequence] = None,
    yearly_renderers: Optional[Sequence] = None,
    yearly_fit_sources: Optional[Sequence] = None,
    yearly_fit_renderers: Optional[Sequence] = None,
    dir_yearly_sources: Optional[Sequence] = None,
    dir_yearly_renderers: Optional[Sequence] = None,
    dir_yearly_fit_sources: Optional[Sequence] = None,
    dir_yearly_fit_renderers: Optional[Sequence] = None,
) -> None:
    """Clear and hide yearly lines (raw + fit) for main and direction figures."""
    def _clear_pair(srcs, rnds):
        if not srcs or not rnds:
            return
        for s, r in zip(srcs, rnds):
            try:
                s.data = dict(t=[], y=[])
            except Exception:
                pass
            try:
                r.visible = False
            except Exception:
                pass

    _clear_pair(yearly_sources, yearly_renderers)
    _clear_pair(yearly_fit_sources, yearly_fit_renderers)
    _clear_pair(dir_yearly_sources, dir_yearly_renderers)
    _clear_pair(dir_yearly_fit_sources, dir_yearly_fit_renderers)


def get_selected_years(w_years) -> list:
    """
    Supports both:
      - Panel MultiChoice / widgets with .value -> list
      - legacy list-of-widgets where each has .value
    """
    if hasattr(w_years, "value"):
        return list(w_years.value or [])
    try:
        return [w.value for w in w_years]
    except Exception:
        return []


def get_dummy_year_window(w_yearly_timerange) -> Optional[Tuple[datetime, datetime]]:
    """Return (start,end) from dummy-year slider, if available."""
    try:
        v = w_yearly_timerange.value
        if v and len(v) == 2 and v[0] and v[1]:
            return v[0], v[1]
    except Exception:
        pass
    return None
