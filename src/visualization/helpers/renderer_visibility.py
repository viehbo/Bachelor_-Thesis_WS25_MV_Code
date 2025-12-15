# src/visualization/helpers/renderer_visibility.py
from __future__ import annotations


def set_main_series_visibility(
    *,
    yearly_enabled: bool,
    ts_fig,
    ts_source,
    ts_source_fit,
    ts_fig_dir=None,
    ts_source_dir=None,
    ts_source_dir_fit=None,
    hide_glacier_in_yearly: bool = True,
) -> None:
    """
    Toggle visibility of:
      - main raw series + fit (by matching renderer.data_source)
      - optional direction raw series + fit
      - optional glacier (by y_range_name == 'glacier')

    This is intentionally conservative: it only touches renderers that match the provided sources.
    """
    vis = bool(yearly_enabled)

    try:
        for r in getattr(ts_fig, "renderers", []):
            ds = getattr(r, "data_source", None)
            y_range_name = getattr(r, "y_range_name", None)

            # main series & its fit
            if ds is ts_source or ds is ts_source_fit:
                r.visible = not vis

            # glacier line (secondary axis)
            if hide_glacier_in_yearly and y_range_name == "glacier":
                r.visible = not vis
    except Exception:
        pass

    if ts_fig_dir is None:
        return

    try:
        for r in getattr(ts_fig_dir, "renderers", []):
            ds = getattr(r, "data_source", None)
            if ds is ts_source_dir or ds is ts_source_dir_fit:
                r.visible = not vis
    except Exception:
        pass
