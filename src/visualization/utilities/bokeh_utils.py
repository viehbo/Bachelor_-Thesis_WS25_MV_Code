"""
Bokeh-specific utility functions for common operations on ColumnDataSource objects.
Reduces code duplication across visualization modules.
"""
from typing import List, Optional, Iterable
import pandas as pd
import numpy as np


def clear_source(source) -> None:
    """Clear a ColumnDataSource by setting empty data."""
    if source is not None:
        source.data = dict(t=[], y=[])


def clear_sources(sources: Iterable) -> None:
    """Clear multiple ColumnDataSource objects."""
    if sources:
        for src in sources:
            clear_source(src)


def update_timeseries_source(
        source,
        t_data: Iterable,
        y_data: Iterable,
        *,
        convert_datetime: bool = True
) -> None:
    """
    Update a ColumnDataSource with time series data.

    Parameters
    ----------
    source : ColumnDataSource
        Bokeh data source to update
    t_data : Iterable
        Time data (will be converted to datetime if needed)
    y_data : Iterable
        Y-axis values
    convert_datetime : bool
        Whether to convert t_data to Python datetime objects
    """
    if source is None:
        return

    if convert_datetime:
        t_list = pd.to_datetime(t_data).to_pydatetime().tolist()
    else:
        t_list = list(t_data)

    y_list = [float(v) for v in y_data]
    source.data = dict(t=t_list, y=y_list)


def hide_renderer(renderer) -> None:
    """Hide a Bokeh renderer and set its alpha to 0."""
    if renderer is not None:
        try:
            renderer.visible = False
            renderer.glyph.line_alpha = 0.0
        except Exception:
            pass


def show_renderer(renderer, alpha: float = 1.0) -> None:
    """Show a Bokeh renderer and set its alpha."""
    if renderer is not None:
        try:
            renderer.visible = True
            renderer.glyph.line_alpha = float(alpha)
        except Exception:
            pass


def clear_and_hide_renderer_pair(source, renderer) -> None:
    """Clear a source and hide its corresponding renderer."""
    clear_source(source)
    hide_renderer(renderer)


def clear_and_hide_pairs(sources: Iterable, renderers: Iterable) -> None:
    """Clear multiple source/renderer pairs."""
    if sources and renderers:
        for src, rnd in zip(sources, renderers):
            clear_and_hide_renderer_pair(src, rnd)


def set_renderer_alpha(renderer, alpha: float) -> None:
    """
    Set the line alpha of a renderer's glyph.

    Only sets alpha if the renderer has data (non-empty source).
    """
    if renderer is None:
        return

    try:
        # Only set alpha if there's actual data
        has_data = (
                hasattr(renderer, 'data_source') and
                renderer.data_source.data.get('t')
        )

        if has_data:
            renderer.glyph.line_alpha = float(alpha)
        else:
            renderer.glyph.line_alpha = 0.0
    except Exception:
        pass


def find_renderers_by_source(figure, source) -> List:
    """
    Find all renderers in a figure that use the given data source.

    Parameters
    ----------
    figure : bokeh Figure
        Figure to search
    source : ColumnDataSource
        Source to match

    Returns
    -------
    list
        List of matching renderers
    """
    matching = []
    try:
        for r in getattr(figure, 'renderers', []):
            if getattr(r, 'data_source', None) is source:
                matching.append(r)
    except Exception:
        pass
    return matching


def find_renderers_by_y_range(figure, y_range_name: str) -> List:
    """
    Find all renderers in a figure that use the given y_range_name.

    Parameters
    ----------
    figure : bokeh Figure
        Figure to search
    y_range_name : str
        Y-range name to match (e.g., 'glacier')

    Returns
    -------
    list
        List of matching renderers
    """
    matching = []
    try:
        for r in getattr(figure, 'renderers', []):
            if getattr(r, 'y_range_name', None) == y_range_name:
                matching.append(r)
    except Exception:
        pass
    return matching


def set_figure_axes_labels(figure, *, x_label: Optional[str] = None, y_label: Optional[str] = None) -> None:
    """Set figure axis labels safely."""
    if figure is None:
        return

    try:
        if x_label is not None:
            figure.xaxis.axis_label = x_label
        if y_label is not None:
            figure.yaxis.axis_label = y_label
    except Exception:
        pass


def set_figure_title(figure, title: str) -> None:
    """Set figure title safely."""
    if figure is not None:
        try:
            figure.title.text = title
        except Exception:
            pass