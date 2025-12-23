# make_temperature_plot.py — temperature-only Bokeh renderer
import numpy as np
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
from bokeh.palettes import Turbo256

from src.visualization.plots.glacier_overlay import load_glacier_points, add_glacier_layer
from src.visualization.helpers.mercator_transformer import lon_to_x, lat_to_y

from src.visualization.utilities.plot_utilities import add_tiles

def make_temperature_plot(
    avg2d,
    extent,
    title,
    cbar_label,
    *,
    lon,
    lat,
    glacier_dir=None,
):
    """
    Render a temperature raster on a WebMercator map.

    Parameters

    """
    left, right, bottom, top = extent

    # Convert extent to WebMercator and set up map
    x0, x1 = lon_to_x(np.array([left, right]))
    y0, y1 = lat_to_y(np.array([bottom, top]))

    p = figure(
        x_axis_type="mercator", y_axis_type="mercator",
        x_range=(float(min(x0, x1)), float(max(x0, x1))),
        y_range=(float(min(y0, y1)), float(max(y0, y1))),
        width=900, height=560, title=title,
        tools="pan,wheel_zoom,reset,save,tap"
    )
    p.xaxis.axis_label = "Latidute"
    p.yaxis.axis_label = "Lonitude"

    add_tiles(p)

    # Data -> quads
    arr = np.array(avg2d, copy=True)

    # set the scale to a fix value
    vmin = -5
    vmax = 25

    mapper = LinearColorMapper(palette=list(Turbo256), low=vmin, high=vmax)

    # centers -> edges
    def _edges_from_centers(c):
        c = np.asarray(c, dtype=float)
        if c.size < 2:
            dc = 0.001
            return np.array([c[0] - dc, c[0] + dc])
        mid = (c[:-1] + c[1:]) / 2.0
        first = c[0] - (c[1] - c[0]) / 2.0
        last = c[-1] + (c[-1] - c[-2]) / 2.0
        edges = np.empty(c.size + 1, dtype=float)
        edges[1:-1] = mid
        edges[0] = first
        edges[-1] = last
        return edges

    if lon is None or lat is None:
        raise ValueError("Provide lon and lat center arrays matching avg2d.")

    lon_edges = _edges_from_centers(lon)
    lat_edges = _edges_from_centers(lat)

    LonL, LatB = np.meshgrid(lon_edges[:-1], lat_edges[:-1])
    LonR, LatT = np.meshgrid(lon_edges[1:],  lat_edges[1:])

    LeftX  = lon_to_x(LonL)
    RightX = lon_to_x(LonR)
    BotY   = lat_to_y(LatB)
    TopY   = lat_to_y(LatT)

    LONc, LATc = np.meshgrid(lon, lat)

    mask = np.isfinite(arr)
    cds = ColumnDataSource(dict(
        left_x=LeftX[mask].ravel(),
        right_x=RightX[mask].ravel(),
        bot_y=BotY[mask].ravel(),
        top_y=TopY[mask].ravel(),
        val=arr[mask].ravel(),
        lon=LONc[mask].ravel(),
        lat=LATc[mask].ravel(),
    ))

    r = p.quad(
        left="left_x", right="right_x", bottom="bot_y", top="top_y",
        source=cds,
        fill_color={"field": "val", "transform": mapper},
        fill_alpha=0.6,
        line_color="black",
        line_width=0.2,
        line_alpha=0.5,
    )

    r.selection_glyph = r.glyph.clone()
    r.selection_glyph.fill_alpha = 0.8  # clicked cell opacity
    r.selection_glyph.line_alpha = 1
    r.selection_glyph.line_width = 0.8

    r.nonselection_glyph = r.glyph.clone()
    r.nonselection_glyph.fill_alpha = 0.40  # all other cells opacity
    r.nonselection_glyph.line_alpha = 0.50
    r.nonselection_glyph.line_width = 0.2

    color_bar = ColorBar(
        color_mapper=mapper,
        label_standoff=8,
        location=(0, 0),
        title=cbar_label or "Temperature (°C)",

    )
    p.add_layout(color_bar, "right")

    p.add_tools(HoverTool(
        tooltips=[("value", "@val{0.00}"), ("lon", "@lon{0.000}"), ("lat", "@lat{0.000}")],
        mode="mouse",
        point_policy="follow_mouse",
        renderers=[r],
    ))

    try:
        if glacier_dir is not None:
            gp = load_glacier_points(glacier_dir)
            if gp:
                add_glacier_layer(p, gp, lon_to_x, lat_to_y)
    except Exception as e:
        print("Glacier overlay skipped:", e)
    return p
