
import numpy as np
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
)
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap

from src.visualization.plots.glacier_overlay import load_glacier_points, add_glacier_layer
from src.visualization.features.custom_markers import arrow_head_marker

from src.visualization.helpers.mercator_transformer import lon_to_x, lat_to_y

from src.visualization.utilities.plot_utilities import add_tiles


# --- public API ---------------------------------------------------------------
def make_wind_plot(
    avg2d,
    extent,
    title,
    cbar_label,
    *,
    lon,
    lat,
    u2d,
    v2d,
    quiver_stride=22,
    glacier_dir=None,
):
    """
    Render wind arrows with color by speed on a WebMercator map.

    Parameters
    ----------
    avg2d : (ny, nx) array-like
        Scalar speed field used for coloring (e.g., |u,v| or other).
    extent : [left, right, bottom, top]
        Geographic bounding box (degrees lon/lat).
    title : str
        Figure title.
    cbar_label : str
        Colorbar label (units).
    lon, lat : 1D arrays
        Center coordinates for grid columns/rows.
    u2d, v2d : (ny, nx) arrays
        Wind components.
    quiver_stride : int
        Decimation stride (not applied here; kept for compatibility).
    glacier_dir : str or None
        Directory with glacier data for overlay (optional).

    Returns
    -------
    p : bokeh.plotting.Figure
    """
    left, right, bottom, top = extent
    if any(x is None for x in (lon, lat, u2d, v2d)):
        raise ValueError("Wind plot requires lon, lat, u2d, v2d.")

    LON, LAT = np.meshgrid(lon, lat)
    sl = (slice(None, None, None), slice(None, None, None))  # keep all; adjust if you want decimation
    lon_d = LON[sl].ravel()
    lat_d = LAT[sl].ravel()
    u_d   = np.asarray(u2d)[sl].ravel()
    v_d   = np.asarray(v2d)[sl].ravel()
    sp_d  = np.asarray(avg2d)[sl].ravel()

    keep = np.isfinite(sp_d)
    lon_d, lat_d, u_d, v_d, sp_d = lon_d[keep], lat_d[keep], u_d[keep], v_d[keep], sp_d[keep]

    # Convert to WebMercator meters
    x = lon_to_x(lon_d)
    y = lat_to_y(lat_d)
    x0, x1 = lon_to_x(np.array([left, right]))
    y0, y1 = lat_to_y(np.array([bottom, top]))

    p = figure(
        x_axis_type="mercator",
        y_axis_type="mercator",
        x_range=(float(min(x0, x1)), float(max(x0, x1))),
        y_range=(float(min(y0, y1)), float(max(y0, y1))),
        width=900,
        height=560,
        title=title,
        tools="pan,wheel_zoom,reset,save,tap",
    )
    p.xaxis.axis_label = "WebMercator X (m)"
    p.yaxis.axis_label = "WebMercator Y (m)"

    add_tiles(p)

    # Arrow geometry in meters; length scales with vector magnitude (u,v)
    base = 5000.0  # meters per (m/s) — tweak for your data density
    xe = x + u_d * base
    ye = y + v_d * base

    if sp_d.size == 0 or not np.isfinite(sp_d).any():
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(np.nanmin(sp_d)), float(np.nanmax(sp_d))
        if vmin == vmax:
            vmin, vmax = 0.0, vmin or 1.0

    mapper = LinearColorMapper(palette=list(Viridis256), low=vmin, high=vmax)

    cds = ColumnDataSource(dict(xs=x, ys=y, xe=xe, ye=ye, sp=sp_d, u=u_d, v=v_d, lon=lon_d, lat=lat_d))

    # Single colormap for shaft/head
    cmap = linear_cmap(field_name="sp", palette=Viridis256, low=vmin, high=vmax)

    # Precompute angles
    angles = np.arctan2(cds.data["ye"] - cds.data["ys"], cds.data["xe"] - cds.data["xs"])
    cds.data["angle"] = angles

    a = p.scatter(
            x="xe", y="ye",
            source=cds,
            marker="@arrow_head_marker",
            size=12,
            angle="angle",
            angle_units="rad",
            fill_color=cmap,
            fill_alpha=0.5,
            line_color="black",
            line_width=0.2,
            defs={"@arrow_head_marker": arrow_head_marker()},
        )

    # clicked cell opacity
    a.selection_glyph = a.glyph.clone()
    a.selection_glyph.fill_alpha = 0.8
    a.selection_glyph.line_alpha = 1
    a.selection_glyph.line_width = 0.8

    # all other cells opacity
    a.nonselection_glyph = a.glyph.clone()
    a.nonselection_glyph.fill_alpha = 0.40
    a.nonselection_glyph.line_alpha = 0.50
    a.nonselection_glyph.line_width = 0.2


    color_bar = ColorBar(
        color_mapper=mapper,
        label_standoff=8,
        location=(0, 0),
        title=cbar_label or "Wind speed (m/s)",

    )
    p.add_layout(color_bar, "right")

    p.add_tools(HoverTool(
        tooltips=[
            ("speed", "@sp{0.00} m/s"),
            ("u", "@u{0.00}"),
            ("v", "@v{0.00}"),
            ("lon", "@lon{0.00}"),
            ("lat", "@lat{0.00}"),
        ],
        mode="mouse",
    ))

    try:
        if glacier_dir is not None:
            gp = load_glacier_points(glacier_dir)
            if gp:
                add_glacier_layer(p, gp, lon_to_x, lat_to_y)
    except Exception as e:
        print("Glacier overlay skipped:", e)

    return p
