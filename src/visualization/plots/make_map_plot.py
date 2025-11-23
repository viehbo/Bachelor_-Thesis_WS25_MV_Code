
from src.visualization.plots.make_temperature_plot import make_temperature_plot  # ← NEW
from src.visualization.plots.make_wind_plot import make_wind_plot

def make_map_plot(avg2d, extent, title, cbar_label, mode,
                  lon=None, lat=None, u2d=None, v2d=None, quiver_stride=22,
                  glacier_dir=None):
    """
    extent: [left, right, bottom, top] in degrees (lon/lat)
    mode:   "temperature" or "wind"
    """
    left, right, bottom, top = extent

    if mode == "temperature":
        return make_temperature_plot(
            avg2d,
            extent,
            title,
            cbar_label,
            lon=lon,
            lat=lat,
            glacier_dir=glacier_dir,
        )

    if mode == "wind":
        return make_wind_plot(
            avg2d,
            extent,
            title,
            cbar_label,
            lon=lon,
            lat=lat,
            u2d=u2d,
            v2d=v2d,
            quiver_stride=quiver_stride,
            glacier_dir=glacier_dir,
        )
