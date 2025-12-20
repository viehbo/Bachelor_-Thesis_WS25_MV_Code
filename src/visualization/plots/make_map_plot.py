from src.visualization.plots.make_temperature_plot import make_temperature_plot
from src.visualization.plots.make_wind_plot import make_wind_plot


def make_map_plot(
    avg2d,
    extent,
    title,
    cbar_label,
    mode,
    lon=None,
    lat=None,
    u2d=None,
    v2d=None,
    quiver_stride=22,
    glacier_dir=None,
):
    """
    Create a map plot for either temperature/scalar or wind data.

    mode:
      - "temperature"
      - "wind"
      - "glacier" (point data; not supported by grid plotters yet)
    """
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
        if u2d is None or v2d is None:
            raise ValueError(
                "Wind plot requested but u2d/v2d are None. "
                "Check compute_average() return values."
            )

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

    if mode == "glacier":
        raise ValueError(
            "Glacier NetCDF files (annual_mass_balance with latitude/longitude per 'name') "
            "are point data, not a 2D grid. A separate glacier scatter map plot is required. "
            "Temperature and wind maps are supported."
        )

    raise ValueError(f"Unknown plot mode: {mode}")
