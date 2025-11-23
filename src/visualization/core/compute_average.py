import numpy as np
import pandas as pd

from src.visualization.helpers.find_var import _find_var
from src.visualization.helpers.lon_lat_names import _lon_lat_names
from src.visualization.helpers.time_name import _time_name
from src.visualization.helpers.open_all import _open_all

def compute_average(DATASETS, selected_files, ds_key, t_range,
                    hour_start=None, hour_end=None):
    if not selected_files:
        raise ValueError("No files selected.")
    cfg = DATASETS[ds_key]

    with _open_all(selected_files) as ds:
        lon_name, lat_name = _lon_lat_names(ds)
        extent = [
            float(ds[lon_name].min().values),
            float(ds[lon_name].max().values),
            float(ds[lat_name].min().values),
            float(ds[lat_name].max().values),
        ]
        lon = ds[lon_name].values
        lat = ds[lat_name].values

        tname = _time_name(ds)
        if t_range is not None:
            t0, t1 = t_range
            ds = ds.sel({tname: slice(np.datetime64(pd.Timestamp(t0)), np.datetime64(pd.Timestamp(t1)))})


        # --- NEW: filter by daily time-of-day (e.g. 05:00–15:00) ---
        if hour_start is not None and hour_end is not None:
            # hour_start/hour_end are usually datetime.time from TimePicker
            hs = hour_start
            he = hour_end

            # In case Panel gives datetime-like, normalize to .time()
            if hasattr(hs, "time"):
                hs = hs.time()
            if hasattr(he, "time"):
                he = he.time()

            # if user accidentally selects start > end, swap them
            if hs > he:
                hs, he = he, hs

            # convert coordinate to pandas index to use .time
            times = ds[tname].to_index()
            mask = (times.time >= hs) & (times.time <= he)

            # apply boolean mask on the time coordinate
            ds = ds.sel({tname: mask})


        if cfg["mode"] == "wind":
            u_name = _find_var(ds, cfg["candidates"]["u"])
            v_name = _find_var(ds, cfg["candidates"]["v"])
            mean_u = ds[u_name].mean(dim=tname)
            mean_v = ds[v_name].mean(dim=tname)
            speed  = np.hypot(mean_u, mean_v)

            # Ensure lat,lon are last:
            mean_u_2d = mean_u.transpose(..., lat_name, lon_name).values
            mean_v_2d = mean_v.transpose(..., lat_name, lon_name).values
            speed_2d  = speed.transpose(...,  lat_name, lon_name).values
            units = getattr(ds[u_name], "attrs", {}).get("units", "")

            # Return speed for coloring, plus u/v for arrows
            return speed_2d, extent, units, lon, lat, mean_u_2d, mean_v_2d
        else:
            x_name = _find_var(ds, cfg["candidates"]["x"])
            avg = ds[x_name].mean(dim=tname)
            avg2d = avg.transpose(..., lat_name, lon_name).values
            units = getattr(ds[x_name], "attrs", {}).get("units", "")
            # No vectors for temperature
            return avg2d, extent, units, lon, lat, None, None