
import pandas as pd
import numpy as np

from src.visualization.core.dataset import DATASETS, GLACIERS

from src.visualization.helpers.open_all import _open_all
from src.visualization.helpers.lon_lat_names import _lon_lat_names
from src.visualization.helpers.time_name import _time_name
from src.visualization.helpers.find_var import _find_var
from src.visualization.helpers.find_nearest_index import _nearest_idx


def _extract_timeseries_grid(files, ds_key, t_range, lon_val, lat_val):
    """Return a pd.Series (scalar) or pd.DataFrame with ['u','v'] (uv) plus meta."""
    cfg = DATASETS[ds_key]
    if not files:
        raise ValueError("No files selected.")
    with _open_all(files) as ds:
        lon_name, lat_name = _lon_lat_names(ds)
        tname = _time_name(ds)

        # time slice
        if t_range is not None and all(t_range):
            t0, t1 = t_range
            ds = ds.sel({tname: slice(np.datetime64(pd.Timestamp(t0)),
                                      np.datetime64(pd.Timestamp(t1)))})

        # nearest grid point
        lons = ds[lon_name].values
        lats = ds[lat_name].values
        i_lon = _nearest_idx(lons, lon_val)
        i_lat = _nearest_idx(lats, lat_val)

        if cfg["mode"] == "wind":
            u_name = _find_var(ds, cfg["candidates"]["u"])
            v_name = _find_var(ds, cfg["candidates"]["v"])
            u1 = ds[u_name].isel({lon_name: i_lon, lat_name: i_lat}).to_series()
            v1 = ds[v_name].isel({lon_name: i_lon, lat_name: i_lat}).to_series()
            df = pd.DataFrame({"u": u1, "v": v1}).sort_index().dropna()
            units = getattr(ds[u_name], "attrs", {}).get("units", "m/s")
            return df, {"units": units, "kind": "uv"}
        else:
            x_name = _find_var(ds, cfg["candidates"]["x"])
            s = ds[x_name].isel({lon_name: i_lon, lat_name: i_lat}).to_series().sort_index().dropna()
            units = getattr(ds[x_name], "attrs", {}).get("units", "")
            return s, {"units": units, "kind": "scalar"}
