def _lon_lat_names(ds):
    for lon_name in ("longitude", "lon", "x"):
        if lon_name in ds:
            break
    else:
        raise KeyError("Longitude coordinate not found.")
    for lat_name in ("latitude", "lat", "y"):
        if lat_name in ds:
            break
    else:
        raise KeyError("Latitude coordinate not found.")
    return lon_name, lat_name
