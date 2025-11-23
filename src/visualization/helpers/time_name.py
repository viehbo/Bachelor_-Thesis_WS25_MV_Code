def _time_name(ds):
    for tname in ("time", "valid_time", "t"):
        if tname in ds.coords:
            return tname
    for name in ds.dims:
        if "time" in name:
            return name
    raise KeyError("Time coordinate not found.")