import xarray as xr

def _open_all(files):
    return xr.open_mfdataset(
        [str(p) for p in files],
        combine="by_coords",
        parallel=False,
        coords="minimal",
        data_vars="minimal",
    )