import numpy as np

def _nearest_idx(arr, val):
    arr = np.asarray(arr)
    m = np.isfinite(arr)
    if not m.any():
        raise ValueError("All-NaN lon/lat grid.")
    idxs = np.nonzero(m)[0]
    local = int(np.argmin(np.abs(arr[m] - val)))
    return int(idxs[local])