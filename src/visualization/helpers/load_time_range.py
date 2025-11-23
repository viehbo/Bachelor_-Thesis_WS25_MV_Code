from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

from src.visualization.helpers.open_all import _open_all
from src.visualization.helpers.time_name import _time_name

def _to_py_naive(dt_any):

    # pandas / numpy case
    if isinstance(dt_any, (np.datetime64, pd.Timestamp)):
        ts = pd.to_datetime(dt_any)
        if ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return ts.to_pydatetime().replace(tzinfo=None)

    # plain python datetime
    if isinstance(dt_any, datetime):
        return dt_any.replace(tzinfo=None)

    # cftime object (no tz), convert by components
    # Avoid importing cftime unless needed
    mod = type(dt_any).__module__
    if "cftime" in mod:
        return datetime(
            dt_any.year, dt_any.month, dt_any.day,
            getattr(dt_any, "hour", 0),
            getattr(dt_any, "minute", 0),
            getattr(dt_any, "second", 0),
            getattr(dt_any, "microsecond", 0),
        )

    # last resort via pandas
    ts = pd.to_datetime(dt_any, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Cannot convert time value {dt_any!r} to python datetime.")
    return ts.to_pydatetime()

def load_time_range(w_files, w_status, w_timerange):
    selected = [Path(p) for p in w_files.value]
    if not selected:
        w_status.object = "Select one or more files first."
        return


    try:
        # Use your context manager; rely on xarray's CF decoding
        with _open_all(selected) as ds:
            tname = _time_name(ds)

            # Prefer the decoded coordinate from xarray directly
            tcoord = ds[tname].to_index() if hasattr(ds[tname], "to_index") else None
            if tcoord is not None and len(tcoord) > 0:
                tmin_raw = tcoord.min()
                tmax_raw = tcoord.max()
            else:
                # Fallback to the raw array
                vals = ds[tname].values
                tmin_raw = vals.min()
                tmax_raw = vals.max()

        # Convert to tz-naive python datetimes (Panel slider wants naive dt)
        t0 = _to_py_naive(tmin_raw)
        t1 = _to_py_naive(tmax_raw)

        # Make sure start <= end
        if t0 > t1:
            t0, t1 = t1, t0

        # Choose a sensible step (keep your 6h default)
        step = pd.Timedelta(hours=6)

        # Wire into the Panel slider
        w_timerange.start = t0
        w_timerange.end   = t1
        w_timerange.value = (t0, t1)
        w_timerange.step  = int(step.total_seconds() * 1000)  # ms
        w_timerange.disabled = False
        w_timerange.visible = True
        w_status.object = f"Time range loaded: {t0} → {t1}"

    except Exception as e:
        w_status.object = f"**Error loading time range:** {e}"
