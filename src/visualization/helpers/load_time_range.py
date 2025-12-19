from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

from src.visualization.helpers.open_all import _open_all  # unused here but kept for compatibility
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


def load_time_range(w_files, w_status, w_timerange, w_hours=None, *, w_files_summary=None, state=None):

    """
    Load global time range and populate available hours from the selected files.
    Uses a tolerant multi-file open: combine='by_coords', join='outer'.
    """
    import xarray as xr
    try:
        paths = [Path(p) for p in w_files.value]
        if not paths:
            w_status.object = "**No files selected.**"
            return

        from src.visualization.helpers.detect_netcdf_kind import detect_kind_for_files

        paths = [Path(p) for p in w_files.value]
        if not paths:
            w_status.object = "**No files selected.**"
            if w_files_summary is not None:
                w_files_summary.object = "**Files:** 0 — **Type:** –"
            if state is not None:
                state["data_kind"] = None
            return

        # validate kind consistency
        try:
            kind, _ = detect_kind_for_files(paths)
            if w_files_summary is not None:
                w_files_summary.object = f"**Files:** {len(paths)} — **Type:** {kind}"
            if state is not None:
                state["data_kind"] = kind
        except Exception as e:
            w_status.object = f"**Error:** {e}"
            if w_files_summary is not None:
                w_files_summary.object = f"**Files:** {len(paths)} — **Type:** (error)"
            if state is not None:
                state["data_kind"] = None
            return

        # Robust open across files with differing time stamps
        with xr.open_mfdataset(
            [str(p) for p in paths],
            combine="by_coords",   # align by coordinate labels
            join="outer",          # allow non-identical coordinate sets
            parallel=False,
        ) as ds:
            tname = _time_name(ds)
            # use to_index() to get a single pandas DatetimeIndex
            t = pd.to_datetime(ds[tname].to_index())

        # Slider range
        w_timerange.start = t.min().to_pydatetime()
        w_timerange.end   = t.max().to_pydatetime()
        w_timerange.value = (w_timerange.start, w_timerange.end)
        w_timerange.disabled = False
        w_timerange.visible  = True

        # Available hours (as strings "00".."23")
        if w_hours is not None:
            hours = pd.Index(t.hour).unique().sort_values().tolist()
            # Map 24->0 if a 1..24 convention appears (rare)
            if (24 in hours) and (0 not in hours):
                hours = sorted({0 if h == 24 else h for h in hours})
            labels = [f"{h:02d}" for h in hours]
            w_hours.options = labels
            w_hours.value = labels  # default: "all hours"

        w_status.object = "Loaded time range and hours."
    except Exception as e:
        w_status.object = f"**Error:** {e}"
