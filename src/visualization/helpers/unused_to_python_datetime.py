import pandas as pd

WIN_SAFE_MIN = pd.Timestamp("1970-01-01T00:00:00Z")
WIN_SAFE_MAX = pd.Timestamp("2262-04-11T23:47:16.854Z")  # numpy/pandas upper limit

def _to_py_dt_safe(x):
    # x may be numpy.datetime64, pandas.Timestamp, or python datetime
    ts = pd.to_datetime(x, utc=True, errors="coerce")
    if pd.isna(ts):
        # fall back to epoch if something is odd
        ts = WIN_SAFE_MIN
    ts = ts.tz_convert(None)  # drop tz -> naive
    # clamp
    if ts < WIN_SAFE_MIN.tz_convert(None):
        ts = WIN_SAFE_MIN.tz_convert(None)
    if ts > WIN_SAFE_MAX.tz_convert(None):
        ts = WIN_SAFE_MAX.tz_convert(None)
    return ts.to_pydatetime()