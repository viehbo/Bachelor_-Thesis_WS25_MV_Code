import pandas as pd
import numpy as np

def set_timeseries(ts_source, series_or_df, *, kind, units="", title=None, fig=None):
    """
    kind:
      - "scalar": series_or_df is a pd.Series (e.g., temperature)
      - "uv":     series_or_df is a pd.DataFrame with columns ['u','v'] (speed magnitude will be plotted)
    """
    if kind == "scalar":
        s = series_or_df
    elif kind == "uv":
        df = series_or_df
        s = pd.Series(np.hypot(df["u"].to_numpy(), df["v"].to_numpy()), index=df.index, name="speed")
    else:
        raise ValueError(f"Unsupported kind={kind!r} in set_timeseries")

    #t = pd.to_datetime(s.index).to_pydatetime().tolist()
    t = pd.to_datetime(s.index).to_pydatetime()
    t = t.tolist()
    y = np.asarray(s.values, dtype=float).tolist()
    ts_source.data = dict(t=t, y=y)
    if fig is not None:
        fig.yaxis.axis_label = units or ""
        if title:
            fig.title.text = title

