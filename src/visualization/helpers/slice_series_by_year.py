
def slice_series_by_year(series_or_df, year):
    """Return a pd.Series (scalar) or DataFrame row-axis slice for a single calendar year."""
    import pandas as pd
    idx = pd.to_datetime(series_or_df.index)
    m = (idx.year == int(year))
    if hasattr(series_or_df, "loc"):
        return series_or_df.loc[m]
    return series_or_df[m]
