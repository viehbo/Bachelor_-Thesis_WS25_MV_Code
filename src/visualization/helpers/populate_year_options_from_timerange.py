import pandas as pd

def populate_year_options_from_timerange(w_timerange, _year_fields):
    # Read the slider tuple (start, end)
    try:
        start, end = w_timerange.value
    except Exception:
        return  # slider not ready yet
    if not (start and end):
        return
    y0 = int(pd.to_datetime(start).year)
    y1 = int(pd.to_datetime(end).year)
    if y1 < y0:
        y0, y1 = y1, y0
    years_available = list(range(y0, y1 + 1))
    # Set options on all 10 selects and reset selection to None
    for w in _year_fields:
        w.options = years_available
        w.value = None  # default: no selection