import pandas as pd
from datetime import datetime

def _normalize_to_dummy_year(index_like, dummy_year=2000):


    t = pd.to_datetime(index_like)
    # handle leap day: shift 29-Feb to 28-Feb to keep axis sane
    def _mk(dt):
        m, d = dt.month, dt.day
        if m == 2 and d == 29:
            d = 28
        return datetime(dummy_year, m, d,
                        dt.hour, dt.minute, dt.second, dt.microsecond)
    return t.map(_mk)