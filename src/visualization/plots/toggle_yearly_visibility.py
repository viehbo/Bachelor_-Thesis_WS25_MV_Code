
import pandas as pd
from datetime import datetime
from src.visualization.helpers.populate_year_options_from_timerange import populate_year_options_from_timerange

def toggle_yearly_visibility(w_yearly_mode,
                             _year_fields,
                             w_set_alpha,
                             w_alpha_value,
                             w_timerange,
                             ts_fig,
                             ts_source,
                             ts_source_fit,
                             ts_fig_dir,
                             ts_source_dir,
                             ts_source_dir_fit,
                             w_yearly_timerange,
                             ):

    DUMMY_START = datetime(2000, 1, 1)
    DUMMY_END = datetime(2000, 12, 31, 23, 59, 59)
    print("FRITZ: ", DUMMY_START, DUMMY_END)
    vis = bool(w_yearly_mode.value)
    # Show timeframe slider only when NOT in yearly mode
    w_timerange.visible = not vis

    for field in _year_fields:
        field.visible = vis
    #w_set_alpha.visible = vis
    #w_alpha_value.visible = vis
    if vis and (_year_fields[0].options == []):
        populate_year_options_from_timerange(w_timerange, _year_fields)



    # show/hide yearly dummy-year slider
    try:
        w_yearly_timerange.visible = vis
        if vis:
            # ensure its domain matches the dummy year
            w_yearly_timerange.start = DUMMY_START
            w_yearly_timerange.end = DUMMY_END
            if not w_yearly_timerange.value or not all(w_yearly_timerange.value):
                w_yearly_timerange.value = (DUMMY_START, DUMMY_END)
    except Exception:
        pass


    # --- Hide/show main lines depending on mode ---
    try:
        for r in ts_fig.renderers:
            ds = getattr(r, "data_source", None)
            if ds is ts_source or ds is ts_source_fit:
                r.visible = not vis  # hide in yearly mode, show otherwise
        for r in ts_fig_dir.renderers:
            ds = getattr(r, "data_source", None)
            if ds is ts_source_dir or ds is ts_source_dir_fit:
                r.visible = not vis
    except Exception:
        pass

    # --- Adjust x-range ---
    try:
        if vis:
            ts_fig.x_range.start = DUMMY_START
            ts_fig.x_range.end = DUMMY_END
            if ts_fig_dir is not None:
                ts_fig_dir.x_range.start = DUMMY_START
                ts_fig_dir.x_range.end = DUMMY_END
        else:
            start, end = w_timerange.value
            ts_fig.x_range.start = pd.to_datetime(start)
            ts_fig.x_range.end = pd.to_datetime(end)
            if ts_fig_dir is not None:
                ts_fig_dir.x_range.start = pd.to_datetime(start)
                ts_fig_dir.x_range.end = pd.to_datetime(end)
    except Exception:
        pass

