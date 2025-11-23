

def update_poly_fit_only(w_fit_degree,
                         poly_fit_datetime,
                         ts_source,
                         ts_source_fit,
                         ts_source_dir,
                         ts_source_dir_fit
                         ):

    deg = int(w_fit_degree.value)

    print("FRITZ: deg = ", deg)

    def update_one(src_raw, src_fit):
        if src_fit is None or src_raw is None:
            return
        t = src_raw.data.get("t", []) or []
        y = src_raw.data.get("y", []) or []
        if len(t) >= deg + 1:
            t_fit, y_fit = poly_fit_datetime(t, y, degree=deg, points=400)
            src_fit.data = dict(t=t_fit, y=y_fit)
        else:
            src_fit.data = dict(t=[], y=[])

    # speed plot
    update_one(ts_source, ts_source_fit)

    # optional: direction plot (only if you created it)
    try:
        update_one(ts_source_dir, ts_source_dir_fit)
    except NameError:
        pass