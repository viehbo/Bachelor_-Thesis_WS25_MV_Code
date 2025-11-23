

def apply_alpha_to_raw_lines(w_alpha_value,
                             ts_fig,
                             ts_source,
                             ts_year_renderers,
                             ts_fig_dir,
                             ts_source_dir,
                             ts_dir_year_renderers
                             ):
    # only change raw/yearly; leave fit lines unchanged
    try:
        a = float(w_alpha_value.value)
    except Exception:
        a = 0.35

    # clamp to [0, 1] without rounding
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0

    # main ts raw line is the first renderer bound to ts_source in make_line_plot_1;
    # easiest: set datasource-level alpha by updating the renderer glyphs we returned for yearly
    # For the raw "observed" line, we adjust the glyph on the figure by finding the renderer that uses ts_source.
    for r in ts_fig.renderers:
        try:
            if getattr(r, "data_source", None) is ts_source:
                r.glyph.line_alpha = a
        except Exception:
            pass
    for r in ts_year_renderers:
        r.glyph.line_alpha = a if r.data_source.data.get("t") else 0.0

    # Direction plot (if visible)
    for r in ts_fig_dir.renderers:
        try:
            if getattr(r, "data_source", None) is ts_source_dir:
                r.glyph.line_alpha = a
        except Exception:
            pass
    for r in ts_dir_year_renderers:
        r.glyph.line_alpha = a if r.data_source.data.get("t") else 0.0