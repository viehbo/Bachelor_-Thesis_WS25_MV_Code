import pandas as pd
import numpy as np
import math

from bokeh.palettes import Category10

from src.visualization.helpers.normalize_to_dummy_year import _normalize_to_dummy_year
from src.visualization.helpers.slice_series_by_year import slice_series_by_year

from src.visualization.core.poly_fit import poly_fit_datetime

PALETTE10 = list(Category10[10])


def set_yearly_overlays(
    fig,
    base_series_or_df,
    *,
    kind: str,
    years: list,
    yearly_sources: list,
    yearly_renderers: list,
    yearly_fit_sources: list,
    yearly_fit_renderers: list,
    fit_degree: int,
    alpha: float = 0.35,
    units: str = "",
    title_prefix: str = "Yearly",
    dummy_range=None,  # (start, end) in dummy year 2000
):
    """
    Draw yearly overlays (and per-year poly fits) of a scalar or uv time series.

    dummy_range: tuple(start, end) where both are datetimes in dummy year 2000.
                 The same (month/day[/time]) window is applied to every selected
                 real year before normalising to dummy year.
    """

    if (
        yearly_sources is None
        or yearly_renderers is None
        or yearly_fit_sources is None
        or yearly_fit_renderers is None
    ):
        return

    if kind == "scalar":
        s_all = pd.Series(base_series_or_df).sort_index()
    elif kind == "uv":
        df = pd.DataFrame(base_series_or_df).sort_index()
        s_all = pd.Series(
            np.hypot(df["u"].to_numpy(), df["v"].to_numpy()),
            index=df.index,
            name="speed",
        )
    else:
        raise ValueError(f"Unsupported kind={kind!r} for yearly overlays.")

    # Color map for selected (non-empty) years
    valid_years = []
    for y in years:
        if y is None:
            continue
        if isinstance(y, float) and math.isnan(y):
            continue
        s = str(y).strip()
        if s == "" or s.lower() in ("nan", "none"):
            continue
        try:
            valid_years.append(int(y))
        except Exception:
            pass

    color_map = {yr: PALETTE10[i % len(PALETTE10)] for i, yr in enumerate(valid_years)}

    # Update the 10 overlay + 10 fit renderers
    for i in range(10):
        src = yearly_sources[i]
        rnd = yearly_renderers[i]
        fit_src = yearly_fit_sources[i]
        fit_rnd = yearly_fit_renderers[i]

        # determine selected year for this slot
        ysel = years[i] if i < len(years) else None
        is_empty = (
            ysel is None
            or (isinstance(ysel, float) and math.isnan(ysel))
            or (isinstance(ysel, str) and ysel.strip().lower() in ("", "nan", "none"))
        )

        if is_empty:
            # hide both raw and fit for this slot
            src.data = dict(t=[], y=[])
            rnd.visible = False
            rnd.glyph.line_alpha = 0.0

            fit_src.data = dict(t=[], y=[])
            fit_rnd.visible = False
            fit_rnd.glyph.line_alpha = 0.0

            try:
                rnd.legend_label = None
                fit_rnd.legend_label = None
            except Exception:
                pass
            continue

        # parse year
        try:
            yint = int(ysel)
        except Exception:
            src.data = dict(t=[], y=[])
            rnd.visible = False
            rnd.glyph.line_alpha = 0.0

            fit_src.data = dict(t=[], y=[])
            fit_rnd.visible = False
            fit_rnd.glyph.line_alpha = 0.0

            try:
                rnd.legend_label = None
                fit_rnd.legend_label = None
            except Exception:
                pass
            continue

        # extract this year's data
        s_one = slice_series_by_year(s_all, yint)
        if len(s_one) == 0:
            src.data = dict(t=[], y=[])
            rnd.visible = False
            rnd.glyph.line_alpha = 0.0

            fit_src.data = dict(t=[], y=[])
            fit_rnd.visible = False
            fit_rnd.glyph.line_alpha = 0.0

            try:
                rnd.legend_label = None
                fit_rnd.legend_label = None
            except Exception:
                pass
            continue

        # Apply dummy-year window in real-year space
        # dummy_range is (start, end) in dummy year 2000, e.g. 2000-07-01..2000-07-31
        s_window = s_one
        if dummy_range is not None:
            try:
                dr_start, dr_end = dummy_range

                # Ensure tz-naive Timestamps
                dr_start = pd.to_datetime(dr_start)
                dr_end = pd.to_datetime(dr_end)
                if getattr(dr_start, "tz", None) is not None:
                    dr_start = dr_start.tz_localize(None)
                if getattr(dr_end, "tz", None) is not None:
                    dr_end = dr_end.tz_localize(None)

                # Map dummy window back to this real year
                real_start = dr_start.replace(year=yint)
                real_end = dr_end.replace(year=yint)

                idx_real = pd.to_datetime(s_one.index)
                if getattr(idx_real, "tz", None) is not None:
                    idx_real = idx_real.tz_localize(None)

                mask = (idx_real >= real_start) & (idx_real <= real_end)
                s_window = s_one.loc[mask]

                # If nothing remains in the window: hide this slot
                if len(s_window) == 0:
                    src.data = dict(t=[], y=[])
                    rnd.visible = False
                    rnd.glyph.line_alpha = 0.0

                    fit_src.data = dict(t=[], y=[])
                    fit_rnd.visible = False
                    fit_rnd.glyph.line_alpha = 0.0

                    try:
                        rnd.legend_label = None
                        fit_rnd.legend_label = None
                    except Exception:
                        pass
                    continue

            except Exception:
                # On failure, just keep full-year data (s_window = s_one)
                s_window = s_one


        # raw yearly line (normalised to dummy year)
        t_norm = _normalize_to_dummy_year(s_window.index)
        y_arr = np.asarray(s_window.values, dtype=float)

        t_norm = pd.to_datetime(t_norm)
        if getattr(t_norm, "tz", None) is not None:
            t_norm = t_norm.tz_localize(None)

        t_norm_list = t_norm.to_pydatetime().tolist()
        y_list = y_arr.tolist()

        src.data = dict(t=t_norm_list, y=y_list)

        color = color_map.get(yint, PALETTE10[i % len(PALETTE10)])
        rnd.glyph.line_color = color
        try:
            rnd.legend_label = str(yint)
        except Exception:
            pass
        rnd.visible = True
        rnd.glyph.line_alpha = float(alpha)

        # per-year poly-fit over the same window
        if len(s_window) >= fit_degree + 1:
            t_fit, y_fit = poly_fit_datetime(
                s_window.index, s_window.values, degree=int(fit_degree), points="original"
            )

            t_fit_norm = _normalize_to_dummy_year(t_fit)
            y_fit_arr = np.asarray(y_fit, dtype=float)

            t_fit_norm = pd.to_datetime(t_fit_norm)
            if getattr(t_fit_norm, "tz", None) is not None:
                t_fit_norm = t_fit_norm.tz_localize(None)

            t_fit_list = t_fit_norm.to_pydatetime().tolist()
            y_fit_list = y_fit_arr.tolist()

            fit_src.data = dict(t=t_fit_list, y=y_fit_list)

            fit_rnd.glyph.line_color = color
            try:
                fit_rnd.legend_label = f"Poly {yint}"
            except Exception:
                pass
            fit_rnd.visible = True
            fit_rnd.glyph.line_alpha = 0.25
            fit_rnd.glyph.line_dash = "solid"
        else:
            # not enough points for the requested degree
            fit_src.data = dict(t=[], y=[])
            fit_rnd.visible = False
            fit_rnd.glyph.line_alpha = 0.0
            try:
                fit_rnd.legend_label = None
            except Exception:
                pass

    # axis cosmetics
    if units:
        fig.yaxis.axis_label = units
    if title_prefix:
        selected_count = len(
            set(
                [
                    y
                    for y in years
                    if y not in (None, "", "NaN")
                    and not (isinstance(y, float) and math.isnan(y))
                ]
            )
        )
        fig.title.text = f"{title_prefix} overlays ({selected_count} selected)"
