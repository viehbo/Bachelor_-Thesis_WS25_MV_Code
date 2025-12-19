import xarray as xr
import panel as pn
#from src.visualization.helpers.scan_files import scan_files
from src.visualization.helpers.pick_netcdf_files import pick_netcdf_files

from src.visualization.helpers.load_time_range import load_time_range

from src.visualization.plots.make_lineplot import make_line_plot_1

from src.visualization.core.dataset import DATASETS

from datetime import datetime, timedelta, time
import panel.util as _pn_util
from panel.widgets.slider import DatetimeRangeSlider
from bokeh.models import Range1d, LinearAxis, ColumnDataSource

from pathlib import Path
from src.visualization.helpers.detect_netcdf_kind import detect_kind_for_files


from src.visualization.utilities.set_datetime import safe_value_as_datetime

from src.visualization.utilities.do_render import do_render

from src.visualization.helpers.glacier_secondary_axis import (
    normalize_glacier_series_for_secondary_axis,
    update_glacier_secondary_axis,
)

from src.visualization.core.poly_fit import poly_fit_datetime

from src.visualization.helpers.apply_alpha_to_raw_lines import apply_alpha_to_raw_lines
from src.visualization.helpers.update_poly_fit_only import update_poly_fit_only
from src.visualization.helpers.populate_year_options_from_timerange import populate_year_options_from_timerange

from src.visualization.plots.toggle_yearly_visibility import toggle_yearly_visibility

from src.visualization.helpers.on_yearly_timerange_change import _on_yearly_timerange_change

from src.visualization.plots.yearly_overlays import set_yearly_overlays


# https://panel.holoviz.org/reference/panes/Bokeh.html

# Panel config (serve JS locally, single extension call)

#pn.config.resources = "inline"
pn.extension(design="material", sizing_mode="stretch_width")
pn.extension()

#https://docs.xarray.dev/en/stable/generated/xarray.set_options.html
xr.set_options(use_new_combine_kwarg_defaults=True)





# 1) Patch Panel's util function (for any code path that consults it)
_pn_util.value_as_datetime = safe_value_as_datetime
# 2) Patch the slider class attribute that Panel actually uses
DatetimeRangeSlider._property_conversion = staticmethod(safe_value_as_datetime)



# Widgets
w_dataset = pn.widgets.RadioButtonGroup(
    name="Dataset", options=list(DATASETS.keys()), button_type="primary")

#w_data_dir = pn.widgets.TextInput(name="Data folder", value=str(DATASETS[w_dataset.value]["dir"]))
#w_pattern  = pn.widgets.TextInput(name="Filename pattern", value=DATASETS[w_dataset.value]["pattern"])
#w_scan     = pn.widgets.Button(name="Scan", button_type="primary")
w_pick_files = pn.widgets.Button(name="Select .nc files (OS)", button_type="primary")
w_clear_files = pn.widgets.Button(name="Clear selection", button_type="light")

w_files    = pn.widgets.MultiSelect(name="NetCDF files", options=[], size=12)
w_files_summary = pn.pane.Markdown("**Files:** 0 — **Type:** –", height=30)
w_files.visible = False  # keep it for storage, but do not show it


w_loadtime = pn.widgets.Button(name="Load files and timerange")
# text fields for the click event
w_stat_mean = pn.pane.Markdown("**Mean:** –", height=30)
w_stat_max  = pn.pane.Markdown("**Max:** –",  height=30)
w_stat_min  = pn.pane.Markdown("**Min:** –",  height=30)
w_stat_ndatapoints = pn.pane.Markdown("**Number of datapoints in the range:** -", height=30)
w_fit_degree = pn.widgets.IntInput(name="Poly degree", value=3, start=1, end=10)


# --- Yearly mode UI ----------------------------------------------------------
w_yearly_mode = pn.widgets.Checkbox(name="Yearly mode", value=False)

# One widget for all selected years
# https://docs.bokeh.org/en/latest/docs/reference/models/widgets/inputs.html#bokeh.models.MultiChoice
w_years = pn.widgets.MultiChoice(
    name="Years",
    options=[],      # will be filled from timerange
    value=[],        # selected years
    solid=True,
    width=300,
)
w_years.visible = False  # only when yearly mode is ON


# Alpha controls (affects raw/yearly lines only; fit stays unchanged)
w_alpha_value = pn.widgets.FloatInput(name="Line alpha (0.1)", value=0.35, step=0.05, start=0.0, end=1.0, width=150)
w_set_alpha   = pn.widgets.Button(name="Set alpha", button_type="light")
w_set_alpha.visible = True
w_alpha_value.visible = True





_placeholder_start = datetime(2000, 1, 1)
_placeholder_end   = _placeholder_start + timedelta(days=1)
w_timerange = pn.widgets.DatetimeRangeSlider(
    name="Timeframe",
    start=datetime(2000, 1, 1),
    end=datetime(2000, 1, 1) + timedelta(days=1),
    #value=(_placeholder_start, _placeholder_end),
    disabled=True,
    visible=False,
    sizing_mode="stretch_width",
)


DUMMY_START = datetime(2000, 1, 1)
DUMMY_END   = datetime(2000, 12, 31, 23, 59, 59)

w_yearly_timerange = pn.widgets.DatetimeRangeSlider(
    name="Yearly timeframe (dummy year)",
    start=DUMMY_START,
    end=DUMMY_END,
    value=(DUMMY_START, DUMMY_END),
    disabled=False,
    visible=False,   # shown only when Yearly mode is ON
)



# Hours-of-day filter (populated from the selected files)
w_hours = pn.widgets.MultiChoice(
    name="Daily hours",
    options=[],           # filled by load_time_range()
    value=[],             # default -> all (we'll set in load_time_range)
    solid=True,
    width=300,
)

w_glacier_multiplier = pn.widgets.FloatSlider(
    name="Multiplier",
    start=-10.0, end=10.0, value=10.0, step=0.1,
    orientation="horizontal",
    width=220,
    visible=False,
)

w_glacier_offset = pn.widgets.FloatSlider(
    name="Offset",
    start=-20.0, end=20.0, value=0.0, step=0.1,
    orientation="horizontal",
    width=220,
    visible=False,
)





w_render = pn.widgets.Button(name="Render", button_type="success")
w_set_poly = pn.widgets.Button(name="set ploy degree", button_type="primary", button_style="outline")
w_status = pn.pane.Markdown("", height=40)
w_citetext = pn.pane.Markdown("Cite of the ECMWF", height=40)
w_sampletext = pn.pane.Markdown("Sampletext", height=60)

# Map pane: will be set in do_render()
plot_pane = pn.pane.Bokeh(sizing_mode="stretch_width")

# Main (speed/temperature) plot
(ts_fig, ts_source, ts_source_fit,
 ts_year_sources, ts_year_renderers,
 ts_year_fit_sources, ts_year_fit_renderers) = make_line_plot_1(
    show_fit=True, y_label=""
)
ts_fig.sizing_mode = "stretch_width"


# --- Glacier secondary axis + line (always on main figure) ---
ts_glacier_source = ColumnDataSource(dict(t=[], y=[]))

# Extra y-range for glacier mass balance
ts_fig.extra_y_ranges["glacier"] = Range1d(start=0, end=1)

# Right-side axis for glacier data
ts_glacier_axis = LinearAxis(
    y_range_name="glacier",
    axis_label="Glacier mass balance (mm w.e.)"
)
ts_fig.add_layout(ts_glacier_axis, "right")

# Glacier line, tied to the secondary y-axis
ts_glacier_line = ts_fig.line(
    x="t",
    y="y",
    source=ts_glacier_source,
    line_width=2,
    line_alpha=0.9,
    line_color="red",        # glacier series color
    y_range_name="glacier",
    legend_label="Glacier mass balance",
)



# Direction plot
(ts_fig_dir, ts_source_dir, ts_source_dir_fit,
 ts_dir_year_sources, ts_dir_year_renderers,
 ts_dir_year_fit_sources, ts_dir_year_fit_renderers) = make_line_plot_1(
    show_fit=True, y_label="Direction (°)", x_range=ts_fig.x_range
)


w_yearly_timerange.param.watch(
    lambda e: _on_yearly_timerange_change(
        w_yearly_mode=w_yearly_mode,
        _last=_last,
        w_years=w_years,
        w_yearly_timerange=w_yearly_timerange,
        w_alpha_value=w_alpha_value,
        set_yearly_overlays=set_yearly_overlays,
        ts_fig=ts_fig,
        ts_fig_dir=ts_fig_dir,
        ts_year_sources=ts_year_sources,
        ts_year_renderers=ts_year_renderers,
        ts_year_fit_sources=ts_year_fit_sources,
        ts_year_fit_renderers=ts_year_fit_renderers,
        ts_dir_year_sources=ts_dir_year_sources,
        ts_dir_year_renderers=ts_dir_year_renderers,
        ts_dir_year_fit_sources=ts_dir_year_fit_sources,
        ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
        w_fit_degree=w_fit_degree,
    ),
    "value",
)





# Make the direction figure responsive in width
ts_fig_dir.sizing_mode = "stretch_width"
# Now wrap them in responsive Panel panes
ts_pane = pn.pane.Bokeh(ts_fig, sizing_mode="stretch_width")
ts_pane_dir = pn.pane.Bokeh(ts_fig_dir, sizing_mode="stretch_width")
ts_pane_dir.visible = False  # shown only for kind="uv"


w_yearly_mode.param.watch(lambda e: toggle_yearly_visibility(w_yearly_mode=w_yearly_mode,
                                                             w_years=w_years,
                                                             w_set_alpha=w_set_alpha,
                                                             w_alpha_value=w_alpha_value,
                                                             w_timerange=w_timerange,
                                                             ts_fig=ts_fig,
                                                             ts_source=ts_source,
                                                             ts_source_fit=ts_source_fit,
                                                             ts_fig_dir=ts_fig_dir,
                                                             ts_source_dir=ts_source_dir,
                                                             ts_source_dir_fit=ts_source_dir_fit,
                                                             w_yearly_timerange=w_yearly_timerange,
                                                             ), "value")



def on_pick_files(event=None):
    cfg = DATASETS[w_dataset.value]
    initial_dir = str(cfg["dir"])

    try:
        paths = pick_netcdf_files(initial_dir=initial_dir)
    except Exception as e:
        w_status.object = f"File picker failed: {e}"
        return

    if not paths:
        w_status.object = "No files selected."
        return

    w_files.options = paths
    w_files.value = list(paths)

    # NEW: detect kind + update summary (simple, immediate feedback)
    try:
        kind, _ = detect_kind_for_files([Path(p) for p in w_files.value])
        w_files_summary.object = f"**Files:** {len(paths)} — **Type:** {kind}"
        _last["data_kind"] = kind
    except Exception as e:
        # Keep selection, but show error and mark kind unknown
        _last["data_kind"] = None
        w_files_summary.object = f"**Files:** {len(paths)} — **Type:** (error)"
        w_status.object = f"**Error:** {e}"
        return

    w_status.object = f"Selected {len(paths)} file(s). Click **Load files and timerange** → **Render**."



w_pick_files.on_click(on_pick_files)

def on_clear_files(event=None):
    w_files.options = []
    w_files.value = []
    w_files_summary.object = "**Files:** 0 — **Type:** –"
    _last["data_kind"] = None
    w_status.object = "Cleared file selection."


w_clear_files.on_click(on_clear_files)


def on_dataset_change(event=None):
    # reset timeframe slider to placeholder + disable (keep your existing logic)
    w_timerange.start = _placeholder_start
    w_timerange.end = _placeholder_end
    w_timerange.value = (_placeholder_start, _placeholder_end)
    w_timerange.disabled = True
    w_timerange.visible = not bool(w_yearly_mode.value)

    # Clear current file selection; dataset changed
    w_files.options = []
    w_files.value = []

    cfg = DATASETS[w_dataset.value]
    w_status.object = f"Dataset set to '{w_dataset.value}'. Click **Select .nc files (OS)** to choose files."


def _update_glacier_overlay_from_sliders(event=None):
    s_raw = _last.get("glacier_series_raw")
    if s_raw is None or len(s_raw) == 0:
        return

    mult = float(w_glacier_multiplier.value)
    off  = float(w_glacier_offset.value)

    s_norm = normalize_glacier_series_for_secondary_axis(s_raw, multiplier=mult, offset=off)
    update_glacier_secondary_axis(s_norm=s_norm, ts_glacier_source=ts_glacier_source, ts_fig=ts_fig)


w_glacier_multiplier.param.watch(_update_glacier_overlay_from_sliders, "value")
w_glacier_offset.param.watch(_update_glacier_overlay_from_sliders, "value")


w_dataset.param.watch(lambda e: on_dataset_change(), 'value')
#
# comment the scan button
# w_scan.on_click(lambda e: scan_files(w_data_dir, w_pattern, w_files, w_status))
w_loadtime.on_click(lambda e: load_time_range(
    w_files, w_status, w_timerange, w_hours,
    w_files_summary=w_files_summary,
    state=_last,
))


w_render.on_click(lambda e: do_render(w_timerange=w_timerange,
                                      w_hours=w_hours,
                                      w_files=w_files,
                                      w_status=w_status,
                                      w_dataset=w_dataset,
                                      _last=_last,
                                      w_stat_mean=w_stat_mean,
                                      w_stat_max=w_stat_max,
                                      w_stat_min=w_stat_min,
                                      plot_pane=plot_pane,
                                      ts_fig=ts_fig,
                                      ts_source=ts_source,
                                      ts_source_fit=ts_source_fit,
                                      fit_degree=w_fit_degree.value,
                                      ts_fig_dir=ts_fig_dir,
                                      ts_source_dir=ts_source_dir,
                                      ts_source_dir_fit=ts_source_dir_fit,
                                      ts_pane_dir=ts_pane_dir,
                                      w_yearly_mode=w_yearly_mode,
                                      w_years=w_years,
                                      w_alpha_value=w_alpha_value,
                                      w_glacier_multiplier=w_glacier_multiplier,
                                      w_glacier_offset=w_glacier_offset,
                                      ts_year_sources=ts_year_sources,
                                      ts_year_renderers=ts_year_renderers,
                                      ts_dir_year_sources=ts_dir_year_sources,
                                      ts_dir_year_renderers=ts_dir_year_renderers,
                                      ts_year_fit_sources=ts_year_fit_sources,
                                      ts_year_fit_renderers=ts_year_fit_renderers,
                                      ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                                      ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
                                      ts_glacier_source=ts_glacier_source,
                                      w_stat_ndatapoints=w_stat_ndatapoints,
                                      ))

# wire the button
w_set_poly.on_click(lambda e: update_poly_fit_only(w_fit_degree=w_fit_degree,
                                                   poly_fit_datetime=poly_fit_datetime,
                                                   ts_source=ts_source,
                                                   ts_source_fit=ts_source_fit,
                                                   ts_source_dir=ts_source_dir,
                                                   ts_source_dir_fit=ts_source_dir_fit, ))

w_set_alpha.on_click(lambda e: apply_alpha_to_raw_lines(w_alpha_value=w_alpha_value,
                                                        ts_fig=ts_fig,
                                                        ts_source=ts_source,
                                                        ts_year_renderers=ts_year_renderers,
                                                        ts_fig_dir=ts_fig_dir,
                                                        ts_source_dir=ts_source_dir,
                                                        ts_dir_year_renderers=ts_dir_year_renderers, ))
# Recompute year options whenever the time range changes
w_timerange.param.watch(lambda e: populate_year_options_from_timerange(w_timerange, w_years), "value")

# call dataset change
on_dataset_change()

_last = {
    "files": [],
    "ds_key": None,
    "time_range": None,
    "lon": None,   # 1D lon centers
    "lat": None,   # 1D lat centers
    "figure": None,
    "glaciers": None,  # cached glacier points for click-near-glacier
}



glacier_slider_col = pn.Column(
    w_glacier_multiplier,
    w_glacier_offset,
    width=110,
    sizing_mode="fixed",
    margin=(0, 0, 0, 10),
)


controls = pn.Column(
    #pn.Row(w_files),
    pn.Row(w_hours),
    pn.Row(w_alpha_value, w_set_alpha),
    pn.Row(w_fit_degree, w_set_poly),
    pn.Row(w_yearly_mode),
    w_yearly_timerange,
    w_years,
    pn.Row(w_render),
    w_status,
    pn.Row(w_stat_mean, w_stat_max, w_stat_min),
    w_stat_ndatapoints,
    w_sampletext,
    w_citetext,
    width=420,
    sizing_mode="stretch_height"
)


top_section_controls = pn.Column(
    #pn.Row(w_dataset),
    pn.Row(w_pick_files, w_files_summary, w_loadtime, w_clear_files),
)


glacier_slider_left = pn.Column(
    "Glacier overlay",
    w_glacier_multiplier,
    w_glacier_offset,
    width=250,
    sizing_mode="fixed",
    styles={
        "border": "1px solid #ddd",
        "padding": "10px",
        "background": "white",
    },
)




ts_with_left_sliders = pn.Row(
    glacier_slider_left,
    ts_pane,
    sizing_mode="stretch_width",
)

plots = pn.Column(
    plot_pane,
    ts_with_left_sliders,
    ts_pane_dir,
    sizing_mode="stretch_both",
    min_height=820,
)



# TOP
top_slider = pn.Row(
    w_timerange,
    sizing_mode="stretch_width"
)


# PAGE LAYOUT
template = pn.template.MaterialTemplate(
    title="Glacier Project",
    header_background="#014f86"
)

template.main.append(
    pn.Column(
        top_section_controls,
        top_slider,
        pn.Row(
            controls,
            plots,
            sizing_mode="stretch_both"
        ),
        sizing_mode="stretch_both"
    )
)

template.servable()


if __name__ == "__main__":
    pn.serve(template, title="Atmos — Time-Subset Average", show=True, autoreload=False)
