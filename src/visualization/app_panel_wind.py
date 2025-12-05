import xarray as xr
import panel as pn
from src.visualization.helpers.scan_files import scan_files
from src.visualization.helpers.load_time_range import load_time_range

from src.visualization.plots.make_lineplot import make_line_plot_1

from src.visualization.core.dataset import DATASETS

from datetime import datetime, timedelta, time
import panel.util as _pn_util
from panel.widgets.slider import DatetimeRangeSlider
from bokeh.models import Range1d, LinearAxis, ColumnDataSource



from src.visualization.utilities.set_datetime import safe_value_as_datetime

from src.visualization.utilities.do_render import do_render

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

w_data_dir = pn.widgets.TextInput(name="Data folder", value=str(DATASETS[w_dataset.value]["dir"]))
w_pattern  = pn.widgets.TextInput(name="Filename pattern", value=DATASETS[w_dataset.value]["pattern"])
w_scan     = pn.widgets.Button(name="Scan", button_type="primary")
w_files    = pn.widgets.MultiSelect(name="NetCDF files", options=[], size=12)
w_loadtime = pn.widgets.Button(name="Load files and timerange")
# text fields for the click event
w_stat_mean = pn.pane.Markdown("**Mean:** –", height=30)
w_stat_max  = pn.pane.Markdown("**Max:** –",  height=30)
w_stat_min  = pn.pane.Markdown("**Min:** –",  height=30)

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
w_alpha_value = pn.widgets.FloatInput(name="Line alpha (0..1)", value=0.35, step=0.05, start=0.0, end=1.0, width=150)
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



w_hour_start = pn.widgets.TimePicker(
    name="Daily start time",
    value=time(0, 0),
    width=150
)

w_hour_end = pn.widgets.TimePicker(
    name="Daily end time",
    value=time(23, 59),
    width=150
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


def on_dataset_change(event=None):
    cfg = DATASETS[w_dataset.value]
    # Update folder and pattern from DATASETS config
    w_data_dir.value = str(cfg["dir"])
    w_pattern.value  = cfg["pattern"]

    # Reset timeframe slider to placeholder + disable
    w_timerange.start = _placeholder_start
    w_timerange.end = _placeholder_end
    w_timerange.value = (_placeholder_start, _placeholder_end)
    w_timerange.disabled = True
    # only show when NOT in yearly mode
    w_timerange.visible = not bool(w_yearly_mode.value)

    # Automatically scan all files in the folder/pattern
    scan_files(w_data_dir, w_pattern, w_files, w_status)

    # Optionally auto-select all found files
    if w_files.options:
        # options are plain strings (paths), so this is safe
        w_files.value = list(w_files.options)


w_dataset.param.watch(lambda e: on_dataset_change(), 'value')
#
# comment the scan button
# w_scan.on_click(lambda e: scan_files(w_data_dir, w_pattern, w_files, w_status))
w_loadtime.on_click(lambda e: load_time_range(w_files, w_status, w_timerange))
w_render.on_click(lambda e: do_render(w_timerange=w_timerange,
                                      w_hour_start=w_hour_start,
                                      w_hour_end=w_hour_end,
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
                                      ts_year_sources=ts_year_sources,
                                      ts_year_renderers=ts_year_renderers,
                                      ts_dir_year_sources=ts_dir_year_sources,
                                      ts_dir_year_renderers=ts_dir_year_renderers,
                                      ts_year_fit_sources=ts_year_fit_sources,
                                      ts_year_fit_renderers=ts_year_fit_renderers,
                                      ts_dir_year_fit_sources=ts_dir_year_fit_sources,
                                      ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
                                      ts_glacier_source=ts_glacier_source,
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


# left control column ~1/4 width
controls = pn.Column(
    #pn.Row(w_dataset),
    #pn.Row(w_data_dir, w_pattern),
    #pn.Row(w_loadtime),

    #pn.Row(
    #    w_timerange,
    #    sizing_mode="stretch_width"),

    # w_timerange,

    pn.Row(w_hour_start, w_hour_end),
    pn.Row(w_alpha_value, w_set_alpha),
    pn.Row(w_fit_degree, w_set_poly),

    # --- Yearly UI block
    pn.Row(w_yearly_mode),
    w_yearly_timerange,
    #pn.Row(w_year_1, w_year_2, w_year_3, w_year_4, w_year_5),
    #pn.Row(w_year_6, w_year_7, w_year_8, w_year_9, w_year_10),
    w_years,

    pn.Row(w_render),
    w_status,
    pn.Row(w_stat_mean, w_stat_max, w_stat_min),
    w_sampletext,
    w_citetext,

    width=420,                   # << keeps controls ≤ 1/4 screen
    sizing_mode="stretch_height"
)

top_section_controls = pn.Column(
    pn.Row(w_dataset),
    pn.Row(w_data_dir, w_pattern, w_loadtime),
    #pn.Row(w_loadtime),

    width=420,  # << keeps controls ≤ 1/4 screen
    #sizing_mode="stretch_height",

    #pn.Row(
    #    w_timerange,
    #    sizing_mode="stretch_width"),

)


# --- RIGHT PLOT COLUMN (~3/4 width) ---
plots = pn.Column(
    plot_pane,
    ts_pane,
    ts_pane_dir,
    sizing_mode="stretch_both",
    min_height=820
)


# --- TOP: FULL-WIDTH TIME SLIDER ---
top_slider = pn.Row(
    w_timerange,
    sizing_mode="stretch_width"
)


# --- PAGE LAYOUT (top slider + left controls + right plots) ---
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
