
from bokeh.models import DatetimeTickFormatter
from bokeh.plotting import figure as bk_figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10


PALETTE10 = list(Category10[10])

def make_line_plot_1(title="Clicked point: time series", y_label="", x_label="Time", show_fit=True, x_range=None):

    kwargs = dict(height=260,
                  width=900,
                  x_axis_type="datetime",
                  tools="pan,wheel_zoom,reset,save",
                  )

    if x_range is not None:
        kwargs["x_range"] = x_range

    fig = bk_figure(**kwargs)

    src_raw = ColumnDataSource(dict(t=[], y=[]))
    src_fit = ColumnDataSource(dict(t=[], y=[])) if show_fit else None

    # set parameter for the basic from-to lineplot
    line_raw = fig.line(x="t",
                        y="y",
                        source=src_raw,
                        line_width=0.5,
                        line_alpha=0.75,
                        line_color='blue',
                        legend_label="Observed",
                        )

    # set the parameter for the fitted polynomial
    if show_fit:
        fig.line(x="t",
                 y="y",
                 source=src_fit,
                 line_width=0.8,
                 line_alpha=1,
                 line_color='black',
                 #line_dash="solid",
                 legend_label="Poly fit",
                 )

    hover = HoverTool(
        tooltips=[("Date", "@t{%F %H:%M}"), (y_label or "Value", "@y{0.00}")],
        formatters={"@t": "datetime", "@y": "numeral"},
        mode="vline",
        renderers=[line_raw],
    )
    fig.add_tools(hover)

    fig.title.text = title
    fig.yaxis.axis_label = y_label
    fig.xaxis.axis_label = x_label

    fig.legend.click_policy = "hide"
    fig.xaxis.formatter = DatetimeTickFormatter(hours="%Y-%m-%d %H:%M", days="%Y-%m-%d")

    yearly_sources = []
    yearly_renderers = []
    yearly_fit_sources = []
    yearly_fit_renderers = []

    for k in range(10):
        # raw yearly line
        ys = ColumnDataSource(dict(t=[], y=[]))
        r = fig.line(
            x="t", y="y", source=ys,
            line_width=0.6,
            line_alpha=0.0,
            visible=False,
            muted_alpha=0.1,
        )
        yearly_sources.append(ys)
        yearly_renderers.append(r)

        # dashed poly-fit for that yearly line
        yfs = ColumnDataSource(dict(t=[], y=[]))
        rf = fig.line(
            x="t", y="y", source=yfs,
            line_width=2,
            line_alpha=0.0,
            visible=False,
            #line_dash="solid",
            muted_alpha=0.1,
        )
        yearly_fit_sources.append(yfs)
        yearly_fit_renderers.append(rf)

    return (fig, src_raw, src_fit,
            yearly_sources, yearly_renderers,
            yearly_fit_sources, yearly_fit_renderers)



