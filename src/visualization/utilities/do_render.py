# do_render
from pathlib import Path

from src.visualization.core.compute_average import compute_average
from src.visualization.plots.make_map_plot import make_map_plot
from src.visualization.core.dataset import DATASETS, GLACIERS

from bokeh.events import Tap
from src.visualization.actions.on_tap import _on_tap  # new single import




def do_render(w_timerange,
              w_files,
              w_dataset,
              w_status,
              _last,
              w_stat_mean,
              w_stat_max,
              w_stat_min,
              w_stat_ndatapoints,
              plot_pane,
              ts_fig,
              ts_source,
              ts_source_fit=None,
              fit_degree: int = 3,
              ts_fig_dir=None,
              ts_source_dir=None,
              ts_source_dir_fit = None,
              ts_pane_dir=None,
              w_hours=None,
              w_yearly_mode=None,
              w_years=None,
              w_yearly_timerange=None,
              w_alpha_value=None,
              w_glacier_multiplier=None,
              w_glacier_offset=None,
              ts_year_sources=None,
              ts_year_renderers=None,
              ts_dir_year_sources=None,
              ts_dir_year_renderers=None,
              ts_year_fit_sources=None,
              ts_year_fit_renderers=None,
              ts_dir_year_fit_sources=None,
              ts_dir_year_fit_renderers=None,
              ts_glacier_source=None,
              ):
    try:
        selected = [Path(p) for p in w_files.value]
        t_val = None
        if (not w_timerange.disabled) and all(w_timerange.value):
            t_val = w_timerange.value

        hour_list = None
        if w_hours is not None and w_hours.value:
            # values are strings like "00", convert to ints
            hour_list = sorted({int(h) for h in w_hours.value})

        mode = _last.get("data_kind") or DATASETS[w_dataset.value]["mode"]

        avg2d, extent, units, lon, lat, u2d, v2d = compute_average(
            DATASETS,
            selected,
            w_dataset.value,
            mode,  # ✅ pass detected type (wind/temperature/glacier)
            t_val,  # ✅ t_range argument
            hours=hour_list,
        )

        label_base = DATASETS[w_dataset.value]["label"]
        label = f"{label_base} [{units}]" if units else label_base
        title = f"{label_base} — {len(selected)} file(s)"
        mode = _last.get("data_kind") or DATASETS[w_dataset.value]["mode"]

        # Show direction plot only for wind (uv)
        if ts_pane_dir is not None:
            ts_pane_dir.visible = (mode == "wind")

        bkplot = make_map_plot(
            avg2d, extent, title, label, mode,
            lon=lon, lat=lat, u2d=u2d, v2d=v2d, quiver_stride=8,
            glacier_dir=GLACIERS["dir"]
        )

        w_status.object = "Rendered."

        _last["files"] = selected
        _last["ds_key"] = w_dataset.value
        _last["time_range"] = t_val
        _last["hours"] = hour_list
        _last["lon"] = lon
        _last["lat"] = lat
        _last["figure"] = bkplot

        # Cache glacier catalog for 'near a glacier' detection
        try:
            from src.visualization.plots.glacier_overlay import load_glacier_points
            _last["glaciers"] = load_glacier_points(GLACIERS["dir"])
        except Exception as _e:
            _last["glaciers"] = None


        _stat_panes = {"mean": w_stat_mean, "max": w_stat_max, "min": w_stat_min}

        # Detach previous and attach new handler
        bkplot.on_event(Tap, lambda evt: _on_tap(
            evt, w_status, _last,
            ts_source,  # raw series source
            ts_fig,  # figure
            _stat_panes,
            ts_source_fit,  # pass fit source
            fit_degree,  # pass degree
            ts_source_dir,  # direction raw
            ts_source_dir_fit,  # direction fit
            ts_fig_dir,  # direction fig (for y-label/title updates)
            w_stat_ndatapoints=w_stat_ndatapoints,
            w_glacier_multiplier=w_glacier_multiplier,
            w_glacier_offset=w_glacier_offset,

            # --- yearly wiring (added) ---
            yearly_enabled_widget = w_yearly_mode,
            year_fields=w_years,
            alpha_widget = w_alpha_value,
            yearly_window_widget=w_yearly_timerange,
            ts_year_sources = ts_year_sources,
            ts_year_renderers = ts_year_renderers,
            ts_dir_year_sources = ts_dir_year_sources,
            ts_dir_year_renderers = ts_dir_year_renderers,
            ts_year_fit_sources=ts_year_fit_sources,
            ts_year_fit_renderers=ts_year_fit_renderers,
            ts_dir_year_fit_sources=ts_dir_year_fit_sources,
            ts_dir_year_fit_renderers=ts_dir_year_fit_renderers,
            ts_glacier_source=ts_glacier_source,
        ))
        plot_pane.object = bkplot




    except Exception as e:
        print("Error 1: ", e)
        w_status.object = f"**Error:** {e}"
