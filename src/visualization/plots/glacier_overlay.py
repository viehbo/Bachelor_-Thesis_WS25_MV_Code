# glacier_overlay.py
from pathlib import Path
import numpy as np
import xarray as xr

def _read_one_year(fp: Path):

    out = {}
    with xr.open_dataset(fp) as ds:
        var_name = None
        for cand in ("annual_balance", "annual_mass_balance"):
            if cand in ds:
                var_name = cand
                break
        if var_name is None:
            return out  # skip unknown

        v = ds[var_name]
        if "time" in v.dims:
            v = v.mean("time", skipna=True)

        if "name" not in v.dims:
            return out

        names = v["name"].values
        lats = ds["latitude"].values if "latitude" in ds else np.full_like(names, np.nan, dtype=float)
        lons = ds["longitude"].values if "longitude" in ds else np.full_like(names, np.nan, dtype=float)
        vals = v.values

        for nm, lo, la, va in zip(names, lons, lats, vals):
            if not (np.isfinite(lo) and np.isfinite(la) and np.isfinite(va)):
                continue
            out[str(nm)] = (float(lo), float(la), float(va))
    return out


def load_glacier_points(base_dir: Path):

    per_name_vals, per_name_pos = {}, {}

    for fp in sorted(Path(base_dir).glob("mass_balance_*.nc")):
        year_data = _read_one_year(fp)
        for nm, (lo, la, va) in year_data.items():
            per_name_pos.setdefault(nm, (lo, la))
            per_name_vals.setdefault(nm, []).append(va)

    if not per_name_vals:
        return None

    names, lons, lats, avgs = [], [], [], []
    for nm, vals in per_name_vals.items():
        lo, la = per_name_pos[nm]
        names.append(nm); lons.append(lo); lats.append(la)
        avgs.append(float(np.nanmean(np.asarray(vals, dtype=float))))

    lons = np.asarray(lons, dtype=float)
    lats = np.asarray(lats, dtype=float)
    avgs = np.asarray(avgs, dtype=float)
    names = np.asarray(names, dtype=object)

    mag = np.abs(avgs)
    sizes = np.full_like(mag, 12.0, dtype=float) if (not np.isfinite(mag).any() or mag.min()==mag.max()) \
            else 6.0 + 20.0 * (mag - mag.min()) / (mag.max() - mag.min())

    return dict(lon=lons, lat=lats, avg=avgs, size=sizes, name=names)


def add_glacier_layer(p, glacier_points, lon_to_x, lat_to_y):
    if not glacier_points or glacier_points["lon"].size == 0:
        return
    from bokeh.models import ColumnDataSource, HoverTool
    gx = lon_to_x(glacier_points["lon"])
    gy = lat_to_y(glacier_points["lat"])
    cds = ColumnDataSource(dict(
        x=gx, y=gy, lon=glacier_points["lon"], lat=glacier_points["lat"],
        mb=glacier_points["avg"], size=glacier_points["size"], name=glacier_points["name"]
    ))
    r = p.scatter(x='x', y='y',
                  size='size',
                  source=cds,
                  fill_alpha=0.8,
                  line_color="black",
                  line_alpha=0.6,
                  line_width=0.2,
                  )

    r.selection_glyph = r.glyph.clone()
    r.selection_glyph.fill_alpha = 0.8  # clicked cell opacity
    r.selection_glyph.line_alpha = 1
    r.selection_glyph.line_width = 0.8

    r.nonselection_glyph = r.glyph.clone()
    r.nonselection_glyph.fill_alpha = 0.80  # all other cells opacity
    r.nonselection_glyph.line_alpha = 0.60
    r.nonselection_glyph.line_width = 0.2

    p.add_tools(HoverTool(renderers=[r],
        tooltips=[("Glacier", "@name"),
                  ("Mass balance (avg)", "@mb{0.0}"),
                  ("lon", "@lon{0.000}"),
                  ("lat", "@lat{0.000}")
                  ]))

def glacier_series_by_name(base_dir: Path, name: str):
    import re
    import pandas as pd
    vals = []
    times = []

    for fp in sorted(Path(base_dir).glob("mass_balance_*.nc")):
        with xr.open_dataset(fp) as ds:
            var_name = None
            for cand in ("annual_balance", "annual_mass_balance"):
                if cand in ds:
                    var_name = cand; break
            if var_name is None or "name" not in ds[var_name].dims:
                continue

            v = ds[var_name]
            # select glacier by name (coordinate or index)
            try:
                gi = int(np.where(ds["name"].values.astype(object) == name)[0][0])
            except Exception:
                continue
            vv = v.isel(name=gi)
            # average over time if a time dim exists
            if "time" in vv.dims:
                vv = vv.mean("time", skipna=True)

            # choose timestamp
            if "time" in ds.coords and ds["time"].size > 0:
                t = pd.to_datetime(ds["time"].values).mean()
            else:
                # parse 4-digit year from filename
                m = re.search(r"(\d{4})", fp.stem)
                if m:
                    t = pd.Timestamp(int(m.group(1)), 7, 1)
                else:
                    continue

            times.append(t)
            vals.append(float(vv.values))

    if not vals:
        return pd.Series(dtype=float)

    s = pd.Series(vals, index=pd.to_datetime(times)).sort_index()
    s.name = name
    return s
