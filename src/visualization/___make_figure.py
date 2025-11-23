
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from matplotlib.figure import Figure

def make_figure(avg2d, extent, title, cbar_label, mode,
                lon=None, lat=None, u2d=None, v2d=None, quiver_stride=8):
    print("_________________  make figure called ___________________________________")
    proj = ccrs.PlateCarree()
    fig = Figure(figsize=(9.5, 6.0))
    #FigureCanvas(fig)
    ax = fig.add_subplot(111, projection=proj)

    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), alpha=0.25)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"), linewidth=0.4, alpha=0.4)

    left, right, bottom, top = extent
    ax.set_extent([left - 1, right + 1, bottom - 1, top + 1], crs=proj)

    if mode == "scalar":
        # TEMPERATURE: same look, but blue→red
        im = ax.imshow(
            avg2d,
            origin="upper",
            extent=extent,
            transform=proj,
            interpolation="nearest",
            cmap="coolwarm",
            #hovercolor="red",
        )
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label(cbar_label)
    else:
        # WIND: arrows colored by speed (avg2d), oriented by mean(u,v)
        if any(x is None for x in (lon, lat, u2d, v2d)):
            raise ValueError("Wind plot requires lon, lat, u2d, v2d.")

        # Build 2D lon/lat grid and decimate to avoid too many arrows
        LON, LAT = np.meshgrid(lon, lat)
        sl = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))

        q = ax.quiver(
            LON[sl], LAT[sl],
            u2d[sl], v2d[sl],
            avg2d[sl],                # color by speed
            transform=proj,
            pivot="middle",
            cmap="viridis",           # pick any colormap you like
            scale=None,               # auto scaling (data units)
            width=0.002,              # arrow thickness
        )
        cb = fig.colorbar(q, ax=ax, shrink=0.9, pad=0.02)
        cb.set_label(cbar_label if cbar_label else "Wind speed")

    ax.set_title(title)
    fig.tight_layout()

    # stash metadata for hover/click callbacks
    ax._grid_lon = lon
    ax._grid_lat = lat
    ax._grid_vals = avg2d
    ax._mode = mode  # <— add back

    if mode == "wind":
        # Keep decimated lon/lat + values for the actually drawn arrows
        LON, LAT = np.meshgrid(lon, lat)
        sl = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
        ax._quiver_lon = LON[sl].ravel()
        ax._quiver_lat = LAT[sl].ravel()
        ax._quiver_vals = avg2d[sl].ravel()

    # --- reusable annotation (hidden until hover hits data) ---
    ax._hover_annot = ax.annotate(
        "",
        xy=(0, 0), xytext=(10, 10), textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->")
    )
    ax._hover_annot.set_visible(False)

    return fig