# --- tile helper (WMTS fallback kept robust) ---------------------------------
XYZTileSource = None
WMTSTileSource = None
try:
    from bokeh.models.tiles import XYZTileSource as _XYZ  # Bokeh 3.4+
    XYZTileSource = _XYZ
except Exception:
    pass
try:
    from bokeh.models.tiles import WMTSTileSource as _WMTS
    WMTSTileSource = _WMTS
except Exception:
    pass

def add_tiles(p):
    # Fallback: WMTS (commonly available)
    if WMTSTileSource is not None:
        try:
            osm_wmts = WMTSTileSource(
                url="https://tile.openstreetmap.org/{Z}/{X}/{Y}.png",
                attribution="© OpenStreetMap contributors"
            )
            tr = p.add_tile(osm_wmts)
            tr.level = "underlay"
            return True
        except Exception:
            pass
    return False