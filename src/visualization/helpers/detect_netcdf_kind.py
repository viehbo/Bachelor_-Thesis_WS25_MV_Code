from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, List, Optional
import xarray as xr


# Keep this intentionally simple and robust:
# - We only inspect variable names (no heavy reads)
# - We classify as "wind" if we can find both u and v
# - We classify as "temperature" if we can find a temperature-like variable
WIND_U_CANDIDATES = ("u", "u10", "u100", "10u", "u_component_of_wind", "eastward_wind")
WIND_V_CANDIDATES = ("v", "v10", "v100", "10v", "v_component_of_wind", "northward_wind")

TEMP_CANDIDATES = (
    "t", "t2m", "2t", "tas",
    "temperature", "air_temperature", "temp",
    "temperature_2m", "2m_temperature", "t_2m"
)



def _has_any(ds: xr.Dataset, names: tuple[str, ...]) -> Optional[str]:
    vars_lower = {k.lower(): k for k in ds.data_vars.keys()}
    for n in names:
        if n.lower() in vars_lower:
            return vars_lower[n.lower()]
    return None


def detect_kind_for_one_file(path: Path) -> str:
    """
    Return 'wind' or 'temperature' based on variables in a single NetCDF file.
    Raises ValueError if neither pattern matches or if it's ambiguous.

    Important: decode_cf=False avoids CF-decoding errors (e.g., 'step') when we only need var names.
    """
    try:
        with xr.open_dataset(path, decode_cf=False) as ds:
            u = _has_any(ds, WIND_U_CANDIDATES)
            v = _has_any(ds, WIND_V_CANDIDATES)
            t = _has_any(ds, TEMP_CANDIDATES)
    except Exception as e:
        raise ValueError(f"Cannot open {path.name} for kind detection: {e}") from e

    is_wind = (u is not None) and (v is not None)
    is_temp = (t is not None)

    if is_wind and not is_temp:
        return "wind"
    if is_temp and not is_wind:
        return "temperature"

    if is_wind and is_temp:
        raise ValueError(
            f"Ambiguous variables in {path.name}: found wind (u/v) and temperature-like variables."
        )

    raise ValueError(
        f"Cannot classify {path.name}: no wind u/v variables and no temperature-like variable found."
    )



def detect_kind_for_files(paths: Iterable[Path]) -> Tuple[str, List[Path]]:
    """
    Detect kind for all files and ensure consistency.
    Returns (kind, paths_list).
    Raises ValueError if inconsistent or unclassifiable.
    """
    paths_list = list(paths)
    if not paths_list:
        raise ValueError("No files selected.")

    kinds = {}
    for p in paths_list:
        k = detect_kind_for_one_file(p)
        kinds.setdefault(k, []).append(p)

    if len(kinds) != 1:
        # produce a compact, useful error
        parts = []
        for k, ps in kinds.items():
            parts.append(f"{k}: {len(ps)} file(s) (e.g. {ps[0].name})")
        raise ValueError("Selected files are not all the same kind. " + " | ".join(parts))

    only_kind = next(iter(kinds.keys()))
    return only_kind, paths_list
