from pathlib import Path

# Setup
project_root = Path(__file__).resolve().parents[3]
BASE = project_root / "data" / "interim"
GLACIERS = {
    "dir": BASE / "mass_balance_netcdf",
    "pattern": "mass_balance_*.nc",   # used only for consistency
}

DATASETS = {
    "Wind": {
        "dir": BASE / "wind_10m_netcdf",
        "pattern": "wind_10m_*.nc",
        "mode": "wind",
        "candidates": {"u": ["u10", "u_10m", "u"], "v": ["v10", "v_10m", "v"]},
        "label": "Avg. wind speed",
    },
    "Temperature": {
        "dir": BASE / "2m_temperature_netcdf_C",
        "pattern": "*.nc",
        "mode": "temperature",
        "candidates": {"x": ["t2m", "2t", "t", "temperature"]},
        "label": "Avg. temperature",
    },
}