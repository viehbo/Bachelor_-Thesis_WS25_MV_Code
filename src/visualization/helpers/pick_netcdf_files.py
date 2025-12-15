from __future__ import annotations

from pathlib import Path
from typing import List, Optional


def pick_netcdf_files(initial_dir: Optional[str] = None) -> List[str]:
    """
    Open a native OS file picker (Explorer/Finder) and return selected NetCDF file paths.

    Notes:
    - Works when running locally with a GUI available.
    - Will not work on headless servers (no display).
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        raise RuntimeError(
            "tkinter is not available. Install/enable Tk on your system or use a Panel FileInput/FileSelector alternative."
        ) from e

    root = tk.Tk()
    root.withdraw()           # hide main window
    root.attributes("-topmost", True)

    init = None
    if initial_dir:
        p = Path(initial_dir).expanduser()
        if p.exists():
            init = str(p)

    paths = filedialog.askopenfilenames(
        title="Select NetCDF (.nc) files",
        initialdir=init,
        filetypes=[
            ("NetCDF files", "*.nc"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()
    return list(paths)
