from pathlib import Path

def scan_files(w_data_dir, w_pattern, w_files, w_status):
    folder = Path(w_data_dir.value).expanduser()
    files = sorted(folder.glob(w_pattern.value.strip() or "*.nc"))
    w_files.options = [str(p) for p in files]
    w_status.object = (
        f"Found {len(files)} file(s). Select some, then **Load time range** → **Render**."
        if files else "No files found. Check folder/pattern."
    )