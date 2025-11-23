
def _find_var(ds, candidates):
    names = {k.lower(): k for k in ds.variables}
    for cand in candidates:
        for var in names:
            if cand in var:
                return names[var]
    raise KeyError(f"None of {candidates} found in dataset vars: {list(ds.variables)}")
