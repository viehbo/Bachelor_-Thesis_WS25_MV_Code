import pandas as pd


def populate_year_options_from_timerange(w_timerange, w_years):
    """
    Update the options of the MultiChoice `w_years` based on the main
    timeframe slider `w_timerange`.

    - Uses the start/end years of the slider.
    - Fills `w_years.options` with the list of years.
    - Keeps already-selected years if they are still valid.
    """

    try:
        start, end = w_timerange.value
    except Exception:
        # Slider not initialized yet
        return

    if not (start and end):
        return

    # Normalize and make sure start <= end
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if end < start:
        start, end = end, start

    years = list(range(start.year, end.year + 1))

    # Update options
    w_years.options = years

    # Keep existing selections but drop anything outside the new range
    if w_years.value:
        w_years.value = [y for y in w_years.value if y in years]
