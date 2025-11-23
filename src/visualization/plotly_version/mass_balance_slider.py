# mass_balance_slider.py
from dash import html, dcc

def mass_balance_slider_component(min_year_mb, max_year_mb):
    return html.Div(
        [
            html.Label('Mass Balance Dataset Year Range'),
            dcc.RangeSlider(
                id='mass_balance-range-slider',
                min=min_year_mb,
                max=max_year_mb,
                value=[min_year_mb, max_year_mb],
                marks={str(y): str(y) for y in range(min_year_mb, max_year_mb + 1, 5)},
                step=1,
                allowCross=False
            )
        ],
        style={'margin-top': '20px'}
    )