# temperature_slider.py
from dash import html, dcc

def temperature_slider_component(min_year_temp, max_year_temp):
    return html.Div(
        [
            html.Label('Temperature Dataset Year Range'),
            dcc.RangeSlider(
                id='temp-year-range-slider',
                min=min_year_temp,
                max=max_year_temp,
                value=[min_year_temp, max_year_temp],
                marks={str(y): str(y) for y in range(min_year_temp, max_year_temp + 1, 5)},
                step=1,
                allowCross=False
            )
        ],
        id='temp-slider-div',
        style={'margin-top': '20px', 'display': 'block'})
