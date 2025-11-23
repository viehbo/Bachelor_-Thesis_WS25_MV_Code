# wind_controls.py
from dash import html, dcc

def wind_controls_component(min_year_wind, max_year_wind, unique_months, unique_days, unique_hours):

    return html.Div(
        [
            html.Label("Wind Dataset Year Range"),
            dcc.RangeSlider(
                id='wind-year-range-slider',
                min=min_year_wind,
                max=max_year_wind,
                value=[min_year_wind, max_year_wind],
                marks={str(y): str(y) for y in range(min_year_wind, max_year_wind + 1, 5)},
                step=1,
                allowCross=False
            ),

            html.Div([
                html.Label("Select Months:"),
                dcc.Checklist(
                    id='wind-month-checklist',
                    options=[{'label': str(m), 'value': m} for m in unique_months],
                    value=[],
                    inline=True
                ),
                dcc.Checklist(
                    id='select-all-months',
                    options=[{'label': 'Select ALL months', 'value': 'all'}],
                    value=[]
                )
            ], style={'margin-top': '10px'}),

            html.Div([
                html.Label("Select Days:"),
                dcc.Checklist(
                    id='wind-day-checklist',
                    options=[{'label': str(d), 'value': d} for d in unique_days],
                    value=[],
                    inline=True
                ),
                dcc.Checklist(
                    id='select-all-days',
                    options=[{'label': 'Select ALL days', 'value': 'all'}],
                    value=[]
                )
            ], style={'margin-top': '10px'}),

            html.Div([
                html.Label("Select Hours:"),
                dcc.Checklist(
                    id='wind-hour-checklist',
                    options=[{'label': str(h), 'value': h} for h in unique_hours],
                    value=[],
                    inline=True
                ),
            ], style={'margin-top': '10px'}),
        ],
        id='wind-controls',
        style={'display': 'none', 'margin-top': '20px'}
    )
