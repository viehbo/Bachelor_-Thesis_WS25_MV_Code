from dash import dcc, html

def dropdown_menu(options):
    return html.Div([
            html.Label("Select Dataset:"),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=options,
                value='temp',  # default
                clearable=False,
                style={'width': '300px'}
            )
        ], style={'margin-bottom': '20px'})
