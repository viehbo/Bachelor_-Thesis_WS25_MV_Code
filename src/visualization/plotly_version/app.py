import pandas as pd
from pathlib import Path
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Output, Input, State
from dropdown_menu import dropdown_menu
from src.visualization.plotly_version.wind_trace import create_wind_trace_mapbox
from temperature_slider import temperature_slider_component
from mass_balance_slider import mass_balance_slider_component
from wind_controls import wind_controls_component
from mass_balance_trace import create_mass_balance_trace_mapbox, create_mass_balance_trace_geo
from temperature_trace import create_temperature_trace_mapbox, create_temperature_trace_geo
from wind_trace import create_wind_trace_geo
from temperature_lineplot import temperature_lineplot_trace
from wind_lineplot_vector import wind_lineplot_magnitude_trace, wind_lineplot_direction_trace
from mass_balance_lineplot import mass_balance_lineplot_trace
#from text_annotation import text_annotation

# because of the problem with scattergeo and scattermapbox here are the options
is_scattermapbox = False

# Idea with .parent.parent comes from ChatGPT the plain use of Path. dos not work cross-platform
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

#temperature_path = 'Thesis_Project_Glacier\data\processed\temperature.csv'
temperature_path = ROOT_DIR / 'data' / 'processed' / 'temperature_yearly.csv'
mass_balance_path = ROOT_DIR / 'data' / 'processed' / 'mass_balance_mean.csv'
wind_data_path = ROOT_DIR / 'data' / 'processed' / 'wind_yearly.csv'

# Convert to absolute path
absolute_temperature_path = temperature_path.resolve()


alps_df = pd.read_csv(temperature_path)
# casting to int is already outdated for the recent version but still works
alps_df['date'] = alps_df['date'].astype(int)

mass_balance_df = pd.read_csv(mass_balance_path)

# wind date
wind_data_df = pd.read_csv(wind_data_path)
# need to split the dat eto use it for the checkboxes
wind_data_df['date'] = pd.to_datetime(wind_data_df['date'])
wind_data_df['year'] = wind_data_df['date'].dt.year
wind_data_df['month'] = wind_data_df['date'].dt.month
wind_data_df['day'] = wind_data_df['date'].dt.day
wind_data_df['hour'] = wind_data_df['date'].dt.hour

# get the time range off the datasets
min_year_temp = alps_df['date'].min()
max_year_temp = alps_df['date'].max()
min_year_mb = mass_balance_df['date'].min()
max_year_mb = mass_balance_df['date'].max()
min_year_wind = wind_data_df['year'].min()
max_year_wind = wind_data_df['year'].max()

# get the overall min/max
min_date = min(min_year_temp, min_year_mb, min_year_wind)
max_date = max(max_year_temp, max_year_mb, max_year_wind)




# Sort unique months, days, hours for checklists to prevent zick-zack linepolts
unique_months = sorted(wind_data_df['month'].unique())
unique_days = sorted(wind_data_df['day'].unique())
unique_hours = sorted(wind_data_df['hour'].unique())

# default values for the click data to visualize smth
clicked_point = {'lon_wind': 11.999999999999982, 'lat_wind': 47.39999999999998,
                 'lon_temp': 11.999999999999982, 'lat_temp': 47.39999999999998, 'text': 'PASTERZE'}


# init Dash app | basic structure taken from https://dash.plotly.com/tutorial
app = Dash(__name__)

# Layout
app.layout = html.Div([

    # dataset dropdown (temp or wind only)
    dropdown_menu(options=[{'label': 'Temperature', 'value': 'temp'},{'label': 'Wind', 'value': 'wind'}]),

    dcc.Graph(id='main-map'),

    # output of the click event | basic structure from Div containers was created with ChatGPT and adapted
    html.Div([
        html.Label('Clicked Data:'),
        html.Div(id='clicked-data-output')
    ]),

    # legal
    html.Div([
        html.Div('The results contain modified Copernicus Climate '
                 'Change Service information 2020. Neither the European Commission nor ECMWF is responsible for any '
                 'use that may be made of the Copernicus information or data it contains.')],
        style={'position': 'absolute',
               'margin-left': '1320px',
               'margin-right': '40px',
               'margin-top': '-300px',
               'border': '1px solid black',
               'fontSize': '9px',
               }),

    # Sliders and check boxes
    temperature_slider_component(min_year_temp, max_year_temp),
    mass_balance_slider_component(min_year_mb, max_year_mb),
    wind_controls_component(min_year_wind, max_year_wind, unique_months, unique_days, unique_hours),
])

#Show/Hide the temperature slider when user picks "Temperature" or "Wind"
# callbacks are created with the help of ChatGPT (debugging) seems a bit unintuitive to me...
@app.callback(
    Output('temp-slider-div', 'style'),
    Input('dataset-dropdown', 'value')
)
def toggle_temp_slider(selected_dataset):
    if selected_dataset == 'temp':
        return {'margin-top': '20px', 'display': 'block'}
    else:
        return {'display': 'none'}


# Show/Hide the wind controls when "Wind" is selected
@app.callback(
    Output('wind-controls', 'style'),
    Input('dataset-dropdown', 'value')
)
def show_hide_wind_controls(selected_dataset):
    if selected_dataset == 'wind':
        return {'display': 'block', 'margin-top': '20px'}
    else:
        return {'display': 'none'}

# Select ALL months for wind
@app.callback(
    Output('wind-month-checklist', 'value'),
    Input('select-all-months', 'value'),
    State('wind-month-checklist', 'value')
)
def select_all_months(select_all, current_months):
    if 'all' in select_all:
        return unique_months
    else:
        return current_months

# "Select ALL days" for wind
@app.callback(
    Output('wind-day-checklist', 'value'),
    Input('select-all-days', 'value'),
    State('wind-day-checklist', 'value')
)
def select_all_days(select_all, current_days):
    if 'all' in select_all:
        return unique_days
    else:
        return current_days

# main callback
@app.callback(
    Output('main-map', 'figure'),
    [
        Input('dataset-dropdown', 'value'),
        Input('temp-year-range-slider', 'value'),
        Input('mass_balance-range-slider', 'value'),
        Input('wind-year-range-slider', 'value'),
        Input('wind-month-checklist', 'value'),
        Input('wind-day-checklist', 'value'),
        Input('wind-hour-checklist', 'value'),
    ],
    State('main-map', 'figure')
)
def update_map(
    selected_dataset,
    temp_year_range,
    mb_year_range,
    wind_year_range,
    wind_months,
    wind_days,
    wind_hours,
    current_figure
):
    # extract the current state and reuse is to do not reset pan and zoom
    # The structure for getting current figure values was given by ChatGPT
    current_geo = current_figure.get('layout', {}).get('geo', {}) if current_figure else {}
    current_layout = current_figure.get('layout', {}) if current_figure else {}

    # how to apply rowspan and add a secondary y-axis was implemented with ChatGPT
    if is_scattermapbox:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.09,
            vertical_spacing=0.09,
            print_grid=True,
            specs=[
                [{"rowspan": 2, "type": "mapbox"}, {"type": "xy", "secondary_y": True}],
                [None, {"type": "xy", "secondary_y": True}]]
            )
    else:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.09,
            vertical_spacing=0.09,
            print_grid=True,
            specs=[
                [{"rowspan": 2, "type": "geo"}, {"type": "xy", "secondary_y": True}],
                [None, {"type": "xy", "secondary_y": True}]]
            )
    '''
    Fix for the disappearing map:
    Problem: After one update loop, switch between temp and wind and back, the map is not been rendered any more.
    Fix: The two line plots are not longer in the if branch any more, so there are always rendered immediately.
        I don not know why, but this solves the problem.
    '''
    
    # temperature
    start_year_temp, end_year_temp = temp_year_range
    temp_mask = (alps_df['date'] >= start_year_temp) & (alps_df['date'] <= end_year_temp)
    temp_df_filtered = alps_df[temp_mask]
    temp_df_mean = temp_df_filtered.groupby(['latitude', 'longitude'], as_index=False)['temperature'].mean()

    if selected_dataset == 'wind':
        if is_scattermapbox:
            wind_trace = create_wind_trace_mapbox(wind_data_df, wind_year_range, wind_months, wind_days, wind_hours)
        else:
            wind_trace = create_wind_trace_geo(wind_data_df, wind_year_range, wind_months, wind_days, wind_hours)
        fig.add_trace(wind_trace, row=1, col=1)

    if selected_dataset == 'temp':
        print('starting with temp trace')
        if is_scattermapbox:
            temperature_trace = create_temperature_trace_mapbox(temp_df_mean)
        else:
            temperature_trace = create_temperature_trace_geo(temp_df_mean)
        fig.add_trace(temperature_trace, row=1, col=1)

    # temp lineplot
    temp_line = temperature_lineplot_trace(temp_df=temp_df_filtered, longitude=clicked_point['lon_temp'],
                                           latitude=clicked_point['lat_temp'])
    fig.add_trace(temp_line, row=1, col=2)

    # wind
    wind_lineplot_magnitude = wind_lineplot_magnitude_trace(wind_data_df, wind_year_range,
                                        wind_months, wind_days, wind_hours, longitude=clicked_point['lon_wind'],
                                        latitude=clicked_point['lat_wind'])

    fig.add_trace(wind_lineplot_magnitude, row=2, col=2, secondary_y=False)

    wind_lineplot_direction = wind_lineplot_direction_trace(wind_data_df, wind_year_range,
                                        wind_months, wind_days, wind_hours, longitude=clicked_point['lon_wind'],
                                        latitude=clicked_point['lat_wind'])
    fig.add_trace(wind_lineplot_direction, row=2, col=2, secondary_y=True)

    #Mass Balance trace
    start_year_mb, end_year_mb = mb_year_range
    mb_mask = (mass_balance_df['date'] >= start_year_mb) & (mass_balance_df['date'] <= end_year_mb)
    mb_df_filtered = mass_balance_df[mb_mask]
    mb_df_mean = mb_df_filtered.groupby(['latitude', 'longitude', 'name'], as_index=False)['annual_balance'].mean()
    if is_scattermapbox:
        mass_balance_trace = create_mass_balance_trace_mapbox(mb_df_mean)
    else:
        mass_balance_trace = create_mass_balance_trace_geo(mb_df_mean)
    fig.add_trace(mass_balance_trace, row=1, col=1)

    mb_lineplot = mass_balance_lineplot_trace(mb_df=mb_df_filtered,name=clicked_point['text'])
    fig.add_trace(mb_lineplot, row=1, col=2, secondary_y=True)

    fig.update_geos(
        visible=False,
        resolution=current_geo.get('resolution', 50),
        scope=current_geo.get('scope', "europe"),
        showcountries=True,
        countrycolor="Black",
        showsubunits=True,
        subunitcolor="Blue",
        landcolor=current_geo.get('landcolor', 'rgb(217, 217, 217)'),
        center=current_geo.get('center', {'lat': 46.5, 'lon': 10}),
        # uirevision ensures to keep the zoom level after updating the map
        uirevision=current_geo.get('uirevision', 'constant'),

    )

    # the fact that I can just numerate y-axis is from ChatGPT
    fig.update_layout(
        title_text="Map and Line Plot",
        title_x=0.5,
        yaxis1=dict(title=dict(text='temperature in °C'),),
        yaxis2=dict(title=dict(text='mass balance in mm w.e.')),
        yaxis3=dict(title=dict(text='magnitude in km/h')),
        yaxis4=dict(title=dict(text='direction in °')),

        yaxis4_range=[0, 360],

        showlegend=True,
        mapbox1=dict(
            style="carto-positron",
            center=current_geo.get('center', {'lat': 47, 'lon': 12}),
            zoom=5,
            layers=[]
        ),
        mapbox2=dict(
            style="carto-positron",
            center=current_geo.get('center', {'lat': 47, 'lon': 12},),
            zoom=5,
            layers=[]
        ),
        margin=current_layout.get('margin', {"r": 10, "t": 1, "l": 0, "b": 10})
    )
    return fig

# The basic structure for click events is from ChatGPT. But there is not much left...77
@app.callback(
    Output('clicked-data-output', 'children'),
    Input('main-map', 'clickData'),
    State('main-map', 'figure')  # Add the figure state
)
def display_click_data(clickData, fig):
    if not clickData:
        click_text_mass_balance = '1'
        click_text_wind = '2'
        click_text_temp = '3'
        return f"Mass Balance: {click_text_mass_balance} \n Wind: {click_text_wind} \n Temperature: {click_text_temp}"
    point = clickData['points'][0]
    curve_number = point['curveNumber']

    # dynamically get trace names from the figure
    try:
        trace_name = fig['data'][curve_number]['name']
    except IndexError:
        return "Invalid trace index."

    # access attributes
    lon = point.get('lon', None)
    lat = point.get('lat', None)
    text = point.get('text', 'No text available')

    # Store the lon/lat in the global variable
    if trace_name == 'Temperature':
        clicked_point['lon_temp'] = lon
        clicked_point['lat_temp'] = lat
    elif trace_name == 'Wind':
        clicked_point['lon_wind'] = lon
        clicked_point['lat_wind'] = lat
    elif trace_name == 'Mass Balance (Reference)':
        clicked_point['text'] = text

    # Return based on trace name
    if trace_name == 'Mass Balance (Reference)':
        return f"Mass Balance: {text}"
    elif trace_name == 'Temperature':
        return f"Temperature: {text}"
    elif trace_name == 'Wind':
        return f"Wind Magnitude: {text}"

    return "No data"

if __name__ == '__main__':
    app.run_server(debug=True)
