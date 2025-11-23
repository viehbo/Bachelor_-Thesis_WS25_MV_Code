# temperature_trace.py
import plotly.graph_objects as go

def create_temperature_trace_mapbox(temp_df_mean):

    return go.Scattermapbox(
        lon=round(temp_df_mean['longitude'], 1),
        lat=round(temp_df_mean['latitude'], 1),
        text=round(temp_df_mean['temperature'], 2),
        mode='markers',
        marker=dict(
            size=10,
            allowoverlap=True,
            color=temp_df_mean['temperature'],
            colorscale=[[0, 'blue'], [0.5, 'yellow'], [1, 'red']],
            opacity=0.5,
            cmin=-5,
            cmax=10,
            colorbar = dict(
                title='Mean Temperature in °C',
                x=-0.05,
                xanchor='left',
                thickness=15,
                len=1.15
                ),
        ),
        name='Temperature',
        visible=True
    )


def create_temperature_trace_geo(temp_df_mean):

    return go.Scattergeo(
        lon=temp_df_mean['longitude'],
        lat=temp_df_mean['latitude'],
        text=temp_df_mean['temperature'],
        mode='markers',
        marker=dict(
            size=10,
            symbol='square',
            color=temp_df_mean['temperature'],
            colorscale=[[0, 'blue'], [0.5, 'yellow'], [1, 'red']],
            opacity=0.5,
            cmin=-5,
            cmax=10,
            colorbar = dict(
                title='Mean Temperature in °C',
                x=0.0,
                xanchor='left',
                thickness=15,
                len=1.15
                ),
        ),
        name='Temperature',
        visible=True
    )
