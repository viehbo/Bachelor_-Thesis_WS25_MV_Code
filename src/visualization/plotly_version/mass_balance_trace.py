# mass_balance_trace.py
import plotly.graph_objects as go

def create_mass_balance_trace_mapbox(mb_df_filtered):

    return go.Scattermapbox(
        lon=mb_df_filtered['longitude'],
        lat=mb_df_filtered['latitude'],
        text=mb_df_filtered['name'],
        mode='markers',
        marker=dict(
            allowoverlap=True,
            size=10,
            color='red',
            opacity=0.8
        ),
        name='Mass Balance (Reference)',
        visible=True
    )

def create_mass_balance_trace_geo(mb_df_filtered):

    return go.Scattergeo(
        lon=mb_df_filtered['longitude'],
        lat=mb_df_filtered['latitude'],
        text=mb_df_filtered['name'],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            opacity=0.8
        ),
        name='Mass Balance (Reference)',
        visible=True
    )
