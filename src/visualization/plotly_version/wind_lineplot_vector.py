# wind_lineplot.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

def wind_lineplot_magnitude_trace(
    wind_data_df,
    wind_year_range,
    wind_months=1,
    wind_days=1,
    wind_hours=1,
    longitude= 49.0,
    latitude= 13.0,
):
    start_year_wind, end_year_wind = wind_year_range

    wind_mask = (
            (wind_data_df['year'] >= start_year_wind) &
            (wind_data_df['year'] <= end_year_wind)
    )
    if wind_months:
        wind_mask &= wind_data_df['month'].isin(wind_months)
    if wind_days:
        wind_mask &= wind_data_df['day'].isin(wind_days)

    # filter and group the data
    filtered_wind = wind_data_df[wind_mask]
    filtered_wind = filtered_wind.loc[(filtered_wind['longitude'] >= longitude-0.001) & (filtered_wind['longitude'] <= longitude+0.001)
                                    & (filtered_wind['latitude'] >= latitude-0.001) & (filtered_wind['latitude'] <= latitude+0.001)]

    return go.Scatter(
        x=filtered_wind['date'],
        y=filtered_wind['magnitude'],
        mode='lines',
        name='Wind speed',
        line=dict(color='red'),
    )

def wind_lineplot_direction_trace(
    wind_data_df,
    wind_year_range,
    wind_months=None,
    wind_days=None,
    wind_hours=None,
    longitude= 49.0,
    latitude= 13.0,
):

    start_year_wind, end_year_wind = wind_year_range

    # apply filtering
    wind_mask = (
            (wind_data_df['year'] >= start_year_wind) &
            (wind_data_df['year'] <= end_year_wind)
    )
    if wind_months:
        wind_mask &= wind_data_df['month'].isin(wind_months)
    if wind_days:
        wind_mask &= wind_data_df['day'].isin(wind_days)
    if wind_hours:
        wind_mask &= wind_data_df['hour'].isin(wind_hours)

    # filter and group the data
    filtered_wind = wind_data_df[wind_mask]
    filtered_wind = filtered_wind.loc[(filtered_wind['longitude'] >= longitude-0.001) & (filtered_wind['longitude'] <= longitude+0.001)
                                    & (filtered_wind['latitude'] >= latitude-0.001) & (filtered_wind['latitude'] <= latitude+0.001)]

    return go.Scatter(
        x=filtered_wind['date'],
        y=filtered_wind['direction'],
        mode='lines',
        name='Wind direction',
        line=dict(color='green'),
        marker=dict(cmin=0, cmax=360),
    )