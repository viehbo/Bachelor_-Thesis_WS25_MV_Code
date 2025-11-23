# wind_trace.py
import plotly.graph_objects as go

def create_wind_trace_mapbox(
    wind_data_df,
    wind_year_range,
    wind_months=None,
    wind_days=None,
    wind_hours=None
):

    # because I have split the date before, now need to work with individual values
    start_year_wind, end_year_wind = wind_year_range

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

    # Filter and group the wind data
    filtered_wind = wind_data_df[wind_mask]

    return go.Scattermapbox(

        lon=filtered_wind['longitude'],
        lat=filtered_wind['latitude'],
        text=[f"Magnitude: {m:.2f} km/h<br>Direction: {d:.2f}°"
          for m, d in zip(filtered_wind['magnitude'], filtered_wind['direction'])],
        mode='markers',
        marker=dict(
            #symbolsrc=f"path://{custom_path}",
            symbol='circle',
            size=5,
            color=filtered_wind['magnitude'],
            colorscale='Viridis',
            cmin=min(filtered_wind['magnitude']),
            cmax=max(filtered_wind['magnitude']),
            colorbar=dict(title='Wind Magnitude in km/h',
                          x=-0.1,
                          xanchor='left',
                          thickness=15,
                          len=1.15
                          ),
        ),
        name='Wind',
        visible=True,
    )

def create_wind_trace_geo(
    wind_data_df,
    wind_year_range,
    wind_months=None,
    wind_days=None,
    wind_hours=None
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
    if wind_hours:
        wind_mask &= wind_data_df['hour'].isin(wind_hours)

    # Filter and group the wind data
    filtered_wind = wind_data_df[wind_mask]

    # The idea that 'len(filtered_wind.latitude)' can be used as a measure comes from ChatGPT
    num_points = len(filtered_wind.latitude)

    # in case the wind dataset is too big.
    if num_points > 20**4:
        #print('filtered_wind: ', filtered_wind['longitude'][1])
        return go.Scattergeo(
            mode='text',
            lon=[12],
            lat=[49],
            text=[f'dataset too big: {num_points} data points'],
            textfont=dict(color='red', size=15),
        )

    else:
        return go.Scattergeo(

            lon=round(filtered_wind['longitude'], 2),
            lat=round(filtered_wind['latitude'], 2),
            text=[f"Magnitude: {m:.2f} km/h<br>Direction: {d:.2f}°"
              for m, d in zip(filtered_wind['magnitude'], filtered_wind['direction'])],
            mode='markers',
            marker=dict(
                #symbolsrc=f"path://{custom_path}",
                symbol='arrow-up',
                size=5,
                color=filtered_wind['magnitude'],
                colorscale='Viridis',
                cmin=min(filtered_wind['magnitude']),
                cmax=max(filtered_wind['magnitude']),
                colorbar=dict(title='Wind Magnitude in km/h',
                              x=-0.1,
                              xanchor='left',
                              thickness=15,
                              len=1.15
                              ),
            ),

            marker_angle=filtered_wind['direction'],
            name='Wind',
            visible=True,
        )
