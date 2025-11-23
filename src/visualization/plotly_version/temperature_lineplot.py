# temperature_lineplot.py.py
import pandas as pd
import plotly.graph_objects as go

def temperature_lineplot_trace(
        temp_df: pd.DataFrame,
        longitude: float,
        latitude: float):
    """
    Returns a Scattergeo trace for plotting temperature lines.

    Parameters:
        temp_df (pd.DataFrame): full temperature data
        start_date (pd.Timestamp): start date
        end_date (pd.Timestamp): end date

    Returns:
        go.Scatter: A Scatter trace for the temperature line plot.
        :param temp_df:
        :param latitude:
        :param longitude:
    """
    temp_df_filtered = temp_df.loc[(temp_df['longitude'] >= longitude-0.001) & (temp_df['longitude'] <= longitude+0.001)
                                    & (temp_df['latitude'] >= latitude-0.001) & (temp_df['latitude'] <= latitude+0.001)]


    return go.Scatter(
        x=temp_df_filtered['date'],
        y=temp_df_filtered['temperature'],
        mode='lines',
        name='Temperature',
        line=dict(color='blue')
    )