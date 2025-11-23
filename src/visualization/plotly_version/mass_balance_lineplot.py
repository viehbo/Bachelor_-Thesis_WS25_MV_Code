import pandas as pd
import plotly.graph_objects as go

def mass_balance_lineplot_trace(
        mb_df: pd.DataFrame,
        name):
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
    mb_df_filtered = mb_df.loc[(mb_df['name'] == name)]

    return go.Scatter(
        x=mb_df_filtered['date'],
        y=mb_df_filtered['annual_balance'],
        mode='lines',
        name='Mass Balance',
        line=dict(color='orange'),
    )