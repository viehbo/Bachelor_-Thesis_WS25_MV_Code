# Just an example annotation
from plotly.graph_objects import Figure
import plotly.graph_objects as go
def text_annotation(text = 'Hier könnte Ihre Werbung stehen'):
    fig = Figure()
    return fig.add_annotation(
        x=0.8,
        y=0.4,
        text=text,
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="#ffffff"
        ),
        align="center",
        ax=20,
        ay=-30,
        bordercolor="#c7c7c7",
        borderwidth=2,
        borderpad=4,
        bgcolor="#ff7f0e",
        opacity=0.8
    )
