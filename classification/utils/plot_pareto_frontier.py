from typing import List
from utils import *
import plotly.graph_objects as go

def plot_to_file(x, y, x_pareto, y_pareto,path):
    layout = go.Layout(
        xaxis=dict(
            title='Size (MB)'
        ),
        yaxis=dict(
            title='Sensitivity'
        )
    )
    t1 = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Configs',
        marker={"size" : 3}
    )
    t2 = go.Scatter(
        x=x_pareto,
        y=y_pareto,
        mode='markers+lines',
        name='Pareto Frontier',
        marker={"size" : 3}
    )
    fig = go.Figure([t1, t2], layout)
    fig.write_image(path)