import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import math
import numpy as np
from constants import *
from plotly.subplots import make_subplots
from application_dash import patient_id, all_distances_icp, all_distances_center

app = dash.Dash(__name__)
server = app.server
app.title = "resetScale2d Test"

# _config = {"modeBarButtons": [["resetScale2d"]]}
#
#
# def make_fig(plot_type="linear", x=range(21), y=range(1, 211, 10)):
#     xvals = list(x)
#     yvals = list(y)
#     xrange = [0, 50]
#     yrange = [1, 500]
#     if plot_type == "log":
#         yrange = [math.log10(yr) for yr in yrange]
#     print(plot_type, xrange, yrange)
#     traces = [go.Scatter(x=xvals, y=yvals, marker={"size": 8}, name="Tens")]
#
#     layout = go.Layout(
#         xaxis=dict(range=xrange),
#         yaxis=dict(type=plot_type,
#                    range=yrange),
#         # uirevision=plot_type
#     )
#     fig = go.Figure(data=traces, layout=layout)
#
#     fig_div = html.Div(
#         [dcc.Graph(id='linlogplot', figure=fig, config=_config)],
#         id='fig_div',
#     )
#
#     return fig_div
#
#
# @app.callback(Output("fig_div", "children"), [Input("linlog", "value")])
# def change_type(plot_type):
#     return make_fig(plot_type=plot_type)


def make_plot():
    colors = [[LIGHT_BLUE] * 13, [GREEN] * 13, [RED] * 13]
    sizes = [[0] * 13, [0] * 13, [0] * 13]
    distances_icp = all_distances_icp[PATIENTS.index(patient_id)]
    prostate, bladder, rectum = distances_icp[0], distances_icp[1], distances_icp[2]

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, shared_xaxes=True)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=prostate, mode="lines+markers", name="Prostate",
                               marker=dict(color=LIGHT_BLUE, symbol="x", line=dict(width=sizes[0], color=colors[0]))), row=1, col=1)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder",
                               marker=dict(color=GREEN, symbol="square", line=dict(width=sizes[1], color=colors[1]))), row=1, col=1)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum",
                               marker=dict(color=RED, symbol="diamond", line=dict(width=sizes[2], color=colors[2]))), row=1, col=1)

    distances_center = all_distances_center[PATIENTS.index(patient_id)]
    bones, bladder, rectum = distances_center[0], distances_center[1], distances_center[2]

    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bones, mode="lines+markers", name="Bones",
                               marker=dict(color=PURPLE, symbol="circle", line=dict(width=sizes[0], color=colors[0]))), row=1, col=2)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder",
                               marker=dict(color=GREEN, symbol="square", line=dict(width=sizes[1], color=colors[1]))), row=1, col=2)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum",
                               marker=dict(color=RED, symbol="diamond", line=dict(width=sizes[2], color=colors[2]))), row=1, col=2)

    fig.update_layout(height=600, width=1200, yaxis2=dict(range=[0, 50]), yaxis2_showticklabels=True,
                      xaxis2_showticklabels=True)
    fig.update_xaxes(matches='x', tick0=0, dtick=2)
    fig.update_yaxes(matches='y')

    return fig


app.layout = html.Div(
    [
        dcc.Graph(figure=make_plot()),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
