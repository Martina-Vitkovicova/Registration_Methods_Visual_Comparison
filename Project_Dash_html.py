from dash import Dash, html, dcc, Output, Input, callback_context

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = Dash(__name__, external_stylesheets=external_stylesheets)

PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]
TIMESTAMPS = list(range(1, 14))

layout = html.Div(className="row", children=[
    html.Div(className="row", children=[
        html.H2("Comparison of ICP and centering aligning methods for prostate cancer treatment",
                style={'textAlign': 'center', "color": "#081e5e", "font-family": "Bahnschrift", 'font-weight': 'bold',
                       "padding": "20px 0px 0px 80px"}),
        html.Div(className="row", children=[
            html.H6("""The difference is shown on 8 patients with each of them having CT scans from 13 different
                    treatment appointments.
                    The ICP (Iterative closest point algorithm) method aligns patient
                    bones in each timestamp to the position of the bones in plan CT scan.
                    The second method aligns center of the prostate in each timestamp to the center of the plan"
                    prostate. Which means it does not consider the rotation of the organs, only the translation.""",
                    style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e", 'font-weight': 'bold',
                           "background-color": "#c0d9f2", "border-radius": "5px",
                           "padding": "10px 30px 10px 30px"})], )]),  # "white-space": "pre-wrap"

    html.Div(className="row", children=[

        html.Div(className='six columns', children=[
            dcc.Graph(id="heatmap-icp", style={'display': 'inline-block', "padding": "20px 30px 0px 40px"}),

            dcc.Graph(id="heatmap-center",
                      style={'display': 'inline-block', "padding": "20px 0px 0px 40px"}),

            dcc.Graph(id="average-icp", style={'display': 'inline-block', "padding": "20px 30px 0px 40px"}),

            html.H6("Select the patient:",
                    style={'display': 'inline-block', "padding": "20px 101px 0px 45px"}),
            dcc.Dropdown(options=PATIENTS, value="137", searchable=False,
                         id="patient-dropdown", style={'display': 'inline-block', "width": "80px", "font-size": "16px",
                                                       "padding": "20px 80px 0px 85px"}),

            dcc.Loading(
                [dcc.Graph(id="organs-icp", style={'display': 'inline-block', "padding": "30px 30px 30px 40px"})],
                type="circle"),

            dcc.Graph(id="alignment-differences", style={"padding": "0px 0px 30px 40px"}),

            html.H6("Select the mode:", style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
            dcc.RadioItems(options=["Average (plan organs showing)", "Two timestamps"],
                           value="Average (plan organs showing)", id="mode-radioitems", inline=True,
                           style={'display': 'inline-block', "padding": "0px 0px 0px 20px", "font-size": "18px"},
                           inputStyle={"margin-left": "20px"}),

            html.H6("Select the first and the second timestamp:",
                    style={'display': 'inline-block', "padding": "20px 20px 0px 45px"}, id="timestamp"),
            dcc.Dropdown(options=TIMESTAMPS, value=1, searchable=False, id="fst-timestamp-dropdown",
                         style={'display': 'inline-block', "width": "50px",
                                "height": "30px", "font-size": "16px", "padding": "0px 0px 0px 0px"}),
            dcc.Dropdown(options=TIMESTAMPS, value=1, searchable=False, id="snd-timestamp-dropdown",
                         style={'display': 'inline-block', "width": "50px", "height": "30px", "font-size": "16px",
                                "padding": "0px 0px 0px 30px"}),

            html.H6("Select the method of alignment:",
                    style={'display': 'inline-block', "padding": "0px 50px 0px 45px"}, id="method"),
            dcc.RadioItems(options=["ICP", "Center point"], value="ICP", inline=True, id="alignment-radioitems",
                           style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"}),

            html.H6("Select the visibility of organs/bones:",
                    style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
            dcc.Checklist(options=["Bones", "Prostate", "Bladder", "Rectum"], value=["Prostate"], inline=True,
                          id="organs-checklist",
                          style={'display': 'inline-block', "font-size": "18px", "padding": "0px 0px 0px 25px"}),

            html.H6("Show movements:", id="movements-header",
                    style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
            dcc.RadioItems(options=["all", "average"], value="average", inline=True, id="movements-radioitems",
                           style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"},
                           inputStyle={"margin-left": "20px"}),

            html.H6("Adjust the opacity of the organs:",
                    style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
            html.Div(dcc.Slider(min=0, max=1, value=1, id="opacity-slider", marks=None),
                     style={"width": "40%", "height": "10px", 'display': 'inline-block',
                            "padding": "0px 0px 0px 40px"}),

            dcc.Graph(id="main-graph", style={'display': 'inline-block', "padding": "20px 30px 0px 40px"})]),

        html.Div(className='six columns', children=[
            dcc.Graph(id="average-center", style={'display': 'inline-block', "padding": "774px 0px 0px 0px"}),

            dcc.Graph(id="organs-center", style={'display': 'inline-block', "padding": "102px 30px 30px 0px"}),

            html.Div(className="row", children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(id="x-slice-graph", style={'display': 'inline-block', "padding": "420px 0px 10px 0px"}),
                    dcc.Graph(id="y-slice-graph", style={'display': 'inline-block', "padding": "0px 0px 10px 0px"})]),

                html.Div(className='six columns', children=[
                    html.Div(className="row", children=[
                        html.H6("X axes slice:", style={'display': 'inline-block', "padding": "450px 0px 0px 30px"},
                                id="x-slice-header"),
                        dcc.Slider(min=0, max=1, value=0.5, id="x-slice-slider", marks=None, updatemode="drag"),
                        html.H6("Y axes slice:", style={'display': 'inline-block', "padding": "2px 0px 0px 35px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="y-slice-slider", marks=None),
                        html.H6("Z axes slice:", style={'display': 'inline-block', "padding": "2px 0px 0px 30px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="z-slice-slider", marks=None)],
                             style={'width': '80%', 'display': 'inline-block', "padding": "2px 0px 0px 5px"}),

                    dcc.Graph(id="z-slice-graph",
                              style={'display': 'inline-block', "padding": "35px 0px 0px 0px"})])])])])])
