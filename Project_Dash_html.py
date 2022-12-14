from dash import html, dcc
import constants

TIMESTAMPS = ["plan"] + list(range(1, 14))

layout = html.Div(className="row", children=[
    html.Div(className="row", children=[

        # INTRO -----------------------------------------------------------------------------------------------------

        html.H2("Comparison of the ICP and the Centring Registration Methods in Radiotherapy",
                style={'textAlign': 'center', "color": "#081e5e", "font-family": "Bahnschrift", 'font-weight': 'bold',
                       "padding": "30px 0px 30px 0px"}),

        html.H6("""In radiotherapy, it is fundamental to position the patient before delivering the irradiation dose 
        to both: affect the tumour and avoid the healthy organs as much as possible. A medical image registration 
        aligns the patient's position at a given time with their position on the treatment plan CT image. It matches 
        them in the best way possible to acquire the desired position of the patient. The goal of this page is to 
        show the difference between the two registration methods to improve prostate cancer patients' treatment 
        planning.""",
                style={"margin-left": "40px", "margin-right": "40px", "margin-top": "10px", "margin-bottom": "10px",
                       "color": "#081e5e", "font-size": "15pt", "background-color": constants.LIGHT_GREY,
                       "border-radius": "5px", "padding": "10px 30px 10px 30px"}),

        dcc.Markdown("""We use **ICP** (Iterative Closest Point) registration method to align the anatomy based on 
        the patient's bones. **Prostate centring**, the second registration method, considers the prostate's position 
        instead of the bones. The difference between these two approaches is that bones have a rigid position within 
        the human body, while the prostate can move within the pelvic region. Furthermore, the ICP method also 
        considers the rotation of the organs with respect to bones, while the second method only considers 
        translation.""",
                     style={"margin-left": "40px", "margin-right": "40px", "margin-top": "10px",
                            "margin-bottom": "10px", "color": "#081e5e", "font-size": "15pt",
                            "background-color": constants.LIGHT_GREY, "border-radius": "5px",
                            "padding": "10px 30px 10px 30px"}),

        dcc.Markdown("""The comparison between these two methods is demonstrated on eight patients during 13 treatment 
        appointments. There are several graphs that depict the distinction from different points of view, 
        divided into three sections. The **patients overview section** depicts all patients' data at the same time; 
        the **individual patient section** shows all the data of one patient, and the **timestamp section** compares 
        only one or two patient's images. All graphs on this page are interactive and mutually connected, so one can 
        click on one chart and see the data highlighted in other charts. For example, in the first heatmap, 
        by clicking on the first cell in line with number 722, all graphs in the individual patient section will show 
        the patient number 722 instead of the number 137 displayed by default.""",
                     style={"margin-left": "40px", "margin-right": "40px", "margin-top": "10px",
                            "margin-bottom": "10px",
                            "color": "#081e5e", "font-size": "15pt", "background-color": constants.LIGHT_GREY,
                            "border-radius": "5px",
                            "padding": "10px 30px 10px 30px"}),

        html.H6("Patients overview section", style={"margin": "20px 40px 20px 40px", "color": "#171717", "font-size":
                "20pt", "font-family": "Bahnschrift", 'font-weight': 'bold', "background-color": constants.LIGHT_GREEN,
                                                    "border-radius": "5px", "padding": "10px 0px 10px 0px",
                                                    "text-align": "center"}),

        # AVERAGES ------------------------------------------------------------------------------------------

        html.H6("""The first two graphs show the average distance patients' organs have moved away from the plan 
        organs' positions during the time of the treatment. For every patient, there are three relevant organs in both 
        graphs. In the prostate centring chart, bones are included instead of the prostate because the prostate is the 
        centre of the alignment; therefore, the distance would be zero every time.""",
                style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                       "background-color": constants.LIGHT_GREY, "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.Div(className="row", style={"textAlign": "center"}, children=[
            html.H6("Show scale:", style={'display': 'inline-block'}),
            dcc.RadioItems(options=["uniform", "individual"], value="uniform", inline=True, id="scale-average",
                           style={'display': 'inline-block', "font-size": "18px"})]),

        html.Div(style={'textAlign': 'center'}, children=[
            dcc.Graph(id="average-distances", config=dict(modeBarButtonsToRemove=constants.MODEBAR),
                      style={'display': 'inline-block', "width": "95%", "padding": "20px 0px 20px 0px"})]),

        # HEATMAPS ---------------------------------------------------------------------------------------------------

        html.H6("""The two heatmaps represent how much patients' organs moved with respect to their 
        equivalent plan organs during the treatment. In both graphs, the Y-axis consists of patients' IDs, 
        and the X-axis comprises 13 timestamps when the patients' scans have been made. Hovering over any of the 
        heatmaps' cells, one can see exact information about the cell, such as the organ one is looking at, 
        movement distance, patient ID, and others.""",
                style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                       "background-color": constants.LIGHT_GREY, "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.Div(className="row", style={"textAlign": "center"}, children=[
            html.H6("Show scale:", style={'display': 'inline-block'}),
            dcc.RadioItems(options=["uniform", "individual"], value="uniform", inline=True, id="scale-heatmap",
                           style={'display': 'inline-block', "font-size": "18px"})]),

        html.Div(style={'textAlign': 'center'}, children=[
            dcc.Graph(id="heatmap-icp", config=dict(modeBarButtonsToRemove=constants.MODEBAR),
                      style={'display': 'inline-block', "padding": "20px 0px 10px 0px",
                             "width": "95%"})]),
        html.Div(style={'textAlign': 'center'}, children=[
            dcc.Graph(id="heatmap-center", config=dict(modeBarButtonsToRemove=constants.MODEBAR),
                      style={'display': 'inline-block', "padding": "0px 0px 20px 0px", "width": "95%"})]),

        html.H6("Individual patient section",
                style={"margin": "20px 40px 20px 40px", "color": "#171717",
                       "font-family": "Bahnschrift", 'font-weight': 'bold',
                       "background-color": constants.LIGHT_GREEN, "border-radius": "5px",
                       "padding": "10px 0px 10px 0px", "text-align": "center", "font-size": "20pt"}),

        # ORGANS DISTANCES --------------------------------------------------------------------------------------------

        html.H6("""The following charts depict only the organs and bones of one 
        patient during the treatment period - 13 appointments. By clicking on the traces in the legend, 
        one can regulate their visibility.""",
                style={"margin-left": "40px", "margin-right": "40px", "margin-top": "30px", "color": "#081e5e",
                       "background-color": constants.LIGHT_GREY, "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"})]),

    html.Div(className="row", style={"textAlign": "center"}, children=[
        html.H6("Show scale:", style={'display': 'inline-block'}),
        dcc.RadioItems(options=["uniform", "individual"], value="uniform", inline=True, id="scale-organs",
                       style={'display': 'inline-block', "font-size": "18px"})]),

    html.Div(className="row", children=[
        html.Div(style={'textAlign': 'center'}, children=[
            dcc.Graph(id="organ-distances", config=dict(modeBarButtonsToRemove=constants.MODEBAR),
                      style={'display': 'inline-block', "padding": "20px 40px 20px 40px", "width": "95%"})]),

        # DIFFERENCES GRAPH -----------------------------------------------------------------------------------------

        html.H6("""The graph below shows the distance difference between the two registration methods, which are 
        depicted in the previous two graphs. In the positive part of the vertical axis, the organs moved more during 
        ICP aligning. In the negative part, there was a greater movement during the centring method.""",
                style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                       "background-color": constants.LIGHT_GREY, "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.Div(className="row", style={'textAlign': 'center'}, children=[
            dcc.Graph(id="alignment-differences", config=dict(modeBarButtonsToRemove=constants.MODEBAR),
                      style={"padding": "20px 40px 30px 40px", 'display': 'inline-block', "width": "95%"})])]),

    # ROTATIONS GRAPH -----------------------------------------------------------------------------------------------
    html.H6("""This graph depicts how the organs in a given time rotated in relation to the organs in the plan 
    images. The rotations along the three axes were computed with the ICP algorithm, and they are shown as angles in 
    degrees.""",
            style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                   "background-color": constants.LIGHT_GREY, "border-radius": "5px",
                   "padding": "10px 30px 10px 30px"}),

    html.Div(className="row", style={'textAlign': 'center'}, children=[
        dcc.Graph(id="rotations-graph", config=dict(modeBarButtonsToRemove=constants.MODEBAR),
                  style={"padding": "20px 40px 10px 40px", "width": "95%"})]),

    html.H6("Timestamp section",
            style={"margin": "20px 40px 20px 40px", "color": "#171717",
                   "font-family": "Bahnschrift", 'font-weight': 'bold',
                   "background-color": constants.LIGHT_GREEN, "border-radius": "5px",
                   "padding": "10px 0px 10px 0px", "text-align": "center", "font-size": "20pt"}),

    # 3D GRAPH --------------------------------------------------------------------------------------------------

    html.Div(className="row", children=[
        html.Div(className="row", children=[
            html.H6("""The 3D chart depicts the organs loaded from patients' anatomy scans. It has two modes, 
            the first showing the organs and the bones from the plan CT scans and in the second mode, one can select 
            two different scans, the alignment method and then compare the organ positions. In both modes, 
            rendering only some organs and bones is possible. The last group of graphs represents organs selected in 
            the 3D graph but in the 2D slices along the main anatomical planes, and a rotation axes graph. There are 
            three sliders that can be used to move the imaged slice in the direction of the given axis. The rotation 
            graph shows the angles in which the organs selected in the second timestamp rotated after the ICP 
            alignment.""",
                    style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                           "background-color": constants.LIGHT_GREY, "border-radius": "5px",
                           "padding": "20px 20px 20px 20px"})]),

        html.Div(className="row", children=[
            html.Div(className="six columns", children=[

                html.H6("Select the mode:", style={'display': 'inline-block', "padding": "20px 0px 0px 45px"}),
                dcc.RadioItems(options=["Plan organs", "Two timestamps"],
                               value="Plan organs", id="mode-radioitems", inline=True,
                               style={'display': 'inline-block', "padding": "0px 0px 0px 20px", "font-size": "18px"},
                               inputStyle={"margin-left": "20px"}),
                html.Div(className="row", style={"display": "flex", "align-items": "center"}, children=[
                    html.H6("Select the first and the second image:",
                            style={'display': 'inline-block', "padding": "0px 20px 0px 45px"}, id="timestamp"),
                    dcc.Dropdown(options=TIMESTAMPS, value="plan", searchable=False, id="fst-timestamp-dropdown",
                                 style={'display': 'inline-block', "font-size": "16px", "padding": "0px 0px 0px 0px"}),
                    dcc.Dropdown(options=TIMESTAMPS, value=1, searchable=False,
                                 id="snd-timestamp-dropdown",
                                 style={'display': 'inline-block', "font-size": "16px",
                                        "padding": "0px 0px 0px 30px"})]),

                html.H6("Select the method of alignment:",
                        style={'display': 'inline-block', "padding": "0px 50px 0px 45px"}, id="method"),
                dcc.RadioItems(options=["ICP", "Prostate Centring"], value="ICP", inline=True,
                               id="alignment-radioitems",
                               style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"}),

                html.H6("Select the visibility of organs/bones:",
                        style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
                dcc.Checklist(options=["Bones", "Prostate", "Bladder", "Rectum"], value=["Prostate"], inline=True,
                              id="organs-checklist",
                              style={'display': 'inline-block', "font-size": "18px", "padding": "0px 0px 0px 25px"}),

                dcc.Graph(id="main-graph", config=dict(modeBarButtonsToRemove=constants.D3_MODEBAR)),
            ]),

            # SLICES & 3D ROTATION -----------------------------------------------------------------------------------

            html.Div(className='six columns', style={"display": "flex", "flex-wrap": "wrap",
                                                     "justify-content": "space-evenly",
                                                     "margin": "20px 0px 0px 45px"}, children=[

                dcc.Graph(id="rotations-axes", config=dict(modeBarButtonsToRemove=constants.D3_MODEBAR + ["pan3d"])),

                html.Div(id="x-slice", children=[
                    dcc.Slider(min=0, max=1, value=0.5, id="x-slice-slider", marks=None),
                    dcc.Graph(id="x-slice-graph", config=dict(modeBarButtonsToRemove=constants.MODEBAR))]),

                html.Div(style={"padding": "20px 0px 10px 0px"}, children=[
                    dcc.Slider(min=0, max=1, value=0.5, id="y-slice-slider", marks=None),
                    dcc.Graph(id="y-slice-graph", config=dict(modeBarButtonsToRemove=constants.MODEBAR))]),

                html.Div(style={"padding": "20px 0px 10px 0px"}, children=[
                    dcc.Slider(min=0, max=1, value=0.5, id="z-slice-slider", marks=None),
                    dcc.Graph(id="z-slice-graph", config=dict(modeBarButtonsToRemove=constants.MODEBAR))])
            ])
        ])
    ]),
])
