from dash import html, dcc

PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]
TIMESTAMPS = list(range(1, 14))

layout = html.Div(className="row", children=[
    html.Div(className="row", children=[

        # INTRO -----------------------------------------------------------------------------------------------------

        html.H2("Comparison of the ICP and the Centering Registration Methods in Radiotherapy",
                style={'textAlign': 'center', "color": "#081e5e", "font-family": "Bahnschrift", 'font-weight': 'bold',
                       "padding": "30px 0px 30px 0px"}),

        html.H6("""In radiotherapy, it is fundamental to position the patient before delivering the irradiation dose 
        to both affect the tumor and avoid the healthy organs as much as possible. Medical image registration is a 
        method that overlays the current patient's position with the one on the treatment plan CT and matches them in 
        the best way possible to acquire the desired position of the patient. 
        The function of this page is to show the difference between the two registration methods to improve prostate 
        cancer patients' treatments.""",
                style={"margin-left": "40px", "margin-right": "40px", "margin-top": "10px", "margin-bottom": "10px",
                       "color": "#081e5e", 'font-weight': 'bold',
                       "background-color": "#c0d9f2", "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.H6("""The ICP (Iterative Closest Point algorithm) is the registration method based on aligning patient's 
        bones to the position of their bones during plan CT image creation. The second registration method considers 
        the prostate's position instead of the location of the bones. It aligns the center of the patient's prostate 
        to the center of the plan prostate. The main difference between these methods is that the first one does 
        consider the rotation of the organs in the patient's body; however, the second only takes into consideration 
        the translation. Another distinction is that bones in the human body move a lot less in relation to organs 
        than the prostate; therefore, there is a difference when one makes both of them the centers of the 
        alignment.""",
                style={"margin-left": "40px", "margin-right": "40px", "margin-top": "10px", "margin-bottom": "10px",
                       "color": "#081e5e", 'font-weight': 'bold',
                       "background-color": "#c0d9f2", "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.H6("""The comparison between these two methods is shown among eight patients, each having CT scans from 
        13 treatment appointments. There are several graphs that depict the distinction from different points of 
        view. They are interactive and mutually connected, so one can click on a chart and see the data highlighted 
        in other charts.""",
                style={"margin-left": "40px", "margin-right": "40px", "margin-top": "10px", "margin-bottom": "10px",
                       "color": "#081e5e", 'font-weight': 'bold',
                       "background-color": "#c0d9f2", "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.H6("", style={"margin": "15px 40px 15px 40px", "color": "#081e5e",
                           "background-color": "#c0f2f2", "border-radius": "5px",
                           "padding": "10px 0px 0px 0px"}),

        # HEATMAPS ---------------------------------------------------------------------------------------------------

        html.H6("""The following two heatmaps represent how much patients' organs moved according to their equivalent 
        plan organs during the treatment. In both graphs, the Y-axis consists of patients' IDs, and the X-axis 
        comprises 13 timestamps when the CT scans have been made. Hovering over any of the heatmaps' cells, one can see 
        exact information about the cell, such as the organ one is looking at, movement distance, patient ID, along 
        with others.""",
                style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                       "background-color": "#c0f2d9", "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        dcc.Graph(id="heatmap-icp", style={'display': 'inline-block', "padding": "20px 30px 0px 40px"}),

        dcc.Graph(id="heatmap-center",
                  style={'display': 'inline-block', "padding": "20px 0px 10px 40px"}),

        # AVERAGE ICP ------------------------------------------------------------------------------------------

        html.H6("""The next two graphs show the average distance patients' organs have moved away from the plan 
        organs' positions during the time of the treatment. For every patient, there are three relevant organs in both 
        graphs. In the centering method chart, bones are included instead of the prostate because the prostate is the 
        center of the alignment; therefore, it would be zero everytime.""",
                style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                       "background-color": "#c0f2cc", "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        dcc.Graph(id="average-icp", style={'display': 'inline-block', "padding": "20px 40px 20px 40px"}),

        # AVERAGE CENTER ----------------------------------------------------------------------------------------

        dcc.Graph(id="average-center", style={'display': 'inline-block', "padding": "20px 10px 20px 20px"}),

        # ORGANS ICP ---------------------------------------------------------------------------------------------

        html.H6("""On the contrary to the previous graphs, the consecutive charts only depict organs and bones of one 
        patient during the treatment period - 13 appointments. By clicking on the traces in the legend, 
        one can regulate their visibility.""",
                style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                       "background-color": "#c0f2cc", "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.Div(className="select", children=[
            html.H6("Select the patient:",
                    style={'display': 'inline-block', "padding": "20px 101px 0px 45px"}),
            dcc.Dropdown(options=PATIENTS, value="137", searchable=False,
                         id="patient-dropdown", style={'display': 'inline-block', "width": "80px", "font-size": "16px",
                                                       "padding": "20px 80px 0px 85px"})
        ], style={
            "display": "flex", "align-items": "center", "justify-content": "center"})

    ]),

    html.Div(className="row", children=[
        dcc.Graph(id="organs-icp", style={'display': 'inline-block', "padding": "20px 40px 20px 40px"}),

        # ORGANS CENTER -----------------------------------------------------------------------------------------

        dcc.Graph(id="organs-center", style={'display': 'inline-block', "padding": "20px 10px 20px 20px"}),

        # DIFFERENCES GRAPH -----------------------------------------------------------------------------------------

        html.H6("""The graph below shows the difference in the moved distance of the organs between the two methods. 
        In the positive part of the vertical axis, the organs moved more during ICP aligning. In the negative part,
         there was a greater movement during the centering method.""",
                style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                       "background-color": "#c0f2cc", "border-radius": "5px",
                       "padding": "10px 30px 10px 30px"}),

        html.Div(className="row", children=[
            dcc.Graph(id="alignment-differences",
                      style={"padding": "20px 0px 30px 40px", 'display': 'inline-block'})])]),

    # 3D GRAPH --------------------------------------------------------------------------------------------------

    html.Div(className="row", children=[
        html.Div(className="row", children=[
            html.H6("""In the 3D chart are depicted organs loaded from patients' CT images. It has two modes, 
            the first showing the organs and the bones from the plan CTs along with the movement vectors. The user 
            can select which vectors they want to see, either ones from all the timestamps, the average ones, 
            or none. In the second mode, one can select two different CT scans, the alignment method and then compare 
            the results. In both modes, selecting only some organs and bones or adjusting their opacity is 
            possible.""",
                    style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                           "background-color": "#c0f2cc", "border-radius": "5px",
                           "padding": "20px 20px 20px 20px"}),

            html.H6("""The last group of graphs represents organs selected in the 3D graph but in the 2D slices. 
            There are three sliders that can be used to move the imaged slice in the direction of the axis.""",
                    style={"margin-left": "40px", "margin-right": "40px", "color": "#081e5e",
                           "background-color": "#c0f2cc", "border-radius": "5px",
                           "padding": "20px 20px 20px 20px"})]),

        html.Div(className="row", children=[
            html.Div(className="six columns", children=[

                html.H6("Select the mode:", style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
                dcc.RadioItems(options=["Plan organs", "Two timestamps"],
                               value="Plan organs", id="mode-radioitems", inline=True,
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
                dcc.RadioItems(options=["all", "average", "none"], value="none", inline=True, id="movements-radioitems",
                               style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"},
                               inputStyle={"margin-left": "20px"}),

                html.H6("Adjust the opacity of the organs:",
                        style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
                html.Div(dcc.Slider(min=0, max=1, value=1, id="opacity-slider", marks=None),
                         style={"width": "40%", "height": "10px", 'display': 'inline-block',
                                "padding": "0px 0px 0px 40px"}),

                dcc.Graph(id="main-graph", style={"padding": "20px 0px 0px 45x", "display": "inline-block",
                                                  "margin": "20px 0px 45px 45px", "width": "92.4%"})
            ]),

            # SLICES ---------------------------------------------------------------------------------------------

            html.Div(className='six columns', style={"display": "flex", "flex-wrap": "wrap",
                                                     "justify-content": "space-evenly",
                                                     "margin": "20px 0px 0px 45px"}, children=[

                html.Div(className='six columns', style={"height": "310px", "width": "320px",
                                                         "margin": "0px 0px 0px 0px"}, children=[
                    html.Div(className="row", children=[
                        html.H6("X axes slice:", id="x-slice-header"),
                        dcc.Slider(min=0, max=1, value=0.5, id="x-slice-slider", marks=None),
                        html.H6("Y axes slice:"),
                        dcc.Slider(min=0, max=1, value=0.5, id="y-slice-slider", marks=None),
                        html.H6("Z axes slice:"),
                        dcc.Slider(min=0, max=1, value=0.5, id="z-slice-slider", marks=None)
                    ], style={"margin": "5px 0px 0px 0px", "padding": "20px 20px 20px 20px"})
                ]),

                dcc.Graph(id="x-slice-graph"),
                dcc.Graph(id="y-slice-graph", style={"margin": "20px 0px 0px 0px"}),
                dcc.Graph(id="z-slice-graph", style={"margin": "20px 0px 0px 0px"})
            ])
        ])
    ]),
])
