from copy import deepcopy
import trimesh
from dash import Dash, html, dcc, Output, Input, callback_context
import numpy as np
import plotly.graph_objects as go
import Project_2

FILEPATH = "C:\\Users\\vitko\\Desktop\\ProjetHCI\\Organs\\"
PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]
TIMESTAMPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

LIGHT_BLUE = "#636EFA"
GREEN = "#EF553B"
RED = "#00CC96"
BLUE = "#1F77B4"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)

patient_id = "137"


def main():
    app.layout = html.Div(className="row", children=[
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

            dcc.Loading([dcc.Graph(id="organs-icp", style={'display': 'inline-block', "padding": "30px 30px 30px 40px"})], type="circle"),

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

            html.H6("Adjust opacity of the organs:", style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
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
                              style={'display': 'inline-block', "padding": "35px 0px 0px 0px"})])])])])


def create_meshes_from_objs(objects, color):
    meshes = []
    for elem in objects:
        x, y, z = np.array(elem[0]).T
        i, j, k = np.array(elem[1]).T
        pl_mesh = go.Mesh3d(x=x, y=y, z=z, color=color, flatshading=True, i=i, j=j, k=k, showscale=False)
        meshes.append(pl_mesh)
    return meshes


def order_slice_vertices(vertices, indices):
    ordered_vertices = []
    for index in indices:
        ordered_vertices.append(vertices[index])

    return ordered_vertices


@app.callback(
    Output(component_id='method', component_property='style'),
    Output(component_id='alignment-radioitems', component_property='style'),
    Output(component_id='timestamp', component_property='style'),
    Output(component_id='fst-timestamp-dropdown', component_property='style'),
    Output(component_id='snd-timestamp-dropdown', component_property='style'),
    Output(component_id='x-slice-graph', component_property='style'),
    Output(component_id='x-slice-header', component_property='style'),
    Input(component_id='mode-radioitems', component_property='value'))
def options_visibility(mode):
    if mode == "Two timestamps":
        return {'display': 'inline-block', "padding": "0px 50px 0px 45px"}, \
               {'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"}, \
               {'display': 'inline-block', "padding": "20px 20px 0px 45px"}, \
               {'display': 'inline-block', "width": "50px",
                "height": "30px", "font-size": "16px", "padding": "0px 0px 0px 0px"}, \
               {'display': 'inline-block', "width": "50px",
                "height": "30px", "font-size": "16px", "padding": "0px 0px 0px 30px"}, \
               {'display': 'inline-block', "padding": "500px 0px 10px 0px"}, \
               {'display': 'inline-block', "padding": "530px 0px 0px 30px"}

    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, \
               {'display': 'inline-block', "padding": "420px 0px 10px 0px"}, {'display': 'inline-block',
                                                                              "padding": "450px 0px 0px 30px"}


@app.callback(
    Output("main-graph", "figure"),
    Input("patient-dropdown", "value"),
    Input("alignment-radioitems", "value"),
    Input("organs-checklist", "value"),
    Input("mode-radioitems", "value"),
    Input("fst-timestamp-dropdown", "value"),
    Input("snd-timestamp-dropdown", "value"),
    Input("opacity-slider", "value"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"))
def update_3dgraph(patient, alignment_radioitems, organs, mode, fst_timestamp, snd_timestamp, opacity_slider,
                   average_icp, average_center, heatmap_icp, heatmap_center):
    global patient_id
    objects_fst = import_selected_organs(organs, fst_timestamp, patient_id)
    objects_snd = import_selected_organs(organs, snd_timestamp, patient_id)
    fst_meshes, snd_meshes, center1_after, center2_after = [], [], [], []

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.5, y=-2, z=0))
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', height=520, width=680,
                       plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=40, r=40, t=60, b=40), showlegend=True)
    fig = go.Figure(layout=layout)
    fig.update_layout(scene_camera=camera)

    if "Two timestamps" in mode:
        if "ICP" in alignment_radioitems:
            meshes, center1_before, center1_after = get_meshes_after_icp(fst_timestamp, objects_fst, patient_id)
            fst_meshes.extend(meshes)
            meshes, center2_before, center2_after = get_meshes_after_icp(snd_timestamp, objects_snd, patient_id, "orange")
            snd_meshes.extend(meshes)

        if "Center point" in alignment_radioitems:
            meshes, center1_before, center1_after = get_meshes_after_centering(fst_timestamp, objects_fst, patient_id,
                                                                               BLUE)
            fst_meshes.extend(meshes)
            meshes, center2_before, center2_after = get_meshes_after_centering(snd_timestamp, objects_snd, patient_id)
            snd_meshes.extend(meshes)
        x1, y1, z1 = center1_after[0][0][0]
        x2, y2, z2 = center2_after[0][0][0]
        fig.add_trace(go.Scatter3d(x=[x1, x2], y=[y1, y2], z=[z1, z2]))

    else:
        objects = import_selected_organs(organs, "_plan", patient_id)
        fst_meshes = create_meshes_from_objs(objects, BLUE)

    for mesh in fst_meshes:
        mesh.update(cmin=-7, opacity=opacity_slider, lightposition=dict(x=100, y=200, z=0),
                    lighting=dict(ambient=0.4, diffuse=1, fresnel=0.1, specular=1, roughness=0.5,
                                  facenormalsepsilon=1e-15, vertexnormalsepsilon=1e-15))
        fig.add_trace(mesh)

    for mesh in snd_meshes:
        mesh.update(cmin=-7, opacity=opacity_slider, lightposition=dict(x=100, y=200, z=0),
                    lighting=dict(ambient=0.4, diffuse=1, fresnel=0.1, specular=1, roughness=0.5,
                                  facenormalsepsilon=1e-15, vertexnormalsepsilon=1e-15))

        fig.add_trace(mesh)
    fig.update_layout(title_text="Patient {}, timestamp number {} (blue) and number {} (orange)"
                      .format(patient_id, fst_timestamp, snd_timestamp), title_x=0.5,
                      title_y=0.95)

    # fig.add_trace(go.Scatter3d(x=[0, 20], y=[0, 0], z=[50, 50]))

    return fig


def import_selected_organs(organs, time_or_plan, patient):
    objects = []

    if "Bones" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, time_or_plan)]))
    if "Prostate" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, time_or_plan)]))
    if "Bladder" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, time_or_plan)]))
    if "Rectum" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, time_or_plan)]))

    return objects


def get_meshes_after_icp(timestamp, objects, patient, color=BLUE):
    plan_bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
    bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, timestamp)])
    bones_center_before = Project_2.find_center_point(bones[0][0])

    transform_matrix, vec = Project_2.icp_rot_vec(bones[0][0], plan_bones[0][0])
    transfr_objects = Project_2.vertices_transformation(transform_matrix, deepcopy(objects))
    after_icp_meshes = create_meshes_from_objs(transfr_objects, color)

    bones_center_after = Project_2.vertices_transformation(transform_matrix, [[[bones_center_before]]])
    # print(vec, timestamp)

    return after_icp_meshes, bones_center_before, bones_center_after


def get_meshes_after_centering(timestamp, objects, patient, color="orange"):
    prostate = Project_2.import_obj([FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, timestamp)])
    center_before = Project_2.find_center_point(prostate[0][0])

    plan_center = Project_2.find_center_point(Project_2.import_obj(
        [FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)])[0][0])
    other_center = Project_2.find_center_point(prostate[0][0])
    center_matrix = Project_2.create_translation_matrix(plan_center, other_center)
    center_transfr_objects = Project_2.vertices_transformation(center_matrix, deepcopy(objects))
    after_center_meshes = create_meshes_from_objs(center_transfr_objects, color)

    center_after = Project_2.vertices_transformation(center_matrix, [[[plan_center]]])

    return after_center_meshes, center_before, center_after


def timestamp_click_data(icp_click_data, center_click_data):
    icp_click_data = int(icp_click_data["points"][0]["x"]) if icp_click_data is not None else 1
    center_click_data = int(center_click_data["points"][0]["x"]) if center_click_data is not None else 1

    input_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    timestamp = 1

    if input_id == "organs-center":
        timestamp = center_click_data
    elif input_id == "organs-icp":
        timestamp = icp_click_data

    return timestamp


def load_organs_average(patient, organs):
    meshes = []

    if "Prostate" in organs:
        meshes.append(trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)))

    if "Bladder" in organs:
        meshes.append(trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder_plan.obj".format(patient)))

    if "Rectum" in organs:
        meshes.append(trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum_plan.obj".format(patient)))

    if "Bones" in organs:
        meshes.append(trimesh.load_mesh(FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)))

    return meshes


def two_slices_mode(method, patient, organs, timestamp):
    if "ICP" in method:
        plan_bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
        bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, timestamp)])
        icp_matrix = Project_2.icp_transformation_matrices(bones[0][0], plan_bones[0][0], False)
        meshes = selected_organs_slices(icp_matrix, organs, timestamp, patient)

    else:
        plan_prostate = trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient))
        prostate = trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, timestamp))
        key_center = Project_2.find_center_point(plan_prostate.vertices)
        other_center = Project_2.find_center_point(prostate.vertices)
        center_matrix = Project_2.create_translation_matrix(key_center, other_center)
        meshes = selected_organs_slices(center_matrix, organs, timestamp, patient)

    return meshes


@app.callback(
    Output("x-slice-graph", "figure"),
    Output("y-slice-graph", "figure"),
    Output("z-slice-graph", "figure"),
    Input("x-slice-slider", "value"),
    Input("y-slice-slider", "value"),
    Input("z-slice-slider", "value"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("organs-checklist", "value"),
    Input("patient-dropdown", "value"),
    Input("alignment-radioitems", "value"),
    Input("mode-radioitems", "value"),
    Input("fst-timestamp-dropdown", "value"),
    Input("snd-timestamp-dropdown", "value"))
def create_graph_slices(x_slider, y_slider, z_slider, icp_click_data, center_click_data, organs, patient, method, mode,
                        fst_timestamp, snd_timestamp):
    figures, snd_meshes = [], []
    names = ["X axis slice", "Y axis slice", "Z axis slice"]
    timestamp = timestamp_click_data(icp_click_data, center_click_data)

    for i in range(3):
        layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', height=310,
                           width=320, plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=30, r=20, t=60, b=40),
                           showlegend=False, title=dict(text=names[i]))
        fig = go.Figure(layout=layout)
        fig.update_layout(title_x=0.5)
        figures.append(fig)

    if "Two timestamps" in mode:
        fst_meshes = two_slices_mode(method, patient, organs, fst_timestamp)
        snd_meshes = two_slices_mode(method, patient, organs, snd_timestamp)
    else:
        fst_meshes = load_organs_average(patient, organs)

    x_fig = create_slice_final(x_slider, fst_meshes, snd_meshes, figures[0], "x")
    y_fig = create_slice_final(y_slider, fst_meshes, snd_meshes, figures[1], "y")
    z_fig = create_slice_final(z_slider, fst_meshes, snd_meshes, figures[2], "z")

    return x_fig, y_fig, z_fig


def selected_organs_slices(matrix, organs, timestamp, patient):
    meshes = []

    if "Prostate" in organs:
        mesh = trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, timestamp))
        meshes.append(deepcopy(mesh).apply_transform(matrix))

    if "Bladder" in organs:
        mesh = trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, timestamp))
        meshes.append(deepcopy(mesh).apply_transform(matrix))

    if "Rectum" in organs:
        mesh = trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, timestamp))
        meshes.append(deepcopy(mesh).apply_transform(matrix))

    if "Bones" in organs:
        mesh = trimesh.load_mesh(FILEPATH + "{}\\bones\\bones{}.obj".format(patient, timestamp))
        meshes.append(deepcopy(mesh).apply_transform(matrix))

    return meshes


def create_slice(mesh, slice_slider, params):
    min_val, max_val, plane_origin, plane_normal, axis = params
    slope = (max_val - 2.5) - (min_val + 0.5)

    if axis == "x":
        plane_origin[0] = (min_val + 0.5) + slope * slice_slider
    elif axis == "y":
        plane_origin[1] = (min_val + 0.5) + slope * slice_slider
    else:
        plane_origin[2] = (min_val + 0.5) + slope * slice_slider

    axis_slice = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    slices = []
    for entity in axis_slice.entities:
        ordered_slice = order_slice_vertices(axis_slice.vertices, entity.points)
        i, j, k = np.array(ordered_slice).T
        slices.append((i, j, k))

    return slices


def create_slice_helper(meshes, slice_slider, fig, color, axis):
    for mesh in meshes:
        if axis == "x":
            params = mesh.bounds[0][0], mesh.bounds[1][0], [0, mesh.centroid[1], mesh.centroid[2]], [1, 0, 0], "x"
            slices = create_slice(mesh, slice_slider, params)
            for _, x, y in slices:
                fig.add_trace(go.Scatter(x=x, y=y, line=go.scatter.Line(color=color, width=3)))

        elif axis == "y":
            params = mesh.bounds[0][1], mesh.bounds[1][1], [mesh.centroid[0], 0, mesh.centroid[2]], [0, 1, 0], "y"
            slices = create_slice(mesh, slice_slider, params)
            for x, _, y in slices:
                fig.add_trace(go.Scatter(x=x, y=y, line=go.scatter.Line(color=color, width=3)))
        else:
            params = mesh.bounds[0][2], mesh.bounds[1][2], [mesh.centroid[0], mesh.centroid[1], 0], [0, 0, 1], "z"
            slices = create_slice(mesh, slice_slider, params)
            for x, y, _ in slices:
                fig.add_trace(go.Scatter(x=x, y=y, line=go.scatter.Line(color=color, width=3)))

    return fig


def create_slice_final(slice_slider, icp_meshes, centered_meshes, fig, axis):
    if icp_meshes:
        create_slice_helper(icp_meshes, slice_slider, fig, BLUE, axis)
    if centered_meshes:
        create_slice_helper(centered_meshes, slice_slider, fig, "orange", axis)

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    return fig


def add_planes(point, normal):
    d = -point.dot(normal)
    xx, yy = np.meshgrid(range(10), range(10))
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]

    fig = go.Surface(x=xx, y=yy, z=z)

    return fig


@app.callback(
    Output("organs-icp", "figure"),
    Input("patient-dropdown", "value"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("alignment-differences", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"))
def create_distances_after_icp(dropdown, click_data, center_click_data, differences, average_icp, average_center,
                               heatmap_icp, heatmap_center):
    global patient_id
    all_click_data = [click_data, center_click_data, differences, average_icp, average_center,
                      heatmap_icp, heatmap_center]
    all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)
    colors1, colors2, colors3 = [LIGHT_BLUE] * 13, [GREEN] * 13, [RED] * 13

    if click_data:
        data = click_data["points"][0]
        if "heatmap" in click_id:
            patient_id = PATIENTS[data["y"]]
            flag = 0
        elif "average" in click_id:
            patient_id = data["x"]
            flag = 2

        # colors1[x], colors2[x], colors3[x] = "white", "white", "white"

    distances_icp = all_distances_icp[PATIENTS.index(patient_id)]
    prostate, bladder, rectum = distances_icp[0], distances_icp[1], distances_icp[2]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(
                           text="Distances of icp aligned organs and the plan organs of patient {}".format(patient_id),
                           font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=prostate, mode="lines+markers", name="Prostate"))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder"))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum"))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    if click_data and "average" not in click_id:
        if "organs" in click_id or "alignment-differences" in click_id:
            x = int(click_data["points"][0]["x"])
        if "heatmap" in click_id:
            x = int(click_data["points"][0]["x"]) // 3 + 1
        fig.add_trace(
            go.Scattergl(x=[x], y=[prostate[x - 1]], mode="lines+markers", name="Prostate", showlegend=False,
                         marker=dict(size=10, color="white")))
        fig.add_trace(
            go.Scattergl(x=[x], y=[bladder[x - 1]], mode="lines+markers", name="Bladder", showlegend=False,
                         marker=dict(size=10, color="white")))
        fig.add_trace(go.Scattergl(x=[x], y=[rectum[x - 1]], mode="lines+markers", name="Rectum", showlegend=False,
                                   marker=dict(size=10, color="white")))

    return fig


@app.callback(
    Output("organs-center", "figure"),
    Input("patient-dropdown", "value"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("alignment-differences", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"))
def create_distances_after_centering(dropdown, icp_click_data, click_data, differences, average_icp, average_center,
                               heatmap_icp, heatmap_center):
    global patient_id
    all_click_data = [icp_click_data, click_data, differences, average_icp, average_center, heatmap_icp, heatmap_center]
    all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    if click_data:
        data = click_data["points"][0]
        if click_id == "heatmap-icp" or click_id == "heatmap-center":
            patient_id = PATIENTS[data["y"]]
        elif click_id == "average-icp" or click_id == "average-center":
            patient_id = data["x"]

    distances_center = all_distances_center[PATIENTS.index(patient_id)]
    prostate, bladder, rectum, bones = distances_center[0], distances_center[1], distances_center[2], distances_center[3]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(
                           text="Distances of the centered organs and the plan organs of patient {}".format(patient_id),
                           font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=prostate, mode="lines+markers", name="Prostate"))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder"))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum"))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    if click_data and "average" not in click_id:
        if "organs" in click_id or "alignment-differences" in click_id:
            x = int(click_data["points"][0]["x"])
        if "heatmap" in click_id:
            x = int(click_data["points"][0]["x"]) // 3 + 1
        fig.add_trace(
            go.Scattergl(x=[x], y=[prostate[x - 1]], mode="lines+markers", name="Prostate", showlegend=False,
                         marker=dict(size=10, color="white")))
        fig.add_trace(
            go.Scattergl(x=[x], y=[bladder[x - 1]], mode="lines+markers", name="Bladder", showlegend=False,
                         marker=dict(size=10, color="white")))
        fig.add_trace(go.Scattergl(x=[x], y=[rectum[x - 1]], mode="lines+markers", name="Rectum", showlegend=False,
                                   marker=dict(size=10, color="white")))

    return fig


@app.callback(
    Output("alignment-differences", "figure"),
    Input("patient-dropdown", "value"),
    Input("alignment-differences", "clickData"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"))
def create_distances_between_alignments(patient, differences, organs_icp, organs_center, average_icp, average_center,
                               heatmap_icp, heatmap_center):
    global patient_id
    dist_icp = all_distances_icp[PATIENTS.index(patient_id)]
    dist_center = all_distances_center[PATIENTS.index(patient_id)]
    distances = np.array(dist_icp) - np.array(dist_center[:3])
    prostate, bladder, rectum = distances[0], distances[1], distances[2]

    colors1, colors2, colors3 = [LIGHT_BLUE] * 13, [RED] * 13, [GREEN] * 13

    # min_val = min(min(np.min(distances_center), np.min(distances_icp)), np.min(distances))
    # max_val = max(max(np.max(distances_center), np.max(distances_icp)), np.max(distances))

    all_click_data = [differences, organs_icp, organs_center, average_icp, average_center, heatmap_icp, heatmap_center]
    all_ids = ["alignment-differences", "organs-icp", "organs-center", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    if click_data:
        data = click_data["points"][0]
        if "heatmap" in click_id:
            x = int(click_data["points"][0]["x"]) // 3
            colors1[x], colors2[x], colors3[x] = "white", "white", "white"
        elif "organs" in click_id or click_id == "alignment-differences":
            x = int(data["x"]) - 1
            colors1[x], colors2[x], colors3[x] = "white", "white", "white"

        # fig['data'][0]['marker']['color'][click_data['points'][0]['pointNumber']] = 'red'
        # fig.add_trace(go.Bar(x=[x], y=[prostate[x-1]], name="Prostate", marker_color="white", showlegend=False))
        # fig.add_trace(go.Bar(x=[x], y=[bladder[x-1]], name="Bladder", marker_color="white", showlegend=False))
        # fig.add_trace(go.Bar(x=[x], y=[rectum[x-1]], name="Rectum", marker_color="white", showlegend=False))

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90, r=40),
                       plot_bgcolor='rgba(70,70,70,1)', width=1420, height=350,
                       title=dict(text="Difference of distances between two alignments",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=prostate, name="Prostate", marker_color=colors1))
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=bladder, name="Bladder", marker_color=colors2))
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=rectum, name="Rectum", marker_color=colors3))

    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2)   # , range=[min_val - 1, max_val + 1]
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


avrg_prostate_icp, avrg_bladder_icp, avrg_rectum_icp, all_distances_icp = [], [], [], []
for pat in PATIENTS:
    dist_icp = Project_2.compute_distances_after_icp(pat)
    all_distances_icp.append(dist_icp)
    prostate, bladder, rectum = Project_2.compute_average_distances(dist_icp)
    avrg_prostate_icp.append(prostate)
    avrg_bladder_icp.append(bladder)
    avrg_rectum_icp.append(rectum)

avrg_prostate_cent, avrg_bladder_cent, avrg_rectum_cent, all_distances_center = [], [], [], []

for pat in PATIENTS:
    dist_center = Project_2.compute_distances_after_centering(pat)
    all_distances_center.append(dist_center)
    prostate, bladder, rectum = Project_2.compute_average_distances(dist_center)
    avrg_prostate_cent.append(prostate)
    avrg_bladder_cent.append(bladder)
    avrg_rectum_cent.append(rectum)


@app.callback(
    Output("average-icp", "figure"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"))
def average_distances_icp(click_data, center_click_data, heatmap_icp, heatmap_center):
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350, showlegend=True,
                       title=dict(text="Average movements of patients' organs after ICP aligning",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_prostate_icp, mode="markers", name="Prostate",
                             marker=dict(symbol="circle", color=LIGHT_BLUE), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bladder_icp, mode="markers", name="Bladder",
                             marker=dict(symbol="square", color=GREEN), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_rectum_icp, mode="markers", name="Rectum",
                             marker=dict(symbol="diamond", color=RED), line=dict(width=5)))

    all_click_data = [click_data, center_click_data, heatmap_icp, heatmap_center]
    all_ids = ["average-icp", "average-center", "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    # highlight data
    if click_data:
        data = click_data["points"][0]
        if "heatmap" in click_id:
            x, y = [int(PATIENTS[data["y"]])], data["y"]
        else:
            x, y = [data["x"]], PATIENTS.index(data["x"])
        fig.add_trace(go.Scatter(x=x, y=[avrg_prostate_icp[y]], mode="markers", showlegend=False,
                                 name="Prostate", marker=dict(symbol="circle", color=LIGHT_BLUE,
                                 line=dict(width=3, color="white"))))
        fig.add_trace(go.Scatter(x=x, y=[avrg_bladder_icp[y]], mode="markers", showlegend=False,
                                 name="Bladder", marker=dict(symbol="square", color=GREEN,
                                 line=dict(width=3, color="white"))))
        fig.add_trace(go.Scatter(x=x, y=[avrg_rectum_icp[y]], mode="markers", showlegend=False,
                                 name="Rectum", marker=dict(symbol="diamond", color=RED,
                                                            line=dict(width=3, color="white"))))

    fig.update_traces(marker=dict(size=12))

    fig.update_xaxes(title_text="Patient")
    fig.update_yaxes(title_text="Average distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


@app.callback(
    Output("average-center", "figure"),
    Input("average-center", "clickData"),
    Input("average-icp", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"))
def average_distances_center(click_data, icp_click_data, heatmap_icp, heatmap_center):
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350, showlegend=True,
                       title=dict(text="Average movements of patients' organs after Centering",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_prostate_cent, mode="markers", name="Prostate",
                             marker=dict(symbol="circle", color=LIGHT_BLUE), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bladder_cent, mode="markers", name="Bladder",
                             marker=dict(symbol="square", color=GREEN), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_rectum_cent, mode="markers", name="Rectum",
                             marker=dict(symbol="diamond", color=RED), line=dict(width=5)))

    all_click_data = [icp_click_data, click_data, heatmap_icp, heatmap_center]
    all_ids = ["average-icp", "average-center", "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    # highlight data
    if click_data:
        data = click_data["points"][0]
        if "heatmap" in click_id:
            x, y = [int(PATIENTS[data["y"]])], data["y"]
        else:
            x, y = [data["x"]], PATIENTS.index(data["x"])
        fig.add_trace(go.Scatter(x=x, y=[avrg_prostate_cent[y]], mode="markers", showlegend=False,
                                 name="Prostate", marker=dict(symbol="circle", color=LIGHT_BLUE,
                                                              line=dict(width=3, color="white"))))
        fig.add_trace(go.Scatter(x=x, y=[avrg_bladder_cent[y]], mode="markers", showlegend=False,
                                 name="Bladder", marker=dict(symbol="square", color=GREEN,
                                                             line=dict(width=3, color="white"))))
        fig.add_trace(go.Scatter(x=x, y=[avrg_rectum_cent[y]], mode="markers", showlegend=False,
                                 name="Rectum", marker=dict(symbol="diamond", color=RED,
                                                            line=dict(width=3, color="white"))))

    fig.update_traces(marker=dict(size=12))

    fig.update_xaxes(title_text="Patient")
    fig.update_yaxes(title_text="Average distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


def resolve_click_data(click_data, ids):
    input_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    for i, click_id in zip(range(len(ids)), ids):
        if input_id == click_id:
            return click_data[i], input_id
    return None, None


@app.callback(
    Output("heatmap-icp", "figure"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("alignment-differences", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"))
def heatmap_icp(organs_icp, organs_center, differences, click_data, center_click_data, average_icp, average_center):
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=1420, height=350, showlegend=True,
                       title=dict(text="Movements of patients' organs after ICP aligning",
                                  font=dict(size=18, color='lightgrey')))
    global patient_id

    data, custom_data, hover_text = create_data_for_heatmap_icp()

    fig = go.Figure(data=go.Heatmap(z=data, text=hover_text, customdata=custom_data,
                                    hovertemplate="<b>%{text}</b><br>Patient: %{y}<br>Timestamp: %{customdata}<br>"
                                                  "Distance: %{z:.2f}<extra></extra>", colorscale="Portland"), layout=layout)
    # create borders around cells
    for i in range(1, 13):
        fig.add_vline(x=3 * i - 0.5, line_width=3)

    for i in range(1, 8):
        fig.add_hline(y=i - 0.5, line_width=5)

    # highlight the selected cell
    all_click_data = [organs_icp, organs_center, differences, average_icp, average_center, click_data, center_click_data]
    all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    if click_data:
        cell = click_data["points"][0]
        if "heatmap" in click_id:
            fig.add_shape(type="rect", x0=cell["x"] - 0.43, y0=cell["y"] - 0.41, x1=cell["x"] + 0.43,
                          y1=cell["y"] + 0.41, line_color="white", line_width=4)
        elif "average" in click_id:
                y = PATIENTS.index(cell["x"])
                fig.add_hrect(y0=y - 0.41, y1=y + 0.41, line_width=4, line_color="white")
        else:
            y = PATIENTS.index(patient_id)
            fig.add_shape(type="rect", x0=cell["x"] * 3 - 3.43, y0=y - 0.41, x1=cell["x"] * 3 -0.57,
                          y1=y + 0.41, line_color="white", line_width=4)

    fig.update_xaxes(title_text="Timestamps", ticktext=TIMESTAMPS, tickmode="array", tickvals=np.arange(1, 39, 3),
                     zeroline=False, showgrid=False)
    fig.update_yaxes(title_text="Patients", ticktext=PATIENTS, tickmode="array", tickvals=np.arange(0, 8, 1),
                     zeroline=False, showgrid=False)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)
    fig.update_coloraxes(colorbar_dtick=1)

    return fig


@app.callback(
    Output("heatmap-center", "figure"),
    Input("heatmap-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("alignment-differences", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"))
def heatmap_centering(click_data, icp_click_data, differences, average_icp, average_center, organs_icp, organs_center):
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=1420, height=350, showlegend=True,
                       title=dict(text="Movements of patients' organs after centering on prostate",
                                  font=dict(size=18, color='lightgrey')))
    global patient_id

    data, custom_data, hover_text = create_data_for_heatmap_center()

    fig = go.Figure(data=go.Heatmap(z=data, text=hover_text, customdata=custom_data,
                                    hovertemplate="<b>%{text}</b><br>Patient: %{y}<br>Timestamp: %{customdata}<br>"
                                                  "Distance: %{z:.2f}<extra></extra>",
                                    colorscale="Portland"), layout=layout)   # [[0, BLUE], [0.5, RED], [1, GREEN]]
    for i in range(1, 13):
        fig.add_vline(x=3 * i - 0.5, line_width=3)

    for i in range(1, 8):
        fig.add_hline(y=i - 0.5, line_width=5)

        # highlight the selected cell
        all_click_data = [organs_icp, organs_center, differences, average_icp, average_center,
                          icp_click_data, click_data]
        all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
                   "heatmap-icp", "heatmap-center"]
        click_data, click_id = resolve_click_data(all_click_data, all_ids)

        if click_data:
            cell = click_data["points"][0]
            if "heatmap" in click_id:
                fig.add_shape(type="rect", x0=cell["x"] - 0.43, y0=cell["y"] - 0.41, x1=cell["x"] + 0.43,
                              y1=cell["y"] + 0.41, line_color="white", line_width=4)
            elif "average" in click_id:
                y = PATIENTS.index(cell["x"])
                fig.add_hrect(y0=y - 0.41, y1=y + 0.41, line_width=4, line_color="white")
            else:
                y = PATIENTS.index(patient_id)
                fig.add_shape(type="rect", x0=cell["x"] * 3 - 3.43, y0=y - 0.41, x1=cell["x"] * 3 - 0.57,
                              y1=y + 0.41, line_color="white", line_width=4)

    fig.update_xaxes(title_text="Timestamps", ticktext=TIMESTAMPS, tickmode="array", tickvals=np.arange(1, 39, 3),
                     zeroline=False, showgrid=False)
    fig.update_yaxes(title_text="Patients", ticktext=PATIENTS, tickmode="array", tickvals=np.arange(0, 8, 1),
                     zeroline=False, showgrid=False)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)
    fig.update_coloraxes(colorbar_dtick=1)

    return fig


def create_data_for_heatmap_icp():
    # data is 2d array with distances for the heightmap, custom_data and hover_text are used just for hover labels
    data, custom_data, hover_text = [], [], []
    for i in range(len(PATIENTS)):
        # patient contains three arrays: prostate, bladder, rectum with distances from all the timestamps
        patient = all_distances_icp[i]
        data_row, custom_row, hover_row = [], [], []

        for j in range(len(TIMESTAMPS)):
            data_row.extend([patient[0][j], patient[1][j], patient[2][j]])
            custom_row.extend([j + 1, j + 1, j + 1])
            hover_row.extend(["Prostate", "Bladder", "Rectum"])

        data.append(data_row)
        custom_data.append(custom_row)
        hover_text.append(hover_row)

    return data, custom_data, hover_text


def create_data_for_heatmap_center():
    # data is 2d array with distances for the heightmap, custom_data and hover_text are used just for hover labels
    data, custom_data, hover_text = [], [], []
    for i in range(len(PATIENTS)):
        # patient contains three arrays: prostate, bladder, rectum with distances from all the timestamps
        patient = all_distances_center[i]
        data_row, custom_row, hover_row = [], [], []

        for j in range(len(TIMESTAMPS)):
            data_row.extend([patient[3][j], patient[1][j], patient[2][j]])
            custom_row.extend([j + 1, j + 1, j + 1])
            hover_row.extend(["Bones", "Bladder", "Rectum"])

        data.append(data_row)
        custom_data.append(custom_row)
        hover_text.append(hover_row)

    return data, custom_data, hover_text


if __name__ == '__main__':
    main()
    app.run_server(debug=True)
