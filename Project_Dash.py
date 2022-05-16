from copy import deepcopy
import trimesh
from dash import Dash, html, dcc, Output, Input, callback_context
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import Project_2

FILEPATH = "C:\\Users\\vitko\\Desktop\\ProjetHCI\\Organs\\"
PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]

BLUE = "#636EFA"
GREEN = "#EF553B"
RED = "#00CC96"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)


def create_meshes_from_objs(objects, color):
    meshes = []
    for elem in objects:
        x, y, z = np.array(elem[0]).T
        i, j, k = np.array(elem[1]).T
        pl_mesh = go.Mesh3d(x=x, y=y, z=z, color=color, flatshading=True, i=i, j=j, k=k, showscale=False, opacity=1)
        meshes.append(pl_mesh)
    return meshes


def order_slice_vertices(vertices, indices):
    ordered_vertices = []
    for index in indices:
        ordered_vertices.append(vertices[index])

    return ordered_vertices


def main():
    app.layout = html.Div(className="row", children=[
        html.Div(className='six columns', children=[
            html.H6("Select the patient:",
                    style={'display': 'inline-block', "padding": "20px 101px 0px 45px"}),
            dcc.Dropdown(options=PATIENTS, value="137", searchable=False,
                         id="patient-dropdown", style={'display': 'inline-block', "width": "80px", "font-size": "16px",
                                                       "padding": "0px 80px 0px 85px"}),

            dcc.Graph(figure=create_patients_icp(), style={'display': 'inline-block', "padding": "20px 30px 0px 40px"}),

            dcc.Graph(id="organs-icp", style={'display': 'inline-block', "padding": "30px 30px 30px 40px"}),

            dcc.Graph(id="alignment-differences", style={"padding": "0px 0px 30px 40px"}),

            html.H6("Select the method of alignment:",
                    style={'display': 'inline-block', "padding": "10px 50px 0px 45px"}),
            dcc.Checklist(options=["ICP", "Center point"], value=["ICP"], inline=True, id="alignment-checklist",
                          style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"}),

            html.H6("Select the visibility of organs/bones:",
                    style={'display': 'inline-block', "padding": "0px 0px 0px 45px"}),
            dcc.Checklist(options=["Bones", "Prostate", "Bladder", "Rectum"], value=["Prostate"], inline=True,
                          id="organs-checklist",
                          style={'display': 'inline-block', "font-size": "18px", "padding": "0px 0px 0px 25px"}),

            dcc.Graph(id="main-graph", style={'display': 'inline-block', "padding": "20px 30px 0px 40px"})]),



        html.Div(className='six columns', children=[
            dcc.Graph(figure=create_patients_center(), style={'display': 'inline-block', "padding": "87px 0px 0px 0px"}),

            dcc.Graph(id="organs-center", style={'display': 'inline-block', "padding": "30px 30px 30px 0px"}),

            html.Div(className="row", children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(id="x-slice-graph", style={'display': 'inline-block', "padding": "400px 0px 10px 0px"}),
                    dcc.Graph(id="z-slice-graph", style={'display': 'inline-block', "padding": "0px 0px 0px 0px"})]),
                html.Div(className='six columns', children=[
                    dcc.Graph(id="y-slice-graph", style={'display': 'inline-block', "padding": "400px 0px 10px 0px"}),

                    html.Div(className="row", children=[
                        html.H6("X axes slice:", style={'display': 'inline-block', "padding": "5px 0px 0px 30px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="x-slice-slider", marks=None, updatemode="drag"),
                        html.H6("Y axes slice:", style={'display': 'inline-block', "padding": "5px 0px 0px 35px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="y-slice-slider", marks=None),
                        html.H6("Z axes slice:", style={'display': 'inline-block', "padding": "5px 0px 0px 30px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="z-slice-slider", marks=None)],
                             style={'width': '80%', 'display': 'inline-block', "padding": "5px 0px 0px 5px"})])])])])


@app.callback(
    Output("main-graph", "figure"),
    Input("patient-dropdown", "value"),
    Input("alignment-checklist", "value"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("organs-checklist", "value"))
def update_3dgraph(patient, alignment_checklist, icp_click_data, center_click_data, organs):
    timestamp = timestamp_click_data(icp_click_data, center_click_data)
    objects = import_selected_organs(organs, timestamp, patient)

    after_icp_meshes, vec, center = get_meshes_after_icp(timestamp, objects, patient)
    after_center_meshes = get_meshes_after_centering(timestamp, objects, patient)

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.5, y=-2, z=0))
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', height=520, width=680,
                       plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=40, r=40, t=60, b=40), showlegend=True)
    fig = go.Figure(layout=layout)
    fig.update_layout(scene_camera=camera)

    meshes = []
    if "ICP" in alignment_checklist:
        meshes.extend(after_icp_meshes)

    if "Center point" in alignment_checklist:
        meshes.extend(after_center_meshes)

    for mesh in meshes:
        mesh.update(cmin=-7, lightposition=dict(x=100, y=200, z=0), lighting=dict(ambient=0.4, diffuse=1,
                                                                                  fresnel=0.1, specular=1,
                                                                                  roughness=0.5,
                                                                                  facenormalsepsilon=1e-15,
                                                                                  vertexnormalsepsilon=1e-15))
        fig.add_trace(mesh)
    fig.update_layout(title_text="Patient {}, timestamp number {}".format(patient, timestamp), title_x=0.5,
                      title_y=0.95)

    fig.add_trace(go.Scatter3d(x=[center[0], vec[0]], y=[center[1], vec[1]], z=[center[2], vec[2]]))
    # fig.add_trace(go.Scatter3d(x=[0, 20], y=[0, 0], z=[50, 50]))

    return fig


def import_selected_organs(organs, timestamp, patient):
    objects = []

    if "Bones" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, timestamp)]))
    if "Prostate" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, timestamp)]))
    if "Bladder" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, timestamp)]))
    if "Rectum" in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, timestamp)]))

    return objects


def get_meshes_after_icp(timestamp, objects, patient):
    plan_bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
    bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, timestamp)])

    transform_matrix, vec = Project_2.icp_rot_vec(bones[0][0], plan_bones[0][0])
    transfr_objects = Project_2.vertices_transformation(transform_matrix, deepcopy(objects))
    after_icp_meshes = create_meshes_from_objs(transfr_objects, "#1F77B4")

    bones_center = Project_2.find_center_point(bones[0][0])
    # print(vec, timestamp)

    return after_icp_meshes, vec, bones_center


def get_meshes_after_centering(timestamp, objects, patient):
    prostate = Project_2.import_obj([FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, timestamp)])

    plan_center = Project_2.find_center_point(Project_2.import_obj(
        [FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)])[0][0])
    other_center = Project_2.find_center_point(prostate[0][0])
    center_matrix = Project_2.create_translation_matrix(plan_center, other_center)
    center_transfr_objects = Project_2.vertices_transformation(center_matrix, deepcopy(objects))
    after_center_meshes = create_meshes_from_objs(center_transfr_objects, "orange")

    return after_center_meshes


def timestamp_click_data(icp_click_data, center_click_data):
    icp_click_data = int(icp_click_data["points"][0]["x"]) if icp_click_data is not None else 1
    center_click_data = int(center_click_data["points"][0]["x"]) if center_click_data is not None else 1

    input_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    timestamp = 1

    if input_id == "organs-center":
        timestamp = center_click_data
    elif input_id == "organs-icp":
        timestamp = icp_click_data

    return timestamp


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
    Input("alignment-checklist", "value"))
def create_graph_slices(x_slider, y_slider, z_slider, icp_click_data, center_click_data, organs, patient, method):
    figures, centered_meshes, icp_meshes = [], [], []
    names = ["X axis slice", "Y axis slice", "Z axis slice"]
    timestamp = timestamp_click_data(icp_click_data, center_click_data)

    if "Center point" in method:
        plan_prostate = trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient))
        prostate = trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, timestamp))
        key_center = Project_2.find_center_point(plan_prostate.vertices)
        other_center = Project_2.find_center_point(prostate.vertices)
        center_matrix = Project_2.create_translation_matrix(key_center, other_center)
        centered_meshes = selected_organs_slices(center_matrix, organs, timestamp, patient)

    if "ICP" in method:
        plan_bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
        bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, timestamp)])
        icp_matrix = Project_2.icp_transformation_matrices(bones[0][0], plan_bones[0][0], False)
        icp_meshes = selected_organs_slices(icp_matrix, organs, timestamp, patient)

    for i in range(3):
        layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', height=310,
                           width=320, plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=30, r=20, t=60, b=40),
                           showlegend=False, title=dict(text=names[i]))
        fig = go.Figure(layout=layout)
        fig.update_layout(title_x=0.5)
        figures.append(fig)

    x_fig = create_slice_final(x_slider, centered_meshes, icp_meshes, figures[0], "x")
    y_fig = create_slice_final(y_slider, centered_meshes, icp_meshes, figures[1], "y")
    z_fig = create_slice_final(z_slider, centered_meshes, icp_meshes, figures[2], "z")

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


def create_slice_final(slice_slider, centered_meshes, icp_meshes, fig, axis):
    if centered_meshes:
        create_slice_helper(centered_meshes, slice_slider, fig, "orange", axis)
    if icp_meshes:
        create_slice_helper(icp_meshes, slice_slider, fig, "#1F77B4", axis)

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
    Input("patient-dropdown", "value"))
def create_distances_after_icp(patient):
    distances_icp = Project_2.compute_distances_after_icp(patient)
    prostate, bladder, rectum = distances_icp[0], distances_icp[1], distances_icp[2]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70, l=90),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(text="Distances of icp aligned organs and the plan organs",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=np.array(range(1, 14)), y=prostate, mode="lines+markers", name="Prostate"))
    fig.add_trace(go.Scatter(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder"))
    fig.add_trace(go.Scatter(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum"))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


@app.callback(
    Output("organs-center", "figure"),
    Input("patient-dropdown", "value"))
def create_distances_after_centering(patient):
    distances_center = Project_2.compute_distances_after_centering(patient)
    prostate, bladder, rectum = distances_center[0], distances_center[1], distances_center[2]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70, l=90),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(text="Distances of the centered organs and the plan organs",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=np.array(range(1, 14)), y=prostate, mode="lines+markers", name="Prostate"))
    fig.add_trace(go.Scatter(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder"))
    fig.add_trace(go.Scatter(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum"))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


# TODO: make grraph interactive by changing the time by clicking
@app.callback(
    Output("alignment-differences", "figure"),
    Input("patient-dropdown", "value"))
def create_distances_between_alignments(patient):
    distances_icp = Project_2.compute_distances_after_icp(patient)
    distances_center = Project_2.compute_distances_after_centering(patient)
    distances = np.array(distances_icp) - np.array(distances_center)
    prostate, bladder, rectum = distances[0], distances[1], distances[2]

    min_val = min(min(np.min(distances_center), np.min(distances_icp)), np.min(distances))
    max_val = max(max(np.max(distances_center), np.max(distances_icp)), np.max(distances))

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70, l=90, r=40),
                       plot_bgcolor='rgba(70,70,70,1)', width=1420, height=350,
                       title=dict(text="Difference of distances between two alignments",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=prostate, name="Prostate"))
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=bladder, name="Bladder"))
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=rectum, name="Rectum"))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2, range=[min_val - 1, max_val + 1])
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


def create_patients_icp():
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350, showlegend=True,
                       title=dict(text="Average movements of patients' organs after ICP aligning",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    avrg_prostate, avrg_bladder, avrg_rectum = [], [], []

    for patient in PATIENTS:
        distances_icp = Project_2.compute_distances_after_icp(patient)
        prostate, bladder, rectum = Project_2.compute_average_distances(distances_icp)
        avrg_prostate.append(prostate)
        avrg_bladder.append(bladder)
        avrg_rectum.append(rectum)

    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_prostate, mode="markers", name="Prostate",
                             marker=dict(symbol="circle", color=BLUE), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bladder, mode="markers", name="Bladder",
                             marker=dict(symbol="square", color=GREEN), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_rectum, mode="markers", name="Rectum",
                             marker=dict(symbol="diamond", color=RED), line=dict(width=5)))
    fig.update_traces(marker=dict(size=12))

    fig.update_xaxes(title_text="Patient")
    fig.update_yaxes(title_text="Average distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


def create_patients_center():
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350, showlegend=True,
                       title=dict(text="Average movements of patients' organs after Centering",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)

    avrg_prostate, avrg_bladder, avrg_rectum = [], [], []
    for patient in PATIENTS:
        distances_center = Project_2.compute_distances_after_centering(patient)
        prostate, bladder, rectum = Project_2.compute_average_distances(distances_center)
        avrg_prostate.append(prostate)
        avrg_bladder.append(bladder)
        avrg_rectum.append(rectum)

    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_prostate, mode="markers", name="Prostate",
                             marker=dict(symbol="circle", color=BLUE), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bladder, mode="markers", name="Bladder",
                             marker=dict(symbol="square", color=GREEN), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_rectum, mode="markers", name="Rectum",
                             marker=dict(symbol="diamond", color=RED), line=dict(width=5)))

    fig.update_traces(marker=dict(size=12))

    fig.update_xaxes(title_text="Patient")
    fig.update_yaxes(title_text="Average distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


if __name__ == '__main__':
    main()
    app.run_server(debug=True)

# 722 4 visible circles
