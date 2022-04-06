from copy import deepcopy
import trimesh
from dash import Dash, html, dcc, Output, Input, State, callback_context
import numpy as np
import plotly.graph_objects as go
import Project_2

BONES = "OBJ_images/bones_plan.obj"
PROSTATE = "OBJ_images/prostate_plan.obj"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)


def create_meshes_from_objs(objects, color_scale):
    meshes = []
    for elem in objects:
        x, y, z = np.array(elem[0]).T
        i, j, k = np.array(elem[1]).T
        pl_mesh = go.Mesh3d(x=x, y=y, z=z, colorscale=color_scale,
                            intensity=z, flatshading=True, i=i, j=j, k=k, showscale=False, opacity=1)
        meshes.append(pl_mesh)
    return meshes


def order_slice_vertices(vertices, indices):
    ordered_vertices = []
    for index, _ in indices:
        ordered_vertices.append(vertices[index])

    ordered_vertices.append(vertices[0])
    return ordered_vertices


def main():
    app.layout = html.Div(className="row", children=[
        html.Div(className='six columns', children=[
            html.H6("Select the method of alignment or both for comparison:",
                    style={'display': 'inline-block', "padding": "20px 50px 0px 45px"}),

            dcc.Checklist(options=["ICP", "Center point"], value=["ICP"], inline=True, id="alignment-checklist",
                          style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 45px"}),

            html.H6("Select organs/bones you want to see:",
                    style={'display': 'inline-block', "padding": "20px 50px 0px 45px"}),

            dcc.Checklist(options=["Bones", "Prostate", "Bladder", "Rectum"], value=["Prostate"], inline=True,
                          id="organs-checklist",
                          style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 45px"}),

            dcc.Graph(id="main-graph", style={'display': 'inline-block', "padding": "30px 30px 0px 40px"}),

            dcc.Graph(id="organs-icp", figure=create_distances_after_icp(),
                      style={'display': 'inline-block', "padding": "30px 30px 30px 40px"})]),


        html.Div(className='six columns', children=[
            html.H6("Select organs for slicing:",
                    style={'display': 'inline-block', "padding": "20px 100px 0px 150px"}),

            dcc.Checklist(options=["Prostate", "Bladder", "Rectum"], value=["Prostate"], inline=True,
                          id="slices-checklist",
                          style={'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 45px"}),

            html.Div(className="row", children=[
                html.Div(className='six columns', children=[
                    dcc.Graph(id="x-slice-graph", style={'display': 'inline-block', "padding": "30px 0px 10px 0px"}),

                    dcc.Graph(id="z-slice-graph", style={'display': 'inline-block', "padding": "0px 0px 0px 0px"})]),

                html.Div(className='six columns', children=[
                    dcc.Graph(id="y-slice-graph", style={'display': 'inline-block', "padding": "30px 0px 10px 0px"}),

                    html.Div(className="row", children=[
                        html.H6("X axes slice:", style={'display': 'inline-block', "padding": "5px 0px 0px 50px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="x-slice-slider", marks=None),
                        html.H6("Y axes slice:", style={'display': 'inline-block', "padding": "5px 0px 0px 50px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="y-slice-slider", marks=None),
                        html.H6("Z axes slice:", style={'display': 'inline-block', "padding": "5px 0px 0px 50px"}),
                        dcc.Slider(min=0, max=1, value=0.5, id="z-slice-slider", marks=None)],
                             style={'width': '80%', 'display': 'inline-block', "padding": "5px 0px 0px 25px"})])]),

            dcc.Graph(id="organs-center", figure=create_distances_after_centering(),
                      style={'display': 'inline-block', "padding": "30px 30px 30px 0px"})])])


@app.callback(
    Output("main-graph", "figure"),
    Input("alignment-checklist", "value"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("organs-checklist", "value"))
def update_3dgraph(alignment_checklist, icp_click_data, center_click_data, organs):
    timestamp = timestamp_click_data(icp_click_data, center_click_data)
    objects = import_selected_organs(organs, timestamp)

    after_icp_meshes = get_meshes_after_icp(timestamp, objects)
    after_center_meshes = get_meshes_after_centering(timestamp, objects)

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.5, y=-2, z=0))
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', height=520, width=680,
                       plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=40, r=40, t=40, b=40), showlegend=True)
    fig = go.Figure(layout=layout)
    fig.update_layout(scene_camera=camera)

    meshes = []
    if "ICP" in alignment_checklist:
        meshes.extend(after_icp_meshes)

    if "Center point" in alignment_checklist:
        meshes.extend(after_center_meshes)

    for mesh in meshes:
        mesh.update(cmin=-7, lightposition=dict(x=100, y=200, z=0), lighting=dict(ambient=0.4, diffuse=1,
                    fresnel=0.1, specular=1, roughness=0.5, facenormalsepsilon=1e-15, vertexnormalsepsilon=1e-15))
        fig.add_trace(mesh)

    return fig


def import_selected_organs(organs, timestamp):
    objects = []

    if "Bones" in organs:
        objects.extend(Project_2.import_obj(["137_bones\\bones{}.obj".format(timestamp)]))
    if "Prostate" in organs:
        objects.extend(Project_2.import_obj(["137_prostate\\prostate{}.obj".format(timestamp)]))
    if "Bladder" in organs:
        objects.extend(Project_2.import_obj(["137_bladder\\bladder{}.obj".format(timestamp)]))
    if "Rectum" in organs:
        objects.extend(Project_2.import_obj(["137_rectum\\rectum{}.obj".format(timestamp)]))

    return objects


def get_meshes_after_icp(timestamp, objects):
    plan_bones = Project_2.import_obj([BONES])
    bones = Project_2.import_obj(["137_bones\\bones{}.obj".format(timestamp)])

    transform_matrix = Project_2.icp_transformation_matrices(bones[0][0], plan_bones[0][0], False)
    transfr_objects = Project_2.vertices_transformation(transform_matrix, deepcopy(objects))
    after_icp_meshes = create_meshes_from_objs(transfr_objects, "darkmint")

    return after_icp_meshes


def get_meshes_after_centering(timestamp, objects):
    prostate = Project_2.import_obj(["137_prostate\\prostate{}.obj".format(timestamp)])

    plan_center = Project_2.find_center_point(Project_2.import_obj([PROSTATE])[0][0])
    other_center = Project_2.find_center_point(prostate[0][0])
    center_matrix = Project_2.create_translation_matrix(plan_center, other_center)
    center_transfr_objects = Project_2.vertices_transformation(center_matrix, deepcopy(objects))
    after_center_meshes = create_meshes_from_objs(center_transfr_objects, "peach")

    return after_center_meshes


def timestamp_click_data(icp_click_data, center_click_data):
    icp_click_data = int(icp_click_data["points"][0]["x"]) + 1 if icp_click_data is not None else 1
    center_click_data = int(center_click_data["points"][0]["x"]) + 1 if center_click_data is not None else 1

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
    Input("slices-checklist", "value"))
def create_graph_slices(x_slider, y_slider, z_slider, icp_click_data, center_click_data, organs):
    figures = []
    names = ["X axis slice", "Y axis slice", "Z axis slice"]
    timestamp = timestamp_click_data(icp_click_data, center_click_data)

    key = trimesh.load_mesh(PROSTATE)
    prostate = trimesh.load_mesh("137_prostate/prostate{}.obj".format(timestamp))
    key_center = Project_2.find_center_point(key.vertices)
    other_center = Project_2.find_center_point(prostate.vertices)
    matrix = Project_2.create_translation_matrix(key_center, other_center)

    meshes, centered_meshes = selected_organs_slices(matrix, organs, timestamp, prostate)

    for i in range(3):
        layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', height=300,
                           width=340, plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=30, r=20, t=60, b=40),
                           showlegend=False, title=dict(text=names[i]))
        fig = go.Figure(layout=layout)
        fig.update_layout(title_x=0.5)
        figures.append(fig)

    x_fig = create_x_slice(x_slider, meshes, centered_meshes, figures[0])
    y_fig = create_y_slice(y_slider, meshes, centered_meshes, figures[1])
    z_fig = create_z_slice(z_slider, meshes, centered_meshes, figures[2])

    return x_fig, y_fig, z_fig


def selected_organs_slices(matrix, organs, timestamp, prostate):
    meshes, centered_meshes = [], []

    if "Prostate" in organs:
        meshes.append(prostate)
        centered_meshes.append(deepcopy(prostate).apply_transform(matrix))

    if "Bladder" in organs:
        mesh = trimesh.load_mesh("137_bladder\\bladder{}.obj".format(timestamp))
        meshes.append(mesh)
        centered_meshes.append(deepcopy(mesh).apply_transform(matrix))

    if "Rectum" in organs:
        mesh = trimesh.load_mesh("137_rectum\\rectum{}.obj".format(timestamp))
        meshes.append(mesh)
        centered_meshes.append(deepcopy(mesh).apply_transform(matrix))

    return meshes, centered_meshes


def create_x_slice(slice_slider, meshes, centered_meshes, fig):
    for mesh, centered_mesh in zip(meshes, centered_meshes):
        slope = (mesh.bounds[1][0] - 2.5) - (mesh.bounds[0][0] + 0.5)
        output_x = (mesh.bounds[0][0] + 0.5) + slope * slice_slider
        axis_slice = mesh.section(plane_origin=[output_x, mesh.centroid[1], mesh.centroid[2]], plane_normal=[1, 0, 0])
        ordered_slice = order_slice_vertices(axis_slice.vertices, axis_slice.vertex_nodes)
        _, i, j = np.array(ordered_slice).T
        fig.add_trace(go.Scatter(x=i, y=j, line=go.scatter.Line(color="red")))

        centered_slice = centered_mesh.section(plane_origin=[output_x, centered_mesh.centroid[1],
                                                             centered_mesh.centroid[2]], plane_normal=[1, 0, 0])
        centered_ordered_slice = order_slice_vertices(centered_slice.vertices, centered_slice.vertex_nodes)
        _, k, l = np.array(centered_ordered_slice).T
        fig.add_trace(go.Scatter(x=k, y=l, line=go.scatter.Line(color="orange")))

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    return fig


def create_y_slice(slice_slider, meshes, centered_meshes, fig):
    for mesh, centered_mesh in zip(meshes, centered_meshes):
        slope = (mesh.bounds[1][1] - 2.5) - (mesh.bounds[0][1] + 0.5)
        output_y = (mesh.bounds[0][1] + 0.5) + slope * slice_slider
        axis_slice = mesh.section(plane_origin=[mesh.centroid[0], output_y, mesh.centroid[2]], plane_normal=[0, 1, 0])
        ordered_slice = order_slice_vertices(axis_slice.vertices, axis_slice.vertex_nodes)
        i, _, j = np.array(ordered_slice).T
        fig.add_trace(go.Scatter(x=i, y=j, line=go.scatter.Line(color="yellow")))

        centered_slice = centered_mesh.section(plane_origin=[centered_mesh.centroid[0], output_y,
                                                             centered_mesh.centroid[2]], plane_normal=[0, 1, 0])
        centered_ordered_slice = order_slice_vertices(centered_slice.vertices, centered_slice.vertex_nodes)
        k, _, l = np.array(centered_ordered_slice).T
        fig.add_trace(go.Scatter(x=k, y=l, line=go.scatter.Line(color="lightgreen")))

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    return fig


def create_z_slice(slice_slider, meshes, centered_meshes, fig):
    for mesh, centered_mesh in zip(meshes, centered_meshes):
        slope = (mesh.bounds[1][2] - 2.5) - (mesh.bounds[0][2] + 0.5)
        output_z = (mesh.bounds[0][2] + 0.5) + slope * slice_slider
        axis_slice = mesh.section(plane_origin=[mesh.centroid[0], mesh.centroid[1], output_z], plane_normal=[0, 0, 1])
        ordered_slice = order_slice_vertices(axis_slice.vertices, axis_slice.vertex_nodes)
        i, j, _ = np.array(ordered_slice).T
        fig.add_trace(go.Scatter(x=j, y=i, line=go.scatter.Line(color="blue")))

        centered_slice = centered_mesh.section(plane_origin=[centered_mesh.centroid[0], centered_mesh.centroid[1],
                                                             output_z], plane_normal=[0, 0, 1])
        centered_ordered_slice = order_slice_vertices(centered_slice.vertices, centered_slice.vertex_nodes)
        k, l, _ = np.array(centered_ordered_slice).T
        fig.add_trace(go.Scatter(x=l, y=k, line=go.scatter.Line(color="purple")))

    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x")

    return fig


def create_distances_after_icp():
    distances = Project_2.compute_distances_after_icp()
    prostate, bladder, rectum = distances[0], distances[1], distances[2]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=100, b=70),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(text="Distances of icp aligned organs and the plan organs",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=np.array(range(0, 13)), y=prostate, mode="lines+markers", name="Prostate"))
    fig.add_trace(go.Scatter(x=np.array(range(0, 13)), y=bladder, mode="lines+markers", name="Bladder"))
    fig.add_trace(go.Scatter(x=np.array(range(0, 13)), y=rectum, mode="lines+markers", name="Rectum"))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5)

    return fig


def create_distances_after_centering():
    distances = Project_2.compute_distances_after_centering()
    prostate, bladder, rectum = distances[0], distances[1], distances[2]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=100, b=70),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(text="Distances of the centered organs and the plan organs",
                                  font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=np.array(range(0, 13)), y=prostate, mode="lines+markers", name="Prostate"))
    fig.add_trace(go.Scatter(x=np.array(range(0, 13)), y=bladder, mode="lines+markers", name="Bladder"))
    fig.add_trace(go.Scatter(x=np.array(range(0, 13)), y=rectum, mode="lines+markers", name="Rectum"))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5)

    return fig


if __name__ == '__main__':
    main()
    app.run_server(debug=True)
