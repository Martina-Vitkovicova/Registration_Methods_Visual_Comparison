from copy import deepcopy
import trimesh
from dash import Dash, Output, Input, callback_context
import numpy as np
import plotly.graph_objects as go
import Project_2
import Project_Dash_html
import json

FILEPATH = "C:\\Users\\vitko\\Desktop\\ProjetHCI-BT\\BT_implementation\\Organs\\"
PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]
TIMESTAMPS = list(range(1, 14))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = Project_Dash_html.layout

LIGHT_BLUE = "#636EFA"
GREEN = "#EF553B"
RED = "#00CC96"
BLUE = "#1F77B4"
PURPLE = "#AB63FA"
ORANGE = "#FFA15A"
GREY = "#667299"

CONE_TIP = 0.1
CONE_START = 0.9

patient_id = "137"


def create_meshes_from_objs(objects, color):
    """
    Transforms imported .obj volume to go.Mesh3d
    :param objects: imported .objs
    :param color: mesh color - blue for the first, orange for the second chosen timestamp
    :return: go.Mesh3d meshes
    """
    meshes = []
    for elem in objects:
        x, y, z = np.array(elem[0]).T
        i, j, k = np.array(elem[1]).T
        pl_mesh = go.Mesh3d(x=x, y=y, z=z, color=color, flatshading=True, i=i, j=j, k=k, showscale=False)
        meshes.append(pl_mesh)
    return meshes


def order_slice_vertices(vertices, indices):
    """
    Corrects the order of the vertices from the slice
    :param vertices: vertices of the slice
    :param indices: the order of the vertices
    :return: vertices in the correct order
    """
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
    Output(component_id='movements-radioitems', component_property='style'),
    Output(component_id='movements-header', component_property='style'),
    Input(component_id='mode-radioitems', component_property='value'))
def options_visibility(mode):
    """
    Changes the 3D graph options visibility
    :param mode: either Two timestamps or Plan organs mode
    :return: display style
    """
    if mode == "Two timestamps":
        return {'display': 'inline-block', "padding": "0px 50px 0px 45px"}, \
               {'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"}, \
               {'display': 'inline-block', "padding": "0px 20px 0px 45px"}, \
               {'display': 'inline-block', "width": "50px",
                "height": "30px", "font-size": "16px", "padding": "0px 0px 0px 0px"}, \
               {'display': 'inline-block', "width": "50px",
                "height": "30px", "font-size": "16px", "padding": "0px 0px 0px 30px"}, \
               {'display': 'inline-block', "padding": "10px 0px 10px 0px"}, \
               {'display': 'inline-block', "padding": "10px 0px 0px 0px"}, {'display': 'none'}, {'display': 'none'}

    else:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, \
               {'display': 'none'}, {}, {}, \
               {'display': 'inline-block', "font-size": "18px", "padding": "0px 100px 0px 12px"}, \
               {'display': 'inline-block', "padding": "0px 0px 0px 45px"}


@app.callback(
    Output("main-graph", "figure"),
    Input("alignment-radioitems", "value"),
    Input("organs-checklist", "value"),
    Input("mode-radioitems", "value"),
    Input("fst-timestamp-dropdown", "value"),
    Input("snd-timestamp-dropdown", "value"),
    Input("opacity-slider", "value"),
    Input("movements-radioitems", "value"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"))
def create_3dgraph(method, organs, mode, fst_timestamp, snd_timestamp, opacity_slider, mov,
                   heatmap_icp, heatmap_center, average_icp, average_center):
    """
    Creates the 3D figure and visualises it. The last four arguments are just for updating the graph.
    :param method: ICP or prostate aligning registration method
    :param organs: organs selected by the user
    :param mode: showing either Plan organs or organ in the Two timestamps
    :param mov: showing either all of the movement vectors, only yhe average ones or none
    :return: the 3d figure
    """
    global patient_id

    fst_timestamp = "_plan" if fst_timestamp == "plan" else fst_timestamp
    snd_timestamp = "_plan" if snd_timestamp == "plan" else snd_timestamp

    objects_fst = import_selected_organs(organs, fst_timestamp, patient_id)
    objects_snd = import_selected_organs(organs, snd_timestamp, patient_id)

    camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0.5, y=-2, z=0))
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=40, r=40, t=60, b=40), showlegend=True)
    fig = go.Figure(layout=layout)
    fig.update_layout(scene_camera=camera)

    fst_meshes, snd_meshes = \
        decide_3d_graph_mode(mode, fig, method, organs, mov, fst_timestamp, snd_timestamp, objects_fst, objects_snd)

    for meshes in [fst_meshes, snd_meshes]:
        for mesh in meshes:
            mesh.update(cmin=-7, opacity=opacity_slider, lightposition=dict(x=100, y=200, z=0),
                        lighting=dict(ambient=0.4, diffuse=1, fresnel=0.1, specular=1, roughness=0.5,
                                      facenormalsepsilon=1e-15, vertexnormalsepsilon=1e-15))
            fig.add_trace(mesh)

    return fig


def import_selected_organs(organs, time_or_plan, patient):
    """
    Imports selected organs as .obj files.
    :param organs: selected organs
    :param time_or_plan: chosen timestamp or _plan suffix
    :param patient: id of the patient
    :return: imported objs
    """
    objects = []
    for organ in organs:
        objects.extend(Project_2.import_obj([FILEPATH + "{}\\{}\\{}{}.obj"
                                            .format(patient, organ.lower(), organ.lower(), time_or_plan)]))
    return objects


def decide_3d_graph_mode(mode, fig, method, organs, mov, fst_timestamp, snd_timestamp, objects_fst, objects_snd):
    """
    Helper function to perform commands according to the chosen mode
    :param method: ICP or prostate aligning registration method
    :param organs: organs selected by the user
    :param mode: showing either Plan organs or organ in the Two timestamps
    :param mov: showing either all of the movement vectors, only yhe average ones or none
    :return: created meshes
    """
    fst_meshes, snd_meshes = [], []

    if "Two timestamps" in mode:
        fst_meshes, snd_meshes, center1_after, center2_after = \
            two_timestamps_mode(method, fst_timestamp, snd_timestamp, objects_fst, objects_snd)
        draw_movement_arrow(fig, center1_after, center2_after)

        fst_timestamp = "plan organs" if "_plan" == fst_timestamp else "timestamp number {}".format(fst_timestamp)
        snd_timestamp = "plan organs" if "_plan" == snd_timestamp else "timestamp number {}".format(snd_timestamp)

        fig.update_layout(title_text="Patient {}, {} (blue) and {} (orange)"
                          .format(patient_id, fst_timestamp, snd_timestamp), title_x=0.5, title_y=0.95)

    else:
        objects = import_selected_organs(organs, "_plan", patient_id)
        fst_meshes = create_meshes_from_objs(objects, BLUE)
        text = plan_organs_mode(fig, mov, organs)
        fig.update_layout(title_text="Plan organs of patient {} {}".format(patient_id, text), title_x=0.5, title_y=0.95)

    return fst_meshes, snd_meshes


def plan_organs_mode(fig, mov, organs):
    """
    Decides the text for the 3D graph and mode of the movement vectors
    :param fig: 3D figure
    :param mov: the mode of the movement vectors selected
    :param organs: chosen organs
    :return: the correct text
    """
    text = ""
    if "average" in mov:
        create_average_movement_lines(fig, organs)
        text = "with the vectors of average organs movements"
    elif "all" in mov:
        draw_all_movements(fig, organs)
        text = "with all of the organ movements vectors"

    return text


def two_timestamps_mode(method, fst_timestamp, snd_timestamp, objects_fst, objects_snd):
    """
    Helper to get the chosen meshes and align them according to the selected method. Compute the center of the meshes.
    :param method: ICP or prostate centering
    :param fst_timestamp: number of the first selected timestamp
    :param snd_timestamp:number of the second selected timestamp
    :param objects_fst: objects imported in the time of the first timestamp
    :param objects_snd: objects imported in the time of the second timestamp
    :return: aligned meshes and center of the moved organs
    """
    fst_meshes, snd_meshes = [], []
    if "ICP" in method:
        meshes, center1_before, center1_after = get_meshes_after_icp(fst_timestamp, objects_fst, patient_id)
        fst_meshes.extend(meshes)
        meshes, center2_before, center2_after = get_meshes_after_icp(snd_timestamp, objects_snd, patient_id, "orange")
        snd_meshes.extend(meshes)

    else:
        meshes, center1_before, center1_after = get_meshes_after_centering(fst_timestamp, objects_fst, patient_id,
                                                                           BLUE)
        fst_meshes.extend(meshes)
        meshes, center2_before, center2_after = get_meshes_after_centering(snd_timestamp, objects_snd, patient_id)
        snd_meshes.extend(meshes)

    return fst_meshes, snd_meshes, center1_after, center2_after


def draw_movement_arrow(fig, center1_after, center2_after):
    """
    Create an arrow which show the direction of the moved organs.
    :param fig: the 3d figure
    :param center1_after: center of the first organ after the alignment
    :param center2_after: center of the second organ after the alignment
    """
    x1, y1, z1 = center1_after[0][0][0]
    x2, y2, z2 = center2_after[0][0][0]
    cone_tip = 0.9 * np.sqrt((x2 - x1) ** 2 + (x2 - x1) ** 2)

    fig.add_trace(go.Scatter3d(x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                               showlegend=False, line=dict(width=6, color=BLUE)))
    fig.add_trace(go.Cone(x=[x2], y=[y2], z=[z2], u=[cone_tip * (x2 - x1)], v=[cone_tip * (y2 - y1)],
                          w=[cone_tip * (z2 - z1)], colorscale=[[0, "orange"], [1, "orange"]], showlegend=False,
                          showscale=False))


def get_meshes_after_icp(timestamp, objects, patient, color=BLUE):
    """
    Runs functions which perform the icp aligning.
    :param timestamp: chosen time
    :return: meshes after aligning and the center of the bones before and after the alignment
    """

    plan_bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
    bones = Project_2.import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, timestamp)])
    bones_center_before = Project_2.find_center_point(bones[0][0])

    transform_matrix, vec = Project_2.icp_rot_vec(bones[0][0], plan_bones[0][0])
    transfr_objects = Project_2.vertices_transformation(transform_matrix, deepcopy(objects))
    after_icp_meshes = create_meshes_from_objs(transfr_objects, color)

    bones_center_after = Project_2.vertices_transformation(transform_matrix, [[[bones_center_before]]])

    return after_icp_meshes, bones_center_before, bones_center_after


def get_meshes_after_centering(timestamp, objects, patient, color="orange"):
    """
    Runs functions which perform the aligning on the center of the prostate.
    :param timestamp: chosen time
    :return: meshes after aligning and the center of the prostate before and after the alignment
    """

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


def draw_all_movements(fig, organs):
    # it is only icp movement!!
    for organ in organs:
        for i in range(len(TIMESTAMPS)):
            x1, y1, z1, x2, y2, z2 = 0, 0, 0, 0, 0, 0
            if organ == "Prostate":
                x1, y1, z1 = plan_center_points[PATIENTS.index(patient_id)][1]
                x2, y2, z2 = all_movements_icp[PATIENTS.index(patient_id)][0][i]
            elif organ == "Bladder":
                x1, y1, z1 = plan_center_points[PATIENTS.index(patient_id)][2]
                x2, y2, z2 = all_movements_icp[PATIENTS.index(patient_id)][1][i]
            elif organ == "Rectum":
                x1, y1, z1 = plan_center_points[PATIENTS.index(patient_id)][3]
                x2, y2, z2 = all_movements_icp[PATIENTS.index(patient_id)][2][i]

            fig.add_trace(go.Scatter3d(x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                                       showlegend=False, line=dict(width=6, color=BLUE)))
            # cone_tip = 0.9 * np.sqrt((x2 - x1) ** 2 + (x2 - x1) ** 2)
            fig.add_trace(go.Cone(x=[x2], y=[y2], z=[z2],
                                  u=[CONE_TIP * (x2 - x1)], v=[CONE_TIP * (y2 - y1)], w=[CONE_TIP * (z2 - z1)],
                                  colorscale=[[0, BLUE], [1, BLUE]], showlegend=False, showscale=False))


def create_average_movement_lines(fig, organs):
    for organ in organs:
        name = organ.lower()
        x1, y1, z1 = Project_2.find_center_point([], trimesh.load_mesh(
            FILEPATH + "{}\\{}\\{}_plan.obj".format(patient_id, name, name)).bounds)
        x2, y2, z2 = get_average_vector(name)

        # print(name, x2, y2, z2)

        fig.add_trace(go.Scatter3d(x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                                   showlegend=False, line=dict(width=6, color=LIGHT_BLUE)))
        fig.add_trace(go.Cone(x=[x2], y=[y2], z=[z2],
                              u=[CONE_TIP * (x2 - x1)], v=[CONE_TIP * (y2 - y1)], w=[CONE_TIP * (z2 - z1)],
                              colorscale=[[0, LIGHT_BLUE], [1, LIGHT_BLUE]], showlegend=False, showscale=False))


def get_average_vector(organ_name):
    avrg_movements = all_movements_center[PATIENTS.index(patient_id)]
    avrg_bones, avrg_bladder, avrg_rectum = avrg_movements[0], avrg_movements[1], avrg_movements[2]

    avrg_movements = all_movements_icp[PATIENTS.index(patient_id)]
    avrg_prostate, avrg_bladder, avrg_rectum = avrg_movements[0], avrg_movements[1], avrg_movements[2]

    if "bones" in organ_name:
        mov_matrix = np.array(avrg_bones).T @ avrg_bones
    elif "prostate" in organ_name:
        mov_matrix = np.array(avrg_prostate).T @ avrg_prostate
    elif "bladder" in organ_name:
        mov_matrix = np.array(avrg_bladder).T @ avrg_bladder
    else:
        mov_matrix = np.array(avrg_rectum).T @ avrg_rectum
    # print(mov_matrix_bladder)

    eigen_values, eigen_vectors = np.linalg.eigh(mov_matrix)
    # print(eigen_vectors, "\n", eigen_values)
    val_count = np.count_nonzero(eigen_values == eigen_values[-1])
    if val_count == 1:
        average_vec = eigen_vectors[-1]
    else:
        average_vec = []
        for i in range(1, val_count + 1):
            average_vec += eigen_vectors[-i]
    # print(average_vec)
    # print("")

    return average_vec


@app.callback(
    Output("x-slice-graph", "figure"),
    Output("y-slice-graph", "figure"),
    Output("z-slice-graph", "figure"),
    Input("x-slice-slider", "value"),
    Input("y-slice-slider", "value"),
    Input("z-slice-slider", "value"),
    Input("organs-checklist", "value"),
    Input("alignment-radioitems", "value"),
    Input("mode-radioitems", "value"),
    Input("fst-timestamp-dropdown", "value"),
    Input("snd-timestamp-dropdown", "value"))
def create_graph_slices(x_slider, y_slider, z_slider, organs, method, mode,
                        fst_timestamp, snd_timestamp):
    """
    Creates three figures of slices made in the X, Y, and the Z axis direction. These figures are made according to the
    3D graph.
    :param x_slider: how far on the X axis normal we want to cut the slice
    :param y_slider: how far on the Y axis normal we want to cut the slice
    :param z_slider: how far on the Z axis normal we want to cut the slice
    :param organs: organs chosen for the 3D graph
    :param method: method of alignment in the 3D graph
    :param mode: mode from the 3D graph
    :param fst_timestamp: chosen in the 3D graph
    :param snd_timestamp: chosen in the 3D graph
    :return: the three slices figures
    """
    figures, fst_meshes, snd_meshes = [], [], []
    names = ["X axis slice", "Y axis slice", "Z axis slice"]

    for i in range(3):
        layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', height=310,
                           width=320, plot_bgcolor='rgba(50,50,50,1)', margin=dict(l=30, r=20, t=60, b=40),
                           showlegend=False, title=dict(text=names[i]))
        fig = go.Figure(layout=layout)
        fig.update_layout(title_x=0.5)
        figures.append(fig)

    if "Two timestamps" in mode:
        fst_timestamp = "_plan" if fst_timestamp == "plan" else fst_timestamp
        snd_timestamp = "_plan" if snd_timestamp == "plan" else snd_timestamp
        fst_meshes = two_slices_mode(method, patient_id, organs, fst_timestamp)
        snd_meshes = two_slices_mode(method, patient_id, organs, snd_timestamp)
    else:
        for organ in organs:
            fst_meshes.append(
                trimesh.load_mesh(FILEPATH + "{}\\{}\\{}_plan.obj".format(patient_id, organ.lower(), organ.lower())))

    x_fig = create_slice_final(x_slider, fst_meshes, snd_meshes, figures[0], "x")
    y_fig = create_slice_final(y_slider, fst_meshes, snd_meshes, figures[1], "y")
    z_fig = create_slice_final(z_slider, fst_meshes, snd_meshes, figures[2], "z")

    return x_fig, y_fig, z_fig


def two_slices_mode(method, patient, organs, timestamp):
    """
    Helper function to decide and perform the steps of the chosen method of alignment.
    :param method: method of the alignment
    :param patient: chosen patien id
    :param organs: organs chosen in the 3D graph
    :param timestamp: chosen time of the timestamp
    :return: aligned meshes
    """
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


def selected_organs_slices(matrix, organs, timestamp, patient):
    """
    Applies the transformation to selected organs.
    :param matrix: acquired either from icp algorithm or centering on the prostate
    :return: transformed meshes
    """
    meshes = []

    for organ in organs:
        mesh = trimesh.load_mesh(FILEPATH + "{}\\{}\\{}{}.obj".format(patient, organ.lower(), organ.lower(), timestamp))
        meshes.append(deepcopy(mesh).apply_transform(matrix))

    return meshes


def create_slice(mesh, slice_slider, params):
    """
    Creates the slices from the imported organs
    :param mesh: mesh of the selected organ
    :param slice_slider: where on the normal of the axis we want to make the slice
    :param params: parameters for the computation of the slice
    :return: created slices
    """
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
    """
    Helper function for the axis creation and the creation of the slices traces for the figures.
    :param meshes: meshes of the selected organs
    :param slice_slider: where on the normal of the axis we want to make the slice
    :param fig: slice graph
    :param color: either orange or blue according to the slices order
    :param axis: which axis slice we are creating
    """
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


def create_slice_final(slice_slider, icp_meshes, centered_meshes, fig, axis):
    """
    Calls the function to create the axis slices.
    :param slice_slider: where on the normal of the axis we want to make the slice
    :param fig: slice graph
    :param axis: which axis slice are we making
    :return slice figure
    """
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


def decide_organs_highlights(click_data, click_id, icp):
    """
    Computes what to highlight in the organs_icp or organs_center graphs according to clickData from other graphs
    :param click_data: information about the location of the last click
    :param click_id: id of the last clicked graph
    :param icp: true if the organs graph is the icp version, false otherwise
    :return: colors and sizes of the traces in the organs graphs
    """
    global patient_id
    colors = [[LIGHT_BLUE] * 13, [GREEN] * 13, [RED] * 13] if icp else [[PURPLE] * 13, [GREEN] * 13, [RED] * 13]
    sizes = [[5] * 13, [5] * 13, [5] * 13]
    data = click_data["points"][0]

    if "heatmap" in click_id:
        patient_id = PATIENTS[data["y"]]
        x = int(click_data["points"][0]["x"]) // 4
        if data["text"] == "Bladder":
            colors[1][x], sizes[1][x] = "white", 10
        elif data["text"] == "Rectum":
            colors[2][x], sizes[2][x] = "white", 10
        elif (data["text"] == "Prostate" and icp) or (data["text"] == "Bones" and not icp):
            colors[0][x], sizes[0][x] = "white", 10

    elif "average" in click_id:
        patient_id = data["x"]
        if data["curveNumber"] == 1:
            colors[1], sizes[1] = ["white"] * 13, [10] * 13
        elif data["curveNumber"] == 2:
            colors[2], sizes[2] = ["white"] * 13, [10] * 13
        elif data["curveNumber"] == 0 and ("icp" in click_id and icp) or ("center" in click_id and not icp):
            colors[0], sizes[0] = ["white"] * 13, [10] * 13

    elif "organs" in click_id:
        x = int(click_data["points"][0]["x"]) - 1
        if data["curveNumber"] != 0 or ("icp" in click_id and icp) or ("center" in click_id and not icp):
            colors[data["curveNumber"]][x], sizes[data["curveNumber"]][x] = "white", 10

    elif "alignment-differences" in click_id:
        x = int(click_data["points"][0]["x"]) - 1
        if data["curveNumber"] == 0:
            colors[1][x], sizes[1][x] = "white", 10
        elif data["curveNumber"] == 1:
            colors[2][x], sizes[2][x] = "white", 10

    return colors, sizes


@app.callback(
    Output("organs-icp", "figure"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("alignment-differences", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("scale-organs", "value"))
def create_distances_after_icp(click_data, center_click_data, differences, average_icp, average_center,
                               heatmap_icp, heatmap_center, scale):
    """
    Creates the organs_icp graph which shows how patient's organs moved in the 13 timestamps after icp aligning.
    :param click_data: clickData from this graph
    :param center_click_data: clickData from organs_center graph
    :param differences: clickData from the differences graph
    :param average_icp: clickData from the average_icp graph
    :param average_center: clickData from the average_center graph
    :param heatmap_icp: clickData from the heatmap_icp graph
    :param heatmap_center: clickData from the heatmap_center graph
    :return: organs_icp figure
    """
    global patient_id
    all_click_data = [click_data, center_click_data, differences, average_icp, average_center,
                      heatmap_icp, heatmap_center]
    all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)
    colors = [[LIGHT_BLUE] * 13, [GREEN] * 13, [RED] * 13]
    sizes = [[5] * 13, [5] * 13, [5] * 13]

    if click_data:
        colors, sizes = decide_organs_highlights(click_data, click_id, True)

    distances_icp = all_distances_icp[PATIENTS.index(patient_id)]
    prostate, bladder, rectum = distances_icp[0], distances_icp[1], distances_icp[2]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(
                           text="Distances of ICP aligned organs and the plan organs of patient {}".format(patient_id),
                           font=dict(size=18, color='lightgrey')))

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=prostate, mode="lines+markers", name="Prostate",
                               marker=dict(color=colors[0], size=sizes[0], line_color=colors[0], opacity=1)))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder",
                               marker=dict(color=colors[1], size=sizes[1], line_color=colors[1], opacity=1)))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum",
                               marker=dict(color=colors[2], size=sizes[2], line_color=colors[2], opacity=1)))

    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance")

    if "uniform" in scale:
        fig.update_yaxes(range=[0, 90])
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


@app.callback(
    Output("organs-center", "figure"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("alignment-differences", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("scale-organs", "value"))
def create_distances_after_centering(icp_click_data, click_data, differences, average_icp, average_center,
                                     heatmap_icp, heatmap_center, scale):
    """
    Creates the organs_center graph which shows how patient's organs moved in the 13 timestamps after aligning on
    prostate.
    :param icp_click_data: organs_icp graph clickData
    :param click_data: this graph clickData
    :param differences: clickData from the differences graph
    :param average_icp: clickData from the average_icp graph
    :param average_center: clickData from the average_center graph
    :param heatmap_icp: clickData from the heatmap_icp graph
    :param heatmap_center: clickData from the heatmap_center graph
    :return: organs_center figure
    :return:
    """
    global patient_id
    colors = [[PURPLE] * 13, [GREEN] * 13, [RED] * 13]
    sizes = [[5] * 13, [5] * 13, [5] * 13]

    all_click_data = [icp_click_data, click_data, differences, average_icp, average_center, heatmap_icp, heatmap_center]
    all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    if click_data:
        colors, sizes = decide_organs_highlights(click_data, click_id, False)

    distances_center = all_distances_center[PATIENTS.index(patient_id)]
    bones, bladder, rectum = distances_center[0], distances_center[1], distances_center[2]

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350,
                       title=dict(
                           text="Distances of the centered organs and the plan organs of patient {}".format(patient_id),
                           font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bones, mode="lines+markers", name="Bones",
                               line=dict(color=PURPLE),
                               marker=dict(color=colors[0], size=sizes[0], line_color=colors[0], opacity=1)))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder",
                               marker=dict(color=colors[1], size=sizes[1], line_color=colors[1], opacity=1)))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum",
                               marker=dict(color=colors[2], size=sizes[2], line_color=colors[2], opacity=1)))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance")

    if "uniform" in scale:
        fig.update_yaxes(range=[0, 90])

    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


def decide_differences_highlights(click_data, click_id):
    """
    Computes what to highlight in the differences graph according to clickData from other graphs
    :param click_data: information about the location of the last click
    :param click_id: id of the last clicked graph
    :return: colors of the traces in the differences graph
    """
    colors = [[GREEN] * 13, [RED] * 13]
    data = click_data["points"][0]

    if "heatmap" in click_id:
        x = int(click_data["points"][0]["x"]) // 4
        if data["text"] == "Bladder":
            colors[0][x] = "white"
        elif data["text"] == "Rectum":
            colors[1][x] = "white"

    elif "average" in click_id:
        if data["curveNumber"] == 1:
            colors[0] = ["white"] * 13
        elif data["curveNumber"] == 2:
            colors[1] = ["white"] * 13

    elif "organs" in click_id:
        x = int(data["x"]) - 1
        if data["curveNumber"] == 1:
            colors[0][x] = "white"
        elif data["curveNumber"] == 2:
            colors[1][x] = "white"

    elif "alignment-differences" in click_id:
        x = int(data["x"]) - 1
        colors[data["curveNumber"]][x] = "white"

    return colors


@app.callback(
    Output("alignment-differences", "figure"),
    Input("alignment-differences", "clickData"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"))
def create_distances_between_alignments(differences, organs_icp, organs_center, average_icp, average_center,
                                        heatmap_icp, heatmap_center):
    """
    Creates the differences graph which shows the distinctions between the registration methods.
    :param differences: clickData from this graph
    :param organs_icp: clickData from the organs_icp graph
    :param organs_center: clickData from the organs_center graph
    :param average_icp: clickData from the average_icp graph
    :param average_center: clickData from the average_center graph
    :param heatmap_icp: clickData from the heatmap_icp graph
    :param heatmap_center: clickData from the heatmap_center graph
    :return: differences graph figure
    """
    global patient_id
    dist_icp = all_distances_icp[PATIENTS.index(patient_id)]
    dist_center = all_distances_center[PATIENTS.index(patient_id)]
    distances = np.array(dist_icp) - np.array(dist_center)
    _, bladder, rectum = distances[0], distances[1], distances[2]

    colors = [[GREEN] * 13, [RED] * 13]

    all_click_data = [differences, organs_icp, organs_center, average_icp, average_center, heatmap_icp, heatmap_center]
    all_ids = ["alignment-differences", "organs-icp", "organs-center", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    if click_data:
        colors = decide_differences_highlights(click_data, click_id)

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90, r=40),
                       plot_bgcolor='rgba(70,70,70,1)', width=1420, height=350,
                       title=dict(
                           text="Differences of distances between the alignments of patient {}".format(patient_id),
                           font=dict(size=18, color='lightgrey')))
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=bladder, name="Bladder", marker=dict(color=colors[0])))
    fig.add_trace(go.Bar(x=np.array(range(1, 14)), y=rectum, name="Rectum", marker=dict(color=colors[1])))

    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance center | Distance ICP  ", tick0=0, dtick=2)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


with open("computations_files/icp_distances.txt", "r") as icp_dist, open("computations_files/icp_movements.txt", "r") \
        as icp_mov, open("computations_files/icp_averages.txt", "r") as icp_avrg:
    all_distances_icp = json.load(icp_dist)
    all_movements_icp = json.load(icp_mov)
    lines = icp_avrg.read().splitlines()
    avrg_prostate_icp = list(map(float, lines[::3]))
    avrg_bladder_icp = list(map(float, lines[1::3]))
    avrg_rectum_icp = list(map(float, lines[2::3]))

with open("computations_files/center_distances.txt", "r") as center_dist, open(
        "computations_files/center_movements.txt", "r") as center_mov, \
        open("computations_files/center_averages.txt", "r") as center_avrg:
    all_distances_center = json.load(center_dist)
    all_movements_center = json.load(center_mov)
    lines = center_avrg.read().splitlines()
    avrg_bones_center = list(map(float, lines[::3]))
    avrg_bladder_center = list(map(float, lines[1::3]))
    avrg_rectum_center = list(map(float, lines[2::3]))

with open("computations_files/plan_center_points.txt", "r") as center_points:
    plan_center_points = json.load(center_points)


def decide_average_highlights(data, click_id, icp):
    """
    Computes what to highlight in the average_icp and the average_center graphs according to clickData from other
    graphs.
    :param data: relevant information about the location of the last click
    :param click_id: id of the last clicked graph
    :param icp: whether we highlight in the icp or prostate aligning version of the average graphs
    :return: colors of the traces in the differences graph
    """
    global patient_id
    highlight, x, y = 0, 0, 0

    if "heatmap" in click_id:
        if (icp and data["text"] == "Prostate") or (not icp and data["text"] == "Bones"):
            highlight = 1
        elif data["text"] == "Bladder":
            highlight = 2
        elif data["text"] == "Rectum":
            highlight = 3
        x, y = [int(PATIENTS[data["y"]])], data["y"]
    else:
        highlight = data["curveNumber"] + 1
        if "differences" in click_id:
            highlight = 2 if data["curveNumber"] == 0 else 3

        x, y = [patient_id], PATIENTS.index(patient_id)

    return x, y, highlight


@app.callback(
    Output("average-icp", "figure"),
    Input("alignment-differences", "clickData"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("scale-average", "value"))
def average_distances_icp(differences, organs_icp, organs_center, click_data, center_click_data, heatmap_icp,
                          heatmap_center, scale):
    """
    Creates the average_icp graph which shows the average movements of patient's organs after icp aligning.
    :param differences: clickData from the differences graph
    :param organs_icp: clickData from the organs_icp graph
    :param organs_center: clickData from the organs_center graph
    :param click_data: clickData from this graph
    :param center_click_data: clickData from the average_center graph
    :param heatmap_icp: clickData from the heatmap_icp graph
    :param heatmap_center: clickData from the heatmap_center graph
    :return: the average_icp figure
    """
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350, showlegend=True,
                       title=dict(text="Average difference of patients' organs positions after ICP aligning",
                                  font=dict(size=16, color='lightgrey')))
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_prostate_icp, mode="markers", name="Prostate",
                             marker=dict(symbol="circle", color=LIGHT_BLUE), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bladder_icp, mode="markers", name="Bladder",
                             marker=dict(symbol="square", color=GREEN), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_rectum_icp, mode="markers", name="Rectum",
                             marker=dict(symbol="diamond", color=RED), line=dict(width=5)))

    all_click_data = [differences, organs_icp, organs_center, click_data, center_click_data, heatmap_icp,
                      heatmap_center]
    all_ids = ["alignment-differences", "organs-icp", "organs-center", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    if click_data and not (("average-center" in click_id or "organs-center" in click_id)
                           and click_data["points"][0]["curveNumber"] == 0):  # so every organ other than bones
        data = click_data["points"][0]
        x, y, highlight = decide_average_highlights(data, click_id, True)

        if highlight == 1:
            fig.add_trace(go.Scatter(x=x, y=[avrg_prostate_icp[y]], mode="markers", showlegend=False,
                                     name="Prostate", marker=dict(symbol="circle", color=LIGHT_BLUE,
                                                                  line=dict(width=3, color="white"))))
        if highlight == 2:
            fig.add_trace(go.Scatter(x=x, y=[avrg_bladder_icp[y]], mode="markers", showlegend=False,
                                     name="Bladder", marker=dict(symbol="square", color=GREEN,
                                                                 line=dict(width=3, color="white"))))
        if highlight == 3:
            fig.add_trace(go.Scatter(x=x, y=[avrg_rectum_icp[y]], mode="markers", showlegend=False,
                                     name="Rectum", marker=dict(symbol="diamond", color=RED,
                                                                line=dict(width=3, color="white"))))

    fig.update_traces(marker=dict(size=12))
    fig.update_xaxes(title_text="Patient")

    if "uniform" in scale:
        fig.update_yaxes(range=[-5, 85])

    fig.update_yaxes(title_text="Average distance")
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


@app.callback(
    Output("average-center", "figure"),
    Input("alignment-differences", "clickData"),
    Input("organs-icp", "clickData"),
    Input("organs-center", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("scale-average", "value"))
def average_distances_center(differences, organs_icp, organs_center, icp_click_data, click_data, heatmap_icp,
                             heatmap_center, scale):
    """
    Creates the average_center graph which shows the average movements of patient's organs after centering on prostate.
    :param differences: clickData from the differences graph
    :param organs_icp: clickData from the organs_icp graph
    :param organs_center: clickData from the organs_center graph
    :param icp_click_data: clickData from the average_icp graph
    :param click_data: clickData from this graph
    :param heatmap_icp: clickData from the heatmap_icp graph
    :param heatmap_center: clickData from the heatmap_center graph
    :return: the average_center figure
    """
    global patient_id
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350, showlegend=True,
                       title=dict(text="Average difference of patients' organs positions after centering on prostate",
                                  font=dict(size=16, color='lightgrey')))
    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bones_center, mode="markers", name="Bones",
                             marker=dict(symbol="circle", color=PURPLE), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bladder_center, mode="markers", name="Bladder",
                             marker=dict(symbol="square", color=GREEN), line=dict(width=5)))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_rectum_center, mode="markers", name="Rectum",
                             marker=dict(symbol="diamond", color=RED), line=dict(width=5)))

    all_click_data = [differences, organs_icp, organs_center, icp_click_data, click_data, heatmap_icp,
                      heatmap_center]
    all_ids = ["alignment-differences", "organs-icp", "organs-center", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    # highlight data
    if click_data and not (("average-icp" in click_id or "organs-icp" in click_id)
                           and click_data["points"][0]["curveNumber"] == 0):  # so every organ other than prostate:
        data = click_data["points"][0]
        x, y, highlight = decide_average_highlights(data, click_id, False)

        if highlight == 1:
            fig.add_trace(go.Scatter(x=x, y=[avrg_bones_center[y]], mode="markers", showlegend=False,
                                     name="Bones", marker=dict(symbol="circle", color=PURPLE,
                                                               line=dict(width=3, color="white"))))
        if highlight == 2:
            fig.add_trace(go.Scatter(x=x, y=[avrg_bladder_center[y]], mode="markers", showlegend=False,
                                     name="Bladder", marker=dict(symbol="square", color=GREEN,
                                                                 line=dict(width=3, color="white"))))
        if highlight == 3:
            fig.add_trace(go.Scatter(x=x, y=[avrg_rectum_center[y]], mode="markers", showlegend=False,
                                     name="Rectum", marker=dict(symbol="diamond", color=RED,
                                                                line=dict(width=3, color="white"))))

    fig.update_traces(marker=dict(size=12))
    fig.update_xaxes(title_text="Patient")
    fig.update_yaxes(title_text="Average distance")

    if "uniform" in scale:
        fig.update_yaxes(range=[-5, 85])

    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


def resolve_click_data(click_data, ids):
    """
    Decides which graph was clicked last.
    :param click_data: clickData form every graph
    :param ids: id of every graph
    :return: clickData info and the last clicked graph id or None if nothing was clicked in the graphs
    """
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
    Input("average-center", "clickData"),
    Input("scale-heatmap", "value"))
def create_heatmap_icp(organs_icp, organs_center, differences, click_data, center_click_data, average_icp,
                       average_center, scale):
    """
    Creates the heatmap_icp graph which depicts every patient and their every organ movement after icp aligning.
    :param organs_icp: clickData from the organs_icp graph
    :param organs_center: clickData from the organs_center graph
    :param differences: clickData from the differences graph
    :param click_data: clickData from this graph
    :param center_click_data: clickData from the heatmap_center graph
    :param average_icp: clickData from the average_icp graph
    :param average_center: clickData from the average_center graph
    :return: heatmap_icp figure
    """
    global patient_id

    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90, r=81), plot_bgcolor='rgba(70,70,70,1)', width=1420, height=350,
                       showlegend=True, title=dict(text="Difference of patients' organs positions after ICP "
                                                        "aligning to the bones",
                                                   font=dict(size=18, color='lightgrey')))

    data, custom_data, hover_text = create_data_for_heatmap(True)

    if "uniform" in scale:
        fig = go.Figure(data=go.Heatmap(z=data, zmin=0, zmax=85, text=hover_text, customdata=custom_data,
                                        hovertemplate="<b>%{text}</b><br>Patient: %{y}<br>Timestamp: %{customdata}<br>"
                                                      "Distance: %{z:.2f} mm<extra></extra>", colorscale="Portland"),
                        layout=layout)
    else:
        fig = go.Figure(data=go.Heatmap(z=data, text=hover_text, customdata=custom_data,
                                        hovertemplate="<b>%{text}</b><br>Patient: %{y}<br>Timestamp: %{customdata}<br>"
                                                      "Distance: %{z:.2f} mm<extra></extra>", colorscale="Portland"),
                        layout=layout)

    # create borders around cells
    for i in range(1, 13):
        fig.add_vline(x=4 * i - 0.5, line_width=4, line_color=GREY)

    for i in range(1, 8):
        fig.add_hline(y=i - 0.5, line_width=4, line_color=GREY)

    all_click_data = [organs_icp, organs_center, differences, average_icp, average_center, click_data,
                      center_click_data]
    all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    # highlight the selection
    if click_data:
        data = click_data["points"][0]
        decide_heatmap_highlights(fig, data, click_id)

    fig.update_xaxes(title_text="Timestamp", ticktext=TIMESTAMPS, tickmode="array", tickvals=np.arange(1.5, 52, 4),
                     zeroline=False, showgrid=False)
    fig.update_yaxes(title_text="Patient", ticktext=PATIENTS, tickmode="array", tickvals=np.arange(0, 8, 1),
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
    Input("organs-center", "clickData"),
    Input("scale-heatmap", "value"))
def create_heatmap_centering(click_data, icp_click_data, differences, average_icp, average_center, organs_icp,
                             organs_center, scale):
    """
    Creates the heatmap_center graph which depicts every patient and their every organ movement after centering on
    the prostate registration method.
    :param click_data: clickData from this graph
    :param icp_click_data: clickData from the heatmap_icp graph
    :param differences: clickData from the differences graph
    :param average_icp: clickData from the average_icp graph
    :param average_center: clickData from the average_center graph
    :param organs_icp: clickData from the organs_icp graph
    :param organs_center: clickData from the organs_center graph
    :return: heatmap_center figure
    """
    global patient_id
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)',
                       margin=dict(t=80, b=70, l=90, r=81), plot_bgcolor='rgba(70,70,70,1)', width=1420, height=350,
                       showlegend=True, title=dict(text="Difference of patients' organs positions after centering on "
                                                        "prostate",
                                                   font=dict(size=18, color='lightgrey')))

    data, custom_data, hover_text = create_data_for_heatmap(False)

    if "uniform" in scale:
        fig = go.Figure(data=go.Heatmap(z=data, zmin=0, zmax=85, text=hover_text, customdata=custom_data,
                                        hovertemplate="<b>%{text}</b><br>Patient: %{y}<br>Timestamp: %{customdata}<br>"
                                                      "Distance: %{z:.2f} mm<extra></extra>",
                                        colorscale="Portland"), layout=layout)  # [[0, BLUE], [0.5, RED], [1, GREEN]]
    else:
        fig = go.Figure(data=go.Heatmap(z=data, text=hover_text, customdata=custom_data,
                                        hovertemplate="<b>%{text}</b><br>Patient: %{y}<br>Timestamp: %{customdata}<br>"
                                                      "Distance: %{z:.2f} mm<extra></extra>",
                                        colorscale="Portland"), layout=layout)

    # dividers between cells
    for i in range(1, 13):
        fig.add_vline(x=4 * i - 0.5, line_width=4, line_color=GREY)

    for i in range(1, 8):
        fig.add_hline(y=i - 0.5, line_width=4, line_color=GREY)

    all_click_data = [organs_icp, organs_center, differences, average_icp, average_center,
                      icp_click_data, click_data]
    all_ids = ["organs-icp", "organs-center", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    # highlight the selection
    if click_data:
        data = click_data["points"][0]
        decide_heatmap_highlights(fig, data, click_id)

    fig.update_xaxes(title_text="Timestamp", ticktext=TIMESTAMPS, tickmode="array", tickvals=np.arange(1.5, 52, 4),
                     zeroline=False, showgrid=False)
    fig.update_yaxes(title_text="Patient", ticktext=PATIENTS, tickmode="array", tickvals=np.arange(0, 9, 1),
                     zeroline=False, showgrid=False)
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90,
                      coloraxis_colorbar=dict(title="Distance", ticksuffix=" mm"))
    fig.update_coloraxes(colorbar_dtick=1)

    # WIP trying to add organs labels
    # row = ["Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc",
    #            "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc",
    #            "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc", "Bn", "Pr", "Bl", "Rc",
    #            "Bn", "Pr", "Bl", "Rc"]
    # z_text = [row, "", "", "", "", "", "", ""]
    # fig.update_traces(text=z_text, texttemplate="%{text}")

    return fig


def decide_heatmap_highlights(fig, data, click_id):
    """
    Computes what to highlight in the heaatmaps according to clickData from other graphs.
    :param fig: heatmap figure
    :param data: clickData from the clicked graph
    :param click_id: id of the clicked graph
    """
    if "heatmap" in click_id:
        fig.add_shape(type="rect", x0=data["x"] - 0.43, y0=data["y"] - 0.41, x1=data["x"] + 0.43,
                      y1=data["y"] + 0.41, line_color="white", line_width=4)
    else:
        if data["curveNumber"] == 0 and "center" in click_id:
            x = 0
        elif data["curveNumber"] == 0 and "icp" in click_id:
            x = 1
        else:
            x = data["curveNumber"] + 1
        y = PATIENTS.index(patient_id)

        if "average" in click_id:
            for i in range(13):
                fig.add_shape(type="rect", x0=(x - 0.43) + 4 * i, y0=y - 0.41, x1=(x + 0.43) + 4 * i, y1=y + 0.41,
                              line_width=4, line_color="white")
        elif "organs" in click_id:
            x_0 = data["x"] - 1
            fig.add_shape(type="rect", x0=x_0 * 4 - 0.43 + x, y0=y - 0.41, x1=x_0 * 4 + 0.43 + x,
                          y1=y + 0.41, line_color="white", line_width=4)
        else:
            x_0 = data["x"] - 1
            x = data["curveNumber"] + 2
            fig.add_shape(type="rect", x0=x_0 * 4 - 0.43 + x, y0=y - 0.41, x1=x_0 * 4 + 0.43 + x,
                          y1=y + 0.41, line_color="white", line_width=4)


def create_data_for_heatmap(icp):
    """
    Creates hovertexts and formats the data for the heatmaps
    :param icp: true if our graph is the icp version of the heatmaps, false otherwise
    :return: formatted data for the heatmap and tha hover text
    """
    # data is 2d array with distances for the heightmap, custom_data and hover_text are used just for hover labels
    data, custom_data, hover_text = [], [], []
    for i in range(len(PATIENTS)):
        # patient contains four arrays: bones, prostate, bladder, rectum with distances from all the timestamps
        patient = all_distances_icp[i] if icp else all_distances_center[i]
        data_row, custom_row, hover_row = [], [], []

        for j in range(len(TIMESTAMPS)):
            data_row.extend([0, patient[0][j], patient[1][j], patient[2][j]]) if icp \
                else data_row.extend([patient[0][j], 0, patient[1][j], patient[2][j]])
            custom_row.extend([j + 1, j + 1, j + 1, j + 1])
            hover_row.extend(["Bones", "Prostate", "Bladder", "Rectum"])

        data.append(data_row)
        custom_data.append(custom_row)
        hover_text.append(hover_row)

    return data, custom_data, hover_text


if __name__ == '__main__':
    app.run_server(debug=True)
