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
                                       showlegend=False, line=dict(width=6, color=CYAN1)))
            # cone_tip = 0.9 * np.sqrt((x2 - x1) ** 2 + (x2 - x1) ** 2)
            fig.add_trace(go.Cone(x=[x2], y=[y2], z=[z2],
                                  u=[CONE_TIP * (x2 - x1)], v=[CONE_TIP * (y2 - y1)], w=[CONE_TIP * (z2 - z1)],
                                  colorscale=[[0, CYAN1], [1, CYAN1]], showlegend=False, showscale=False))


def create_average_movement_lines(fig, organs):
    for organ in organs:
        name = organ.lower()
        x1, y1, z1 = Project_2.find_center_point([], trimesh.load_mesh(
            FILEPATH + "{}\\{}\\{}_plan.obj".format(patient_id, name, name)).bounds)
        x2, y2, z2 = get_average_vector(name)

        fig.add_trace(go.Scatter3d(x=[x1, x2], y=[y1, y2], z=[z1, z2], mode='lines',
                                   showlegend=False, line=dict(width=6, color=CYAN1)))
        fig.add_trace(go.Cone(x=[x2], y=[y2], z=[z2],
                              u=[CONE_TIP * (x2 - x1)], v=[CONE_TIP * (y2 - y1)], w=[CONE_TIP * (z2 - z1)],
                              colorscale=[[0, CYAN1], [1, CYAN1]], showlegend=False, showscale=False))


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

    eigen_values, eigen_vectors = np.linalg.eigh(mov_matrix)
    val_count = np.count_nonzero(eigen_values == eigen_values[-1])
    if val_count == 1:
        average_vec = eigen_vectors[-1]
    else:
        average_vec = []
        for i in range(1, val_count + 1):
            average_vec += eigen_vectors[-i]

    return average_vec


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
                               showlegend=False, line=dict(width=6, color=PINK)))
    fig.add_trace(go.Cone(x=[x2], y=[y2], z=[z2], u=[cone_tip * (x2 - x1)], v=[cone_tip * (y2 - y1)],
                          w=[cone_tip * (z2 - z1)], colorscale=[[0, ORANGE], [1, ORANGE]], showlegend=False,
                          showscale=False))


# first in order centering function to update the scale of the icp graph
@app.callback(
    Output("figure"),
    Input("organ-distances", "clickData"),
    Input("clickData"),
    Input("alignment-differences", "clickData"),
    Input("average-icp", "clickData"),
    Input("average-center", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("rotations-graph", "clickData"),
    Input("scale-organs", "value"))
def create_distances_after_centering(icp_click_data, click_data, differences, average_icp, average_center,
                                     heatmap_icp, heatmap_center, rotations_graph, scale):
    """
    Creates the  graph which shows how patient's organs moved in the 13 timestamps after aligning on
    prostate.
    :param icp_click_data: organ_distances graph clickData
    :param click_data: this graph clickData
    :param differences: clickData from the differences graph
    :param average_icp: clickData from the average_icp graph
    :param average_center: clickData from the average_center graph
    :param heatmap_icp: clickData from the heatmap_icp graph
    :param heatmap_center: clickData from the heatmap_center graph
    :return:  figure
    :return:
    """
    global patient_id

    colors = [[PURPLE] * 13, [GREEN] * 13, [RED] * 13]
    sizes = [[0] * 13, [0] * 13, [0] * 13]

    all_click_data = [icp_click_data, click_data, differences, average_icp, average_center, heatmap_icp, heatmap_center,
                      rotations_graph]
    all_ids = ["organ-distances", "alignment-differences", "average-icp", "average-center",
               "heatmap-icp", "heatmap-center", "rotations-graph"]
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
                               marker=dict(color=PURPLE, symbol="circle", line=dict(width=sizes[0], color=colors[0]))))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=bladder, mode="lines+markers", name="Bladder",
                               marker=dict(color=GREEN, symbol="square", line=dict(width=sizes[1], color=colors[1]))))
    fig.add_trace(go.Scattergl(x=np.array(range(1, 14)), y=rectum, mode="lines+markers", name="Rectum",
                               marker=dict(color=RED, symbol="diamond", line=dict(width=sizes[2], color=colors[2]))))
    fig.update_xaxes(title_text="Timestamp", tick0=0, dtick=1)
    fig.update_yaxes(title_text="Distance [mm]")

    if "uniform" in scale:
        fig.update_yaxes(range=[0, constants.scale[patient_id]])

    fig.update_traces(marker=dict(size=10))
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)

    return fig


@app.callback(
    Output("average-center", "figure"),
    Input("alignment-differences", "clickData"),
    Input("organ-distances", "clickData"),
    Input("average-distances", "clickData"),
    Input("heatmap-icp", "clickData"),
    Input("heatmap-center", "clickData"),
    Input("rotations-graph", "clickData"),
    Input("scale-average", "value"),
    Input("average-distances", "relayoutData"))
def average_center(differences, organ_distances, icp_click_data, click_data, heatmap_icp,
                             heatmap_center, rotations_graph, scale, icp_relayout):
    """
    Creates the  graph which shows the average movements of patient's organs after centering on prostate.
    :param differences: clickData from the differences graph
    :param organ_distances: clickData from the organ_distances graph
    :param : clickData from the  graph
    :param icp_click_data: clickData from the average_distances graph
    :param click_data: clickData from this graph
    :param heatmap_icp: clickData from the heatmap_icp graph
    :param heatmap_center: clickData from the heatmap_center graph
    :return: the  figure
    """
    global patient_id
    colors = [[PURPLE] * 8, [GREEN] * 8, [RED] * 8]
    sizes = [[0] * 8, [0] * 8, [0] * 8]
    layout = go.Layout(font=dict(size=12, color='darkgrey'), paper_bgcolor='rgba(50,50,50,1)', margin=dict(t=80, b=70,
                                                                                                           l=90, r=81),
                       plot_bgcolor='rgba(70,70,70,1)', width=680, height=350, showlegend=True,
                       title=dict(text="Average difference of patients' organs positions after centering on prostate",
                                  font=dict(size=16, color='lightgrey')))
    fig = go.Figure(layout=layout)

    all_click_data = [differences, organ_distances, icp_click_data, click_data, heatmap_icp,
                      heatmap_center, rotations_graph]
    all_ids = ["alignment-differences", "organ-distances", "average-distances",
               "heatmap-icp", "heatmap-center", "rotations-graph"]
    click_data, click_id = resolve_click_data(all_click_data, all_ids)

    if click_data:
        colors, sizes = decide_average_highlights(click_data, click_id, False)

    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bones_center, mode="markers", name="Bones",
                             marker=dict(symbol="circle", color=PURPLE, line=dict(width=sizes[0], color=colors[0]))))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_bladder_center, mode="markers", name="Bladder",
                             marker=dict(symbol="square", color=GREEN, line=dict(width=sizes[1], color=colors[1]))))
    fig.add_trace(go.Scatter(x=PATIENTS, y=avrg_rectum_center, mode="markers", name="Rectum",
                             marker=dict(symbol="diamond", color=RED, line=dict(width=sizes[2], color=colors[2]))))

    if icp_relayout and "xaxis.range[0]" in icp_relayout.keys():
        fig.update_xaxes(range=[icp_relayout["xaxis.range[0]"], icp_relayout["xaxis.range[1]"]])
    if icp_relayout and "yaxis.range[0]" in icp_relayout.keys():
        fig.update_yaxes(range=[icp_relayout["yaxis.range[0]"], icp_relayout["yaxis.range[1]"]])

    if "uniform" in scale and not icp_relayout or \
            ("xaxis.range[0]" not in icp_relayout.keys() and "yaxis.range[0]" not in icp_relayout.keys()):
        fig.update_xaxes(range=[-0.5, 7.5])
        fig.update_yaxes(range=[-5, 89])

    fig.update_traces(marker=dict(size=12))
    fig.update_layout(title_x=0.5, font=dict(size=14), title_y=0.90)
    fig.update_xaxes(title_text="Patient", autorange=False)
    fig.update_yaxes(title_text="Average distance [mm]", autorange=False)

    return fig
