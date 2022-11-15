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