def visualize(object_list, align=False):
    """
    Use trimesh library to visualize the objects from the object_list
    Align argument is just for colour changing
    """
    index = 0
    scene = trimesh.scene.scene.Scene()

    for verts, faces in object_list:
        if align:
            if index == 0:
                mesh = trimesh.Trimesh(verts, faces, face_colors=[20, 100, 180, 200])
            elif index == 1:
                mesh = trimesh.Trimesh(verts, faces, face_colors=[240, 50, 50, 200])
            else:
                mesh = trimesh.Trimesh(verts, faces, face_colors=[240, 220, 20, 200])
        else:
            mesh = trimesh.Trimesh(verts, faces, face_colors=[20, 50, 200, 180])

        # mesh = trimesh.smoothing.filter_mut_dif_laplacian(mesh)
        scene.add_geometry(mesh)
        index += 1

    scene.show()


def visualization_method(method, key, other, organs):
    """
    Function used in kivy GUI. That's why it has a little bit weird logic and parameters
    :param method: 'ICP algorithm' or anything else for center point method
    :param key: a list with a path to the key organ file (usually the plan file)
    :param other: a list with a path to the organ which is going to be aligned to the key organ
    :param organs: a list with a path to the other organs which are going to be aligned and visualized

    examples:
        visualization_method("ICP algorithm", ["OBJ_images/bones_plan.obj"], ["OBJ_images/bones_region_grow.obj"], [])
        visualization_method("center", ["OBJ_images/prostate_plan.obj"], ["OBJ_images/prostate.obj"],
                             ["OBJ_images/bones_region_grow.obj", "OBJ_images/bladder.obj"])
    """
    key = import_obj(key)
    other = import_obj(other)
    organs = import_obj(organs)
    all_organs = other + organs

    if method == "ICP algorithm":
        transform_matrix = icp_transformation_matrices(other[0][0], key[0][0])
        transfr_objects = vertices_transformation(transform_matrix, deepcopy(all_organs))
        all_organs.extend(transfr_objects)
        visualize(all_organs, True)

    else:
        key_center = find_center_point(key[0][0])
        other_center = find_center_point(other[0][0])
        matrix = create_translation_matrix(key_center, other_center)
        centered_objects = list(vertices_transformation(matrix, deepcopy(all_organs)))
        all_organs.extend(centered_objects)
        visualize(all_organs, True)



# for simple example of bones icp aligning call:
# objects = import_obj(["OBJ_images/bones_region_grow.obj", "OBJ_images/bladder.obj",
#                       "OBJ_images/prostate.obj", "OBJ_images/rectum.obj"])
# plan = import_obj(["OBJ_images/bones_plan.obj", "OBJ_images/prostate_plan.obj"])


# icp_transformation_matrices takes as arguments only vertices, so when we want to transform for example bladder.obj,
# we write objects[1][0] as the second index [0] means vertices
# transform_matrix, transformed = icp_transformation_matrices(objects[0][0], plan[0][0])

# make copy of the objects because we want to save the unmodified version too
# transfr_objects = vertices_transformation(transform_matrix, deepcopy([objects[0]]))
#
# BEFORE aligning:
# visualize([plan[0], objects[0]], True)
# AFTER aligning:
# visualize([plan[0], transfr_objects[0]], True)