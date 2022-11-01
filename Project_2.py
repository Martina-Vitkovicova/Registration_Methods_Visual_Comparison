import math
import os
import numpy as np
import trimesh.smoothing
import trimesh.registration
import numpy
import pywavefront
import time
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

import constants
from constants import FILEPATH


def import_obj(files):
    """Create a list of vertices and faces from the .obj files"""
    object_list = []

    for name in files:
        organ = pywavefront.Wavefront(name, collect_faces=True)
        object_list.append([organ.vertices, organ.mesh_list[0].faces])

    return object_list


def icp_transformation_matrices(other, key, print_out=False):
    """Create transformation matrix using icp algorithm from trimesh library"""
    matrix, transformed, _ = trimesh.registration.icp(other, key, scale=False)
    rotation_m = numpy.array([matrix[0][:3], matrix[1][:3], matrix[2][:3]])
    translation_m = numpy.array([matrix[0][3], matrix[1][3], matrix[2][3]])

    rotation_m = list(R.from_matrix(rotation_m).as_euler('xyz', degrees=True))

    if print_out:
        with open("old_project_files/rotation_icp.txt", "a") as file:
            file.write("x: {}, y: {}, z: {}\n".format(rotation_m[0], rotation_m[1], rotation_m[2]))

        with open("old_project_files/translation_icp.txt", "w") as file:
            file.write("x: {}, y: {}, z: {}\n".format(translation_m[0], translation_m[1], translation_m[2]))

        # print_matrix_info(rotation_m, translation_m)

    return matrix

def icp_rot_vec(other, key):
    matrix, transformed, _ = trimesh.registration.icp(other, key, scale=False)
    rotation_m = numpy.array([matrix[0][:3], matrix[1][:3], matrix[2][:3]])
    translation_m = numpy.array([matrix[0][3], matrix[1][3], matrix[2][3]])

    rot_vec = R.from_matrix(rotation_m).as_rotvec()

    return matrix, rot_vec


def print_matrix_info(rotation_m, translation_m):
    print("Rotation around x axis in degrees: {}".format(rotation_m[0]))
    print("Rotation around y axis in degrees: {}".format(rotation_m[1]))
    print("Rotation around z axis in degrees: {}".format(rotation_m[2]))
    print()
    print("Translation on x axis: {}".format(translation_m[0]))
    print("Translation on y axis: {}".format(translation_m[1]))
    print("Translation on z axis: {}".format(translation_m[2]))


def vertices_transformation(matrix, object_list):
    """Apply transformation matrix to every point of the objects vertices"""
    for i in range(len(object_list)):
        t_vertices = []
        vertices = object_list[i][0]
        for vertex in vertices:
            t_vertex = list(vertex)
            t_vertex.append(1)
            t_vertex = numpy.matmul(matrix, np.array(t_vertex).T)
            t_vertex /= t_vertex[-1]
            t_vertices.append(tuple(t_vertex[:3]))
        object_list[i][0] = numpy.array(t_vertices)

    return object_list


def find_bounding_box(vertices):
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf

    for x, y, z in vertices:
        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if z < min_z:
            min_z = z

        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y
        if z > max_z:
            max_z = z

    return [min_x, min_y, min_z], [max_x, max_y, max_z]


def find_center_point(vertices, bounding_box=None):
    if bounding_box is None:
        bounding_box = find_bounding_box(vertices)
    minimum = bounding_box[0]
    maximum = bounding_box[1]

    x = (minimum[0] + maximum[0]) / 2
    y = (minimum[1] + maximum[1]) / 2
    z = (minimum[2] + maximum[2]) / 2

    return x, y, z


def create_translation_matrix(center, other_point):
    return [[1, 0, 0, center[0] - other_point[0]],
            [0, 1, 0, center[1] - other_point[1]],
            [0, 0, 1, center[2] - other_point[2]],
            [0, 0, 0, 1]]


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


def write_center_movements():
    """
    Function to write translation matrix between plan prostate center and prostates centres in different timestamps
    in a file
    """
    key_center = find_center_point(plan[1][0])
    with open("old_project_files/translation_center.txt", "a") as file:
        for entry in os.listdir(FILEPATH + "\\137_prostate"):
            prostate = import_obj([FILEPATH + "\\137_prostate\\{}".format(entry)])
            other_center = find_center_point(prostate[0][0])
            matrix = create_translation_matrix(key_center, other_center)
            translation_m = numpy.array([matrix[0][3], matrix[1][3], matrix[2][3]])

            file.write("x: {}, y: {}, z: {}\n".format(translation_m[0], translation_m[1], translation_m[2]))


def write_icp_movements():
    """
    Function that writes to two files translation and rotation matrices between plan bones and bones in different
    timestamps
    """
    for entry in os.listdir(FILEPATH + "\\137_bones"):
        bones = import_obj([FILEPATH + "\\137_bones\\{}".format(entry)])
        transform_matrix = icp_transformation_matrices(bones[0][0], plan[0][0], True)


def write_icp_center_movements():
    """
    Function that writes translation matrices to a file. They are computed by applying transformation matrix between
    plan bones and bones in different timestamps to prostates. Then there is computed a translation matrix between
    plan prostate center and translated prostates centers
    """
    plan_prostate_center = find_center_point(plan[1][0])
    with open("old_project_files/translation_icp_center_prostates", "a") as file:
        for prostate, bone in zip(os.listdir(FILEPATH + "\\137_prostate"), os.listdir(FILEPATH + "\\137_bones")):
            bone = import_obj([FILEPATH + "\\137_bones\\{}".format(bone)])
            transformation_matrix = icp_transformation_matrices(bone[0][0], plan[0][0])

            prostate = import_obj([FILEPATH + "\\137_prostate\\{}".format(prostate)])
            transformed_prost = vertices_transformation(transformation_matrix, [prostate[0]])

            other_center = find_center_point(transformed_prost[0][0])
            matrix = create_translation_matrix(plan_prostate_center, other_center)
            translation_m = numpy.array([matrix[0][3], matrix[1][3], matrix[2][3]])

            file.write("x: {}, y: {}, z: {}\n".format(translation_m[0], translation_m[1], translation_m[2]))


def compute_prostate_center_distances():
    """
    Computes distances between plan prostate center and prostate centers in different timestamps
    :return: list of distances
    """
    plan_prostate_center = find_center_point(plan[1][0])
    distances = []
    for entry in os.listdir(FILEPATH + "\\137_prostate"):
        prostates = import_obj([FILEPATH + "\\137_prostate\\{}".format(entry)])
        prost_center = find_center_point(prostates[0][0])
        distances.append(numpy.linalg.norm(numpy.array(plan_prostate_center) - numpy.array(prost_center)))

    return distances


def compute_distances_after_icp_centroid(patient):
    """
    Function that computes distances between aligned organs and their plan organs. Organs were aligned according to the
    icp transformation matrix computed from plan bones and bones in different timestamps
    :return: 2d list of distances in order: prostate, bladder, rectum
    """
    plan_prostate_center = trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)).centroid
    plan_bladder_center = trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder_plan.obj".format(patient)).centroid
    plan_rectum_center = trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum_plan.obj".format(patient)).centroid

    distances = [[], [], []]
    plan_bone = import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
    keys = [numpy.array(plan_prostate_center), numpy.array(plan_bladder_center), numpy.array(plan_rectum_center)]

    for i in range(1, 14):
        bone = import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, i)])
        transform_matrix = icp_transformation_matrices(bone[0][0], plan_bone[0][0])

        prostate = numpy.array(trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, i)).centroid)
        bladder = numpy.array(trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, i)).centroid)
        rectum = numpy.array(trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, i)).centroid)
        organs = [prostate, bladder, rectum]

        for i in range(3):
            transf_organ_center = vertices_transformation(transform_matrix, [[[organs[i]]]])
            distances[i].append(numpy.linalg.norm(keys[i] - numpy.array(transf_organ_center)))

    return distances, movement_vectors


def compute_distances_after_icp(patient):
    """
    Function that computes distances between aligned organs and their plan organs. Organs were aligned according to the
    icp transformation matrix computed from plan bones and bones in different timestamps
    :return: 2d list of distances in order: prostate, bladder, rectum, bones
    """
    plan_prostate_center = find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)).bounds)
    plan_bladder_center = find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder_plan.obj".format(patient)).bounds)
    plan_rectum_center = find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum_plan.obj".format(patient)).bounds)

    distances, movement_vectors = [[], [], []], [[], [], []]
    plan_bone = import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
    keys = [numpy.array(plan_prostate_center), numpy.array(plan_bladder_center), numpy.array(plan_rectum_center)]

    for i in range(1, 14):
        bone = import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, i)])
        transform_matrix = icp_transformation_matrices(bone[0][0], plan_bone[0][0])

        prostate = numpy.array(find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, i)).bounds))
        bladder = numpy.array(find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, i)).bounds))
        rectum = numpy.array(find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, i)).bounds))
        organs = [prostate, bladder, rectum]

        for i in range(3):
            transf_organ_center = vertices_transformation(transform_matrix, [[[organs[i]]]])
            distances[i].append(numpy.linalg.norm(keys[i] - numpy.array(transf_organ_center)))
            movement_vectors[i].append((np.ravel(keys[i] - numpy.array(transf_organ_center))).tolist())
    # print("")
    # print_mov(movement_vectors)

    return distances, movement_vectors


def print_mov(mov):
    for organ in mov:
        for o in organ:
            print(list(o), end="")
        print("")


def compute_distances_after_centering(patient):
    plan_prostate_center = find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)).bounds)
    plan_bladder_center = find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder_plan.obj".format(patient)).bounds)
    plan_rectum_center = find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum_plan.obj".format(patient)).bounds)
    plan_bones_center = find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)).bounds)

    # movement vectors or distances
    distances, movement_vectors = [[], [], []], [[], [], []]
    keys = [numpy.array(plan_bones_center), numpy.array(plan_bladder_center), numpy.array(plan_rectum_center)]

    for i in range(1, 14):
        prostate = numpy.array(find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, i)).bounds))
        bladder = numpy.array(find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, i)).bounds))
        rectum = numpy.array(find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, i)).bounds))
        bones = numpy.array(find_center_point([], trimesh.load_mesh(FILEPATH + "{}\\bones\\bones{}.obj".format(patient, i)).bounds))

        organs = [bones, bladder, rectum]
        transform_matrix = create_translation_matrix(plan_prostate_center, prostate)

        for i in range(3):
            transf_organ_center = vertices_transformation(transform_matrix, [[[organs[i]]]])
            distances[i].append(numpy.linalg.norm(keys[i] - numpy.array(transf_organ_center)))
            movement_vectors[i].append((np.ravel(keys[i] - numpy.array(transf_organ_center))).tolist())

    return distances, movement_vectors


def timer():
    """Used just for deciding which functions are faster"""
    start = time.time()
    compute_distances_after_centering_old("146")
    end = time.time()
    print(end - start)

    start = time.time()
    compute_distances_after_centering("146")
    end = time.time()
    print(end - start)


def compute_average_distances(distances):
    prostate, bladder, rectum = distances[0], distances[1], distances[2]
    avrg_prostate = np.average(prostate)
    avrg_bladder = np.average(bladder)
    avrg_rectum = np.average(rectum)

    return [avrg_prostate, avrg_bladder, avrg_rectum]


# timer()

# print(find_center_point(import_obj([PATH + "{}\\prostate\\prostate_plan.obj".format("137")])[0][0]))
# print(compute_distances_after_centering("137"))
