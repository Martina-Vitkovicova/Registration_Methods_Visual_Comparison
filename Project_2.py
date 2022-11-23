import numpy as np
import trimesh.registration
import pywavefront
from scipy.spatial.transform import Rotation as R
from constants import FILEPATH


def import_obj(files):
    """Create a list of vertices and faces from the .obj files"""
    object_list = []

    for name in files:
        organ = pywavefront.Wavefront(name, collect_faces=True)
        object_list.append([organ.vertices, organ.mesh_list[0].faces])

    return object_list


def icp_transformation_matrix(other, key):
    """Create transformation matrix using icp algorithm from trimesh library"""
    matrix, transformed, _ = trimesh.registration.icp(other, key, scale=False)
    rotation_m = np.array([matrix[0][:3], matrix[1][:3], matrix[2][:3]])
    translation_m = np.array([matrix[0][3], matrix[1][3], matrix[2][3]])

    rotation_m = list(R.from_matrix(rotation_m).as_euler('xyz', degrees=True))

    return matrix


def vertices_transformation(matrix, object_list):
    """Apply transformation matrix to every point of the objects vertices"""
    for i in range(len(object_list)):
        t_vertices = []
        vertices = object_list[i][0]
        for vertex in vertices:
            t_vertex = list(vertex)
            t_vertex.append(1)
            t_vertex = np.matmul(matrix, np.array(t_vertex).T)
            t_vertex /= t_vertex[-1]
            t_vertices.append(tuple(t_vertex[:3]))
        object_list[i][0] = np.array(t_vertices)

    return object_list


def find_center_of_mass(vertices):
    center_x, center_y, center_z = 0, 0, 0

    for x, y, z in vertices:
        center_x += x
        center_y += y
        center_z += z

    center_x /= len(vertices)
    center_y /= len(vertices)
    center_z /= len(vertices)

    return center_x, center_y, center_z


def create_translation_matrix(center, other_point):
    return [[1, 0, 0, center[0] - other_point[0]],
            [0, 1, 0, center[1] - other_point[1]],
            [0, 0, 1, center[2] - other_point[2]],
            [0, 0, 0, 1]]


def compute_distances_after_icp_centroid(patient):
    """
    Function that computes distances between aligned organs and their plan organs. Organs were aligned according to the
    icp transformation matrix computed from plan bones and bones in different timestamps
    :return: 2d list of distances in order: prostate, bladder, rectum
    """
    plan_prostate_center = find_center_of_mass(trimesh.load_mesh(
        FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)).vertices)
    plan_bladder_center = find_center_of_mass(trimesh.load_mesh(
        FILEPATH + "{}\\bladder\\bladder_plan.obj".format(patient)).vertices)
    plan_rectum_center = find_center_of_mass(trimesh.load_mesh(
        FILEPATH + "{}\\rectum\\rectum_plan.obj".format(patient)).vertices)

    distances = [[], [], []]
    plan_bone = import_obj([FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)])
    keys = [np.array(plan_prostate_center), np.array(plan_bladder_center), np.array(plan_rectum_center)]

    for i in range(1, 14):
        bone = import_obj([FILEPATH + "{}\\bones\\bones{}.obj".format(patient, i)])
        transform_matrix = icp_transformation_matrix(bone[0][0], plan_bone[0][0])

        prostate = np.array(find_center_of_mass(trimesh.load_mesh(
            FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, i)).vertices))
        bladder = np.array(find_center_of_mass(trimesh.load_mesh(
            FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, i)).vertices))
        rectum = np.array(find_center_of_mass(trimesh.load_mesh(
            FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, i)).vertices))
        organs = [prostate, bladder, rectum]

        for j in range(3):
            transf_organ_center = vertices_transformation(transform_matrix, [[[organs[j]]]])
            distances[j].append(np.linalg.norm(keys[j] - np.array(transf_organ_center)))

    return distances


def compute_distances_after_centering_centroid(patient):
    plan_prostate_center = find_center_of_mass(trimesh.load_mesh(
        FILEPATH + "{}\\prostate\\prostate_plan.obj".format(patient)).vertices)
    plan_bladder_center = find_center_of_mass(trimesh.load_mesh(
        FILEPATH + "{}\\bladder\\bladder_plan.obj".format(patient)).vertices)
    plan_rectum_center = find_center_of_mass(trimesh.load_mesh(
        FILEPATH + "{}\\rectum\\rectum_plan.obj".format(patient)).vertices)
    plan_bones_center = find_center_of_mass(trimesh.load_mesh(
        FILEPATH + "{}\\bones\\bones_plan.obj".format(patient)).vertices)

    distances = [[], [], []]
    keys = [np.array(plan_bones_center), np.array(plan_bladder_center), np.array(plan_rectum_center)]

    for i in range(1, 14):
        prostate = np.array(find_center_of_mass(trimesh.load_mesh(
            FILEPATH + "{}\\prostate\\prostate{}.obj".format(patient, i)).vertices))
        bladder = np.array(find_center_of_mass(trimesh.load_mesh(
            FILEPATH + "{}\\bladder\\bladder{}.obj".format(patient, i)).vertices))
        rectum = np.array(find_center_of_mass(trimesh.load_mesh(
            FILEPATH + "{}\\rectum\\rectum{}.obj".format(patient, i)).vertices))
        bones = np.array(find_center_of_mass(trimesh.load_mesh(
            FILEPATH + "{}\\bones\\bones{}.obj".format(patient, i)).vertices))

        organs = [bones, bladder, rectum]
        transform_matrix = create_translation_matrix(plan_prostate_center, prostate)

        for j in range(3):
            transf_organ_center = vertices_transformation(transform_matrix, [[[organs[j]]]])
            distances[j].append(np.linalg.norm(keys[j] - np.array(transf_organ_center)))

    return distances


def compute_average_distances(distances):
    prostate, bladder, rectum = distances[0], distances[1], distances[2]
    avrg_prostate = np.average(prostate)
    avrg_bladder = np.average(bladder)
    avrg_rectum = np.average(rectum)

    return [avrg_prostate, avrg_bladder, avrg_rectum]
