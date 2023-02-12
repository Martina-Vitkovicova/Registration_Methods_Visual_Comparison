import numpy as np
import trimesh.registration
import pywavefront
from scipy.spatial.transform import Rotation
from constants import FILEPATH


def import_obj(files):
    """Create a list of vertices and faces from the .obj files."""
    object_list = []

    for name in files:
        organ = pywavefront.Wavefront(name, collect_faces=True)
        object_list.append([organ.vertices, organ.mesh_list[0].faces])

    return object_list


def icp_transformation_matrix(other, key):
    """Create transformation matrix using icp algorithm from trimesh library."""
    matrix, transformed, _ = trimesh.registration.icp(other, key, scale=False)
    rotation_m = np.array([matrix[0][:3], matrix[1][:3], matrix[2][:3]])

    # translation_m and rotation_m were used and can be used individually, however we currently use only the matrix
    translation_m = np.array([matrix[0][3], matrix[1][3], matrix[2][3]])
    rotation_m = list(Rotation.from_matrix(rotation_m).as_euler('xyz', degrees=True))

    return matrix


def vertices_transformation(matrix, object_list):
    """Apply transformation matrix to every vertex of the objects vertices."""
    for i in range(len(object_list)):
        transformed_vertices = []
        vertices = object_list[i][0]

        for vertex in vertices:
            transformed_vertex = list(vertex)
            transformed_vertex.append(1)
            transformed_vertex = np.matmul(matrix, np.array(transformed_vertex).T)
            transformed_vertex /= transformed_vertex[-1]
            transformed_vertices.append(tuple(transformed_vertex[:3]))
        object_list[i][0] = np.array(transformed_vertices)

    return object_list


def find_center_of_mass(vertices):
    """Find the centroid as sum of the vertices divided by their count."""
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
    """Compute transformation matrix used in prostate centring RM, which is just a translation matrix"""
    return [[1, 0, 0, center[0] - other_point[0]],
            [0, 1, 0, center[1] - other_point[1]],
            [0, 0, 1, center[2] - other_point[2]],
            [0, 0, 0, 1]]


def compute_distances_after_icp_centroid(patient):
    """
    Compute distances between aligned organs and their equivalent plan organs. Organs were aligned according to the
    icp transformation matrix computed from plan bones and bones in different timestamps.
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
            transform_organ_center = vertices_transformation(transform_matrix, [[[organs[j]]]])
            distances[j].append(np.linalg.norm(keys[j] - np.array(transform_organ_center)))

    return distances


def compute_distances_after_centering_centroid(patient):
    """
    Compute distances between aligned organs and their equivalent plan organs. Organs were aligned according to the
    prostate centring translation matrix computed from plan prostate and prostate in different timestamps.
    :return: 2d list of distances in order: bones, bladder, rectum
    """
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
            transform_organ_center = vertices_transformation(transform_matrix, [[[organs[j]]]])
            distances[j].append(np.linalg.norm(keys[j] - np.array(transform_organ_center)))

    return distances


def compute_average_distances(distances):
    """Compute average distances of patient's organs during the treatment time. The first organ in RM-dependent"""
    prostate_or_bones, bladder, rectum = distances[0], distances[1], distances[2]
    avrg_prostate = np.average(prostate_or_bones)
    avrg_bladder = np.average(bladder)
    avrg_rectum = np.average(rectum)

    return [avrg_prostate, avrg_bladder, avrg_rectum]
