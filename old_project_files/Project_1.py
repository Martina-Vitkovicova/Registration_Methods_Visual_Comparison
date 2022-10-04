import math
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import trimesh.smoothing
import trimesh.registration
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy
from skimage import measure
from vtkmodules.vtkIOGeometry import vtkOBJReader


ds = pydicom.dcmread("137.dcm")
ds2 = pydicom.dcmread("rtst.dcm")
rt_dose = pydicom.dcmread("rtdose.dcm")


# print(ds)
# print(ds2)


def plot_3d(image, threshold=0):
    p = image.transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.5)

    mesh.set_facecolor([0.5, 0.5, 1])
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()


def plt_3d(verts, faces):
    print("Drawing")
    x, y, z = zip(*verts)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.5, alpha=0.5)
    face_color = [0.2, 0.1, 0.7]
    mesh.set_facecolor(face_color)
    mesh.set_edgecolor([0.1, 0.1, 0.1])
    ax.add_collection3d(mesh)

    ax.set_xlim(min(x), max(x))
    ax.set_ylim(min(y), max(y))
    ax.set_zlim(min(z), max(z))
    plt.show()


def visualize(object_array):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    index = 0
    min_x = min_y = min_z = math.inf
    max_x = max_y = max_z = -math.inf

    for verts, faces in object_array:
        x, y, z = zip(*verts)

        mesh2 = Poly3DCollection(verts[faces], linewidths=0.5, alpha=0.5)
        mesh2.set_facecolor([0.1 + index, 0.1, 0.5])
        mesh2.set_edgecolor([0.1, 0.1, 0.1])

        ax.add_collection3d(mesh2)

        if min(x) < min_x:
            min_x = min(x)
        if min(y) < min_y:
            min_y = min(y)
        if min(z) < min_z:
            min_z = min(z)

        if max(x) > max_x:
            max_x = max(x)
        if max(y) > max_y:
            max_y = max(y)
        if max(z) > max_z:
            max_z = max(z)

        index += 0.2

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_zlim(min_z, max_z)

    plt.show()


def trimesh_vis(object_array):
    scene = trimesh.scene.scene.Scene()
    index = 0
    for verts, faces in object_array:
        mesh = trimesh.Trimesh(verts, faces, face_colors=[50 + index, 150, 255, 200])
        mesh = trimesh.smoothing.filter_mut_dif_laplacian(mesh)
        scene.add_geometry(mesh)
        index += 50

    scene.show()


def trimesh_icp(verts1, verts2):
    matrix, transform, _ = trimesh.registration.icp(verts1, verts2)

    # mesh = trimesh.Trimesh(transform)
    # mesh.show()
    return matrix


def translation(transf_matrix, verts):
    t_vertices = []
    for vertex in verts:
        t_vertex = list(vertex)
        t_vertex.append(1)
        t_vertex = numpy.matmul(transf_matrix, t_vertex)
        t_vertices.append(t_vertex[:3])

    t_vertices = numpy.array(t_vertices)
    return t_vertices


def make_faces(verts):
    faces_dict = {}
    index = 0
    for point in verts:
        if faces_dict.get(tuple(point)) is None:
            faces_dict[tuple(point)] = index
            index += 1
    return faces_dict


def make_mesh(image, threshold=0, step_size=1):
    p = image.transpose(2, 1, 0)
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=False)
    return verts, faces


def obj_reader(obj_file):
    reader = vtkOBJReader()
    reader.SetFileName(obj_file)
    reader.Update(0)
    data = reader.GetOutput()

    faces_dict = {}
    index = 0
    all_vertices = []

    for i in range(data.GetNumberOfCells()):
        pts = data.GetCell(i).GetPoints()
        for j in range(pts.GetNumberOfPoints()):
            point = pts.GetPoint(j)
            all_vertices.append(point)
            if faces_dict.get(point) is None:
                faces_dict[point] = index
                index += 1

    return all_vertices, faces_dict


def merge_dcm(dcm1, dcm2):
    for i in range(len(dcm1)):
        for j in range(len(dcm1[0])):
            for k in range(len(dcm1[0][0])):
                if i < len(dcm2) and j < len(dcm2[0]) and k < len(dcm2[0][0]) and dcm2[i][j][k] != 0:
                    dcm1[i][j][k] = dcm2[i][j][k]

    return dcm1


def rename_faces(all_vertices, faces_dict):
    result_faces = []
    face = []
    for vertex in all_vertices:
        face.append(faces_dict[tuple(vertex)])
        if len(face) == 3:
            result_faces.append(face)
            face = []
    return result_faces


vertices, dict = obj_reader("bladder.obj")
vertices1, dict1 = obj_reader("prostate.obj")
vertices2, dict2 = obj_reader("rectum.obj")
vertices3, dict3 = obj_reader("bones.obj")

scaling_matrix = [[10, 0, 0, 0],
                  [0, 10, 0, 0],
                  [0, 0, 10, 0],
                  [0, 0, 0, 1]]

# p = ds.pixel_array.transpose((2, 1, 0))
# verts, faces, normals, values = measure.marching_cubes(p, 0)

# matrix = trimesh_icp(vertices3, vertices1)
# trans_verts = translation(matrix, verts)
# scaled_verts = translation(scaling_matrix, trans_verts)

organs = [[vertices, dict], [vertices1, dict1], [vertices2, dict2], [vertices3, dict3]]
# organs_mesh = [[verts, faces]]
organs_mesh = []
# organs_mesh = [[scaled_verts, faces]]

for verts, dict in organs:
    faces = numpy.array(rename_faces(verts, dict))
    vertices = numpy.array(list(dict.keys()))
    organs_mesh.append([vertices, faces])

trimesh_vis(organs_mesh)
# visualize(organs_mesh)
# print(matrix)
