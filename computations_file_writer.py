import trimesh
import trimesh.registration
import numpy
from scipy.spatial.transform import Rotation as R

import Project_2
import json
import constants


def write_computations(distances, movements, averages, icp):
    with open(distances, "w") as dist_f, open(movements, "w") as mov_f, open(averages, "w") as avrg_f:
        all_dist, all_mov = [], []
        for pat in constants.PATIENTS:
            print(pat)
            if icp:
                dist, mov = Project_2.compute_distances_after_icp(pat)
            else:
                dist, mov = Project_2.compute_distances_after_centering(pat)

            prostate_bones, bladder, rectum = Project_2.compute_average_distances(dist)

            all_dist.append(dist)
            all_mov.append(mov)

            print(prostate_bones, file=avrg_f)
            print(bladder, file=avrg_f)
            print(rectum, file=avrg_f)

        json.dump(all_dist, dist_f)
        json.dump(all_mov, mov_f)


def write_computations_centroid(distances, averages, icp):
    with open(distances, "w") as dist_f, open(averages, "w") as avrg_f:
        all_dist = []
        for pat in constants.PATIENTS:
            print(pat)
            if icp:
                dist = Project_2.compute_distances_after_icp_centroid(pat)
            else:
                dist = Project_2.compute_distances_after_centering_centroid(pat)

            prostate_bones, bladder, rectum = Project_2.compute_average_distances(dist)

            all_dist.append(dist)

            print(prostate_bones, file=avrg_f)
            print(bladder, file=avrg_f)
            print(rectum, file=avrg_f)

        json.dump(all_dist, dist_f)


# write_computations_centroid("icp_distances_c.txt", "icp_averages_c.txt", True)
# write_computations_centroid("center_distances_c.txt", "center_averages_c.txt", False)

# write_computations("icp_distances.txt", "icp_movements.txt", "icp_averages.txt", True)
# write_computations("center_distances.txt", "center_movements.txt", "center_averages.txt", False)

# with open("computations_files/icp_movements.txt", "r") as icp_mov:
#     all_movements_icp = json.load(icp_mov)


def write_plan_center_points():
    with open("computations_files/plan_center_points.txt", "w") as cent_points:
        all_cent_p = []
        for pat in constants.PATIENTS:
            cent_p = []
            for organ in ["bones", "prostate", "bladder", "rectum"]:
                x1, y1, z1 = Project_2.find_center_point([], trimesh.load_mesh(
                    constants.FILEPATH + "{}\\{}\\{}_plan.obj".format(pat, organ, organ)).bounds)

                cent_p.append((x1, y1, z1))
            all_cent_p.append(cent_p)
        json.dump(all_cent_p, cent_points)


def write_rotation_file():
    with open("computations_files/rotation_icp.txt", "w") as rots_file:
        all_rot = []
        for pat in constants.PATIENTS:
            plan_bones = Project_2.import_obj([constants.FILEPATH + "{}\\bones\\bones_plan.obj".format(pat)])
            rot_x, rot_y, rot_z = [], [], []
            patient_rot = []
            for i in range(1, 14):
                bones = Project_2.import_obj([constants.FILEPATH + "{}\\bones\\bones{}.obj".format(pat, i)])
                matrix, transformed, _ = trimesh.registration.icp(bones[0][0], plan_bones[0][0], scale=False)
                rotation_m = numpy.array([matrix[0][:3], matrix[1][:3], matrix[2][:3]])

                rotation_m = list(R.from_matrix(rotation_m).as_euler('xyz', degrees=True))
                rot_x.append(rotation_m[0])
                rot_y.append(rotation_m[1])
                rot_z.append(rotation_m[2])

            patient_rot.append([rot_x, rot_y, rot_z])
            all_rot.append(patient_rot)

        json.dump(all_rot, rots_file)


# write_rotation_file()

# write_plan_center_points()
