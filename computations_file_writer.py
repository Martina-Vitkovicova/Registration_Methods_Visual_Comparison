import trimesh

import Project_2
import json
PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]
FILEPATH = "C:\\Users\\vitko\\Desktop\\ProjetHCI-BT\\BT_implementation\\Organs\\"
TIMESTAMPS = list(range(1, 14))


def write_computations(distances, movements, averages, icp):
    with open(distances, "w") as dist_f, open(movements, "w") as mov_f, open(averages, "w") as avrg_f:
        all_dist, all_mov = [], []
        for pat in PATIENTS:
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


# write_computations("icp_distances.txt", "icp_movements.txt", "icp_averages.txt", True)
# write_computations("center_distances.txt", "center_movements.txt", "center_averages.txt", False)

# with open("computations_files/icp_movements.txt", "r") as icp_mov:
#     all_movements_icp = json.load(icp_mov)


def write_plan_center_points():
    with open("computations_files/plan_center_points.txt", "w") as cent_points:
        all_cent_p = []
        for pat in PATIENTS:
            cent_p = []
            for organ in ["bones", "prostate", "bladder", "rectum"]:
                x1, y1, z1 = Project_2.find_center_point([], trimesh.load_mesh(
                    FILEPATH + "{}\\{}\\{}_plan.obj".format(pat, organ, organ)).bounds)

                cent_p.append((x1, y1, z1))
            all_cent_p.append(cent_p)
        json.dump(all_cent_p, cent_points)


write_plan_center_points()
