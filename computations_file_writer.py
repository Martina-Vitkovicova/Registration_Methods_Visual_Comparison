import Project_2
import json
PATIENTS = ["137", "146", "148", "198", "489", "579", "716", "722"]


def write_computations(distances, movements, averages, icp):
    with open(distances, "w") as dist_f, open(movements, "w") as mov_f, \
            open(averages, "w") as avrg_f:

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

