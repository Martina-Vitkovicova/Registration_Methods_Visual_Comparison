import shutil

PATH = "C:\\Users\\vitko\\Documents\\CTs\\"
NEW_PATH = "C:\\Users\\vitko\\Desktop\\ProjetHCI\\Organs\\"


def copy_files(patient):
    for i in range(1, 14):
        old_name = PATH + "{} CBCTs\\{}\\".format(patient, i)

        new_name = NEW_PATH + "{}\\bladder\\bladder{}.obj".format(patient, i)
        shutil.copy(old_name + "bladder.obj", new_name)

        new_name = NEW_PATH + "{}\\prostate\\prostate{}.obj".format(patient, i)
        shutil.copy(old_name + "prostate.obj", new_name)

        new_name = NEW_PATH + "{}\\rectum\\rectum{}.obj".format(patient, i)
        shutil.copy(old_name + "rectum.obj", new_name)

    shutil.copy(PATH + "{} CBCTs\\plan\\prostate.obj".format(patient),
                NEW_PATH + "{}\\prostate\\prostate_plan.obj".format(patient))

    shutil.copy(PATH + "{} CBCTs\\plan\\bladder.obj".format(patient),
                NEW_PATH + "{}\\bladder\\bladder_plan.obj".format(patient))

    shutil.copy(PATH + "{} CBCTs\\plan\\rectum.obj".format(patient),
                NEW_PATH + "{}\\rectum\\rectum_plan.obj".format(patient))


# copy_files("722")

