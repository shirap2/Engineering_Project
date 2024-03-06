import pickle
import os
from common_packages.LongGraphPackage import LoaderSimpleFromJson, DrawerLabels
from common_packages.LongGraphClassification import LongitClassification
from generate_info.gen_single_lesion.gen_single_lesion_pdf import get_dates
import matplotlib.pyplot as plt

# USR = "shira_p/PycharmProjects/engineering_project/matching"
USR = "talia.dym/Desktop/Engineering_Project"

class Organ:
    LIVER = 'liver',
    BRAIN = 'brain',
    LUNGS = 'lungs'


class PatientInput:
    def __init__(self, name, organ: Organ, partial_scans_address, json_input_address,
                 pickle_input_address, graph_image_path):
        self.name = name
        self.organ = organ[0]  # tuple -> str
        self.partial_scans_address = partial_scans_address
        self.json_input_address = json_input_address
        self.pickle_input_address = pickle_input_address
        self.graph_image_path = graph_image_path


def get_patient_input(name_for_path: str, organ: Organ):

    patient_name = name_for_path.replace("_", ". ")

    pickle_input_address = f"/cs/usr/{USR}/input/{organ[0]}/pkl_files/{name_for_path}_graph_class_data.pkl"
    graph_image_path = f"/cs/usr/{USR}/input/{organ[0]}/graph_images/{name_for_path}_graph_image.png"

    json_input_address = f"/cs/casmip/bennydv/{organ[0]}_pipeline/" \
                         f"lesions_matching/longitudinal_gt/original_corrected/{name_for_path}glong_gt.json"
    partial_scans_address = f"/cs/casmip/bennydv/{organ[0]}_pipeline/" \
                            f"gt_data/size_filtered/labeled_no_reg/{name_for_path}"

    return PatientInput(patient_name, organ,
                        partial_scans_address, json_input_address, pickle_input_address, graph_image_path)


def save_patient_input_into_pickle_file(name, patient, ld):
    # data to dump
    lg = LongitClassification(ld, name, get_dates(patient.partial_scans_address))

    # save
    path_to_save_in = patient.pickle_input_address
    if os.path.exists(path_to_save_in):
        os.remove(path_to_save_in)

    with open(path_to_save_in, "wb") as file:
        # pickle.dump(ld, file)
        pickle.dump(lg, file)


def save_patient_input_graph_image(ld, patient):
    plt.figure()
    lg1 = LongitClassification(ld)
    dr_2 = DrawerLabels(lg1)
    dr_2.show_graph(patient.graph_image_path)


def save_patient_input(name: str, organ: Organ):
    patient = get_patient_input(name, organ)
    ld = LoaderSimpleFromJson(patient.json_input_address)
    save_patient_input_graph_image(ld, patient)
    save_patient_input_into_pickle_file(name, patient, ld)


def list_folders(directory):  # used to get all patients names
    folders = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return folders


def create_all_input():
    for organ in ['liver', 'brain', 'lungs']:
        path = f"/cs/casmip/bennydv/{organ}_pipeline/gt_data/size_filtered/labeled_no_reg/"
        for patient in list_folders(path):
            save_patient_input(patient, (organ,))

