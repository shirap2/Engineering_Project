import pickle
import os
from common_packages.LongGraphPackage import LoaderSimpleFromJson, DrawerLabels
from common_packages.LongGraphClassification import LongitClassification
from generate_info.gen_single_lesion.gen_single_lesion_pdf import get_dates
import matplotlib.pyplot as plt


def save_patient_input_into_pickle_file(name, name_for_path, partial_scans_adress, ld):

    # data to dump
    lg = LongitClassification(ld, name, get_dates(partial_scans_adress))

    # save
    path_to_save_in = f"/cs/usr/talia.dym/Desktop/Engineering_Project/input/pkl_files/{name_for_path}_graph_class_data.pkl"
    if os.path.exists(path_to_save_in):
        os.remove(path_to_save_in)

    
    with open(path_to_save_in, "wb") as file:
        # pickle.dump(ld, file)
        pickle.dump(lg, file)

def save_patient_input_graph_image(ld, name_for_path):
    plt.figure()
    lg1 = LongitClassification(ld)
    dr_2 = DrawerLabels(lg1)

    image_path = f"/cs/usr/talia.dym/Desktop/Engineering_Project/input/graph_images/{name_for_path}_graph_image.png"
    dr_2.show_graph(image_path)


def save_patient_input(name, partial_scans_adress, json_input_address):
    name_for_path = name.replace(" ", "_").replace(".", "")
    ld = LoaderSimpleFromJson(json_input_address)
    save_patient_input_graph_image(ld, name_for_path)
    save_patient_input_into_pickle_file(name, name_for_path, partial_scans_adress, ld)



# # A. W.
# NAME = "A. W."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_"

# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

# A. S. H.
NAME = "A. S. H."
JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_S_H_glong_gt.json"
PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_S_H_"

save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

# # # A. S. S.
# NAME = "A. S. S."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_S_S_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_S_S_"

# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

# # B. B. S.
# NAME = "B. B. S."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/B_B_S_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/B_B_S_"

# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

# # B. T.
# NAME = "B. T."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/B_T_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/B_T_"

# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

# # C. A.
# NAME = "C. A."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/C_A_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/C_A_"

# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)