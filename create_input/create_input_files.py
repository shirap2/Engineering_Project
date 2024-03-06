import pickle
import os
from common_packages.LongGraphPackage import LoaderSimpleFromJson, DrawerLabels
from common_packages.LongGraphClassification import LongitClassification
from generate_info.gen_single_lesion.gen_single_lesion_pdf import get_dates
import matplotlib.pyplot as plt

USR = "shira_p/PycharmProjects/engineering_project/matching"
# USR = "talia.dym/Desktop/Engineering_Project"

def save_patient_input_into_pickle_file(name, name_for_path, partial_scans_adress, ld):
    # data to dump
    lg = LongitClassification(ld, name, get_dates(partial_scans_adress))

    # save
    path_to_save_in = f"/cs/usr/{USR}/input/pkl_files/{name_for_path}_graph_class_data.pkl"
    if os.path.exists(path_to_save_in):
        os.remove(path_to_save_in)

    with open(path_to_save_in, "wb") as file:
        # pickle.dump(ld, file)
        pickle.dump(lg, file)


def save_patient_input_graph_image(ld, name_for_path):
    plt.figure()
    lg1 = LongitClassification(ld)
    dr_2 = DrawerLabels(lg1)

    image_path = f"/cs/usr/{USR}/input/graph_images/{name_for_path}_graph_image.png"
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
# NAME = "A. S. H."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_S_H_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_S_H_"
#
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

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

# # E. N.
# NAME = "E. N."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/E_N_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/E_N_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

# NAME = "F. Y. Ga."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/F_Y_Ga_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/F_Y_Ga_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)


## next batch ###
# NAME = "G. B."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/G_B_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/G_B_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)
#
#
# NAME = "G. Y."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/G_Y_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/G_Y_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)
#
# NAME = "H. G."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/H_G_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/H_G_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)
#
# NAME = "M. I."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/M_I_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/M_I_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)
#
# NAME = "M. N."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/M_N_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/M_N_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

### next batch ###
# NAME = "N. M."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/N_M_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/N_M_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)
#
# NAME = "S. I."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/S_I_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/S_I_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)
#
# NAME = "S. N."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/S_N_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/S_N_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)
#
# NAME = "T. N."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/T_N_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/T_N_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

# NAME = "Z. Aa."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/Z_Aa_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/Z_Aa_"
# save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)


NAME = "AA0"
JSON_ADDRESS = "/cs/casmip/bennydv/brain_pipeline/lesions_matching/longitudinal_gt/original_corrected/AA0glong_gt.json"
PARTIAL_ADDRESS = "/cs/casmip/bennydv/brain_pipeline/gt_data/size_filtered/labeled_no_reg/AA0"
save_patient_input(NAME, PARTIAL_ADDRESS, JSON_ADDRESS)

folder_path = "/cs/casmip/bennydv/brain_pipeline/lesions_matching/longitudinal_gt/original_corrected/"

# Iterate over files in the folder
for filename in os.listdir(folder_path):
    if "glong" in filename:
        name_until_glong = filename.split('glong')[0]
        # NAME = "AA0"
        JSON_ADDRESS = folder_path+name_until_glong+"glong_gt.json"
        PARTIAL_ADDRESS = "/cs/casmip/bennydv/brain_pipeline/gt_data/size_filtered/labeled_no_reg/"+name_until_glong
        save_patient_input(name_until_glong, PARTIAL_ADDRESS, JSON_ADDRESS)
