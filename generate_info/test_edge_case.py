import os
import pickle

from common_packages.LongGraphPackage import LoaderSimpleFromJson
from generate_info.gen_all import get_patient_input, create_pdf_file
from patient_summary.classify_changes_in_individual_lesions import gen_dict_classified_nodes_for_layers, \
    count_d_in_d_out_for_test, classify_changes_in_individual_lesions_test
import networkx as nx


def check_for_edge_case(patient_name):
    """
    this function checks for edge case in each CC
    when there is a split/merge/complex and other lesions in the same time stamp.
    need to figure out how to implement percentage arrows in this case
    :param patient_name:
    :return:
    """
    files_to_check = []
    patient = get_patient_input(patient_name)
    pkl_path = patient.pickle_input_address
    with open(pkl_path, "rb") as file:
        lg = pickle.load(file)
    G = lg.get_graph()

    ld = LoaderSimpleFromJson(patient.json_input_address)
    components = list(nx.connected_components(G))
    for cc in components:
        # classify_changes_in_individual_lesions(count_d_in_d_out_for_test(G),ld)
        classified_nodes_dict = gen_dict_classified_nodes_for_layers(
            classify_changes_in_individual_lesions_test(count_d_in_d_out_for_test(G), cc))

        result = check_complex_merge_split(classified_nodes_dict)
        if result:
            # files_to_check.append(patient_name)
            print(cc)
            return True

    return False


def check_complex_merge_split(timestamp_dict):
    """
    this function iterates over the dictionary {timestamp(int):{'class':num}...} and checks if there is a timestamp that
    has at least 2 types of classes where at least one is merge/split/complex
    :param timestamp_dict: dictionary
    :return: true if condition occurs, false otherwise
    """
    for timestamp, classes_dict in timestamp_dict.items():
        if len(classes_dict) >= 2:
            class_list = list(classes_dict.keys())
            if 'complex' in class_list or 'merge' in class_list or 'split' in class_list:
                return True
    return False


folder_path_brain = "/cs/casmip/bennydv/brain_pipeline/lesions_matching/longitudinal_gt/original_corrected/"
folder_path_liver = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/"
# Iterate over files in the folder
for filename in os.listdir(folder_path_liver):
    if "glong" in filename:
        name_until_glong = filename.split('_glong')[0]
        edge_case_res = check_for_edge_case(name_until_glong)
        if edge_case_res:
            print(name_until_glong)
        create_pdf_file(name_until_glong)