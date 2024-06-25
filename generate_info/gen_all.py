from generate_info.gen_single_patient_pdf import create_single_patient_pdf_page
from generate_info.gen_single_lesion.gen_single_lesion_pdf import create_single_lesion_pdf_page
from reportlab.platypus import SimpleDocTemplate, PageBreak
import os
from volume.volume_calculation import generate_longitudinal_volumes_array
from create_input.create_input_files import get_patient_input, Organ
from pathlib import Path


ROOT = str(Path(__file__).resolve().parent).replace("generate_info", "")
output_path = ROOT + "output"


def get_sorted_cc_idx(cc_info_dict):
    # Sort the dictionary items by the first element of the list (volume_sum_of_last_nodes)
    sorted_items = sorted(cc_info_dict.items(), key=lambda item: item[1][0], reverse=True)

    # Extract and return the sorted keys
    sorted_cc_idx = [item[0] for item in sorted_items]
    return sorted_cc_idx


def organize_elements_per_cc(elements_per_cc, cc_info_dict):
    cc_elements_dict = {}
    elements = []

    for elem in elements_per_cc:

        # the cc elements
        if isinstance(elem, tuple):
            cc_idx, element = elem
            if cc_idx in cc_elements_dict:
                cc_elements_dict[cc_idx] += [element]
            else:
                cc_elements_dict[cc_idx] = [element]

        else:
            elements += [elem]

    # order elements by volume
    cc_idx_ordered_by_volume = get_sorted_cc_idx(cc_info_dict)
    for cc_idx in cc_idx_ordered_by_volume:

        if cc_idx in cc_elements_dict:
            elements += cc_elements_dict[cc_idx]

    return elements, cc_elements_dict


def create_pdf_file(patient_name: str, organ: Organ):
    patient = get_patient_input(patient_name, organ)
    pdf_name = f"{output_path}/{patient.organ}/" + patient_name.replace(" ", "_") + "patient_summary.pdf"
    if os.path.exists(pdf_name):
        os.remove(pdf_name)
    doc = SimpleDocTemplate(pdf_name)

    elements = []
    volumes_dict = generate_longitudinal_volumes_array(patient.partial_scans_address)  # returns sorted (by date)
    # array of dictionaries (one for each time stamp), key - lesion idx, value - volume in cm^3
    elements += create_single_patient_pdf_page(patient_name, patient.json_input_address, patient.partial_scans_address,
                                               patient.graph_image_path, volumes_dict)

    elements.append(PageBreak())

    elements_per_cc, cc_info_dict, _ = create_single_lesion_pdf_page(patient, volumes_dict)
    organized_cc_elements, _ = organize_elements_per_cc(elements_per_cc, cc_info_dict)
    elements += organized_cc_elements

    doc.build(elements)


def get_full_display_elements(patient_name: str, organ: Organ):
    patient = get_patient_input(patient_name, organ)

    elements = []
    volumes_dict = generate_longitudinal_volumes_array(patient.partial_scans_address)  # returns sorted (by date)
    # array of dictionaries (one for each time stamp), key - lesion idx, value - volume in cm^3
    elements += create_single_patient_pdf_page(patient_name, patient.json_input_address, patient.partial_scans_address,
                                               patient.graph_image_path, volumes_dict)

    elements_per_cc, cc_info_dict, non_draw_internal_external_names_dict = create_single_lesion_pdf_page(patient, volumes_dict)
    organized_cc_elements, cc_elements_dict = organize_elements_per_cc(elements_per_cc, cc_info_dict)
    elements += organized_cc_elements

    return elements, cc_elements_dict, cc_info_dict, non_draw_internal_external_names_dict


liver = ['E_N_', 'N_M_', 'M_I_', 'M_N_', 'G_Y_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'T_N_', 'G_B_', 'C_A_', 'B_B_S_',
      'A_S_S_', 'A_S_H_', 'Z_Aa_', 'H_G_', 'A_W_', 'B_T_']

brain = ['SZ0', 'VA0', 'MY1', 'RL0', 'DD1', 'MG0', 'AN0', 'BY0', 'LS0', 'SM0', 'SF0', 'LS1', 'TM0', 'HD0', 'AA0', 'IM0',
      'MB0', 'YA0', 'BH0', 'RS0', 'AZ0', 'LA0', 'HS0', 'AF0', 'ZR0', 'NN0', 'NM1', 'DT0', 'HM0', 'ZI0']

lungs = ['M_G_', 'A_Z_A_', 'N_M_R_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'G_B_', 'C_A_', 'B_S_Ya_', 'B_B_S_', 'P_I_', 'N_Na_',
      'A_S_H_', 'Z_Aa_', 'A_Y_', 'A_A_', 'G_Ea_', 'L_I_', 'M_S_']


# name = 'C_A_'
# # name = 'A_S_H_'
# organ = Organ.LIVER
# create_pdf_file(name, organ)  # this doesnt work for the venv!