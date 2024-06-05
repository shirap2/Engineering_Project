import os
import re

from reportlab.lib.pagesizes import letter

from common_packages.LongGraphPackage import LoaderSimpleFromJson
from create_input.create_input_files import Organ, list_folders, get_patient_input
from volume.volume_calculation import generate_longitudinal_volumes_array
from reportlab.lib import colors


from docutils.writers.odf_odt import TableStyle
from reportlab.platypus import Table, SimpleDocTemplate

from patient_summary.classify_changes_in_individual_lesions import gen_dict_classified_nodes_for_layers, \
    changes_in_individual_lesions, count_d_in_d_out, classify_changes_in_individual_lesions
from datetime import datetime

def get_sorted_patient_scans_date(patient):
    # date_pattern = r'_\d{2}_\d{2}_\d{4}'
    # folder_path = patient.partial_scans_address
    # # Counter for files with a different date format
    # unique_dates = set()
    # # Iterate over each file in the folder
    # for filename in os.listdir(folder_path):
    #     file_path = os.path.join(folder_path, filename)
    #     if os.path.isfile(file_path):  # Check if it's a file (not a subdirectory)
    #         # Extract date from filename using regular expression
    #         match = re.search(date_pattern, filename)
    #         if match:
    #             unique_dates.add(match.group()[1:])
    # # return unique_dates
    # unique_dates = list(unique_dates)
    # for d in unique_dates:
    #     print(d)
    # return sorted(unique_dates, key=lambda x: datetime.strptime(x, '%d_%m_%y'))

    date_pattern = r'(\d{2}_\d{2}_\d{4})'
    formatted_dates = set()

    for filename in os.listdir(patient.partial_scans_address):
        match = re.search(date_pattern, filename)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, '%d_%m_%Y')

            # Format the datetime object into "dd.mm.yyyy" and append to the list
            formatted_date = date_obj.strftime('%d_%m_%Y')
            formatted_dates.add(formatted_date)

    formatted_dates = sorted(formatted_dates, key=lambda x: datetime.strptime(x, '%d_%m_%Y'))
    return formatted_dates




def get_max_vol(volumes_dict):
    return max(volumes_dict[-1].values())


def save_info(key, organ):
    """
    this function gets all the information needed for a patient and returns it in an array
    :param key: name of patient
    :param organ: organ type (lungs, liver or brain)
    :return: array containing patient info
    """
    patient = get_patient_input(key, organ)

    volumes_dict = generate_longitudinal_volumes_array(patient.partial_scans_address)  # returns sorted (by date)
    number_of_scans = len(get_sorted_patient_scans_date(patient))
    max_vol = round(get_max_vol(volumes_dict)) - 1

    return [patient, volumes_dict, number_of_scans, max_vol]


# def load_save_patient_data(organ_type):
#     organ = ""
#     if organ_type == Organ.LIVER:
#         organ = "liver"
#     if organ_type == Organ.LUNGS:
#         organ = "lungs"
#     if organ_type == Organ.BRAIN:
#         organ = "brain"
#
#     folder_path = f"/cs/casmip/archive/bennydv/{organ}_pipeline/gt_data/size_filtered/labeled_no_reg/"
#
#     folders = list_folders(folder_path)
#     patient_data_dict = {key: save_info(key, organ) for key in folders}
#
#     return patient_data_dict

def load_save_patient_data(organ_type, patient_name):
    # organ = ""
    # if organ_type == Organ.LIVER:
    #     organ = "liver"
    # if organ_type == Organ.LUNGS:
    #     organ = "lungs"
    # if organ_type == Organ.BRAIN:
    #     organ = "brain"

    patient = get_patient_input(patient_name, organ_type)

    volumes_dict = generate_longitudinal_volumes_array(patient.partial_scans_address)  # returns sorted (by date)
    number_of_scans = len(get_sorted_patient_scans_date(patient))
    max_vol = round(get_max_vol(volumes_dict)) - 1

    return [patient, volumes_dict, number_of_scans, max_vol]

def lesion_counter_and_classifier_table(json_input_address):
    ld = LoaderSimpleFromJson(json_input_address)
    pdf_doc = SimpleDocTemplate("table.pdf", pagesize=letter)
    classified_nodes_dict = (
        gen_dict_classified_nodes_for_layers(classify_changes_in_individual_lesions(count_d_in_d_out(ld),ld)))
    unique_classes = [getattr(changes_in_individual_lesions, attr) for attr in dir(changes_in_individual_lesions)
                      if not callable(getattr(changes_in_individual_lesions, attr)) and not attr.startswith("__")]
    unique_layers = sorted(classified_nodes_dict.keys())

    # Create a 2D list for the data (counts)
    data = [[f"Time Layer"] + unique_classes]
    for layer in unique_layers:
        row = [layer] + [classified_nodes_dict.get(layer, {}).get(class_, 0) for class_ in unique_classes]
        data.append(row)

    elements = []
    table = Table(data)

    # Define table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), (0.8, 0.8, 0.8)),  # Grey background for headers
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold font for headers
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Add padding to headers
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Add gridlines with 1pt width
    ])

    # table.setStyle(table_style)
    pdf_doc.build([table])


# load_save_patient_data(Organ.LIVER)
