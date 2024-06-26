from reportlab.platypus import Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from volume.volume_calculation import get_diff_in_total, generate_longitudinal_volumes_array
from patient_summary.classify_changes_in_individual_lesions import gen_dict_classified_nodes_for_layers, classify_changes_in_individual_lesions, count_d_in_d_out, changes_in_individual_lesions
from common_packages.LongGraphClassification import LongitClassification
from common_packages.LongGraphPackage import LoaderSimpleFromJson, DrawerLabels
import numpy as np

def get_file_title(patient_name: str):
    title_string = patient_name + " Patient Summary"
    sub_title = "Patient Study Based on Lesion Graph"

    title_style = getSampleStyleSheet()['Title']
    title = Paragraph(title_string, title_style)

    subtitle_style = getSampleStyleSheet()['Title']
    sub_title = Paragraph(sub_title, subtitle_style)

    return [title, sub_title]



def get_sub_title(sub_title: str):
    style = getSampleStyleSheet()['Title']
    style.fontSize = 10
    style.spaceAfter = 5
    title = Paragraph(sub_title, style=style)
    return [title]



def get_lesion_counter_and_classifier_table(ld):
    classified_nodes_dict = gen_dict_classified_nodes_for_layers(classify_changes_in_individual_lesions(count_d_in_d_out(ld),ld))
    unique_classes = [getattr(changes_in_individual_lesions, attr) for attr in dir(changes_in_individual_lesions) if not callable(getattr(changes_in_individual_lesions, attr)) and not attr.startswith("__")]
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

    table.setStyle(table_style)
    elements.append(table)
    return elements



def edit_volume_data_to_str(data: list):
    total_vol_cm3, vol_percentage_diff, vol_cm3_diff = data
    diff_is_positive = (vol_percentage_diff > 0)

    total_vol_cm3 = str(round(total_vol_cm3, 2))

    # added this check because of the error: cannot convert float infinity to integer
    if np.isinf(vol_percentage_diff):
        if vol_percentage_diff > 0:
            vol_percentage_diff = 'inf'
        else:
            vol_percentage_diff = '-inf'
    else:
        vol_percentage_diff = str(round(vol_percentage_diff))

    if vol_percentage_diff != "0":
        vol_percentage_diff += "%"
    vol_cm3_diff = str(round(vol_cm3_diff, 2))

    if diff_is_positive:
        vol_percentage_diff = "+" + vol_percentage_diff
        vol_cm3_diff = "+" + vol_cm3_diff

    return [total_vol_cm3, vol_percentage_diff, vol_cm3_diff]

def replace_zeros_with_hyphen(value):
    return '-' if value == "0" else value

def replace_zeros_in_table(table_data):
    for col_idx, cell_value in enumerate(table_data[1]):
        table_data[1][col_idx] = replace_zeros_with_hyphen(cell_value)


def get_volume_changes_per_time_table(patient_partial_path : str,volumes_dict):

    diff_in_total = get_diff_in_total(volumes_dict) # [total_vol_cm3, vol_percentage_diff, vol_cm3_diff]

    table_data = [["Time Stamp", "Total Volume [cc]", "Volume Difference Percentage", "Volume Difference [cc]"]]

    for idx, data in enumerate(diff_in_total):
        data = edit_volume_data_to_str(data)
        table_data.append([idx] + data)


    replace_zeros_in_table(table_data)

    # Create the table and apply styles
    table = Table(table_data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)]))

    return [table]

def get_nodes_graph_image(image_path : str):

    # lg1 = LongitClassification(ld)
    # dr_2 = DrawerLabels(lg1)
    # dr_2.show_graph(image_path)

    graph = Image(image_path, height=300, width=400)

    credit_style = ParagraphStyle('credit_style',fontsize=12)
    credit = Paragraph("Di Veroli B., Joskowicz L. A Graph Theoretic Approach for Analysis of Lesion Changes and Lesions "
                       "Detection Review in Longitudinal Ontological Imaging, CASMIP Hebrew University, 2023",credit_style)
    
    return [graph, credit]


def create_single_patient_pdf_page(patient_name : str, scan_name : str, patient_partial_path : str, png_name : str,volumes_dict):

    ld = LoaderSimpleFromJson(scan_name)
    elements = []

    # title
    elements += get_file_title(patient_name.replace("_", "."))
    elements.append(Spacer(1,20))

    # graph image
    # elements += get_nodes_graph_image(png_name)
    # elements.append(Spacer(1,20))

    # table 1
    elements += get_sub_title("Lesion Count According to Classification")
    elements += get_lesion_counter_and_classifier_table(ld)
    elements.append(Spacer(1,20))

    # table 2
    elements += get_sub_title("Tracking the Changes in the Total Volume of the Tumors From One Scan to the Previous One")
    elements += get_volume_changes_per_time_table(patient_partial_path,volumes_dict)
    elements.append(Spacer(1,20))
    
    return elements


