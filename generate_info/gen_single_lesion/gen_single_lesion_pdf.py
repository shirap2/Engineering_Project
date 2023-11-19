import os
from common_packages.LongGraphPackage import LoaderSimpleFromJson
from generate_info.generate_pdf_base import BasePDFGenerator
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from volume.lesion_volume_changes import check_single_lesion_growth, generate_volume_list_single_lesion, lesion_growth_percentage
from generate_info.gen_single_lesion.gen_single_lesion_graph import get_single_node_graph_image




def get_file_title():
    title_string = "Individual Lesion Changes"

    title_style = getSampleStyleSheet()['Title']
    title = Paragraph(title_string, title_style)

    return [title]



def get_sub_title(sub_title: str, spacer=True):
    elements = []
    title_style = getSampleStyleSheet()['Title']
    title_style.fontSize = 10
    title = Paragraph(sub_title, style=title_style)
    elements.append(title)
    if spacer:
        elements.append(Spacer(1, 5))
    return elements


def get_note(note: str, spacer=False):
    elements = []
    note = Paragraph(note, style=getSampleStyleSheet()['Normal'])
    elements.append(note)
    if spacer:
        elements.append(Spacer(1, 5))
    return elements

def get_graph_title(lesion_idx: int):
    return get_sub_title(f"The History of Lesion {lesion_idx}", False)


def get_lesion_history_text(key, vol_list):
    text_to_add = check_single_lesion_growth(vol_list,key)
    return get_note("Lesion "+ str(key)+ ": "+ text_to_add, True)


def create_single_lesion_pdf_page(patient_name : str, scan_name : str, patient_partial_path : str):

    ld = LoaderSimpleFromJson(scan_name)
    png_name = "output/" + patient_name.replace(" ", "_") + "_lesion_changes.png"
    elements = []

    # title
    elements += get_file_title()
    elements.append(Spacer(1,20))

    # graph image
    elements += get_note("In the following graphs along each edge, the % change in volume between one scan and the next is shown in green/red; On top of each node, the actual volume is shown in cubic cm, and under each node the time stamp appears.")
    elements.append(Spacer(1,20))
    vol_list = generate_volume_list_single_lesion(patient_partial_path)
    cc_idx = 0
    while True:
        graph, lesion_idx = get_single_node_graph_image("output/single_labeled_lesion_graph" , patient_partial_path, scan_name, cc_idx)

        if not graph:
            break
        elements += get_graph_title(lesion_idx)
        elements += [graph]
        elements += get_lesion_history_text(lesion_idx, vol_list)
        cc_idx += 1
    # elements += get_nodes_graph_image("output/single_labeled_lesion_graph" , patient_partial_path, scan_name)



    # total lesion volume change text
    elements+=get_sub_title("Total Lesion Growth History", False)
    elements+=lesion_growth_percentage(patient_partial_path, len(vol_list))
       
    return elements




# create_single_lesion_pdf_page("A. W.", "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json",
#                 "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_")


