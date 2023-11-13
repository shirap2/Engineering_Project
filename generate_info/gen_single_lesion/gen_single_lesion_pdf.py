import os
from common_packages.LongGraphPackage import LoaderSimpleFromJson
from generate_info.generate_pdf_base import BasePDFGenerator
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from volume.lession_volume_changes import check_single_lession_growth, generate_volume_list_single_lesion
from generate_info.gen_single_lesion.gen_single_lesion_graph import get_nodes_graph_image




def get_file_title():
    title_string = "Individual Lession Changes"

    title_style = getSampleStyleSheet()['Title']
    title = Paragraph(title_string, title_style)

    return [title]



def get_sub_title(sub_title: str):
    elements = []
    title = Paragraph(sub_title, style=getSampleStyleSheet()['Title'])
    elements.append(title)
    elements.append(Spacer(1, 5))
    return elements

def get_note(note: str):
    elements = []
    note = Paragraph(note, style=getSampleStyleSheet()['Normal'])
    elements.append(note)
    return elements


def create_single_lesion_pdf_page(patient_name : str, scan_name : str, patient_partial_path : str):

    ld = LoaderSimpleFromJson(scan_name)
    png_name = "output/" + patient_name.replace(" ", "_") + "_lession_changes.png"
    elements = []

    # title
    elements += get_file_title()
    elements.append(Spacer(1,20))

    # graph image
    elements += get_nodes_graph_image("output/single_labeled_lesion_graph.png" , patient_partial_path, ld)
    elements += get_note("On the edges the change in volume between one scan to the next, is shown by percentage.")
    elements += get_note("Under the nodes the actual volume is shown in [cm³].")
    elements.append(Spacer(1,20))


    # lession volume change text
    elements+=get_sub_title("Lession Growth Changes")
    vol_list = generate_volume_list_single_lesion(patient_partial_path)
    for key in vol_list.keys():
         text_to_add =check_single_lession_growth(vol_list,key)
         paragraph = Paragraph("Lession "+str(key)+": "+ text_to_add)
         elements.append(paragraph)
       
    return elements



# create_single_lesion_pdf_page("A. W.", "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json",
#                 "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_")


