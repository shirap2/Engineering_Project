import os
from common_packages.LongGraphPackage import LoaderSimpleFromJson
from generate_info.generate_pdf_base import BasePDFGenerator
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from volume_calculation.lession_volume_changes import check_single_lession_growth, generate_volume_list_single_lesion
from gen_single_lesion_graph import get_nodes_graph_image




def get_file_title(patient_name: str):
    title_string = patient_name + " Individual Lession Changes"
    sub_title = "Patient Study Based on Lesion Graph"

    title_style = getSampleStyleSheet()['Title']
    title = Paragraph(title_string, title_style)

    subtitle_style = getSampleStyleSheet()['Title']
    sub_title = Paragraph(sub_title, subtitle_style)

    return [title, sub_title]



def get_sub_title(sub_title: str):
    elements = []
    title = Paragraph(sub_title, style=getSampleStyleSheet()['Title'])
    elements.append(title)
    elements.append(Spacer(1, 5))
    return elements


def create_pdf_file(patient_name : str, scan_name : str, patient_partial_path : str):
    ld = LoaderSimpleFromJson(scan_name)

    pdf_name = patient_name.replace(" ", "_") + "_lession_changes.pdf"
    png_name = patient_name.replace(" ", "_") + "_lession_changes.png"
    if os.path.exists(pdf_name):
        os.remove(pdf_name)


    doc = SimpleDocTemplate(pdf_name)
    elements = []

    # title
    elements += get_file_title(patient_name)
    elements.append(Spacer(1,20))

    # graph image
    elements += get_nodes_graph_image("single_labeled_lesion_graph.png" , patient_partial_path, ld)
    elements.append(Spacer(1,20))


    # lession volume change text
    elements+=get_sub_title("Lession Growth Changes")
    vol_list = generate_volume_list_single_lesion(patient_partial_path)
    for key in vol_list.keys():
         text_to_add =check_single_lession_growth(vol_list,key)
         paragraph = Paragraph("Lession "+str(key)+": "+ text_to_add)
         elements.append(paragraph)
       


    doc.build(elements)



create_pdf_file("A. W.", "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json",
                "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_")


