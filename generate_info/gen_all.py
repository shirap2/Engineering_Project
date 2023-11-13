from gen_single_patient_pdf import create_single_patient_pdf_page
from gen_single_lesion.gen_single_lesion_pdf import create_single_lesion_pdf_page
from reportlab.platypus import SimpleDocTemplate, PageBreak
import os

def create_pdf_file(patient_name : str, scan_name : str, patient_partial_path : str):

    pdf_name = "output/" + patient_name.replace(" ", "_") + "_patient_summary.pdf"
    if os.path.exists(pdf_name):
        os.remove(pdf_name)
    doc = SimpleDocTemplate(pdf_name)

    elements = []

    elements += create_single_patient_pdf_page(patient_name, scan_name, patient_partial_path)

    elements.append(PageBreak())

    elements += create_single_lesion_pdf_page(patient_name, scan_name, patient_partial_path)

    doc.build(elements)



create_pdf_file("A. W.", "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json",
                "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_")
