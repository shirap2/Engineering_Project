from gen_single_patient_pdf import create_single_patient_pdf_page
from gen_single_lesion.gen_single_lesion_pdf import create_single_lesion_pdf_page
from reportlab.platypus import SimpleDocTemplate, PageBreak
import os
import json

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

# # A. W.
# NAME = "A. W."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_"

# # A. S. H.
# NAME = "A. S. H."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_S_H_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_S_H_"

# # A. S. S.
# NAME = "A. S. S."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_S_S_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_S_S_"

# # B. B. S.
# NAME = "B. B. S."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/B_B_S_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/B_B_S_"

# # B. T.
# NAME = "B. T."
# JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/B_T_glong_gt.json"
# PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/B_T_"

# C. A.
NAME = "C. A."
JSON_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/C_A_glong_gt.json"
PARTIAL_ADDRESS = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/C_A_"

create_pdf_file(NAME, JSON_ADDRESS, PARTIAL_ADDRESS)

# with open(JSON_ADDRESS, 'r') as json_file:
#     loaded_data = json.load(json_file)

# for key, value in loaded_data.items():
#     print(f'{key}: {value}')
#     print()
