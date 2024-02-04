from gen_single_patient_pdf import create_single_patient_pdf_page
from gen_single_lesion.gen_single_lesion_pdf import create_single_lesion_pdf_page
from reportlab.platypus import SimpleDocTemplate, PageBreak
import os
import json

USR = "shira_p/PycharmProjects/engineering_project/matching"


class PatientInput:
    def __init__(self, name, partial_scans_address, json_input_address, pickle_input_address, praph_image_path):
        self.name = name
        self.partial_scans_address = partial_scans_address
        self.json_input_address = json_input_address
        self.pickle_input_address = pickle_input_address
        self.praph_image_path = praph_image_path


def get_patient_input(patient_name: str):
    name_for_path = patient_name.replace(" ", "_").replace(".", "")
    partial_scans_address = f"/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/{name_for_path}_"
    json_input_address = f"/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/{name_for_path}_glong_gt.json"
    pickle_input_address = f"/cs/usr/{USR}/input/pkl_files/{name_for_path}_graph_class_data.pkl"
    praph_image_path = f"/cs/usr/{USR}/input/graph_images/{name_for_path}_graph_image.png"

    return PatientInput(patient_name, partial_scans_address, json_input_address, pickle_input_address, praph_image_path)


def create_pdf_file(patient_name: str):
    patient = get_patient_input(patient_name)

    pdf_name = f"/cs/usr/{USR}/output/" + patient_name.replace(" ", "_") + "_patient_summary.pdf"
    if os.path.exists(pdf_name):
        os.remove(pdf_name)
    doc = SimpleDocTemplate(pdf_name)

    elements = []

    elements += create_single_patient_pdf_page(patient_name, patient.json_input_address, patient.partial_scans_address,
                                               patient.praph_image_path)

    elements.append(PageBreak())

    elements += create_single_lesion_pdf_page(patient_name, patient.json_input_address, patient.pickle_input_address,
                                              patient.partial_scans_address)

    doc.build(elements)


# NAME = "A. W."
# create_pdf_file(NAME)
# NAME = "A. S. H."
# create_pdf_file(NAME)
# NAME = "A. S. S."
# create_pdf_file(NAME)
# NAME = "B. B. S."
# create_pdf_file(NAME)
# NAME = "B. T."
# create_pdf_file(NAME)
NAME = "C. A."
create_pdf_file(NAME)
