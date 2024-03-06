from gen_single_patient_pdf import create_single_patient_pdf_page
from gen_single_lesion.gen_single_lesion_pdf import create_single_lesion_pdf_page
from reportlab.platypus import SimpleDocTemplate, PageBreak

from volume.volume_calculation import generate_longitudinal_volumes_array
import os
from create_input.create_input_files import get_patient_input, Organ

# USR = "shira_p/PycharmProjects/engineering_project/matching"
USR = "talia.dym/Desktop/Engineering_Project"



def create_pdf_file(patient_name: str, organ: Organ):
    patient = get_patient_input(patient_name, organ)
    pdf_name = f"/cs/usr/{USR}/output/{patient.organ}/" + patient_name.replace(" ", "_") + "patient_summary.pdf"
    if os.path.exists(pdf_name):
        os.remove(pdf_name)
    doc = SimpleDocTemplate(pdf_name)

    elements = []
    volumes_dict = generate_longitudinal_volumes_array(patient.partial_scans_address)  # returns sorted (by date)
    # array of dictionaries (one for each time stamp), key - lesion idx, value - volume in cm^3
    elements += create_single_patient_pdf_page(patient_name, patient.json_input_address, patient.partial_scans_address,
                                               patient.graph_image_path, volumes_dict)

    elements.append(PageBreak())

    elements += create_single_lesion_pdf_page(patient, volumes_dict)

    doc.build(elements)


# liver:
# ['E_N_', 'N_M_', 'M_I_', 'M_N_', 'G_Y_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'T_N_', 'G_B_', 'C_A_', 'B_B_S_',
#       'A_S_S_', 'A_S_H_', 'Z_Aa_', 'H_G_', 'A_W_', 'B_T_']

# all checked

# brain:
# ['SZ0', 'VA0', 'MY1', 'RL0', 'DD1', 'MG0', 'AN0', 'BY0', 'LS0', 'SM0', 'SF0', 'LS1', 'TM0', 'HD0', 'AA0', 'IM0',
#       'MB0', 'YA0', 'BH0', 'RS0', 'AZ0', 'LA0', 'HS0', 'AF0', 'ZR0', 'NN0', 'NM1', 'DT0', 'HM0', 'ZI0']


# lungs
# ['M_G_', 'A_Z_A_', 'N_M_R_', 'S_I_', 'S_N_', 'F_Y_Ga_', 'G_B_', 'C_A_', 'B_S_Ya_', 'B_B_S_', 'P_I_', 'N_Na_',
#       'A_S_H_', 'Z_Aa_', 'A_Y_', 'A_A_', 'G_Ea_', 'L_I_', 'M_S_']

name = 'C_A_'
organ = Organ.LIVER
create_pdf_file(name, organ)


