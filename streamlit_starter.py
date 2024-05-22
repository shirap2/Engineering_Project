import os
from interactive_summary.scan_window import open_itksnap_on_slice
import streamlit as st
import sys
import os
import argparse
from pdf2image import convert_from_bytes
# from common_packages.LongGraphPackage import LoaderSimpleFromJson
import base64
from create_input.create_input_files import Organ
from interactive_summary.save_patient_info import load_save_patient_data, lesion_counter_and_classifier_table
from reportlab.platypus import Paragraph, Table, Image
import pandas as pd


# Add the directory containing your module to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/create_input', 'create_input'))
sys.path.append(module_path)



# def get_num_of_scans_for_patient(patient_name):
#     """
#     this function checks how many scans are available for the patient
#     :return: number of scans available
#     """
#     return 9


# def filter_scans(option_name):
#     st.sidebar.header("Filter Scans", divider='blue')
#
#     # st.sidebar.header("Number of Scans")
#
#     num_scans_available = get_num_of_scans_for_patient(option_name)
#     selected_num_scans = range_slider(num_scans_available)
#
#     # st.sidebar.header("Number of Overlapped Scans")
#     # overlap_num = st.sidebar.slider("Select the number of overlapped scans to display in each row",
#     #                                 min_value=0, max_value=3)
#     # st.sidebar.write(f"""<div style="color: black; font-size: 14px;"> <b>Selected:
#     # {selected_num_scans} scans to display with overlap of {overlap_num} scans per row </b>
#     # </div>""", unsafe_allow_html=True)
#
#     st.sidebar.header("Filter by Type of Lesion", divider='blue')
#     # st.sidebar.header("Lesion Volume Filter")
#
#     threshold_size = st.sidebar.slider("Select lesion volume threshold in current scan", min_value=0, max_value=50)
#
#     # st.sidebar.header("Lesion Volume Changes Threshold (%)")
#     vol_change_percentage = st.sidebar.slider("Select percentage threshold for lesion volume changes compared to "
#                                               "previous scan", min_value=0, max_value=100, step=10)
#     st.sidebar.write(f"""
#         <div style="color: black; font-size: 14px;margin-bottom: 35px">
#         <b> Displaying lesions that are currently larger than: {threshold_size}[cc] and
#         with a volume change of at least {vol_change_percentage}% from previous scan </b>
#         </div>
#         """, unsafe_allow_html=True)
#     return []



def main():
    # ***************** pre-loading all patients: Dict {patient_name: PatientInput, volume dict} ***********************
    # organ_type = Organ.LIVER
    # patient_data_dict = load_save_patient_data(
    #     organ_type)  # too slow. i changed the function that goes over folders list to stop after
    # # first one so that only 1 patient is loaded at a time and run time is faster
    # patient_name = next(iter(patient_data_dict.keys()))
    # ******************************************************************************************************************


    # ************************************** configurations - choose patient *******************************************
    if 'initialization_complete' not in st.session_state:

        parser = argparse.ArgumentParser(description='Streamlit App with Command Line Arguments')
        parser.add_argument('--patient_name', type=str, required=True, help='First argument')
        parser.add_argument('--organ_type', type=str, required=True, help='Second argument')
        args = parser.parse_args()

        patient_streamlit_data = load_save_patient_data(args.organ_type, args.patient_name)
        # st.sidebar.header(f'Patient Name: {args.patient_name.replace("_", ".")}', divider='blue')
        st.session_state.initialization_complete = True
    # ******************************************************************************************************************

    # ********************************************* configurations - buttons *******************************************
    add_buttons()
    # ******************************************************************************************************************

    # ****************************************** display - write title *************************************************
    st.write('# Summary of the ', args.organ_type, ' patient ', args.patient_name.replace("_", "."))
    # # ******************************************************************************************************************

    # ****************************************** display - show PDF ****************************************************
    display_pdf(f'output/{args.organ_type}/{args.patient_name}patient_summary.pdf')
    # ******************************************************************************************************************

    # # ****************************************** display - add tables **************************************************
    # lesion_counter_and_classifier_table(patient_streamlit_data[0].json_input_address)
    # images = convert_from_bytes(open("table.pdf", "rb").read())
    # # ******************************************************************************************************************
    #
    #
    #
    # # ****************************************** display - add text & graph ********************************************
    # # Display the first page of the PDF as an image
    # for im in images:
    #     st.image(im)
    # # ******************************************************************************************************************

    # 4. show summary using args

def display_paragraph(paragraph):
    st.write(paragraph.text)

def display_table(table):
    data = table._cellvalues
    df = pd.DataFrame(data)
    st.table(df)

def display_image(image):
    image_path = image.filename
    st.image(image_path)

def display_element(element):
    if isinstance(element, Paragraph):
        display_paragraph(element)
    elif isinstance(element, Table):
        display_table(element)
    elif isinstance(element, Image):
        display_image(element)
    else:
        st.write(f"Unsupported element: {type(element)}")


def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)


def add_buttons():
    st.sidebar.header("Special Cases", divider='blue')
    # session_state = SessionState()
    if st.sidebar.button("Disappeared Lesions"):
        # organ = "liver"
        # name = "C_A_"
        # date = "14_01_2020"
        # open_itksnap_on_slice(organ, name, date)

        st.write("displaying lesions that have disappeared")
    if st.sidebar.button("New Lesions"):
        st.write("displaying new lesions")

    if st.sidebar.button("Lesions that didnt appear in segmentation"):
        st.write("displaying lesions didnt appear in segmentation")
    if st.sidebar.button("Unusual Lesions"):
        st.write("displaying lesions that are unusual")


def range_slider(range_size):
    selected_value = st.sidebar.slider('Select the number of previous scans to display', min_value=0,
                                       max_value=range_size,
                                       value=range_size // 2)
    return selected_value


def display_content(file_path):
    st.title("Main Section")
    # Load and display images or text based on user settings
    if file_path:
        # If file is uploaded, display its content
        # with open(file_path, "rb") as file:
        #     file_content = file.read()
        st.image(file_path, caption='Uploaded Image')

        # You can add more content here based on user settings

    else:
        st.write("Upload a file to display content")


if __name__ == "__main__":
    # TO CONNECT ENV RUN:
    # source ./env/bin/activate
    # streamlit run streamlit_starter.py -- --patient_name C_A_ --organ_type liver
    main()
