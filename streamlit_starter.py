import os

import streamlit as st
import sys
import os
from pdf2image import convert_from_bytes
from common_packages.LongGraphPackage import LoaderSimpleFromJson

# Add the directory containing your module to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/create_input', 'create_input'))
sys.path.append(module_path)
from create_input.create_input_files import Organ
from interactive_summary.save_patient_info import load_save_patient_data, lesion_counter_and_classifier_table


def get_num_of_scans_for_patient(patient_name):
    """
    this function checks how many scans are available for the patient
    :return: number of scans available
    """
    return 9


def filter_scans(option_name):
    st.sidebar.header("Filter Scans", divider='blue')

    # st.sidebar.header("Number of Scans")

    num_scans_available = get_num_of_scans_for_patient(option_name)
    selected_num_scans = range_slider(num_scans_available)

    # st.sidebar.header("Number of Overlapped Scans")
    # overlap_num = st.sidebar.slider("Select the number of overlapped scans to display in each row",
    #                                 min_value=0, max_value=3)
    # st.sidebar.write(f"""<div style="color: black; font-size: 14px;"> <b>Selected:
    # {selected_num_scans} scans to display with overlap of {overlap_num} scans per row </b>
    # </div>""", unsafe_allow_html=True)

    st.sidebar.header("Filter by Type of Lesion", divider='blue')
    # st.sidebar.header("Lesion Volume Filter")

    threshold_size = st.sidebar.slider("Select lesion volume threshold in current scan", min_value=0, max_value=50)

    # st.sidebar.header("Lesion Volume Changes Threshold (%)")
    vol_change_percentage = st.sidebar.slider("Select percentage threshold for lesion volume changes compared to "
                                              "previous scan", min_value=0, max_value=100, step=10)
    st.sidebar.write(f"""
        <div style="color: black; font-size: 14px;margin-bottom: 35px"> 
        <b> Displaying lesions that are currently larger than: {threshold_size}[cm³] and 
        with a volume change of at least {vol_change_percentage}% from previous scan </b>
        </div>
        """, unsafe_allow_html=True)
    return []


def main():
    # 0. pre-loading all patients: Dict {patient_name: PatientInput, volume dict}
    organ_type = Organ.LIVER
    patient_data_dict = load_save_patient_data(
        organ_type)  # too slow. i changed the function that goes over folders list to stop after
    # first one so that only 1 patient is loaded at a time and run time is faster
    patient_name = next(iter(patient_data_dict.keys()))
    # 1. choose patient
    st.sidebar.header(f"Patient Name: {patient_name}", divider='blue')  ### change text
    # option_name = st.sidebar.selectbox(
    #     'Choose a patient to display',
    #     ('choose', 'A. S. H.', 'A. S. W.', 'A. W.', 'C. A.'))

    # 2. write title
    st.write('# You selected:', patient_name)

    # add tables

    lesion_counter_and_classifier_table(patient_data_dict[patient_name][0].json_input_address)
    st.write("Here is a table generated using ReportLab:")
    images = convert_from_bytes(open("table.pdf", "rb").read())

    # Display the first page of the PDF as an image
    st.image(images[0])

    # 3. filter scans
    args = filter_scans(patient_data_dict)
    add_buttons()

    # 4. show summary using args
    volume_th = 9


def add_buttons():
    # session_state = SessionState()
    if st.sidebar.button("Disappeared Lesions"):
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
    main()
