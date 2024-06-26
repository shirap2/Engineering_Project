from create_input.create_input_files import get_patient_input
from generate_info.gen_single_lesion.gen_single_lesion_pdf import get_title, get_sub_title
from interactive_summary.convert_pdf_to_streamlit import display_element
import streamlit as st
import sys
import os
import argparse
from generate_info.gen_all import get_full_display_elements
import base64
from interactive_summary.scan_window import get_segment_info
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from interactive_summary.save_patient_info import get_sorted_patient_scans_date
from interactive_summary.scan_window import get_segment_mapping_table, open_itksnap_on_slice

# Add the directory containing your module to the Python path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/create_input', 'create_input'))
sys.path.append(module_path)


class InteractiveState():
    initialization = 0
    default_full_information_display = 1
    new_lesions = 2
    lone_lesions = 3
    non_consecutive_matched_lesions = 4
    open_itk_snap = 5
    lesion_segmentation_map = 6
    download_version = 2




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

# def range_slider(range_size):
#     selected_value = st.sidebar.slider('Select the number of previous scans to display', min_value=0,
#                                        max_value=range_size,
#                                        value=range_size // 2)
#     return selected_value


def main():
    # ***************** pre-loading all patients: Dict {patient_name: PatientInput, volume dict} ***********************
    # organ_type = Organ.LIVER
    # patient_data_dict = load_save_patient_data(
    #     organ_type)  # too slow. i changed the function that goes over folders list to stop after
    # # first one so that only 1 patient is loaded at a time and run time is faster
    # patient_name = next(iter(patient_data_dict.keys()))
    # ******************************************************************************************************************


    # ************************************** configurations *******************************************
    if 'state' not in st.session_state:
        st.session_state.state = InteractiveState.initialization

    if st.session_state.state == InteractiveState.initialization:
        parser = argparse.ArgumentParser(description='Streamlit App with Command Line Arguments')
        parser.add_argument('--patient_name', type=str, required=True, help='First argument')
        parser.add_argument('--organ_type', type=str, required=True, help='Second argument')
        st.session_state.args = parser.parse_args()

        # save dates
        st.session_state.args.dates = get_sorted_patient_scans_date(get_patient_input(st.session_state.args.patient_name, st.session_state.args.organ_type))[::-1]
        st.session_state.args.dates_for_txt = [d.replace("_", ".") for d in st.session_state.args.dates]
        # patient_streamlit_data = load_save_patient_data(args.organ_type, args.patient_name)

        # save patient elements, and
        # cc info, foe each cc_idx: ( volume_sum_of_last_nodes, internal_external_names_dict, is_cc_non_consecutive, pattern_of_cc )
        # cc_info_dict[cc_idx] = [volume_sum_of_last_nodes, internal_external_names_dict, is_cc_non_consecutive,
        # pattern_of_cc]

        st.session_state.elements, st.session_state.cc_elements_dict, st.session_state.cc_info_dict,\
        non_draw_internal_external_names_dict = get_full_display_elements(
            st.session_state.args.patient_name, st.session_state.args.organ_type)

        st.session_state.internal_external_names_dict = {}
        # add names of the drawn cc's
        for cc_idx in st.session_state.cc_info_dict:
            _, internal_external_names_dict_per_cc, _, _ = st.session_state.cc_info_dict[cc_idx]
            for external_name, internal_name in internal_external_names_dict_per_cc.items():
                st.session_state.internal_external_names_dict[external_name] = internal_name
        # add names of the non drawn cc's
        for cc_idx in non_draw_internal_external_names_dict:
            for external_name, internal_name in non_draw_internal_external_names_dict[cc_idx].items():
                st.session_state.internal_external_names_dict[external_name] = internal_name


        # save lesion segmentation map
        st.session_state.largest_slices_info = {}
        for date in st.session_state.args.dates:
            st.session_state.largest_slices_info[date] = get_segment_info(st.session_state.args.organ_type, st.session_state.args.patient_name, date)

        st.session_state.state = InteractiveState.default_full_information_display
    # ******************************************************************************************************************

    # ********************************************* configurations - buttons *******************************************
    add_sidebar()
    # ******************************************************************************************************************

    # ************************************ display default - full inforamtion ******************************************
    if st.session_state.state == InteractiveState.default_full_information_display:
        for element in st.session_state.elements:
            display_element(element)
    # ******************************************************************************************************************

    # ****************************************** display - show PDF ****************************************************
    elif st.session_state.state == InteractiveState.download_version:
        display_pdf(f'output/{st.session_state.args.organ_type}/{st.session_state.args.patient_name}patient_summary.pdf')
    # ******************************************************************************************************************

    # ************************************** lesion segmentation map ***************************************************
    elif st.session_state.state == InteractiveState.lesion_segmentation_map:
        display_element(get_title('Lesion - Segmentation Map')[0])
        display_element(get_sub_title("Mapping Between Summary Notations and Segmentation Notations", True)[0])
        st.write('Lesion Name: The used lesion name.')
        st.write('Slice Num.: The slice index where the lesions diameter is the largest.')
        st.write('Color: The color of the lesion in the segmentation.')
        st.write('Area: The area of the lesion in the above slice.')
        st.write('Diameter: The diameter of the lesion in the above slice.')
        st.write('')
        st.write('')
        st.write('')
        st.write('')

        time_stamp = len(st.session_state.args.dates) - 1
        for date in st.session_state.args.dates:
            st.write(f'Lesions - Segmentation Map {date.replace("_", ".")}\n')
            get_segment_mapping_table(date, time_stamp)
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            time_stamp -= 1
    # ******************************************************************************************************************

    # ************************************** display only non-consecutive **********************************************
    elif st.session_state.state == InteractiveState.non_consecutive_matched_lesions:
        if 'non_consecutive_elements' not in st.session_state:
            st.session_state.non_consecutive_elements = []
            st.session_state.non_consecutive_elements += get_title('Non Consecutive Lesions')
            for cc_idx in st.session_state.cc_elements_dict:
                _, _, is_cc_non_consecutive, _ = st.session_state.cc_info_dict[cc_idx]
                if is_cc_non_consecutive:
                    st.session_state.non_consecutive_elements += st.session_state.cc_elements_dict[cc_idx]

        for element in st.session_state.non_consecutive_elements:
            display_element(element)
    # ******************************************************************************************************************

    # ************************************** display lone pattern lesions **********************************************
    elif st.session_state.state == InteractiveState.lone_lesions:
        if 'lone_lesions_elements' not in st.session_state:
            st.session_state.lone_lesions_elements = []
            st.session_state.lone_lesions_elements += get_title('Lone Lesions')
            # for cc_idx in st.session_state.cc_elements_dict:
                # _, _, _, pattern = st.session_state.cc_info_dict[cc_idx]
                # print(pattern)
                # st.session_state.non_consecutive_elements += Paragraph('1')

        for element in st.session_state.lone_lesions_elements:
            display_element(element)

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
        st.markdown(pdf_display, unsafe_allow_html=True)


def add_sidebar():

    if st.sidebar.button("Full Information Display", use_container_width=True):
        st.session_state.state = InteractiveState.default_full_information_display

    st.sidebar.header("Special Cases", divider='blue')  # **************************************************************

    # if st.sidebar.button("New Lesions", use_container_width=True):
    #     st.session_state.state = InteractiveState.new_lesions

    if st.sidebar.button("Lone Lesions", use_container_width=True):
        st.session_state.state = InteractiveState.lone_lesions

    if st.sidebar.button("Non-Consecutive Matched Lesions", use_container_width=True):
        st.session_state.state = InteractiveState.non_consecutive_matched_lesions

    st.sidebar.header("Open Scans", divider='blue')  # *****************************************************************

    selected_date = st.sidebar.selectbox("Select Scan Date", ["Select Date"] + st.session_state.args.dates_for_txt)

    if st.sidebar.button(f"Open Scan", use_container_width=True):
        # this button doesnt change the state - it leaves whatever is displayed open
        if selected_date != "Select Date":
            open_itksnap_on_slice(st.session_state.args.organ_type, st.session_state.args.patient_name, selected_date.replace(".", "_"))

    if st.sidebar.button("Lesion - Segmentation Map", use_container_width=True):
        st.session_state.state = InteractiveState.lesion_segmentation_map

    st.sidebar.header("Download", divider='blue')  # *******************************************************************

    if st.sidebar.button("Download Full Information Display", use_container_width=True):
        st.session_state.state = InteractiveState.download_version



if __name__ == "__main__":
    # TO CONNECT ENV RUN:
    # source ./env/bin/activate
    # streamlit run app.py -- --patient_name C_A_ --organ_type liver
    main()