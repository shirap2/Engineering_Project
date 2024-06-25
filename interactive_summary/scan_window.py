import subprocess
import os
import nibabel as nib
import numpy as np
import streamlit as st
from common_packages.BaseClasses import Colors
import pandas as pd
from reportlab.lib.colors import Color
from scipy.spatial import distance_matrix
from create_input.create_input_files import DATASET_ON_CASMIP, dataset_path


def scan_path(organ, name, date):
    if DATASET_ON_CASMIP:
        return f'/cs/casmip/archive/bennydv/{organ}_pipeline/gt_data/size_filtered/labeled_no_reg/{name}/scan_{date}.nii.gz'
    return f"{dataset_path}/{organ}_pipeline/gt_data_nifti/{name}/scan_{date}.nii.gz"


def gt_segmentation_path(organ, name, date):
    if DATASET_ON_CASMIP:
        return f'/cs/casmip/archive/bennydv/{organ}_pipeline/gt_data/size_filtered/labeled_no_reg/{name}/lesions_gt_{date}.nii.gz'
    return f"{dataset_path}/{organ}_pipeline/gt_data_nifti/{name}/lesions_gt_{date}.nii.gz"

def open_itksnap_on_slice(organ, name, date):
    """
    Opens a specified slice of a NIfTI scan and segmentation file in ITK-SNAP.

    Parameters:
    slice_idx (int): The index of the slice to view.
    scan_file (str): The path to the NIfTI scan file (.nii.gz).
    seg_file (str): The path to the NIfTI segmentation file (.nii.gz).
    """
    scan_file = scan_path(organ, name, date)
    seg_file = gt_segmentation_path(organ, name, date)
    # Ensure ITK-SNAP is installed
    if subprocess.run(["which", "itksnap"], capture_output=True).returncode != 0:
        raise EnvironmentError("ITK-SNAP is not installed or not found in the system PATH.")

    # Check if the files exist
    if not os.path.exists(scan_file):
        raise FileNotFoundError(f"Scan file not found: {scan_file}")
    if not os.path.exists(seg_file):
        raise FileNotFoundError(f"Segmentation file not found: {seg_file}")

    # Construct the ITK-SNAP command
    command = [
        "itksnap",
        "-g", scan_file,
        "-s", seg_file
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to open ITK-SNAP: {e}")



# ########################################### Get SLice Number etc. ###############################################
def calculate_diameter(binary_slice, label):
    """Calculate the diameter of the segment in a binary slice."""
    # coords = np.argwhere(binary_slice == label)
    # if coords.size == 0:
    #     return 0
    #
    # y_min, x_min = coords.min(axis=0)
    # y_max, x_max = coords.max(axis=0)
    #
    # diameter = np.sqrt((y_max - y_min) ** 2 + (x_max - x_min) ** 2)
    # return diameter

    # Extract coordinates of non-zero pixels
    points = np.argwhere(binary_slice == label)

    # Compute the pairwise distance matrix
    dist_matrix = distance_matrix(points, points)

    # Find the maximum distance
    max_dist = np.max(dist_matrix)

    return max_dist


def get_segment_info(organ, name, date):

    file_path = gt_segmentation_path(organ, name, date)
    nii = nib.load(file_path)
    segmentation_data = nii.get_fdata()

    # unique segments (labels) in the segmentation mask
    labels = np.unique(segmentation_data)

    largest_slices = {}

    # iterate through each segment label
    for label in labels:
        if label == 0:
            continue  # (assuming 0 is the background label)

        max_area = 0
        slice_with_max_area = 0

        # Iterate through each slice in the 3D volume
        for slice_index in range(segmentation_data.shape[2]):
            # Get the current slice
            current_slice = segmentation_data[:, :, slice_index]

            # Calculate the area of the current segment in this slice
            area = np.sum(current_slice == label)

            # Update if the current slice has a larger area
            if area > max_area:
                max_area = area
                slice_with_max_area = slice_index

        diameter_of_max_area = calculate_diameter(segmentation_data[:, :, slice_with_max_area], label)

        # Store the result in the dictionary
        color = Colors.itk_colors(label)
        largest_slices[label] = [int(slice_with_max_area + 1), color, round(max_area / 100, 2), round(diameter_of_max_area / 10, 2)]

    return largest_slices


def get_segment_mapping_table(date, time_stamp):
    def rgb_to_hex(r, g, b):
        return Color(r, g, b).hexval().replace("0x", "#")

    largest_slices_info = st.session_state.largest_slices_info[date]

    lesions, slices, colors, areas, diameters = [], [], [], [], []
    for idx, val in largest_slices_info.items():

        if f'{int(idx)}_{time_stamp}' in st.session_state.internal_external_names_dict:  # otherwise - Benny ignored those lesions

            lesions.append(st.session_state.internal_external_names_dict[f'{int(idx)}_{time_stamp}'])
            slices.append(val[0])
            colors.append(val[1])
            areas.append(round(val[2], 2))
            diameters.append(round(val[3], 2))


    # df = pd.DataFrame({
    #     "Name": lesions,
    #     "Lesion Area Slice Num.": slices,
    #     "Color in Segmentation": colors,
    #     "Area in Slice [cm²]": areas,
    #     "Diameter in Slice [cm]": diameters
    # })

    df = pd.DataFrame({
        "Lesion Name": lesions,
        "Slice Num.": slices,
        "Color": colors,
        "Area [cm²]": areas,
        "Diameter [cm]": diameters
    })

    def color_square(val):
        hex_color = rgb_to_hex(*val)
        return f'<div style="width: 20px; height: 20px; background-color: {hex_color}; margin-right: 5px;"></div>'

    df['Color'] = df['Color'].apply(color_square)

    #  convert the HTML content to be displayed in Streamlit
    def render_html(df):
        return df.to_html(escape=False, index=False)

    # display the DataFrame as an HTML table in Streamlit
    st.markdown(render_html(df), unsafe_allow_html=True)





organ = "liver"
name = "C_A_"
date = "14_01_2020"
# open_itksnap_on_slice(organ, name, date)