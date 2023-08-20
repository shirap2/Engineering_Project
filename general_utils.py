import os
from datetime import datetime
from time import time, gmtime
import nibabel as nib
import numpy as np
from config import Path, Name
import glob
from skimage.morphology import label
from scipy import ndimage

name = Name()
path = Path()


def get_patients_list():
    """
    Assumes in gt_data/original/labeled_no_reg there are only patient folders
    """
    patient_paths = sorted(glob.glob(f'{path.GT_ORIG_LABELED_NO_REG}/*'))
    patient_names = [os.path.basename(p) for p in patient_paths]
    return patient_names


def get_patient_dates(patient_name):
    """
    Assumes in gt_data/original/labeled_no_reg/{patient_name} there are all the patient's lesion segmentation
    return: the sorted dates of patient's scans
    """
    patient_lesion_scans = sorted(glob.glob(f'{path.GT_ORIG_LABELED_NO_REG}/{patient_name}/*lesion*'), key=sort_dates)
    patient_dates = [get_date_longitudinal(os.path.basename(p)) for p in patient_lesion_scans]
    patient_dates = [d[0] for d in patient_dates]
    return patient_dates

def load_nifti_data(nifti_file_name: str, type_integer=False):
    """
    Loading cropped_data from a nifti file.

    :param nifti_file_name: The path to the desired nifti file.
    :param type_integer: default False. If True, return the image as np.int32, else as np.float32

    :return: A tuple in the following form: (cropped_data, file), where:
        • cropped_data is a ndarray containing the loaded cropped_data.
        • file is the file object that was loaded.
    """

    # loading nifti file
    nifti_file = nib.load(nifti_file_name)
    nifti_file = nib.as_closest_canonical(nifti_file)

    # extracting the cropped_data of the file
    if type_integer:
        data = nifti_file.get_fdata().astype(np.int32)
    else:
        data = nifti_file.get_fdata().astype(np.float32)

    return data, nifti_file


def get_patient(case_name):
    """
    Get the patient's identifier from the case name
    :param case_name: patient name + date. e.g.: A_Y_24_12_2020
    :return: patient identifier. e.g. A_Y_
    """
    components = case_name.split("_")
    res = ""
    for comp in components:
        if comp.isdigit():
            break
        res += comp + "_"
    return res


def get_date(case_name):
    """
    :param case_name: <patient_name>_<date>. E.g = A_A_01_12_17
    :return: date: E.g: 01_12_17
    """
    patient_name = get_patient(case_name)
    scan_date = case_name.replace(patient_name, '').replace('-nifti', '')
    scan_date = scan_date.replace('-', '_')
    return scan_date


def sort_date_only(date1):
    return datetime.strptime(date1, '%d_%m_%Y')


def sort_dates(curr_path):

    img_name = os.path.basename(curr_path)
    date1, _ = get_date_longitudinal(img_name)
    return datetime.strptime(date1, '%d_%m_%Y')


def split_simultaneous_id_to_bl_and_fu(sim_id):
    """
    Get the ID of a pair and split it to the ID of BL and FU scans
    :param sim_id: the id of the pair
    :return: baseline ID, followup ID
    """
    patient = get_patient(sim_id)
    components = sim_id.split(patient)
    if len(components) != 3:
        raise ValueError("simultaneous case id is malformed " + sim_id)
    return patient + components[1], patient + components[2]


def calculate_runtime(t):
    """
    Calculate the time elapsed since t
    :param t: a timestamp
    :return: the time elapsed since t, in string format
    """
    t2 = gmtime(time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'

def is_time_ordered(date1, date2):
    """True if date1 is before date2"""
    prev_date = datetime.strptime(date1, '%d_%m_%Y')
    curr_date = datetime.strptime(date2, '%d_%m_%Y')
    return prev_date < curr_date


def is_first_pair(pair_name):
    """"
    All the pairs are consecutive (in time), except the first pair, that is used to find the segmentation in the first scan
    This function returns True if the pair is the first pair. It checks if the date of the previous scan is posterior to the
    date of the current scan.
    """
    prev, curr = split_simultaneous_id_to_bl_and_fu(pair_name)
    # prev_date = datetime.strptime(get_date(prev), '%d_%m_%Y')
    # curr_date = datetime.strptime(get_date(curr), '%d_%m_%Y')
    # return prev_date > curr_date
    return not is_time_ordered(get_date(prev), get_date(curr))

def get_patient_from_path(file_path):
    """
    :param file_path: str, for example: */*/../A_A_/lesion_gt_25_09_2014_03_08_2015.nii.gz
    :return: patient_name: for example: A_A_
    """
    folder_tree = file_path.split('/')
    return folder_tree[-2]

def get_labeled_segmentation(img, connectivity=1, size_filtering=20):
    """
    :param img: the image to label
    :param connectivity: (default 1)
    :param size_filtering: eliminate all the labels that have less than 20 voxels
    :return: a labeled image
    """
    label_img = label(img, connectivity=connectivity)
    cc_num = label_img.max()
    cc_areas = ndimage.sum(img, label_img, range(cc_num + 1))
    area_mask = (cc_areas < size_filtering)
    label_img[area_mask[label_img]] = 0

    return label_img

def get_date_longitudinal(img_name):
    """
    :param img_name: str, for example: lesion_gt_25_09_2014_03_08_2015.nii.gz ('registered lesion') or lesion_gt_25_09_2014.nii.gz
    :return: date1, date2 (if present), for example: 25_09_2014 [03_08_2015]
    """
    no_file_type = img_name.replace(".nii.gz", "")
    components = no_file_type.split('_')
    date1 = ""
    date2 = ""
    count = 0
    for c in components:
        if c.isdigit():
            if count < 3:
                date1+= f'{c}_'
            else:
                date2+= f'{c}_'
            count+=1
    if len(date1)>0:
        date1 = date1[:-1]
    if len(date2)>0:
        date2 = date2[:-1]
    return date1, date2


def to_mb3(mb2_path):
    """replace pred_data with pred_data_mb3"""
    if isinstance(mb2_path, list):
        return [p.replace("pred_data", "pred_data_mb3") for p in mb2_path if "pred_data_mb3" not in p]
    else:
        if "pred_data_mb3" not in mb2_path:
            return mb2_path.replace("pred_data", "pred_data_mb3")
        else:
            return mb2_path


def to_gt_mapping_mb3(gt_path):
    """replace gt_data/mapped with gt_data/mapped_mb3"""
    if isinstance(gt_path, list):
        return [p.replace("mapped", "mapped_mb3") for p in gt_path if "mapped_mb3" not in p]
    else:
        if "mapped_mb3" not in gt_path:
            return gt_path.replace("mapped", "mapped_mb3")
        else:
            return gt_path


def find_case_folder(all_cases_folder_paths, pat_name, date1, date2, return_case_name=False):
    """
    :params all_cases_folder_paths: example of a path format: */*/.../A_A_03_02_2013-niftiA_A_05-12_2014/. This is a list of all those
    folders
    :params pat_name: the required patient to find: ex: A_A_
    :params date1: the bl date ex: 03_02_2013
    :params date2: the fu date ex: 05_12_2015
    :return: the folder in all_cases_folder_paths that matches pat_name, date1 and date2. ex: A_A_03_02_2013-niftiA_A_05-12_2014
    """
    folders_with_pat_name = [path for path in all_cases_folder_paths if pat_name in path]
    for folder in folders_with_pat_name:
        case_name = os.path.basename(folder)
        bl, fu = split_simultaneous_id_to_bl_and_fu(case_name)
        if get_date(bl) == date1 and get_date(fu) == date2:
            if return_case_name:
                return case_name
            else:
                return folder

    raise ValueError("No folder matches the input!")


def write_to_excel(sheet_name, df, writer):
    columns_order = list(df.columns)
    workbook = writer.book
    cell_format = workbook.add_format({'num_format': '#,##0.00'})
    cell_format.set_font_size(16)

    df.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=1, header=False, index=False)
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'font_size': 16,
        'valign': 'top',
        'border': 1})

    max_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#E6FFCC'})
    min_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#FFB3B3'})
    last_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#C0C0C0',
        'border': 1,
        'num_format': '#,##0.00'})

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 1)

    for col_num, value in enumerate(columns_order):
        worksheet.write(0, col_num + 1, value, header_format)
    for row_num, value in enumerate(df.axes[0].astype(str)):
        worksheet.write(row_num + 1, 0, value, header_format)

    # Fix first column
    column_len = df.axes[0].astype(str).str.len().max() + df.axes[0].astype(str).str.len().max() * 0.5
    worksheet.set_column(0, 0, column_len, cell_format)

    # Fix all  the rest of the columns
    for i, col in enumerate(columns_order):
        # find length of column i
        column_len = df[col].astype(str).str.len().max()
        # Setting the length if the column header is larger
        # than the max column value length
        column_len = max(column_len, len(col))
        column_len += column_len * 0.5
        # set the column length
        worksheet.set_column(i + 1, i + 1, column_len, cell_format)

