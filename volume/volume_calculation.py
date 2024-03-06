import os
import re

import numpy as np
from datetime import datetime

from common_packages.LongGraphPackage import LoaderSimpleFromJson
from general_utils import load_nifti_data, get_labeled_segmentation

class edgeVolumeClassification:
    LINEAR = 'linear'
    MERGED = 'merged'
    SPLITTING = 'splitting'
    COMPLEX = 'complex'

def get_voxels_count_per_label(scan):
    labels, counts = np.unique(scan, return_counts=True)
    combined_labels_and_counts = np.column_stack((labels, counts))
    counts_per_label = dict(
        np.core.records.fromarrays(combined_labels_and_counts.T, names='label, count', formats='i4, i4'))
    del counts_per_label[0]
    return counts_per_label


def from_voxels_count_to_volume_cm_3(counts_per_label: dict, nifti):
    one_voxel_volume = nifti.header.get_zooms()[0] * nifti.header.get_zooms()[1] * nifti.header.get_zooms()[2]
    volume_per_label = {key: val * one_voxel_volume for key, val in counts_per_label.items()}
    volume_per_label = {key: val / 1000 for key, val in volume_per_label.items()} # mm_3 to cm_3
    return volume_per_label


def from_mask_to_volume(path):
    scan, nifti = load_nifti_data(path, type_integer=True)

    labeled_scan = get_labeled_segmentation(scan)
    counts_per_label = get_voxels_count_per_label(labeled_scan)
    volume_per_label = from_voxels_count_to_volume_cm_3(counts_per_label, nifti)
    return volume_per_label


def generate_longitudinal_volumes_array(patient_path: str):
    files = os.listdir(patient_path)
    scans = [f for f in files if f.startswith('lesions_gt_')]
    date_pattern = r'(\d{2}_\d{2}_\d{4})'
    formatted_dates = [(datetime.strptime(re.search(date_pattern, filename).group(), '%d_%m_%Y'), filename) for filename
                       in scans]
    sorted_filenames = [filename for date, filename in sorted(formatted_dates, key=lambda x: x[0])]
    longitudinal_volumes_array = []
    for file_name in sorted_filenames:
        longitudinal_volumes_array.append(from_mask_to_volume(
            f"{patient_path}/{file_name}"))
    return longitudinal_volumes_array


def get_volume_percentage_diff(longitudinal_volumes_array, root_idx, tail_idx, root_time, tail_time):
    if root_idx in longitudinal_volumes_array[root_time]:
        root_volume = longitudinal_volumes_array[root_time][root_idx]
    else:
        root_volume=0
    ## check if tail index is in dictionary - check if lesion has disapeared
    if tail_idx not in longitudinal_volumes_array[tail_time]:
        tail_volume =0
    else:
        tail_volume = longitudinal_volumes_array[tail_time][tail_idx]

    if root_volume ==0:
        percentage_diff = "+inf"
    else:
        percentage_diff = ((tail_volume / root_volume) - 1) * 100

    actual_diff = tail_volume - root_volume

    return (percentage_diff, actual_diff)


def get_dict_of_volume_change_per_edge(ld: LoaderSimpleFromJson, longitudinal_volumes_array):
    volume_change_per_edge_dict = {}
    for edge in ld.get_edges():
        root = edge[0]
        tail = edge[1]
        root_idx = int(root.split("_")[0])
        root_time = int(root.split("_")[1])
        tail_idx = int(tail.split("_")[0])
        tail_time = int(tail.split("_")[1])
        volume_change_per_edge_dict[edge] = get_volume_percentage_diff(longitudinal_volumes_array, root_idx, tail_idx,
                                                                       root_time, tail_time)
    return volume_change_per_edge_dict

def get_edges_to_node_dict(ld: LoaderSimpleFromJson):
    edges_to_node_dict = {}

    for edge in ld.get_edges():

        dest_node = edge[1]

        if dest_node not in edges_to_node_dict:
            edges_to_node_dict[dest_node] = [edge]
        else:
            edges_to_node_dict[dest_node] += [edge]

    return edges_to_node_dict

def get_edges_from_node_dict(ld: LoaderSimpleFromJson):
        edges_from_node_dict = {}

        for edge in ld.get_edges():

            src_node = edge[0]

            if src_node not in edges_from_node_dict:
                edges_from_node_dict[src_node] = [edge]
            else:
                edges_from_node_dict[src_node] += [edge]

        return edges_from_node_dict

def get_volume(longitudinal_volumes_array, node):
    idx = int(node.split("_")[0])
    time = int(node.split("_")[1])
    if idx in longitudinal_volumes_array[time]:
        return longitudinal_volumes_array[time][idx], True
    return 0, False


def get_dict_of_volume_percentage_change_and_classification_per_edge(ld: LoaderSimpleFromJson, longitudinal_volumes_array):
    volume_change_per_edge_dict = {}  # return {edge : [volume percentage change, calssification]}
    edges_to_node_dict = get_edges_to_node_dict(ld)  # {node : [edges to node]}
    edges_from_node_dict = get_edges_from_node_dict(ld)  # {node : [edges from node]}

    for edge in ld.get_edges():
        src_node, dest_node = edge
        if len(edges_from_node_dict[src_node]) > 1:  # split

            src_total_volume, is_existing = get_volume(longitudinal_volumes_array, src_node)
            if not is_existing:
                print("Error in get_dict_of_volume_percentage_change_and_classification_per_edge 1")
                # return {}
            dest_total_volume = 0
            for edge_from_src in edges_from_node_dict[src_node]:
                temp_dest = edge_from_src[1]
                vol, is_existing = get_volume(longitudinal_volumes_array, temp_dest)
                if not is_existing:
                    print("Error in get_dict_of_volume_percentage_change_and_classification_per_edge 2")
                    # return {}
                dest_total_volume += vol

            if src_total_volume == 0:
                percentage_diff = "+inf"
            else:
                percentage_diff = ((dest_total_volume/src_total_volume) - 1) * 100
            volume_change_per_edge_dict[edge] = [percentage_diff, edgeVolumeClassification.SPLITTING]

        elif len(edges_to_node_dict[dest_node]) > 1:  # merged
            dest_total_volume, is_existing = get_volume(longitudinal_volumes_array, dest_node)
            if not is_existing:
                    print("Error in get_dict_of_volume_percentage_change_and_classification_per_edge 3")
                    # return {}
            src_total_volume = 0
            for edge_to_dest in edges_to_node_dict[dest_node]:
                temp_src = edge_to_dest[0]
                vol, is_existing = get_volume(longitudinal_volumes_array, temp_src)
                if not is_existing:
                    print("Error in get_dict_of_volume_percentage_change_and_classification_per_edge 4")
                    # return {}
                src_total_volume += vol
                
            if src_total_volume == 0:
                percentage_diff = "+inf"
            else:
                percentage_diff = ((dest_total_volume/src_total_volume) - 1) * 100
            if edge not in volume_change_per_edge_dict:
                volume_change_per_edge_dict[edge] = [percentage_diff, edgeVolumeClassification.MERGED]
            # else:
            #     volume_change_per_edge_dict[edge] = [percentage_diff, edgeVolumeClassification.COMPLEX] # currently we dont deal with this case properly

        else: # linear
            src_total_volume, is_existing1 = get_volume(longitudinal_volumes_array, src_node)
            dest_total_volume, is_existing2 = get_volume(longitudinal_volumes_array, dest_node)
            if (not is_existing1) or (not is_existing2):
                [None, edgeVolumeClassification.LINEAR]
            else:
                if src_total_volume == 0:
                    percentage_diff = "+inf"
                else:
                    percentage_diff = ((dest_total_volume/src_total_volume) - 1) * 100
                volume_change_per_edge_dict[edge] = [percentage_diff, edgeVolumeClassification.LINEAR]

    return volume_change_per_edge_dict


def get_diff_in_total(longitudinal_volumes_array):
    total_volume_arr = []
    for dict in longitudinal_volumes_array:
        total_vol_cm3 = 0
        for key, val in dict.items():
            total_vol_cm3 += val

        vol_percentage_diff, vol_cm3_diff = 0, 0
        if len(total_volume_arr) > 0:
            prev_total_vol = total_volume_arr[len(total_volume_arr) - 1][0]
            vol_percentage_diff = ((total_vol_cm3 / prev_total_vol)-1)*100
            vol_cm3_diff = total_vol_cm3 - prev_total_vol

        total_volume_arr.append([total_vol_cm3, vol_percentage_diff, vol_cm3_diff])
    return total_volume_arr

def get_percentage_diff_per_edge_dict(ld, partial_patient_path):
    longitudinal_volumes_array = generate_longitudinal_volumes_array(partial_patient_path)
    # remove the difference in cm^3, leave only difference in percentage
    # volume_change_per_edge = get_dict_of_volume_change_per_edge(ld,longitudinal_volumes_array)
    # return {edge: percentage for edge, (percentage, _) in volume_change_per_edge.items()}
    volume_change_and_class_per_edge = get_dict_of_volume_percentage_change_and_classification_per_edge(ld, longitudinal_volumes_array)
    return {edge: percentage for edge, (percentage, _) in volume_change_and_class_per_edge.items()}

