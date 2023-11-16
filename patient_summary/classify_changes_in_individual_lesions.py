import os
from matplotlib import image, pyplot as plt
import numpy as np
from reportlab.platypus import SimpleDocTemplate, Table, Image, Paragraph, TableStyle, Spacer
from common_packages.LongGraphPackage import LoaderSimpleFromJson
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from io import BytesIO
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


D_IN = 0
D_OUT = 1


class changes_in_individual_lesions:
    LONE = "lone"
    NEW = "new"
    DISAPPEARED = "disappeared"
    PERSISTENT = "persistent"
    MERGED = "merged"
    SPLIT = "split"
    COMPLEX = "complex"


def count_d_in_d_out(ld):
    d_in_d_out_per_time_arr = {}
    for edge in ld.get_edges():
        root = edge[0]
        tail = edge[1]
        root_idx = int(root.split("_")[0])
        root_time = int(root.split("_")[1])
        tail_idx = int(tail.split("_")[0])
        tail_time = int(tail.split("_")[1])

        # increase tails' d_in
        if d_in_d_out_per_time_arr.get(tail) is None:
            d_in_d_out_per_time_arr[tail] = [0, 0]
        d_in_d_out_per_time_arr[tail][D_IN] += 1

        # increase roots` d_out
        if d_in_d_out_per_time_arr.get(root) is None:
            d_in_d_out_per_time_arr[root] = [0, 0]
        d_in_d_out_per_time_arr[root][D_OUT] += 1

    return d_in_d_out_per_time_arr


def classify_changes_in_individual_lesions(d_in_d_out_per_time_arr):
    classified_nodes = {}
    print(d_in_d_out_per_time_arr)
    # Extract the highest time stamp value
    highest_y = max(int(key.split('_')[1]) for key in d_in_d_out_per_time_arr.keys())

    # Filter keys based on the highest y value
    filtered_keys = [key for key in d_in_d_out_per_time_arr.keys() if int(key.split('_')[1]) == highest_y]

    
    for node, [d_in, d_out] in d_in_d_out_per_time_arr.items():
        if d_in == 0 and d_out == 0:
            classified_nodes[node] = changes_in_individual_lesions.LONE
            continue

        if d_in == 0 and d_out == 1:
            classified_nodes[node] = changes_in_individual_lesions.NEW
            continue

        if d_in == 1 and d_out == 0 and node not in filtered_keys:
            classified_nodes[node] = changes_in_individual_lesions.DISAPPEARED
            continue

        if d_in == 1 and d_out == 1:
            classified_nodes[node] = changes_in_individual_lesions.PERSISTENT
            continue

        if d_in >= 2 and (d_out == 0 or d_out == 1):
            classified_nodes[node] = changes_in_individual_lesions.MERGED
            continue

        if (d_in == 0 or d_in == 1) and d_out >= 2:
            classified_nodes[node] = changes_in_individual_lesions.SPLIT
            continue

        if d_in >= 2 and d_out >= 2:
            classified_nodes[node] = changes_in_individual_lesions.COMPLEX

        if d_in == 1 and d_out == 0 and node in filtered_keys :
            classified_nodes[node] = changes_in_individual_lesions.PERSISTENT
            continue

    return classified_nodes


def gen_dict_classified_nodes_for_layers(classified_nodes):
    dict_classified_times = {}
    for node, node_class in classified_nodes.items():
        node_time = int(node.split("_")[1])
        if dict_classified_times.get(node_time) is None:
            dict_classified_times[node_time] = {}
        if dict_classified_times[node_time].get(node_class) is None:
            dict_classified_times[node_time][node_class] = 1
            continue
        dict_classified_times[node_time][node_class] += 1
    return dict_classified_times

