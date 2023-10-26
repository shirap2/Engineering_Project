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

    print(d_in_d_out_per_time_arr)
    return d_in_d_out_per_time_arr


def classify_changes_in_individual_lesions(d_in_d_out_per_time_arr):
    classified_nodes = {}
    for node, [d_in, d_out] in d_in_d_out_per_time_arr.items():

        if d_in == 0 and d_out == 0:
            classified_nodes[node] = changes_in_individual_lesions.LONE
            continue

        if d_in == 0 and d_out == 1:
            classified_nodes[node] = changes_in_individual_lesions.NEW
            continue

        if d_in == 1 and d_out == 0:
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




def generate_layer_class_chart(classified_nodes_dict):

    unique_classes = [getattr(changes_in_individual_lesions, attr) for attr in dir(changes_in_individual_lesions) if not callable(getattr(changes_in_individual_lesions, attr)) and not attr.startswith("__")]
    unique_layers = sorted(classified_nodes_dict.keys())

    # Create a 2D list for the data (counts)
    data = [[f"Time Layer"] + unique_classes]
    for layer in unique_layers:
        row = [layer] + [classified_nodes_dict.get(layer, {}).get(class_, 0) for class_ in unique_classes]
        data.append(row)

    # Create a PDF document
    doc = SimpleDocTemplate("layer_class_chart.pdf", pagesize=letter)

    elements = []

    title_string = "Layer Class Chart"
    subtitle = "Lesion Class Type"

    title_style = getSampleStyleSheet()['Title']
    title = Paragraph(title_string, title_style)

    elements.append(title)
    elements.append(Spacer(1, 20))

    table = Table(data)

    # Define table style
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), (0.8, 0.8, 0.8)),  # Grey background for headers
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold font for headers
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Add padding to headers
        ('GRID', (0, 0), (-1, -1), 1, colors.black),  # Add gridlines with 1pt width
    ])

    table.setStyle(table_style)

    elements.append(table)

    doc.build(elements)


ld = LoaderSimpleFromJson(
    f"/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json")
d_in_d_out_per_time_arr = count_d_in_d_out(ld)
print(classify_changes_in_individual_lesions(d_in_d_out_per_time_arr))
classes_dict = gen_dict_classified_nodes_for_layers(classify_changes_in_individual_lesions(d_in_d_out_per_time_arr))
print(classes_dict)

generate_layer_class_chart(classes_dict)