from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from common_packages.LongGraphClassification import LongitClassification
from common_packages.LongGraphPackage import LoaderSimpleFromJson, DrawerLabels
from reportlab.platypus import SimpleDocTemplate, Table, Image, Paragraph, TableStyle, Spacer
import os


def count_total_lesions(ld: LoaderSimpleFromJson):
    layer_counters = {}
    for node in ld.get_nodes():
        layer = node.split("_")[1]
        if layer in layer_counters:
            layer_counters[layer] += 1
        else:
            layer_counters[layer] = 1
    return layer_counters


def new_or_existing_lesion_classification(ld: LoaderSimpleFromJson):
    classified_nodes = {node: True for node in ld.get_nodes()}  # False: existing lesion, True : new lesion
    existing_nodes = [edge[1] for edge in ld.get_edges()]  # the edge destination is an existing node
    for existing_node in existing_nodes:
        classified_nodes[existing_node] = False

    return classified_nodes  # notice: all the nodes of t0 are labeled True, but this does not indicate anything


def count_new_nodes(classified_nodes: dict):
    new_lesion_counters = {}

    for node, is_new_label in classified_nodes.items():
        layer = node.split("_")[1]

        # all the nodes of t0 are labeled True, but this does not indicate anything
        if layer == '0':
            new_lesion_counters[layer] = "-"
            continue

        if layer not in new_lesion_counters:
            new_lesion_counters[layer] = 0

        # count all the new lesions
        if is_new_label:
            new_lesion_counters[layer] += 1

    return new_lesion_counters


def print_counter_into_pdf(layer_counters: dict, new_lesion_counters: dict, pdf_name: str):
    if os.path.exists(pdf_name):
        os.remove(pdf_name)
    doc = SimpleDocTemplate(pdf_name)
    elements = []

    # title
    title_string = "C. A. Patient Summary"
    sub_title = "Patient Study Based on Lesion Graph"

    title_style = getSampleStyleSheet()['Title']
    title = Paragraph(title_string, title_style)

    subtitle_style = getSampleStyleSheet()['Title']
    subtitle = Paragraph(sub_title, subtitle_style)

    elements.append(title)
    elements.append(subtitle)

    # graph image
    graph = Image("C_A_patient_summary.png", height=400, width=500)
    elements.append(graph)

    credit_style = ParagraphStyle('credit_style',fontsize=12)
    credit = Paragraph("Di Veroli B., Joskowicz L. A Graph Theoretic Approach for Analysis of Lesion Changes and Lesions "
                       "Detection Review in Longitudinal Ontological Imaging, CASMIP Hebrew University, 2023",credit_style)
    elements.append(credit)
    elements.append(Spacer(1,20))

    # table
    num_of_time_layers = len(layer_counters)
    data = [[key, layer_counters[str(key)], new_lesion_counters[str(key)]] for key in range(num_of_time_layers)]
    data = [["Time Layer", "Num. of Lesions", "Num. of New Appeared Lesions"]] + data
    table = Table(data)
    style = TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, (0, 0, 0)),  # Add gridlines with 1pt width
        ('BACKGROUND', (0, 0), (-1, 0), (0.8, 0.8, 0.8)),  # Grey background for headers
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),  # Center align all cells
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Bold font for headers
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),  # Add padding to headers
    ])

    # Apply the style to the table
    table.setStyle(style)
    elements.append(table)

    doc.build(elements)


def generate_counter():
    ld = LoaderSimpleFromJson(
        f"/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/C_A_glong_gt.json")
    lg1 = LongitClassification(ld)
    dr_2 = DrawerLabels(lg1)
    dr_2.show_graph("C_A_patient_summary.png")

    layer_counters = count_total_lesions(ld)
    classified_nodes = new_or_existing_lesion_classification(ld)
    new_lesion_counters = count_new_nodes(classified_nodes)
    pdf_name = "C_A_patient_summary.pdf"
    print_counter_into_pdf(layer_counters, new_lesion_counters, pdf_name)


generate_counter()
