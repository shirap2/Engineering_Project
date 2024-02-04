from common_packages.LongGraphPackage import LoaderSimpleFromJson
from common_packages.LongGraphClassification import LongitClassification
from generate_info.gen_single_lesion.drawer_single_lesion_graph import DrawerLabelsAndLabeledEdges
import reportlab.platypus as ply
import os
import networkx as nx
import matplotlib.pyplot as plt
import PIL as pil


def crop_middle_of_image(input_path, output_path, crop_dimensions):
    original_image = pil.Image.open(input_path)

    # Get the dimensions of the original image
    width, height = original_image.size

    # Calculate the left, upper, right, and lower coordinates for cropping the middle region
    left = (width - crop_dimensions[0]) // 2
    upper = (height - crop_dimensions[1]) // 2
    right = left + crop_dimensions[0]
    lower = upper + crop_dimensions[1]

    # Crop the middle region
    cropped_image = original_image.crop((left, upper, right, lower))
    cropped_image.save(output_path)


def get_single_node_graph_image(image_path: str, scan_name: str, cc_idx: int, lg, ld,
                                components: list, nodes_to_put, longitudinal_volumes_array: list,
                                percentage_diff_per_edge_dict):
    dr = DrawerLabelsAndLabeledEdges(lg, cc_idx, ld, components, nodes_to_put, longitudinal_volumes_array,
                                     percentage_diff_per_edge_dict)

    if dr._is_graph_empty:
        return [None, None]

    full_path = f"{image_path}_{cc_idx}.png"
    if os.path.exists(full_path):
        os.remove(full_path)
    plt.figure()
    dr.show_graph(full_path)
    crop_dimensions = (450, 300)
    crop_middle_of_image(full_path, full_path, crop_dimensions)
    graph = ply.Image(full_path, height=200, width=300)

    return [graph, dr.get_lesion_idx()]

# def get_nodes_graph_image(image_path : str, partial_patient_path : str, scan_name: str):

#     cc_idx = 0
#     graphs = []

#     ld = LoaderSimpleFromJson(scan_name)
#     lg = LongitClassification(ld)
#     dr = DrawerLabelsAndLabeledEdges(lg, cc_idx, partial_patient_path, ld)


#     while not dr._is_graph_empty:
#         full_path = f"{image_path}_{cc_idx}.png"
#         if os.path.exists(full_path):
#             os.remove(full_path)
#         plt.figure()
#         dr.show_graph(full_path)
#         crop_dimensions = (600, 200)
#         crop_middle_of_image(full_path, full_path, crop_dimensions)
#         graphs.append(ply.Image(full_path, height=200, width=600))

#         cc_idx += 1
#         ld = LoaderSimpleFromJson(scan_name)
#         lg = LongitClassification(ld)
#         dr = DrawerLabelsAndLabeledEdges(lg, cc_idx, partial_patient_path, ld)

#     return graphs

# scan_name = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json"
# partial_patient_path = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_"
# ld = LoaderSimpleFromJson(scan_name)
# get_nodes_graph_image("single_labeled_lesion_graph.png", partial_patient_path, ld)
