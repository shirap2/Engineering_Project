from generate_info.gen_single_lesion.drawer_single_lesion_graph import (
    DrawerLabelsAndLabeledEdges, PatientData, GraphDisplay)
import reportlab.platypus as ply
import os
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
                                components: list, longitudinal_volumes_array: list,
                                percentage_diff_per_edge_dict, start: int, end_of_patient_dates: int):

    patient_data = PatientData(lg, ld, components,
                 longitudinal_volumes_array, percentage_diff_per_edge_dict)
    graph_display = GraphDisplay(cc_idx, start, end_of_patient_dates)

    dr = DrawerLabelsAndLabeledEdges(patient_data, graph_display)

    if dr._is_graph_empty:
        return [None, None]

    full_path = f"{image_path}_{cc_idx}_{start}.png"
    if os.path.exists(full_path):
        os.remove(full_path)
    plt.figure()
    dr.show_graph(full_path)
    crop_dimensions = (450, 300)
    crop_middle_of_image(full_path, full_path, crop_dimensions)
    graph = ply.Image(full_path, height=200, width=300)

    return [graph, dr.get_lesion_idx()]
