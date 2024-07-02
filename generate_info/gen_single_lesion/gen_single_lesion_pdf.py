from common_packages.LongGraphPackage import LoaderSimpleFromJson
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from generate_info.gen_single_lesion.drawer_single_lesion_graph import PatientData, get_node_volume
from patient_summary.classify_changes_in_individual_lesions import classify_changes_in_individual_lesions, \
    count_d_in_d_out, gen_dict_classified_nodes_for_layers
from volume.lesion_volume_changes import check_single_lesion_growth, generate_volume_list_single_lesion
from generate_info.gen_single_lesion.gen_single_lesion_graph import get_single_node_graph_image
import networkx as nx
from volume.volume_calculation import get_dict_of_volume_percentage_change_and_classification_per_edge, get_percentage_diff_per_edge_dict, generate_longitudinal_volumes_array
from common_packages.BaseClasses import *
from datetime import datetime
import re
import pickle
from pathlib import Path
from text_generation.gen_text import gen_summary_for_cc, get_last_t_node


ROOT = str(Path(__file__).resolve().parent).replace("generate_info/gen_single_lesion", "")
output_path = ROOT + "output"

MAX_SCANS_PER_GRAPH = 5
OVERLAP_BETWEEN_GRAPHS = 1



def get_title(title_string):
    title_style = getSampleStyleSheet()['Title']
    title = Paragraph(title_string, title_style)

    return [title]


def get_sub_title(sub_title: str, no_spaceBefore=True):
    title_style = getSampleStyleSheet()['Heading2']
    # title_style.fontSize = 10
    title_style.spaceAfter = 0
    title_style.spaceBefore = 20
    if no_spaceBefore:
        title_style.spaceBefore = 0
    title = Paragraph(sub_title, style=title_style)
    return [title]


def get_sub_sub_title(sub_title: str, spacer=True):
    # elements = []
    title_style = getSampleStyleSheet()['Heading1']
    title_style.fontSize = 10
    title_style.spaceAfter = 0
    title_style.spaceBefore = 0
    return Paragraph(sub_title, style=title_style)
    # title = Paragraph(sub_title, style=title_style)
    # elements.append(title)
    # if spacer:
    #     elements.append(Spacer(1, 5))
    # return elements


def get_note(note: str, spacer=False):
    elements = []
    note = Paragraph(note, style=getSampleStyleSheet()['Normal'])
    elements.append(note)
    if spacer:
        elements.append(Spacer(1, 5))
    return elements


def get_cc_title(cc_idx, lesions_idx, num_of_all_dates, internal_external_names_dict):
    lesions_idx_string = ''
    for lesion_idx in lesions_idx:
        lesions_idx_string += internal_external_names_dict[f'{str(lesion_idx)}_{num_of_all_dates - 1}']
        lesions_idx_string += ', '

    lesions_idx_string = lesions_idx_string[:-2]
    text = f'Changes over-time of lesion {cc_idx + 1}, appearing at last scan as {lesions_idx_string}:'

    return get_sub_title(text, False)[0]
    # return get_sub_sub_title(f"&#8226; {text}", False)


def get_lesion_history_text(key, vol_list):
    text_to_add = check_single_lesion_growth(vol_list, key)
    # return get_note("Lesion "+ str(key)+ ": "+ text_to_add, True)
    return get_note(text_to_add, True)


def find_max_time_stamp_per_cc_and_total(components):
    max_total_time, max_cc_time = 0, 0
    max_time_per_cc_dict = dict()
    for cc in components:
        max_cc_time = 0
        for node in cc:
            time = int(node.split("_")[1])
            if time > max_cc_time:
                max_cc_time = time
        max_time_per_cc_dict[tuple(cc)] = max_cc_time
        if max_cc_time > max_total_time:
            max_total_time = max_cc_time
    return max_time_per_cc_dict, max_total_time


def find_min_time_stamp_per_cc(components, max_total_time):
    min_time_per_cc_dict = dict()
    for cc in components:
        min_cc_time = max_total_time
        for node in cc:
            time = int(node.split("_")[1])
            if time <= min_cc_time:
                min_cc_time = time
        min_time_per_cc_dict[tuple(cc)] = min_cc_time
    return min_time_per_cc_dict


def devide_components(components, max_time_per_cc_dict, total_max_time):
    disappeared_components, new_single_components, components_to_draw = [], [], []
    for cc in components:
        if total_max_time == max_time_per_cc_dict[tuple(cc)]:
            if len(cc) == 1:
                # new & single
                new_single_components.append(cc)
            else:
                components_to_draw.append(cc)
        else:
            disappeared_components.append(cc)
    return disappeared_components, new_single_components, components_to_draw


def get_new_lesions_text(new_single_components, union_non_draw_internal_external_names_dict):
    num_of_new = len(new_single_components)

    if num_of_new == 0:
        return get_note("No new lesions had appeared.", True)
    lesions_idx = []
    for cc in new_single_components:
        node = cc.pop()
        lesions_idx.append(union_non_draw_internal_external_names_dict[node])
        # lesions_idx.append(int(node.split("_")[0]))
    as_strings = map(str, lesions_idx)
    result_string = ", ".join(as_strings)
    text_to_add = f"Lesions {result_string} appeared for the first time in the last scan."
    if num_of_new == 1:
        text_to_add = f"Lesion {result_string} appeared for the first time in the last scan."
    return get_note(text_to_add, True)


def get_disappeared_lesions_text(disappeared_components, max_time_per_cc_dict, classifed_nodes_dict, lg):
    num_of_disappeared = len(disappeared_components)
    sum_disappeared = sum(entry.get("disappeared", 0) for entry in classifed_nodes_dict.values())
    dates = lg._patient_dates
    if num_of_disappeared == 0:
        return get_note("Over time, no lesions disappeared.", True)

    if num_of_disappeared == 1:
        elements = get_note(f"Over time, one lesion disappeared.", False)
        cc = disappeared_components[0]
        elements += get_note(f"It was last identified in {dates[max_time_per_cc_dict[tuple(cc)]]}.", True)
        return elements

    num_of_disappeared_lesions_per_time = dict()  # key:time, value: num of desappeared lesion
    for cc in disappeared_components:
        time = max_time_per_cc_dict[tuple(cc)]
        if time in num_of_disappeared_lesions_per_time:
            num_of_disappeared_lesions_per_time[time] += 1
        else:
            num_of_disappeared_lesions_per_time[time] = 1

    elements = get_note(f"Over time, {sum_disappeared} lesions disappeared.", False)

    if len(num_of_disappeared_lesions_per_time) == 1:
        # all disappeared in the same time
        cc = disappeared_components[0]
        elements += get_note(f"They were last identified in {dates[max_time_per_cc_dict[tuple(cc)]]}.", False)
    else:
        num_of_disappeared_lesions_per_time = sorted(num_of_disappeared_lesions_per_time.items(),
                                                     key=lambda item: item[0])
        for time in sorted(classifed_nodes_dict.keys()):
            if time + 1 not in classifed_nodes_dict:
                continue
            num_of_dis_lesions = classifed_nodes_dict.get(time + 1, {}).get("disappeared", 0)
            were_or_was = "s were"
            if num_of_dis_lesions == 1:
                were_or_was = " was"
            if num_of_dis_lesions != 0:
                elements += get_note(f"{num_of_dis_lesions} lesion{were_or_was} last identified in {dates[time]}.", False)

    elements.append(Spacer(1, 5))
    return elements


"""
this function genernates the text for classification of connnected component
"""
def cc_class_text(node2cc, nodes2cc_class, lesion_idx_in_last_scan: int):
    lesion_cc_class = ""
    max_time = max(int(key.split('_')[1]) for key in node2cc.keys())
    node_key = f"{lesion_idx_in_last_scan}_{max_time}"
    if node_key in nodes2cc_class:
        lesion_cc_class = nodes2cc_class[node_key]
        elements = get_note("Classification of connected component: " + lesion_cc_class[:-2] + ".", True)
    else:
        elements = get_note("No information for classifiaction", True)
    return elements


def get_dates(patient_path):
    date_pattern = r'(\d{2}_\d{2}_\d{4})'
    formatted_dates = set()

    for filename in os.listdir(patient_path):
        match = re.search(date_pattern, filename)
        if match:
            date_str = match.group(1)
            date_obj = datetime.strptime(date_str, '%d_%m_%Y')

            # Format the datetime object into "dd.mm.yyyy" and append to the list
            formatted_date = date_obj.strftime('%d.%m.%y')
            formatted_dates.add(formatted_date)

    formatted_dates = sorted(formatted_dates, key=lambda x: datetime.strptime(x, '%d.%m.%y'))
    return formatted_dates

# patient_name, patient.json_input_address,
#                                               patient.pickle_input_address, patient.partial_scans_address,
def set_nodes_external_name(cc_idx, nodes):
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    n = len(letters)
    i = 0

    sorted_nodes = sorted(nodes, key=lambda x: int(x.split("_")[1]))
    internal_external_names_dict = {}
    for node in sorted_nodes:
        internal_external_names_dict[node] = f'{cc_idx + 1}{letters[i%n]}'
        i += 1

    return internal_external_names_dict


def set_nodes_external_name(cc_idx, nodes):
    letters = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    n = len(letters)
    i = 0

    sorted_nodes = sorted(nodes, key=lambda x: int(x.split("_")[1]))
    internal_external_names_dict = {}
    for node in sorted_nodes:
        internal_external_names_dict[node] = f'{cc_idx + 1}{letters[i%n]}'
        i += 1

    return internal_external_names_dict

def find_doesnt_appear_vol_pattern(cur_component, longitudinal_volumes_array, nodes2cc_class):
    
    # Determine nodes that do not appear
    doesnt_appear_nodes = [
        node for node in cur_component
        if not get_node_volume(node, longitudinal_volumes_array)[1]
    ]
    
    # Get last nodes and max time
    last_nodes, max_time = get_last_t_node(cur_component)
    
    # Calculate the volume sum of the last nodes
    volume_sum_of_last_nodes = 0
    for node in last_nodes:

        if len(longitudinal_volumes_array)>max_time:
            if int(node.split("_")[0]) in longitudinal_volumes_array[max_time]:
                volume_sum_of_last_nodes+=longitudinal_volumes_array[max_time][int(node.split("_")[0])]
    
    # Determine the pattern of the connected component
    pattern_of_cc = nodes2cc_class[next(iter(cur_component))] if cur_component else None
    
    return doesnt_appear_nodes, volume_sum_of_last_nodes, pattern_of_cc


def create_single_lesion_pdf_page(patient,
                                  longitudinal_volumes_array):
    with open(patient.pickle_input_address, "rb") as file:
        lg = pickle.load(file)

    png_name = "output/" + patient.name.replace(" ", "_") + "_lesion_changes.png"
    elements = []

    # file title
    elements += get_title("Individual Lesion Changes")
    elements.append(Spacer(1, 20))

    # graph image
    # vol_list = generate_volume_list_single_lesion(longitudinal_volumes_array)
    cc_idx = 0
    ld = LoaderSimpleFromJson(patient.json_input_address)

    G = lg.get_graph()
    components = list(nx.connected_components(G))
    all_patient_dates = lg.get_patient_dates()

    max_time_per_cc_dict, total_max_time = find_max_time_stamp_per_cc_and_total(components)
    min_time_per_cc_dict = find_min_time_stamp_per_cc(components, total_max_time)

    disappeared_components, new_single_components, components_to_draw = devide_components(components,
                                                                                          max_time_per_cc_dict,
                                                                                          total_max_time)
    num_of_CCS_to_draw = len(components_to_draw)

    non_draw_cc_idx = 0
    non_draw_internal_external_names_dict = {}
    union_non_draw_internal_external_names_dict = {}
    non_draw_components = disappeared_components + new_single_components
    lone_components_dict = {}
    for cc in non_draw_components:


        cc_dict = set_nodes_external_name(
            non_draw_cc_idx + num_of_CCS_to_draw, non_draw_components[non_draw_cc_idx])
        union_non_draw_internal_external_names_dict = {**union_non_draw_internal_external_names_dict,
                                                 **cc_dict}
        non_draw_internal_external_names_dict[non_draw_cc_idx + num_of_CCS_to_draw] = cc_dict

        if len(cc) == 1:
            internal_name, external_name = next(iter(cc_dict.items()))
            date = all_patient_dates[int(internal_name.split('_')[1])]
            volume, _ = get_node_volume(internal_name, longitudinal_volumes_array)
            lone_components_dict[external_name] = [date, volume]

        non_draw_cc_idx += 1

    # longitudinal_volumes_array = generate_longitudinal_volumes_array(patient_partial_path)
    percentage_diff_per_edge_dict = get_percentage_diff_per_edge_dict(ld, patient.partial_scans_address, longitudinal_volumes_array)

    # add section of new
    elements += get_sub_title("New Lesions", True)
    elements += get_new_lesions_text(new_single_components, union_non_draw_internal_external_names_dict)
    # add section of disappeared
    elements += get_sub_title("Disappeared Lesions", False)

    classified_nodes_dict = gen_dict_classified_nodes_for_layers(
        classify_changes_in_individual_lesions(count_d_in_d_out(ld), ld))
    elements += get_disappeared_lesions_text(disappeared_components, max_time_per_cc_dict, classified_nodes_dict, lg) #TODO: use non_draw_internal_external_names_dict

    ## classification of nodes and cc from benny code
    lg.classify_nodes()
    lg.classify_cc()

    # dictionary of nodes-keys and the index of their cc- values
    node2cc = nx.get_node_attributes(G, NodeAttr.CC_INDEX)

    # set of all cc indices
    cc_set = set(node2cc.values())

    # dictionary of node(key)'s class(value) when part of cc
    nodes2cc_class = nx.get_node_attributes(G, NodeAttr.CC_PATTERNS)
    lesions_idx = 0
    # draw components to drw (existing in last scan + not new- no history)
    elements += get_sub_title("Lesions Appearing in Multiple Scans", False)

    patient_data = PatientData(lg, ld, components_to_draw,
                               longitudinal_volumes_array, percentage_diff_per_edge_dict)

    cc_info_dict = {}

    while True:
        if cc_idx >= num_of_CCS_to_draw:
            return elements, cc_info_dict, non_draw_internal_external_names_dict, lone_components_dict
        internal_external_names_dict = set_nodes_external_name(cc_idx, patient_data.components[cc_idx])

        count = 0
        ran_through_all_scans = False
        CC_first_appeared_in = min_time_per_cc_dict[tuple(components_to_draw[cc_idx])]
        cur_component = patient_data.components[cc_idx]

        cur_elements = []
        num_of_all_dates = len(all_patient_dates)
        while not ran_through_all_scans:

            start = count + CC_first_appeared_in
            end_of_patient_dates = start + MAX_SCANS_PER_GRAPH

            if end_of_patient_dates >= num_of_all_dates:
                end_of_patient_dates = num_of_all_dates
                ran_through_all_scans = True
            lg._patient_dates = all_patient_dates[start:end_of_patient_dates]

            lg._num_of_layers = end_of_patient_dates - start
            # lg._num_of_layers = MAX_SCANS_PER_GRAPH
            path = f"{output_path}/{patient.organ}/sub_graphs/single_labeled_lesion_graph"
            graph, lesions_idx = get_single_node_graph_image(path, cc_idx, start, end_of_patient_dates, patient_data, internal_external_names_dict)

            cur_elements += [(cc_idx, graph)]
            count += MAX_SCANS_PER_GRAPH - OVERLAP_BETWEEN_GRAPHS


        doesnt_appear_nodes, volume_sum_of_last_nodes, pattern_of_cc = find_doesnt_appear_vol_pattern(cur_component, longitudinal_volumes_array, nodes2cc_class)

        # update cc info dict
        is_cc_non_consecutive = (len(doesnt_appear_nodes) != 0)
        cc_info_dict[cc_idx] = [volume_sum_of_last_nodes, internal_external_names_dict, is_cc_non_consecutive, pattern_of_cc]

        # add the cc elements: title, graphs, text, space
        elements += [(cc_idx, get_cc_title(cc_idx, lesions_idx, num_of_all_dates, internal_external_names_dict))]
        elements += cur_elements
        elements.append((cc_idx, gen_summary_for_cc(ld, cur_component, longitudinal_volumes_array,
                                                    nodes2cc_class, all_patient_dates,
                                                    internal_external_names_dict, doesnt_appear_nodes)))
        elements.append((cc_idx, Spacer(1, 20)))

        cc_idx += 1

