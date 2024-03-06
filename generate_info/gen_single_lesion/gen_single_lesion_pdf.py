from common_packages.LongGraphPackage import LoaderSimpleFromJson
from reportlab.platypus import Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from patient_summary.classify_changes_in_individual_lesions import classify_changes_in_individual_lesions, \
    count_d_in_d_out, gen_dict_classified_nodes_for_layers
from volume.lesion_volume_changes import check_single_lesion_growth, generate_volume_list_single_lesion
from generate_info.gen_single_lesion.gen_single_lesion_graph import get_single_node_graph_image
import networkx as nx
from volume.volume_calculation import get_percentage_diff_per_edge_dict, generate_longitudinal_volumes_array
from common_packages.BaseClasses import *
from datetime import datetime
import re
import pickle

# USR = "shira_p/PycharmProjects/engineering_project/matching"
USR = "talia.dym/Desktop/Engineering_Project"


MAX_SCANS_PER_GRAPH = 10
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
    elements = []
    title_style = getSampleStyleSheet()['Heading1']
    title_style.fontSize = 10
    title_style.spaceAfter = 0
    title_style.spaceBefore = 0
    title = Paragraph(sub_title, style=title_style)
    elements.append(title)
    if spacer:
        elements.append(Spacer(1, 5))
    return elements


def get_note(note: str, spacer=False):
    elements = []
    note = Paragraph(note, style=getSampleStyleSheet()['Normal'])
    elements.append(note)
    if spacer:
        elements.append(Spacer(1, 5))
    return elements


def get_graph_title(lesions_idx: list):
    text = ""
    if len(lesions_idx) == 1:
        text += f"The History of Lesion {lesions_idx[0]}"
    else:
        as_strings = map(str, lesions_idx)
        result_string = ", ".join(as_strings)
        text += f"The History of Lesions {result_string}"

    return get_sub_sub_title(f"&#8226; {text}", False)


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
    disappeared_components, mew_single_components, components_to_draw = [], [], []
    for cc in components:
        if total_max_time == max_time_per_cc_dict[tuple(cc)]:
            if len(cc) == 1:
                # new & single
                mew_single_components.append(cc)
            else:
                components_to_draw.append(cc)
        else:
            disappeared_components.append(cc)
    return disappeared_components, mew_single_components, components_to_draw


def get_new_lesions_text(new_single_components):
    num_of_new = len(new_single_components)

    if num_of_new == 0:
        return get_note("No new lesions had appeared.", True)
    lesions_idx = []
    for cc in new_single_components:
        node = cc.pop()
        lesions_idx.append(int(node.split("_")[0]))
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
    # vol_list = generate_volume_list_single_lesion(patient.partial_scans_address, longitudinal_volumes_array)
    cc_idx = 0
    ld = LoaderSimpleFromJson(patient.json_input_address)

    G = lg.get_graph()
    components = list(nx.connected_components(G))

    max_time_per_cc_dict, total_max_time = find_max_time_stamp_per_cc_and_total(components)
    min_time_per_cc_dict = find_min_time_stamp_per_cc(components, total_max_time)

    disappeared_components, new_single_components, components_to_draw = devide_components(components,
                                                                                          max_time_per_cc_dict,
                                                                                          total_max_time)
    # longitudinal_volumes_array = generate_longitudinal_volumes_array(patient_partial_path)
    percentage_diff_per_edge_dict = get_percentage_diff_per_edge_dict(ld, patient.partial_scans_address)

    # add section of new
    elements += get_sub_title("New Lesions", True)
    elements += get_new_lesions_text(new_single_components)
    # add section of disappeared
    elements += get_sub_title("Disappeared Lesions", False)

    classified_nodes_dict = gen_dict_classified_nodes_for_layers(
        classify_changes_in_individual_lesions(count_d_in_d_out(ld), ld))
    elements += get_disappeared_lesions_text(disappeared_components, max_time_per_cc_dict, classified_nodes_dict, lg)

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
    all_patient_dates = lg.get_patient_dates()
    num_of_CCS_to_draw = len(components_to_draw)
    while True:
        if cc_idx >= num_of_CCS_to_draw:
            return elements
        count = 0
        ran_through_all_scans = False
        CC_first_appeared_in = min_time_per_cc_dict[tuple(components_to_draw[cc_idx])]

        while not ran_through_all_scans:

            start = count + CC_first_appeared_in
            end_of_patient_dates = start + MAX_SCANS_PER_GRAPH

            if end_of_patient_dates >= len(all_patient_dates):
                end_of_patient_dates = len(all_patient_dates)
                ran_through_all_scans = True
            lg._patient_dates = all_patient_dates[start:end_of_patient_dates]

            lg._num_of_layers = end_of_patient_dates - start
            # lg._num_of_layers = MAX_SCANS_PER_GRAPH
            path = f"/cs/usr/{USR}/output/{patient.organ}/sub_graphs/single_labeled_lesion_graph"
            graph, lesions_idx = get_single_node_graph_image(path,
                                                             patient.json_input_address, cc_idx, lg, ld, components_to_draw,
                                                             longitudinal_volumes_array, percentage_diff_per_edge_dict,
                                                             start, end_of_patient_dates)
            if not graph:
                return elements

            # elements += get_graph_title(lesions_idx)
            elements += [graph]
            # elements += get_lesion_history_text(lesions_idx[0], vol_list)  # todo
            #
            # ## shira added text for classification of connected component
            # elements += cc_class_text(node2cc, nodes2cc_class, lesions_idx[0])

            elements.append(Spacer(1, 20))
            count += MAX_SCANS_PER_GRAPH - OVERLAP_BETWEEN_GRAPHS

        # elements += get_lesion_history_text(lesions_idx[0], vol_list)  # todo

        # shira added text for classification of connected component
        # elements += cc_class_text(node2cc, nodes2cc_class, lesions_idx[0])

        cc_idx += 1
        # return elements #todo remove

    # return elements