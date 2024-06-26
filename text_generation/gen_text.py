import math
import re
from volume.volume_calculation import generate_longitudinal_volumes_array, get_diff_in_total, \
    get_dict_of_volume_percentage_change_and_classification_per_edge, get_edges_to_node_dict
from reportlab.platypus import Paragraph
from generate_info.gen_single_lesion.drawer_single_lesion_graph import get_node_volume
from reportlab.lib.styles import getSampleStyleSheet


def get_last_t_node(cur_componenet):
    max_time = 0
    last_nodes_i = []
    for node in cur_componenet:
        time = int(node.split("_")[1])
        if time > max_time:
            max_time = time
    for node in cur_componenet:
        if int(node.split("_")[1]) == max_time:
            last_nodes_i.append(node)
    return last_nodes_i, max_time


def get_vol_change_percentage(src, dest, vol_array):
    percentage_diff = 0
    src_vol, _ = get_node_volume(src, vol_array)
    dest_vol, _ = get_node_volume(dest, vol_array)

    if src_vol != 0:
        percentage_diff = ((dest_vol / src_vol) - 1) * 100
    if percentage_diff == 100:
        return 99
    return percentage_diff


def join_with_and(strings):
    if not strings:
        return ""
    elif len(strings) == 1:
        return strings[0]
    else:
        return ", ".join(strings[:-1]) + " and " + strings[-1]


def get_prev_appeared_nodes(node, edges_to_node_dict, vol_array):
    if node not in edges_to_node_dict:
        return []
    prev_nodes = [src for src, _ in edges_to_node_dict[node]]
    for prev in prev_nodes:
        src_vol, _ = get_node_volume(prev, vol_array)
        if src_vol == 0:
            # remove src
            prev_nodes.remove(prev)
            prev_nodes += get_prev_appeared_nodes(prev, edges_to_node_dict, vol_array)
    return prev_nodes


def find_when_last_appeared_text(node, edges_to_node_dict, vol_array, all_patient_dates, last_node):
    text = ""
    prev_nodes = get_prev_appeared_nodes(node, edges_to_node_dict, vol_array)
    if len(prev_nodes) > 0:
        time = prev_nodes[0].split("_")[1]
        date = all_patient_dates[int(time)]
        text += f"It previously appeared in the scan taken on {date}. "
        percentage = get_vol_change_percentage(prev_nodes[0], last_node, vol_array)  # only works for single prev node
        changed = "increased"
        if percentage < 0:
            changed = "decreased"
        if time != int(node.split("_")[1]):
            text += f"This lesion’s volume has {changed} by {abs(int(percentage))}% between the previous scan where it last appeared ({date}) and the current scan. "

    return text


def doesnt_appear_or_lone_lesion(ld, last_node, pattern, vol_array, all_patient_dates, cur_component,
                                 doesnt_appear_nodes={}):
    edges_to_node_dict = get_edges_to_node_dict(ld)
    last_seen, _ = edges_to_node_dict[last_node][0]
    last_seen_time = last_seen.split("_")[1]
    last_node_time = last_node.split("_")[1]
    last_node_idx = last_node.split("_")[0]
    text = ""
    doesnt_appear_nodes_times = {int(node.split("_")[1]) for node in doesnt_appear_nodes.keys()}
    if pattern == "lone":
        text += f"This is a lone lesion. It appeared in the scan taken on {all_patient_dates[int(last_node_time)]}. "
        if len(vol_array) == int(last_node_time) - 1:
            if int(last_node.split("_")[0]) in vol_array[int(last_node_time) - 1]:
                cur_vol = vol_array[int(last_node_time) - 1][int(last_node.split("_")[0])]
                text += f"This lesion's volume is {cur_vol} cc. "
                return text
    if last_node != next(iter(doesnt_appear_nodes.keys())):
        if len(doesnt_appear_nodes_times) > 1:

            text += "This lesion doesn't appear in the previous scans, taken on " + ", ".join(
                [all_patient_dates[int(last_seen_time)] for last_seen_time in doesnt_appear_nodes_times]) + ". "
        else:
            text += f"This lesion doesn't appear in the previous scan, taken on {all_patient_dates[int(next(iter(doesnt_appear_nodes_times)))]}. "
            text += find_when_last_appeared_text(next(iter(doesnt_appear_nodes.keys())), edges_to_node_dict, vol_array,
                                                 all_patient_dates, last_node)
        if int(last_node_idx) in vol_array[int(last_node_time)]:
            cur_node_vol = vol_array[int(last_node_time)][int(last_node_idx)]
            if len(last_node) == 1:
                text += f"The currrent lesion volume is {round(cur_node_vol, 2)} cc. "
        # text+= get_volume_diff_from_first_scan(ld, last_node,cur_component,vol_array)
        return text
    if int(last_node_idx) in vol_array[int(last_node_time)]:
        cur_node_vol = vol_array[int(last_node_time)][int(last_node_idx)]
        if len(doesnt_appear_nodes_times) > 1:

            text += "This lesion doesn't appear in the previous scans, taken on " + ", ".join(
                [all_patient_dates[int(last_seen_time)] for last_seen_time in doesnt_appear_nodes_times]) + "."
        else:
            text += f"This lesion doesn't appear in the previous scan, taken on {all_patient_dates[int(last_seen_time)]}. "
        text += find_when_last_appeared_text(next(iter(doesnt_appear_nodes.keys())), edges_to_node_dict, vol_array,
                                             all_patient_dates, last_node)
        if len(last_node) == 1:
            text += f"The currrent lesion volume is {round(cur_node_vol, 2)} cc. "
        return text

    text += f"The lesion does not appear in the scan taken on {all_patient_dates[int(last_node_time)]}. "
    if last_seen:
        text += f"It last appeared in the scan taken on {all_patient_dates[int(last_seen_time)]}. "

    return text


def get_first_node(cur_component):
    node_numbers = [(node, int(node.split('_')[1])) for node in cur_component]

    # Find the minimum numerical part
    min_num = min(node_numbers, key=lambda x: x[1])[1]

    # Collect all nodes with the minimum numerical part
    min_nodes = [node for node, num in node_numbers if num == min_num]

    # Return the list of nodes with the smallest number
    return min_nodes


def is_monotonic(cur_vol, prev_nodes, vol_array, cur_component):
    total_volumes_by_time = {}

    # Sum the volumes for each timestamp
    for node in cur_component:
        _, node_time = map(int, node.split("_"))
        node_vol, _ = get_node_volume(node, vol_array)

        if node_vol == 0:
            if node_time in total_volumes_by_time:
                del total_volumes_by_time[node_time]
            continue

        if node_time not in total_volumes_by_time:
            total_volumes_by_time[node_time] = node_vol
        else:
            total_volumes_by_time[node_time] += node_vol

    # Extract the total volumes in order of their timestamps
    sorted_times = sorted(total_volumes_by_time.keys())
    sorted_volumes = [total_volumes_by_time[time] for time in sorted_times]

    # Check if the volumes are monotonically increasing or decreasing
    is_increasing = all(x <= y for x, y in zip(sorted_volumes, sorted_volumes[1:]))
    is_decreasing = all(x >= y for x, y in zip(sorted_volumes, sorted_volumes[1:]))

    # Return True if either condition is satisfied
    return is_increasing or is_decreasing


def get_volume_diff_from_first_scan(ld, cur_node, cur_component, vol_array):
    first_nodes = get_first_node(cur_component)
    cur_vol = 0
    # for doesn't appear
    edges_to_node_dict = get_edges_to_node_dict(ld)

    # only gives nodes from the prevous time stamp where they last appeared
    prev_nodes = get_prev_appeared_nodes(cur_node, edges_to_node_dict, vol_array)

    cur_node_idx = int(cur_node.split("_")[0])
    if cur_node_idx in vol_array[-1]:
        cur_vol = vol_array[-1][cur_node_idx]

    # get volume for first node
    if len(first_nodes) == 1:
        first_node = first_nodes[0]
        first_node_vol = 0
        first_node_idx, first_node_time = first_node.split("_")
        # if int(first_node_time) in vol_array:
        if int(first_node_idx) in vol_array[int(first_node_time)]:
            first_node_vol = vol_array[int(first_node_time)][int(first_node_idx)]
        percentage = abs(math.floor(get_vol_change_percentage(first_node, cur_node, vol_array)))

    if len(first_nodes) > 1:
        ## TO DO!! case where there is more than one first node
        first_node_vol = 0
        for node in first_nodes:
            first_node_idx, first_node_time = node.split("_")
            if int(first_node_idx) in vol_array[int(first_node_time)]:
                first_node_vol += vol_array[int(first_node_time)][int(first_node_idx)]
        if first_node_vol != 0:
            percentage = abs(math.floor(((cur_vol / first_node_vol) - 1) * 100))

    if not first_node_vol:
        return "There is no volume data for this lesion from the first scan. "
    if not cur_vol:
        if first_node_vol:
            return f"The lesion's volume in the first scan was {round(first_node_vol,2)}"
        return ""
    change = "decreased"
    if first_node_vol < cur_vol:
        change = "increased"

    monotonic_text = ""
    monotonic_flag = is_monotonic(cur_vol, prev_nodes, vol_array, cur_component)
    if monotonic_flag:
        monotonic_text = "monotonically"
    if percentage == 100:
        percentage = 99
    text = f"The total lesion burden has {monotonic_text} {change}, from the first scan to the current scan, by {percentage}%. "
    return text


def find_linear_node(last_nodes, edges_list, volume_change_per_edge_dict):
    # for edge in iter(edges_list.values()):
    #     if volume_change_per_edge_dict[edge[0]][1]!="splitting":
    #         linear_node = edge[0]
    #         return linear_node
    # return last_nodes[0]

    def trace_back_pre_split(lesion, timestamp):
        for (src, tgt) in volume_change_per_edge_dict.keys():
            src_lesion, src_time = src.split('_')
            tgt_lesion, tgt_time = tgt.split('_')

            if int(tgt_lesion) == lesion and int(tgt_time) == timestamp:
                # Check if the source lesion has multiple outgoing edges
                outgoing_edges = [(s, t) for (s, t) in volume_change_per_edge_dict.keys() if s == src]
                if len(outgoing_edges) > 1:
                    # This is the split point, return the lesion just before split
                    return src
                else:
                    # Continue tracing back
                    return trace_back_pre_split(int(src_lesion), int(src_time))
        # If no predecessor is found, return this lesion (origin lesion)
        return f"{lesion}_{timestamp}"

    pre_split_lesions = {}

    # For each lesion in the last timestamp, trace back to the lesion just before the split
    for lesion in last_nodes:
        lesion_id, timestamp = map(int, lesion.split('_'))
        pre_split_lesion = trace_back_pre_split(lesion_id, timestamp)
        pre_split_lesions[lesion] = pre_split_lesion

    return pre_split_lesions[last_nodes[0]]


def get_text_splitting(last_nodes, vol_change_percentage, longitudinal_volumes_array, edges_list,
                       volume_change_per_edge_dict):
    text = ""
    linear_node = find_linear_node(last_nodes, edges_list, volume_change_per_edge_dict)
    joined_nodes = join_with_and(last_nodes)
    text += f"The lesion {linear_node} split into {len(last_nodes)} lesions- {joined_nodes} . "

    # find total currrent volume
    total_cur_vol = 0
    for node in last_nodes:
        i, time = node.split("_")
        i = int(i)
        time = int(time)
        if time <= len(longitudinal_volumes_array) - 1:
            if i in longitudinal_volumes_array[time]:
                total_cur_vol += longitudinal_volumes_array[time][i]
    text += f"The total current lesion burden is {round(total_cur_vol, 2)} cc. "
    total_vol_first_scan = 0
    for vol in longitudinal_volumes_array[0]:
        total_vol_first_scan += vol
    change = "decreased"
    if vol_change_percentage > 0:
        change = "increased"
    text += f"The total current lesion burden has {change} by {abs(round(vol_change_percentage))}% since the previous scan "

    # find difference from first scan

    return text


def gen_text_single_node(ld, last_node, nodes2cc_class, edge_vol_change_class, edges_list, longitudinal_volumes_array,
                         all_patient_dates, cur_component, volume_change_per_edge_dict, doesnt_appear_nodes={}):
    text = ""
    if len(last_node) < 1:
        return text
    last_node_val = last_node[0]
    # check pattern type of cc
    pattern = nodes2cc_class[last_node_val]
    doesnt_appear_cc_dict = {node: nodes2cc_class[node] for node in doesnt_appear_nodes}
    vol_change_percentage = 0
    if len(edges_list[last_node_val]) > 0:
        vol_change_percentage, edge_class = edge_vol_change_class[edges_list[last_node_val][0]]
    # vol_change_percentage = abs(round(vol_change_percentage))
    # doesnt_appear_flag =False
    if len(doesnt_appear_nodes) > 0:
        doesnt_appear_flag = True
        text += doesnt_appear_or_lone_lesion(ld, last_node_val, pattern, longitudinal_volumes_array, all_patient_dates,
                                             cur_component, doesnt_appear_cc_dict)
        # return text
    change = ""
    if vol_change_percentage < 0:
        change = "decreased"

    if vol_change_percentage > 0:
        change = "increased"

    if pattern == 'merge_p':
        return merge_pattern_text(change, cur_component, edges_list, last_node_val, ld, longitudinal_volumes_array,
                                  text, vol_change_percentage, volume_change_per_edge_dict)

    ## todo
    if pattern == 'split_p':
        text += f"The lesions are a result of a split of a lesion from a previous scan into multiple lesions. "

    if pattern == 'complex_p':
        text += f"The lesion classification is complex. "
        if edge_class == "splitting":
            # find total tumor burden in current scan
            text += get_text_splitting(last_node, vol_change_percentage, longitudinal_volumes_array, edges_list,
                                       volume_change_per_edge_dict)
            # compare to total tumor burden in first scan and return text for volume change since first scan

            return text
        if edge_class == "merged":
            return merge_pattern_text(change, cur_component, edges_list, last_node_val, ld, longitudinal_volumes_array,
                                      text, vol_change_percentage, volume_change_per_edge_dict)

    if change != "":
        text += f"The current lesion volume has {change} over time by {abs(round(vol_change_percentage))}% relative to the previous scan. "
    text += get_volume_diff_from_first_scan(ld, last_node_val, cur_component, longitudinal_volumes_array)

    return text


def merge_pattern_text(change, cur_component, edges_list, last_node_val, ld, longitudinal_volumes_array, text,
                       vol_change_percentage, volume_change_per_edge_dict):
    connected_nodes = [node1 for node1, node2 in edges_list[last_node_val]]
    prev_lesions_for_text = connected_nodes
    lesion_text = "relative to the combined (summed) previous volumes of lesions"
    prev_lesions_for_text_list = join_with_and(prev_lesions_for_text)
    if len(prev_lesions_for_text) == 1:
        lesion_text = "relative to the previous scan, of lesion"
        prev_lesions_for_text_list = prev_lesions_for_text[0]
    while True:
        if len(connected_nodes) > 1:
            break
        edges_list = [key for key in volume_change_per_edge_dict.keys() if connected_nodes[0] in key]
        prev_node = connected_nodes[0]
        connected_nodes = [node1 for node1, node2 in edges_list]
        if prev_node in connected_nodes:
            connected_nodes.remove(prev_node)
    connected_nodes_list = join_with_and(connected_nodes)
    text += f"Lesion {last_node_val} is a merged lesion resulting from a merge of lesions {connected_nodes_list}. "
    if vol_change_percentage != 0:
        text += f"Lesion {last_node_val}'s total volume, {lesion_text} {prev_lesions_for_text_list}, has {change} by {abs(round(vol_change_percentage))}%. "
        text += get_volume_diff_from_first_scan(ld, last_node_val, cur_component, longitudinal_volumes_array)
    return text


def replace_lesion_names(text, internal_external_names_dict):
    # Regular expression to find patterns of the type num_num
    pattern = re.compile(r'\b\d+_\d+\b')

    # Function to replace match with dictionary value
    def replace_match(match):
        return internal_external_names_dict.get(match.group(0), match.group(0))

    # Perform the replacement
    return pattern.sub(replace_match, text)


def gen_summary_for_cc(ld, cur_component, longitudinal_volumes_array, nodes2cc_class, all_patient_dates,
                       internal_external_names_dict, doesnt_appear_nodes):
    text = ""
    last_nodes, max_time = get_last_t_node(cur_component)
    volume_change_per_edge_dict = get_dict_of_volume_percentage_change_and_classification_per_edge(ld,
                                                                                                   longitudinal_volumes_array)
    edges_dict = {}
    for node in last_nodes:
        matching_keys = [key for key in volume_change_per_edge_dict.keys() if node in key]
        edges_dict[node] = matching_keys
    # if len(last_nodes)==1:
    #     last_node_par=last_nodes[0]
    #     edges_list =edges_dict[last_node_par]

    # if len(doesnt_appear_nodes)>0:
    #     text+= gen_text_single_node(ld,last_node_par,nodes2cc_class,(0,"empty"),edges_list,longitudinal_volumes_array,all_patient_dates,cur_component,volume_change_per_edge_dict,doesnt_appear_nodes)

    # else:
    #     input_par =volume_change_per_edge_dict[edges_list[0]]
    #     text += gen_text_single_node(ld,last_node_par,nodes2cc_class,input_par,edges_list,longitudinal_volumes_array,all_patient_dates,cur_component,volume_change_per_edge_dict)

    text += gen_text_single_node(ld, last_nodes, nodes2cc_class, volume_change_per_edge_dict, edges_dict,
                                 longitudinal_volumes_array, all_patient_dates, cur_component,
                                 volume_change_per_edge_dict, doesnt_appear_nodes)

    text2 = replace_lesion_names(text, internal_external_names_dict)
    paragraph_text = Paragraph(text2, getSampleStyleSheet()['Normal'])

    return paragraph_text
