import math
import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from volume.volume_calculation import generate_longitudinal_volumes_array, get_diff_in_total,get_dict_of_volume_percentage_change_and_classification_per_edge,get_edges_to_node_dict
from reportlab.platypus import Paragraph
from generate_info.gen_single_lesion.drawer_single_lesion_graph import get_node_volume
from reportlab.lib.styles import getSampleStyleSheet


# nltk.download('wordnet')

def check_lesion_growth_from_last_scan(lesion_volumes):
    """
    this function gets a list of a lesion's volume changes over time and
    checks the difference from last 2 consecutive scans
    :param lesion_volumes:
    :return:
    """
    text = ""
    if not lesion_volumes:
        return "No volume data available for the lesion."
    cur_volume = lesion_volumes[-1]
    if len(lesion_volumes) < 2:
        return "No volume data available for the lesion from previous scan. "

    prev_volume = lesion_volumes[-2]
    vol_change_type = ""
    change_percentage = round(abs((cur_volume / prev_volume) - 1) * 100)
    if cur_volume < prev_volume:
        #The volume has increased by {change_percentage}% between the previous scan and the current scan.
        text += f"The lesion volume has increased by {change_percentage}% from previous scan to current scan. "
    elif cur_volume > prev_volume:
        text += f"Lesion volume has decreased by {change_percentage}% from previous scan to current scan. "
    else:
        text += "No change in lesion volume from previous scan to current scan. "

    return text

def check_single_lesion_growth(lesion_volumes):
    """
    this function gets a list of a lesion's volume changes over time
    and checks if the volume increased/decreased
    """
    
    if not lesion_volumes:
        return "No volume data available for the lesion. "

    increasing = decreasing = True

    text = check_lesion_growth_from_last_scan(lesion_volumes)

    for i in range(1, len(lesion_volumes) - 1):
        if lesion_volumes[i] > lesion_volumes[i - 1]:
            decreasing = False
        elif lesion_volumes[i] < lesion_volumes[i - 1]:
            increasing = False
    change_percentage = round(abs((lesion_volumes[-1] / lesion_volumes[0]) - 1) * 100)
    if increasing:
        text += f"The total lesion burden has monotonically increased, from first scan to last scan, by {change_percentage}%. "
    elif decreasing:
        text += f"The total lesion burden has monotonically decreased, from first scan to last scan, by {change_percentage}%. "
    else:
        text += f"Volume shows both increases and decreases over time from first scan to last scan. "
    return text


"""
this function adds text describing percentage of volume growth for each time stamp
"""

def lesion_growth_percentage(longitudinal_volumes_array, num_of_tumors):
    paragraph = []
    volumes_array = get_diff_in_total(longitudinal_volumes_array)  # [total_vol_cm3, vol_percentage_diff, vol_cm3_diff]
    if len(volumes_array) < 2:
        return "Insufficient data for calculating percentage change."

    # Extract the first and last elements
    initial_value, _, _ = volumes_array[0]
    final_value, vol_percentage_diff, vol_cm3_diff = volumes_array[-1]

    # Determine the change type based on the sign of vol_percentage_diff
    change_type = "increased" if vol_percentage_diff >= 0 else "decreased"
    vol_percentage_diff_abs = round(abs((final_value / initial_value) - 1) * 100)
    # Format the message based on the presence of a minus sign
    vol_diff_text = f"{vol_percentage_diff_abs}%"

    text = f"Tumor burden of {num_of_tumors} lesions {change_type} over time by {vol_diff_text}\n"
    paragraph.append(Paragraph(text))

    return paragraph


def get_last_t_node(cur_componenet):
    max_time = 0
    last_nodes_i = []
    for node in cur_componenet:
        time = int(node.split("_")[1])
        if time > max_time:
            max_time=time
    for node in cur_componenet:
        if int(node.split("_")[1])==max_time:
            last_nodes_i.append(node)
    return last_nodes_i,max_time


def get_vol_change_percentage(src,dest,vol_array):
    percentage_diff = 0
    src_vol, _ = get_node_volume(src,vol_array)
    dest_vol, _ = get_node_volume(dest,vol_array)

    if src_vol != 0:
        percentage_diff = ((dest_vol/src_vol) - 1) * 100

    return percentage_diff

def join_with_and(strings):
    if not strings:
        return ""
    elif len(strings) == 1:
        return strings[0]
    else:
        return ", ".join(strings[:-1]) + " and " + strings[-1]

def get_prev_appeared_nodes(node,edges_to_node_dict,vol_array):
    if node not in edges_to_node_dict:
        return []
    prev_nodes = [src for src, _ in edges_to_node_dict[node]]
    for prev in prev_nodes:
        src_vol, _ = get_node_volume(prev,vol_array)
        if src_vol == 0:
            # remove src
            prev_nodes.remove(prev)
            prev_nodes += get_prev_appeared_nodes(prev,edges_to_node_dict,vol_array)
    return prev_nodes


def doesnt_appear_lesion(ld,last_node,pattern,vol_array,all_patient_dates):
    edges_to_node_dict = get_edges_to_node_dict(ld)
    last_seen,_ = edges_to_node_dict[last_node][0]
    last_seen_time = last_seen.split("_")[1]
    cur_time = last_node.split("_")[1]
    
    text =""
    if pattern =="lone":
        text+= f"This is a lone lesion. It appeared in the scan taken on {all_patient_dates[int(cur_time)]}. "
        if len(vol_array)==int(cur_time)-1:
            if int(last_node.split("_")[0]) in vol_array[int(cur_time)-1]:
                cur_vol = vol_array[int(cur_time)-1][int(last_node.split("_")[0])]
                text+= f"This lesion's volume is {cur_vol} cc. "
    date = "date"
    text += f"The lesion does not appear in the scan taken on {all_patient_dates[int(cur_time)]}."
    if last_seen:
        text+=f"It last appeared in the scan taken on {all_patient_dates[int(last_seen_time)]}"

    return text

def get_first_node(cur_component):
    node_numbers = [(node, int(node.split('_')[1])) for node in cur_component]
    
    # Find the minimum numerical part
    min_num = min(node_numbers, key=lambda x: x[1])[1]
    
    # Collect all nodes with the minimum numerical part
    min_nodes = [node for node, num in node_numbers if num == min_num]
    
    # Return the list of nodes with the smallest number
    return min_nodes


def is_monotonic(cur_vol,prev_nodes,vol_array,cur_component):
    total_volumes_by_time = {}

    # Sum the volumes for each timestamp
    for node in cur_component:
        _, node_time = map(int, node.split("_"))
        node_vol,_ = get_node_volume(node, vol_array)

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


def get_volume_diff_from_first_scan(ld,cur_node,cur_component,vol_array):
    first_nodes = get_first_node(cur_component)
    cur_vol=0
    # for doesn't appear
    edges_to_node_dict = get_edges_to_node_dict(ld)

    # only gives nodes from the prevous time stamp where they last appeared
    prev_nodes = get_prev_appeared_nodes(cur_node,edges_to_node_dict,vol_array)

    cur_node_idx = int(cur_node.split("_")[0])
    if cur_node_idx in vol_array[-1]:
        cur_vol = vol_array[-1][cur_node_idx]
    
     # get volume for first node
    if len(first_nodes) ==1:
        first_node = first_nodes[0]
        first_node_vol = 0
        first_node_idx, first_node_time = first_node.split("_")
        # if int(first_node_time) in vol_array:
        if int(first_node_idx) in vol_array[int(first_node_time)]:
            first_node_vol = vol_array[int(first_node_time)][int(first_node_idx)]
        percentage = abs(math.floor(get_vol_change_percentage(first_node, cur_node,vol_array)))
    

    if len(first_nodes)>1:
        ## TO DO!! case where there is more than one first node
        first_node_vol = 0
        for node in first_nodes:
            first_node_idx, first_node_time = node.split("_")
            if int(first_node_idx) in vol_array[int(first_node_time)]:
                first_node_vol += vol_array[int(first_node_time)][int(first_node_idx)]
        if first_node_vol != 0:
            percentage = abs(math.floor(((cur_vol/first_node_vol) - 1) * 100))

    if not first_node_vol:
        return "There is no volume data for this lesion from first scan"
   
    change = "decreased"
    if first_node_vol< cur_vol:
        change = "increased"

    monotonic_text = ""
    monotonic_flag = is_monotonic(cur_vol,prev_nodes,vol_array,cur_component)
    if monotonic_flag:
        monotonic_text = "monotonically"
    
    text =f"The total lesion burden has {monotonic_text} {change}, from the first scan to the current scan, by {percentage}%. "
    return text


def gen_text_single_node(ld,last_node,nodes2cc_class,edge_vol_change_class,edges_list,longitudinal_volumes_array,all_patient_dates,cur_component,volume_change_per_edge_dict):
    text =""
    # check pattern type of cc
    pattern = nodes2cc_class[last_node]
    vol_change_percentage,edge_class = edge_vol_change_class
    vol_change_percentage = abs(round(vol_change_percentage))
    doesnt_appear_flag =False
    if edge_class == "empty":
        doesnt_appear_flag=True
        text = doesnt_appear_lesion(ld,last_node,pattern,longitudinal_volumes_array,all_patient_dates)
        return text

    if vol_change_percentage<0:
        change = "decreased"
        
    if vol_change_percentage>0:
        change ="increased"

    if pattern =='merge_p':
        # connected_nodes =[]
        connected_nodes = [node1 for node1, node2 in edges_list]
        prev_lesions_for_text = connected_nodes
        lesion_text = "relative to the combined (summed) previous volumes of lesions"
        prev_lesions_for_text_list =join_with_and(prev_lesions_for_text)
        if len(prev_lesions_for_text)==1:
            lesion_text="relative to the previous scan, of lesion"
            prev_lesions_for_text_list =prev_lesions_for_text[0]
        while True:            
            if len(connected_nodes)>1:
                break
            edges_list = [key for key in volume_change_per_edge_dict.keys() if connected_nodes[0] in key]
            prev_node =connected_nodes[0]
            connected_nodes = [node1 for node1, node2 in edges_list]
            if prev_node in connected_nodes:
                connected_nodes.remove(prev_node)
        connected_nodes_list = join_with_and(connected_nodes)
        
        text +=f"Lesion {last_node} is a merged lesion resulting from a merge of lesions {connected_nodes_list}. "
        if not doesnt_appear_flag:
            text += f"Lesion {last_node}'s total volume, {lesion_text} {prev_lesions_for_text_list}, has {change} by {vol_change_percentage}%. "
            text+= get_volume_diff_from_first_scan(ld, last_node,cur_component,longitudinal_volumes_array)
        return text
    
    ## todo
    if pattern == 'split_p':
        pass

    if pattern == 'complex_p':
        pass
    
    # check if total vol is increase/decrease and monotonic
    # percentage_diff = get_vol_change_percentage(prev_node,last_node)

    # if vol_change_percentage>0:
    text += f"The current lesion volume has {change} over time by {vol_change_percentage}% relative to the previous scan. "
    text+= get_volume_diff_from_first_scan(ld, last_node,cur_component,longitudinal_volumes_array)



    return text

def replace_lesion_names(text,internal_external_names_dict):
     # Regular expression to find patterns of the type num_num
    pattern = re.compile(r'\b\d+_\d+\b')
    
    # Function to replace match with dictionary value
    def replace_match(match):
        return internal_external_names_dict.get(match.group(0), match.group(0))

    # Perform the replacement
    return pattern.sub(replace_match, text)

def gen_summary_for_cc(ld,cur_component,longitudinal_volumes_array,max_time_per_cc_dict,nodes2cc_class,all_patient_dates,internal_external_names_dict):
    text =""
    last_nodes,max_time = get_last_t_node(cur_component)
    volume_change_per_edge_dict= get_dict_of_volume_percentage_change_and_classification_per_edge(ld,longitudinal_volumes_array)
    edges_dict ={}
    for node in last_nodes:
        matching_keys = [key for key in volume_change_per_edge_dict.keys() if node in key]
        edges_dict[node]=matching_keys
    if len(last_nodes)==1:
        last_node_par=last_nodes[0]
        edges_list =edges_dict[last_node_par]

        # check if there are no edges from this node. meaning node either doesnt appear or is lone
        if len(edges_list)==0:
            text+= gen_text_single_node(ld,last_node_par,nodes2cc_class,(0,"empty"),edges_list,longitudinal_volumes_array,all_patient_dates,cur_component,volume_change_per_edge_dict)

        else:
            input_par =volume_change_per_edge_dict[edges_list[0]]
            text += gen_text_single_node(ld,last_node_par,nodes2cc_class,input_par,edges_list,longitudinal_volumes_array,all_patient_dates,cur_component,volume_change_per_edge_dict)

    ## need to support case of several last nodes in cc
    # check_single_lesion_growth(vol_list)
    text2 = replace_lesion_names(text,internal_external_names_dict)
    paragraph_text =Paragraph(text2,getSampleStyleSheet()['Normal'])

    return paragraph_text
