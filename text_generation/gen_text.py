import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from volume.volume_calculation import generate_longitudinal_volumes_array, get_diff_in_total,get_dict_of_volume_percentage_change_and_classification_per_edge
from reportlab.platypus import Paragraph
from generate_info.gen_single_lesion.drawer_single_lesion_graph import get_node_volume


nltk.download('wordnet')

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
        text += f"Volume monotonically increased over time by {change_percentage}% from first scan to last scan. "
    elif decreasing:
        text += f"Volume monotonically decreased over time by {change_percentage}% from first scan to last scan. "
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


def get_vol_change_percentage(src,dest):
    src_vol, _ = get_node_volume(src)
    dest_vol, _ = get_node_volume(dest)

    if src_vol != 0:
        percentage_diff = ((dest_vol/src_vol) - 1) * 100

    return percentage_diff


def gen_text_single_node(last_node,nodes2cc_class,edge_vol_change_class):
    text =[]
    vol_change_percentage,edge_class = edge_vol_change_class
    # check pattern type of cc
    pattern = nodes2cc_class[last_node]
    if pattern =='merge_p':
        pass

    if pattern == 'split_p':
        pass

    if pattern == 'complex_p':
        pass
    
    # check if total vol is increase/decrease and monotonic
    # percentage_diff = get_vol_change_percentage(prev_node,last_node)

    if vol_change_percentage>0:
        text += f"The current lesion volume has increased over time by {vol_change_percentage}% relative to the previous scan. "
    elif vol_change_percentage<0:
        text += f"Volume monotonically decreased over time by {vol_change_percentage}% relative to the previous scan. "

    # gen base case text

    # if not monotonic add text 

    # continue base case text


    return text


def gen_summary_for_cc(ld,cur_component,longitudinal_volumes_array,max_time_per_cc_dict,nodes2cc_class):
    text =[]
    last_nodes,max_time = get_last_t_node(cur_component)
    volume_change_per_edge_dict= get_dict_of_volume_percentage_change_and_classification_per_edge(ld,longitudinal_volumes_array)
    edges_dict ={}
    for node in last_nodes:
        matching_keys = [key for key in volume_change_per_edge_dict.keys() if node in key]
        edges_dict[node]=matching_keys
    if len(last_nodes)==1:
        gen_text_single_node(last_nodes[0],nodes2cc_class,volume_change_per_edge_dict[edges_dict[0]])

    ## need to support case of several last nodes in cc


    return text
