
import math
from volume.volume_calculation import generate_longitudinal_volumes_array, get_diff_in_total
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image

def generate_volume_list_single_lesion(patient_path):
    longitudinal_volumes_array = generate_longitudinal_volumes_array(patient_path)  # returns sorted (by date) array of
    # dictionaries (one for each time stamp), key - lesion idx, value - volume in cm^3

    grouped_volumes = {}

    for time_stamp in longitudinal_volumes_array:
       for lesion_idx, volume in time_stamp.items():
           if lesion_idx not in grouped_volumes:
               grouped_volumes[lesion_idx] = []

           grouped_volumes[lesion_idx].append(volume)

    # Sort the volumes by lesion index
    sorted_grouped_volumes = dict(sorted(grouped_volumes.items()))

    return sorted_grouped_volumes
"""
this function gets a list of a lesion's volume changes over time and 
checks the difference from last 2 consecutive scans"""
def check_lesion_growth_from_last_scan(lesion_volumes):
    text = ""
    if not lesion_volumes:
        return "No volume data available for the lesion."
    cur_volume = lesion_volumes[-1]
    if len(lesion_volumes)<2:
        return "No volume data available for the lesion from previous scans"
    
    prev_volume = lesion_volumes[-2]
    vol_change_type =""
    change_percentage = round(abs((cur_volume/prev_volume)-1)*100)
    if cur_volume<prev_volume:
        text += f"Lesion volume has decreased by {change_percentage}% from previous scan to current scan. "
    elif cur_volume>prev_volume:
        text += f"Lesion volume has decreased by {change_percentage}% from previous scan to current scan. "
    else:
        text += "No change in lesion volume from previous scan to current scan. "

    return text

"""
this function gets a list of a lession's volume changes over time 
and checks if the volume increased/decreased
"""
def check_single_lesion_growth(vol_list,lesion_idx):
    lesion_volumes = 0
    if lesion_idx in vol_list:
        lesion_volumes = vol_list[lesion_idx]
    if not lesion_volumes:
        return "No volume data available for the lesion."
        
    increasing = decreasing = True

    text = check_lesion_growth_from_last_scan(lesion_volumes)

    for i in range(1, len(lesion_volumes)-1):
        if lesion_volumes[i] > lesion_volumes[i - 1]:
            decreasing = False
        elif lesion_volumes[i] < lesion_volumes[i - 1]:
            increasing = False
    change_percentage = round(abs((lesion_volumes[-1]/lesion_volumes[0])-1)*100)
    if increasing:
        text +=f"Volume consistently increased over time by {change_percentage}% from first scan to last scan."
    elif decreasing:
        text+= f"Volume consistently decreased over time by {change_percentage}% from first scan to last scan."
    else:
        text +=f"Volume shows both increases and decreases over time from first scan to last scan."
    return text

"""
this function adds text describing percentage of volume growth for each time stamp
"""
def lesion_growth_percentage(patient_partial_path,num_of_tumors):
    paragraph = []
    volumes_array = get_diff_in_total(generate_longitudinal_volumes_array(patient_partial_path)) # [total_vol_cm3, vol_percentage_diff, vol_cm3_diff]
    if len(volumes_array) < 2:
        return "Insufficient data for calculating percentage change."

    # Extract the first and last elements
    initial_value, _, _ = volumes_array[0]
    final_value, vol_percentage_diff, vol_cm3_diff = volumes_array[-1]

    # Determine the change type based on the sign of vol_percentage_diff
    change_type = "increased" if vol_percentage_diff >= 0 else "decreased"
    vol_percentage_diff_abs = round(abs((final_value / initial_value)-1) * 100)
    # Format the message based on the presence of a minus sign
    vol_diff_text = f"{vol_percentage_diff_abs}%"

    text = f"Tumor burden of {num_of_tumors} lesions {change_type} over time by {vol_diff_text}\n"
    paragraph.append(Paragraph(text))

    return paragraph



# vol_list = generate_volume_list_single_lesion("/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_")
# print(vol_list)
# check_single_lession_growth(vol_list,2)

