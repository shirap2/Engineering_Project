
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
this function gets a list of a lession's volume changes over time 
and checks if the volume increased/decreased
"""
def check_single_lession_growth(vol_list,lession_idx):
    lession_volumes = vol_list[lession_idx]
    if not lession_volumes:
        return "No volume data available for the lesion."
        
    increasing = decreasing = True

    for i in range(1, len(lession_volumes)-1):
        if lession_volumes[i] > lession_volumes[i - 1]:
            decreasing = False
        elif lession_volumes[i] < lession_volumes[i - 1]:
            increasing = False
    text =""
    if increasing:
        text +="Volume consistently increased over time."
    elif decreasing:
        text+= "Volume consistently decreased over time."
    else:
        text +="Volume shows both increases and decreases over time."
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
    vol_percentage_diff_abs = int(abs((final_value - initial_value) / abs(initial_value)) * 100)
    # Format the message based on the presence of a minus sign
    vol_diff_text = f"{vol_percentage_diff_abs}%"

    text = f"Tumor burden of {num_of_tumors} lesions {change_type} over time by {vol_diff_text}\n"
    paragraph.append(Paragraph(text))

    return paragraph



# vol_list = generate_volume_list_single_lesion("/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_")
# print(vol_list)
# check_single_lession_growth(vol_list,2)

