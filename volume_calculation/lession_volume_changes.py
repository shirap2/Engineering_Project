
from volume_calculation.volume_calculation import generate_longitudinal_volumes_array


def generate_volume_list_single_lesion(patient_path):
    longitudinal_volumes_array = generate_longitudinal_volumes_array(patient_path)  # returns sorted (by date) array of
    # dictionaries (one for each time stamp), key - lesion idx, value - volume in mm^3

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
        print("No volume data available for the lesion.")
        return
    increasing = decreasing = True

    for i in range(1, len(lession_volumes)-1):
        if lession_volumes[i] > lession_volumes[i - 1]:
            decreasing = False
        elif lession_volumes[i] < lession_volumes[i - 1]:
            increasing = False
    text =""
    if increasing:
        text +="Volumes are consistently increasing over time."
    elif decreasing:
        text+= "Volumes are consistently decreasing over time."
    else:
        text +="Volumes show both increases and decreases over time."
    return text

# vol_list = generate_volume_list_single_lesion("/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_")
# print(vol_list)
# check_single_lession_growth(vol_list,2)

