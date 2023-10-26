
from volume_calculation import generate_longitudinal_volumes_array


def generate_volume_list_single_lesion():
    longitudinal_volumes_array = generate_longitudinal_volumes_array()  # returns sorted (by date) array of
    # dictionaries (one for each time stamp), key - lesion idx, value - volume in mm^3

    print(longitudinal_volumes_array)

    grouped_volumes = {}

    for time_stamp in longitudinal_volumes_array:
       for lesion_idx, volume in time_stamp.items():
           if lesion_idx not in grouped_volumes:
               grouped_volumes[lesion_idx] = []

           grouped_volumes[lesion_idx].append(volume)

    # Sort the volumes by lesion index
    sorted_grouped_volumes = dict(sorted(grouped_volumes.items()))

    return sorted_grouped_volumes


print(generate_volume_list_single_lesion())


"""
this fucntion gets a list of a lession's volume changes over time 
and checks if the volume increased/decreased
"""
def check_single_lession_growth(lession_volumes):
    if not lession_volumes:
        print("No volume data available for the lesion.")
        return
    


