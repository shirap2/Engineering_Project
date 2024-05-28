import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from volume_cal.volume_calculation import generate_longitudinal_volumes_array, get_diff_in_total
from reportlab.platypus import Paragraph


nltk.download('wordnet')

def check_lesion_growth_from_last_scan(lesion_volumes):
    """
    this function gets a list of a lesion's volume_cal changes over time and
    checks the difference from last 2 consecutive scans
    :param lesion_volumes:
    :return:
    """
    text = ""
    if not lesion_volumes:
        return "No volume_cal data available for the lesion."
    cur_volume = lesion_volumes[-1]
    if len(lesion_volumes) < 2:
        return "No volume_cal data available for the lesion from previous scan. "

    prev_volume = lesion_volumes[-2]
    vol_change_type = ""
    change_percentage = round(abs((cur_volume / prev_volume) - 1) * 100)
    if cur_volume < prev_volume:
        #The volume_cal has increased by {change_percentage}% between the previous scan and the current scan.
        text += f"The lesion volume_cal has increased by {change_percentage}% from previous scan to current scan. "
    elif cur_volume > prev_volume:
        text += f"Lesion volume_cal has decreased by {change_percentage}% from previous scan to current scan. "
    else:
        text += "No change in lesion volume_cal from previous scan to current scan. "

    return text

def check_single_lesion_growth(vol_list, lesion_idx):
    """
    this function gets a list of a lesion's volume_cal changes over time
    and checks if the volume_cal increased/decreased
    """
    lesion_volumes = [0]
    if lesion_idx in vol_list:
        lesion_volumes = vol_list[lesion_idx]
    if not lesion_volumes:
        return "No volume_cal data available for the lesion. "

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
this function adds text describing percentage of volume_cal growth for each time stamp
"""


def lesion_growth_percentage(patient_partial_path, num_of_tumors):
    paragraph = []
    volumes_array = get_diff_in_total(
        generate_longitudinal_volumes_array(patient_partial_path))  # [total_vol_cm3, vol_percentage_diff, vol_cm3_diff]
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


def format_lesion_data(lesion_list):
    data = []
    for lesion in lesion_list:
        pass

        

    return data


lesions_data = [
    {
        'id': 1,
        'changes': [
            {'timestamp': '2023-01-01', 'volume_cal': 2.0},
            {'timestamp': '2023-02-01', 'volume_cal': 2.5}
        ]
    },
    {
        'id': 2,
        'changes': [
            {'timestamp': '2023-01-01', 'volume_cal': 1.5},
            {'timestamp': '2023-02-01', 'volume_cal': 0.0}
        ]
    },
    {
        'id': 3,
        'changes': [
            {'timestamp': '2023-01-01', 'volume_cal': 3.0},
            {'timestamp': '2023-02-01', 'volume_cal': 3.5}
        ]
    }
]
      


def gen_summary_for_cc(lesion_data):
    pass


def gen_text_summaries(connected_components, lesions_data):
    summaries = []
    for cc in connected_components:
        summary = gen_summary_for_cc(cc, lesions_data)
        summaries.append(summary)
    return summaries