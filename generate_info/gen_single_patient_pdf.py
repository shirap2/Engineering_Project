from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os
from volume_calculation.volume_calculation import get_diff_in_total, generate_longitudinal_volumes_array

def get_file_title():
    return []


def get_lesion_counter_and_classifier_table():
    return []


def edit_volume_data_to_str(data: list):
    total_vol_mm3, vol_percentage_diff, vol_mm3_diff = data
    diff_is_positive = (vol_percentage_diff > 0)
    
    total_vol_mm3 = str(round(total_vol_mm3, 2)) + " [mm³]"
    vol_percentage_diff = str(round(vol_percentage_diff, 2)) + "%"
    vol_mm3_diff = str(round(vol_mm3_diff, 2)) + " [mm³]"

    if diff_is_positive:
        vol_percentage_diff = "+" + vol_percentage_diff
        vol_mm3_diff = "+" + vol_mm3_diff

    return [total_vol_mm3, vol_percentage_diff, vol_mm3_diff]


def get_volume_changes_per_time_table():

    diff_in_total = get_diff_in_total(generate_longitudinal_volumes_array()) # [total_vol_mm3, vol_percentage_diff, vol_mm3_diff]

    table_data = [["Time Stamp", "Total Volume [mm³]", "Volume Difference Percentage", "Volume Difference [mm³]"]]

    for idx, data in enumerate(diff_in_total):
        data = edit_volume_data_to_str(data)
        table_data.append([idx] + data)

    # Create the table and apply styles
    table = Table(table_data)
    table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                            ('GRID', (0, 0), (-1, -1), 1, colors.black)]))
    
    styles = getSampleStyleSheet()
    para_style = styles['Normal']
    table_title = "Tracking the Changes in the Total Volume of the Tumors From One Scan to the Previous One"
    table_title = Paragraph(table_title, style=para_style)

    return [table_title, table]


def create_pdf_file(pdf_name: str):

    if os.path.exists(pdf_name):
        os.remove(pdf_name)

    doc = SimpleDocTemplate(pdf_name)
    elements = []

    elements += get_file_title()

    elements += get_lesion_counter_and_classifier_table()

    elements += get_volume_changes_per_time_table()


    doc.build(elements)


create_pdf_file("a.pdf")