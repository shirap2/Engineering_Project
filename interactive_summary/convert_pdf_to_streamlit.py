import streamlit as st
from reportlab.platypus import Paragraph, Table, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import os
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT


def paragraph_to_html(paragraph):
    # Extract the paragraph style
    style = paragraph.style
    # font_name = style.fontName
    font_size = style.fontSize * 1.5
    text_color = style.textColor
    # alignment = style.alignment
    # if alignment == TA_CENTER:
    #     align = "center"
    # elif alignment == TA_JUSTIFY:
    #     align = "justify"
    # elif alignment == TA_LEFT:
    #     align = "left"
    # elif alignment == TA_RIGHT:
    #     align = "right"
    # else:
    #     align = "left"
    align = "left"

    # Construct the HTML string
    html_string = f"""
    <p style="
    font-size: {font_size}px;
    color: {text_color};
    text-align: {align};
    ">{paragraph.text}</p>
    """
    return html_string
# def paragraph_to_html(paragraph):
#     # Extract the paragraph style
#     style = paragraph.style
#     text_color = style.textColor
#     alignment = style.alignment
#
#     if alignment == TA_CENTER:
#         align = "center"
#     elif alignment == TA_JUSTIFY:
#         align = "justify"
#     elif alignment == TA_LEFT:
#         align = "left"
#     elif alignment == TA_RIGHT:
#         align = "right"
#     else:
#         align = "left"
#
#     # Construct the HTML string
#     html_string = f"""
#     <p style="
#     color: {text_color};
#     text-align: {align};
#     ">{paragraph.text}</p>
#     """
#     return html_string


def display_paragraph(paragraph):
    html_paragraph = paragraph_to_html(paragraph)
    st.markdown(html_paragraph, unsafe_allow_html=True)


def split_table(table):
    first_row = table._cellvalues[0]  # Get the first row of data
    remaining_table = table._cellvalues[1:]  # Table containing data except for the first row
    return first_row, remaining_table


def display_table(table):
    columns, data = split_table(table)
    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_image(image):
    # Since ReportLab's Image object holds a path to the image,
    # we can use this path to display the image in Streamlit
    image_path = image.filename
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.write(f"Image not found: {image_path}")


def display_spacer(spacer):
    st.write("\n\n\n\n")


def display_element(element):
    if isinstance(element, Paragraph):
        display_paragraph(element)
    elif isinstance(element, Table):
        display_table(element)
    elif isinstance(element, Image):
        display_image(element)
    elif isinstance(element, Spacer):
        display_spacer(element)
    else:
        st.write(f"Unsupported element: {type(element)}")

