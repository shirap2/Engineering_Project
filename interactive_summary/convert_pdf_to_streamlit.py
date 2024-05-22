import streamlit as st
from reportlab.platypus import Paragraph, Table, Image
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd
import os

def display_paragraph(paragraph):
    st.write(paragraph.text)

def display_table(table):
    data = table._cellvalues
    df = pd.DataFrame(data)
    st.table(df)

def display_image(image):
    # Since ReportLab's Image object holds a path to the image,
    # we can use this path to display the image in Streamlit
    image_path = image.filename
    if os.path.exists(image_path):
        st.image(image_path)
    else:
        st.write(f"Image not found: {image_path}")

def display_element(element):
    if isinstance(element, Paragraph):
        display_paragraph(element)
    elif isinstance(element, Table):
        display_table(element)
    elif isinstance(element, Image):
        display_image(element)
    else:
        st.write(f"Unsupported element: {type(element)}")

# Example usage
elements = []

# Sample ReportLab elements
styles = getSampleStyleSheet()
elements.append(Paragraph("Hello, World!", styles['Title']))

data = [['00', '01', '02', '03', '04'],
        ['10', '11', '12', '13', '14'],
        ['20', '21', '22', '23', '24'],
        ['30', '31', '32', '33', '34']]

table = Table(data)
elements.append(table)

# Add an image element
image_path = 'path_to_image.png'  # Replace with the actual image path
image = Image(image_path)
elements.append(image)

# Display elements in Streamlit
for element in elements:
    display_element(element)
