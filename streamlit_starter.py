import streamlit as st


def get_num_of_scans_for_patient(patient_name):
    """
    this function checks how many scans are available for the patient
    :return: number of scans available
    """
    return 8


def main():

    st.sidebar.header("Patient Name", divider='blue')
    option_name = st.sidebar.selectbox(
        'Choose a patient to display',
        ('A. S. H.', 'A. S. W.', 'A. W.')
    )

    st.write('# You selected:', option_name)
    # Display image based on the selected option
    if option_name == 'A. S. H.':
        st.image('input/graph_images/A_S_H_graph_image.png', caption='Graph Image for A.S.H.', use_column_width=True)
    elif option_name == 'A. S. W.':
        st.image('input/graph_images/A_S_S_graph_image.png', caption='Graph Image for A.S.W.', use_column_width=True)
    elif option_name == 'A. W.':
        st.image('input/graph_images/A_W_graph_image.png', caption='Graph Image for A.W.', use_column_width=True)
    st.sidebar.header("Filter Scans", divider='blue')

    # st.sidebar.header("Number of Scans")

    num_scans_available = get_num_of_scans_for_patient(option_name)
    selected_num_scans = range_slider(num_scans_available)

    # st.sidebar.header("Number of Overlapped Scans")
    overlap_num = st.sidebar.slider("Select the number of overlapped scans to display in each row",
                                    min_value=0, max_value=3)
    st.sidebar.write(f"""<div style="color: black; font-size: 14px;"> <b>Selected: 
    {selected_num_scans} scans to display with overlap of {overlap_num} scans per row </b>
    </div>""", unsafe_allow_html=True)

    st.sidebar.header("Filter by Type of Lesion", divider='blue')
    # st.sidebar.header("Lesion Volume Filter")

    threshold_size = st.sidebar.slider("Select lesion volume threshold", min_value=0, max_value=50)

    # st.sidebar.header("Lesion Volume Changes Threshold (%)")
    vol_change_percentage = st.sidebar.slider("Select lesion growth percentage threshold", min_value=0, max_value=100,step=10)
    st.sidebar.write(f"""
        <div style="color: black; font-size: 14px;margin-bottom: 35px"> 
        <b> Displaying lesions that are larger than: {threshold_size} [cm^3]  and 
        with a volume change of at least {vol_change_percentage} % from previous scan </b>
        </div>
        """, unsafe_allow_html=True)

    add_buttons()

def add_buttons():
    # session_state = SessionState()
    if st.sidebar.button("Disappeared Lesions"):
        st.write("displaying lesions that have disappeared")
    if st.sidebar.button("New Lesions"):
        st.write("displaying new lesions")

    if st.sidebar.button("Lesions that didnt appear in segmentation"):
        st.write("displaying lesions didnt appear in segmentation")
    if st.sidebar.button("Unusual Lesions"):
        st.write("displaying lesions that are unusual")


def range_slider(range_size):
    selected_value = st.sidebar.slider('Select the number of longitudinal scans to display', min_value=0, max_value=range_size,
                                       value=range_size // 2)
    return selected_value


def display_content(file_path):
    st.title("Main Section")
    # Load and display images or text based on user settings
    if file_path:
        # If file is uploaded, display its content
        # with open(file_path, "rb") as file:
        #     file_content = file.read()
        st.image(file_path, caption='Uploaded Image')

        # You can add more content here based on user settings

    else:
        st.write("Upload a file to display content")


if __name__ == "__main__":
    main()
