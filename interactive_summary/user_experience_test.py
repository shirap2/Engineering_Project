import os

import streamlit as st
import pandas as pd


def render_feedback_form(patient_name):
    st.subheader("User Evaluation")

    # Collect user information
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown("**<span style='font-size:20px;'>Please provide your feedback on the following aspects:</span>**",
                unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .stSlider > div {
            font-size: 20px;  /* Increase font size of the sliders */
        }
        .css-1aumxhk, .css-1aumxhk .stTextInput, .css-1aumxhk .stTextInput div {
            font-size: 20px;  /* Increase font size of text inputs */
        }
        .css-1aumxhk .stButton button {
            font-size: 20px;  /* Increase font size of buttons */
        }
        .css-1aumxhk {
            line-height: 2.8;  /* Increase spacing between lines */
        }
        </style>
        """, unsafe_allow_html=True
    )
    tables_feedback = st.select_slider(
        "Did the tables effectively display the patient's condition?",
        options=["Not at all", "Partially", "Somewhat", "Mostly", "Completely"]
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing

    text_feedback = st.select_slider(
        "Were the textual summaries clear and understandable?",
        options=["Not at all", "Partially", "Somewhat", "Mostly", "Completely"]
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing

    graphs_feedback = st.select_slider(
        "Were the visual graphs easy to understand?",
        options=["Not at all", "Partially", "Somewhat", "Mostly", "Completely"]
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing

    lesion_names_feedback = st.select_slider(
        "Were the names of the lesions easy to follow?",
        options=["Not at all", "Partially", "Somewhat", "Mostly", "Completely"]
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing

    consistency_feedback = st.select_slider(
        "Were the textual summaries consistent with the visual graphs?",
        options=["Not at all", "Partially", "Somewhat", "Mostly", "Completely"]
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing

    overall_feedback = st.select_slider(
        "Did the combination of graphs, tables, and textual summary provide a cohesive and informative picture?",
        options=["Not at all", "Partially", "Somewhat", "Mostly", "Completely"]
    )
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing

    comments = st.text_area("Additional comments")
    st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing

    # Render star rating for satisfaction
    render_satisfaction_rating()

    # Submit feedback button
    if st.button("Submit Feedback"):
        save_feedback(patient_name, name, email,
                      tables_feedback, text_feedback, consistency_feedback, graphs_feedback, lesion_names_feedback,
                      overall_feedback, comments)
        st.success("Thank you for your feedback!")


# Function to render star rating for satisfaction
def render_satisfaction_rating():
    st.write("How satisfied are you with the app?")
    st.markdown(
        """
        <style>
        .stars {
            display: flex;
            flex-direction: row-reverse; /* Stars fill from right to left */
            justify-content: flex-end;
        }
        .stars input[type="radio"] {
            display: none;
        }
        .stars label {
            font-size: 40px;
            color: #cccccc;
            cursor: pointer;
        }
        .stars input[type="radio"]:checked ~ label {
            color: #ffc107; /* Yellow color for selected star */
        }
        </style>

        <div class="stars">
            <input type="radio" id="star5" name="rating" value="5" />
            <label for="star5">&#9733;</label>
            <input type="radio" id="star4" name="rating" value="4" />
            <label for="star4">&#9733;</label>
            <input type="radio" id="star3" name="rating" value="3" />
            <label for="star3">&#9733;</label>
            <input type="radio" id="star2" name="rating" value="2" />
            <label for="star2">&#9733;</label>
            <input type="radio" id="star1" name="rating" value="1" />
            <label for="star1">&#9733;</label>
        </div>
        """,
        unsafe_allow_html=True
    )


# Function to save feedback to CSV
def save_feedback(patient_name, name, email,
                  tables_feedback, text_feedback, consistency_feedback, graphs_feedback, lesion_names_feedback,
                  overall_feedback, comments):
    feedback_df = pd.DataFrame({
        'Name': [name],
        'Email': [email],
        'Tables Feedback': [tables_feedback],
        'Textual Summary Feedback': [text_feedback],
        'Consistency Feedback': [consistency_feedback],
        'Graphs Feedback': [graphs_feedback],
        'Lesion Names Feedback': [lesion_names_feedback],
        'Overall Feedback': [overall_feedback],
        'Comments': [comments]
    })

    # Save to CSV file
    feedback_file = f'{str(patient_name)}_feedback.csv'
    feedback_df.to_csv(feedback_file, mode='a', header=not os.path.exists(feedback_file), index=False)

