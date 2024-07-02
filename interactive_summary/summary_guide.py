import streamlit as st
from PIL import Image

def summary_guide_display():
    st.title('User Guide')

    st.markdown("### Lesion Names:")
    st.markdown("""
    Note that each lesion might appear in more than one scan over time. Additionally, a lesion might split into multiple lesions at some point, or it might result from the merging of several lesions. Due to these reasons, a lesion name consists of two parts:

    1. **Index Number**: This number represents the lesion's index over time and different behaviors (merging, splitting, etc.). This index is common to the entire connected component.
    2. **Identifier Letter**: This letter identifies the specific lesion in the scan.
    """)

    st.markdown("### Lesion Segmentation Map")
    st.markdown("""Now that you have all this information about the lesions,
     how can you connect the lesion information to the lesion segmentations on the scan?""")
    st.markdown("""""")
    st.markdown("""Use the Lesion Segmentation Map provided in the app to map the lesion name from the summary to its appearance in the segmented scan opened threw the app.
    You can locate the lesion on the scan using the lesion's color in the segmentation (in Fig 1: dark blue) and identifying the slice number where the lesion appears largest (in Fig 1: see in red).
    """)

    image = Image.open('interactive_summary/graph_guide_fig1.png')
    st.image(image, caption='Figure 1', use_column_width=True)

    st.markdown("### Lesion Patterns:")
    st.markdown("""
    There are six types of lesion patterns:

    1. **New** – It first appeared in this scan, and if the current scan isn’t the last one, it continues to appear in the next scan.
    2. **Disappeared** – It appeared in the previous scan but does not appear in the next scan.
    3. **Lone** – It did not appear in the previous scan, and it does not appear in the next scan.
    4. **Merged** – It used to be several lesions in the previous scan, and now they have merged into a single lesion (mostly due to the enlargement of the lesions).
    5. **Split** – It used to be a single lesion in the previous scan, and it has split into several lesions in the current scan.
    6. **Complex** – A combination of merging and splitting.
    """)

    st.markdown("### Non-Consecutive Lesions:")
    st.markdown("""
    An algorithm developed at the CASMIP lab identifies lesions that were not segmented by the radiologist by mistake. This mostly happens when a lesion appears in scan x, does not appear in scan x+1, and reappears in scan x+2. If the lesions in scans x and x+2 have properties indicating they are actually the same lesion, a decision is made that the lesion also existed in scan x+1 but was not found due to human error or scanning difficulties.

    Practically, this means:

    - We do not have the lesion segmentation, so it wouldn’t be found in the segmented scans and the lesion-segmentation map.
    - We do not have the lesion segmentation, so we do not know the volume of the lesion in the missing scan and the related volume changes.
    """)

    st.markdown("### Graphs: ")
    image = Image.open('interactive_summary/graph_guide_fig2.png')
    st.image(image, caption='Figure 2', use_column_width=True)
    st.markdown("""
    
    - Each graph represents one connected component.
    - Each circle represents a lesion, labeled by its name (in red, Fig 2).
    - The lesion's volume is printed above it if it is known (in blue, Fig 2). In non-consecutive cases, "doesn’t appear" will be printed (in orange, Fig 3).
    - Each column of the graph represents a single scan, with its date written at the bottom (in purple, Fig2).
    - The arrows represent the change in volume from one lesion to another, and the bottom lines represent the total volume change from one scan to the next. They are marked green when the change is negative (the lesions got smaller) and red otherwise.
    - For more than five scans, we split the graph into rows with an intersection of one time stamp between the rows (in pink, Fig 3).
    """)

    image = Image.open('interactive_summary/graph_guide_fig3.png')
    st.image(image, caption='Figure 3', use_column_width=True)

