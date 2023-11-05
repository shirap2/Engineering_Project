from common_packages.LongGraphPackage import LoaderSimpleFromJson
from common_packages.LongGraphClassification import LongitClassification
from drawer_single_lesion_graph import DrawerLabelsAndLabeledEdges
from reportlab.platypus import Image

def get_nodes_graph_image(image_path : str, partial_patient_path : str, ld):

    lg1 = LongitClassification(ld)
    dr_2 = DrawerLabelsAndLabeledEdges(lg1, partial_patient_path, ld)
    dr_2.show_graph(image_path)

    graph = Image(image_path, height=300, width=400)

    return [graph]

scan_name = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/A_W_glong_gt.json"
partial_patient_path = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_W_"
ld = LoaderSimpleFromJson(scan_name)
get_nodes_graph_image("single_labeled_lesion_graph.png", partial_patient_path, ld)
