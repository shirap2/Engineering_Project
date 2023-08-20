import colorsys

from common_packages.LongGraphPackage import LoaderSimpleWithMapping, LoaderEval_SkipEdgeHandlerSoft
from common_packages.BaseClasses import *
import json

class Cols:
    NUM_TP = '#TP se'
    TP_NUM_SKIPPED_VERT = 'TP se: missed lesions'
    NUM_UNSH = '#UNSHARED se'
    UNSH_NUM_FP_VERT = 'UNS se: fp lesions'
    NUM_FP = '#FP se'
    NUM_FP_PARALLEL = '#FP* se'
    FP_NUM_SKIPPED_VERT = 'FP* se: fp/missed lesions'

def num_skipped_layers(se):
    n_skipped_layers = int(se[1].split('_')[1]) - int(se[0].split('_')[1]) - 1
    return n_skipped_layers

def analyze_skip_edges_patient(pat, df):
    gt_graph_path = f"{path.LESIONS_MATCH_GT_ORIGINAL}/{pat}glong_gt.json"
    pred_graph_path = f"{path.LESIONS_MATCH_RESULTS_DIR}/pred_segmentation_gw13/{pat}/gw13.json"
    mapping_path = f"{path.PRED_FILTERED_LABELED_NO_REG}/{pat}/mapping_list.json"
    with open(gt_graph_path, 'r') as f:
        graph_gt = json.load(f)
    with open(pred_graph_path, 'r') as f:
        graph_pred = json.load(f)

    ld1 = LoaderSimpleWithMapping(label_list=graph_gt["nodes"], edges_list=graph_gt["edges"],
                                  mapping_path=mapping_path, is_gt=True)
    ld2 = LoaderSimpleWithMapping(label_list=graph_pred["nodes"], edges_list=graph_pred["edges"],
                                  mapping_path=mapping_path, is_gt=False)

    ld_ev = LoaderEval_SkipEdgeHandlerSoft(ld1, ld2, patient_name=pat)
    longit = Longit(ld_ev)
    nodes_attr = dict(longit.get_graph().nodes(data=True))
    all_edges_view = longit.get_graph().edges(data=True)
    all_edges = Longit.edgeview2dict(all_edges_view, nodes_attr)

    longit_pr = Longit(ld2)
    nodes_attr_pr = dict(longit_pr.get_graph().nodes(data=True))
    all_edges_view_pr = longit_pr.get_graph().edges(data=True)
    all_edges_pr = Longit.edgeview2dict(all_edges_view_pr, nodes_attr_pr)
    skip_edges_pr = [e for e, attr in all_edges_pr.items() if attr[EdgeAttr.IS_SKIP]]

    for se in skip_edges_pr:
        if all_edges[se][EdgeAttr.DETECTION] == EdgesDetect.TP:
            df.loc[pat, Cols.NUM_TP] += 1
            df.loc[pat, Cols.TP_NUM_SKIPPED_VERT] += num_skipped_layers(se)

        elif all_edges[se][EdgeAttr.DETECTION] == EdgesDetect.UNSHARED:
            df.loc[pat, Cols.NUM_UNSH] += 1
            det_n0 = nodes_attr[se[0]][NodeAttr.DETECTION]
            det_n1 = nodes_attr[se[1]][NodeAttr.DETECTION]
            if det_n0 == NodesDetect.TP or det_n1 == NodesDetect.TP:
                df.loc[pat, Cols.UNSH_NUM_FP_VERT] += 1
            else:
                df.loc[pat, Cols.UNSH_NUM_FP_VERT] += 2

        elif all_edges[se][EdgeAttr.DETECTION] == EdgesDetect.FP:
            df.loc[pat, Cols.NUM_FP] += 1
            if all_edges[se][EdgeAttr.IS_SKIP_EDGE_PATH] == EdgesInSkipPath.PRED:
                df.loc[pat, Cols.NUM_FP_PARALLEL] += 1
                df.loc[pat, Cols.FP_NUM_SKIPPED_VERT] += num_skipped_layers(se)


def analyze_skip_edges():

    rows = get_patients_list()
    cols = [Cols.NUM_TP, Cols.TP_NUM_SKIPPED_VERT, Cols.NUM_UNSH, Cols.UNSH_NUM_FP_VERT, Cols.NUM_FP, Cols.NUM_FP_PARALLEL, Cols.FP_NUM_SKIPPED_VERT]
    data = np.zeros((len(rows), len(cols)))
    df = pd.DataFrame(data=data, index=rows, columns=cols)
    for pat in get_patients_list():
        if pat == 'B_B_S_': continue
        analyze_skip_edges_patient(pat, df)
    df.to_excel(f"{path.LESIONS_MATCHING}/lesion_matching_database_nnunet/skip_edges_stat.xlsx")



if __name__ == "__main__":
    from config import *
    from general_utils import *
    name = Name()
    path = Path()
    analyze_skip_edges()