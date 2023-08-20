from common_packages.Recist import *
import json
from general_utils import *
name = Name()

def some_fun():
    lone_lesions_table = "/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/lone_vert_data.xlsx"
    lesion_data_path = "/cs/casmip/bennydv/lungs_pipeline/lesions_matching/lesion_matching_database/lesions_data.xlsx"
    dates_ph = [None] * 12
    IS_FIRST = True
    all_df = pd.DataFrame()
    for pat in get_patients_list():
        ev = LoaderSimpleFromJson(
            f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json")
        longit = LongitClassification(ev, patient_name=pat)
        df = pd.read_excel(lesion_data_path, sheet_name=pat, index_col=0)
        longit.add_node_attribute_from_dict(
            attr_dict={key: round(val, 1) for key, val in dict(df["extr. diameter"]).items()},
            attr_name=NodeAttr.DIAMETER)
        longit.add_node_attribute_from_dict(
            attr_dict=dict(df["max lesion slice"]), attr_name=NodeAttr.SLICE)

        longit.classify_nodes()
        nodes2diam = nx.get_node_attributes(longit.get_graph(), NodeAttr.DIAMETER)
        nodes2presence = nx.get_node_attributes(longit.get_graph(), NodeAttr.PRESENCE_CLASS)
        nodes2label = nx.get_node_attributes(longit.get_graph(), NodeAttr.LABEL)
        nodes2layer = nx.get_node_attributes(longit.get_graph(), NodeAttr.LAYER)
        nodes2slice = nx.get_node_attributes(longit.get_graph(), NodeAttr.SLICE)

        lone_nodes = [n for n, pres in nodes2presence.items() if pres == NodesPresence.LONE]
        lone_nodes_diam_bigger_than_5 = [n for n in lone_nodes if nodes2diam[n] > 5]
        if len(lone_nodes_diam_bigger_than_5) == 0:
            continue
        header = dates_ph.copy()
        header[0] = pat
        for i, date in enumerate(get_patient_dates(pat)):
            header[i + 1] = date
        # pat_lones = [dates_ph.copy()]*(len(lone_nodes_diam_bigger_than_5) + 1)
        pat_lones = [header]
        for n in lone_nodes_diam_bigger_than_5:
            row = dates_ph.copy()
            row[nodes2layer[n] + 1] = f"{int(nodes2label[n])}, #{int(nodes2slice[n])}"
            pat_lones.append(row)
        pat_df = pd.DataFrame(data=pat_lones)
        if IS_FIRST:
            all_df = pat_df.copy()
            IS_FIRST = False
        else:
            all_df = pd.concat([all_df, pat_df], ignore_index=True)

    all_df.to_excel(lone_lesions_table)


def convert_json_format():
    all_jsons = glob.glob("/cs/usr/bennydv/Desktop/bennydv/liver_pipeline/lesions_matching/results/gt_segmentation_shalom_registration/*/long*json")
    for j in all_jsons:
        with open(j, 'r') as f:
            m = json.load(f)
        new_j = {"nodes": [], "edges": []}
        for i, layer in enumerate(m):
            layer_nodes = layer['nodes']
            new_j["nodes"] += [f"{n}_{i}" for n in layer_nodes]
            if i > 0:
                layer_edges = layer["edges"]
                new_j["edges"] += [[f"{e[0]}_{i-1}", f"{e[1]}_{i}"] for e in layer_edges]
        suffix = os.path.basename(j)
        new_j_name = j.replace("long_", "glong_")
        with open(new_j_name, 'w') as f:
            json.dump(new_j, f)


#convert_json_format()


# r = RecistSimulation(lesion_data_path="/cs/casmip/bennydv/lungs_pipeline/lesions_matching/lesion_matching_database/lesions_data.xlsx",
#                      pat2matching={pat: f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json" for pat in get_patients_list()})
#
# r.show_all()

ld = LoaderSimpleFromJson(f"/cs/usr/bennydv/Desktop/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original_corrected/C_A_glong_gt.json")
lg1 = LongitClassification(ld)
dr_2 = DrawerLabels(lg1)
dr_2.show_graph()

for pat in get_patients_list():
    # for pat in get_patients_list():
    # if pat == 'B_B_S_': continue
# #     #ld1 = LoaderSimpleWithMapping(f"")
#     ld = LoaderSimpleFromJson(f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/results/pred_segmentation_gw4_2/{pat}/gw4_2_intra=11_d=17.json")
#     ld1 = LoaderSimpleWithMapping(ld.get_nodes(), ld.get_edges(),
#                                   mapping_path=f"/cs/casmip/bennydv/lungs_pipeline/pred_data/size_filtered/labeled_no_reg/{pat}/mapping_list.json",
#                                   is_gt=False)
# #     lg1 = LongitClassification(ld1)
# #     dr_1 = DrawerClassification(lg1, attr_to_show=NodeAttr.CHANGES)
# #     dr_1.show_graph()
#     ldd = LoaderSimpleFromJson(f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json")
#     # ld2 = LoaderSimpleWithMapping(ldd.get_nodes(), ldd.get_edges(),
#     #                               mapping_path=f"/cs/casmip/bennydv/lungs_pipeline/pred_data/size_filtered/labeled_no_reg/{pat}/mapping_list.json",
#     #                               is_gt=True)
#     # lg2 = LongitClassification(ld2)
#     # dr_2 = DrawerClassification(lg2, attr_to_show=NodeAttr.CHANGES)
#     dr_2 = DrawerLabels(Longit(ldd))
#     dr_2.show_graph()
#     #l = LoaderEval_SkipEdgeHandlerSoft(gt_loader=ld2, pred_loader=ld1, patient_name=pat, patient_dates=get_patient_dates(pat))
    #dr_gt = DrawerEvalSourceGraphs(Longit(l), SourceGraph.GT)
    #dr_gt.show_graph()
    #dr_pred = DrawerEvalSourceGraphs(Longit(l), SourceGraph.PRED)
    #dr_pred.show_graph()
    #dr = DrawerEval(Longit(l), attr=EdgeAttr.DETECTION)
    #dr.show_graph()
    a = 1
# for p in get_patients_list(testset=True):
#     print(f"working on {p}")
#     if p == 'A_Ab_' or p == 'S_N_':
#         continue
#
#pat = "Z_Aa_"
# lesion_data_path="/cs/casmip/bennydv/lungs_pipeline/lesions_matching/lesion_matching_database/lesions_data.xlsx"
# for pat in get_patients_list():
#     #if pat != "S_N_": continue
#     ev = LoaderSimpleFromJson(f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json")
#     longit = LongitClassification(ev, patient_name=pat)
#     df = pd.read_excel(lesion_data_path, sheet_name=pat, index_col=0)
#     longit.add_node_attribute_from_dict(attr_dict={key: round(val, 1) for key, val in dict(df["extr. diameter"]).items()}, attr_name=NodeAttr.DIAMETER)
#     dr_gt = DrawerClassification(longit, attr_to_show=NodeAttr.PRESENCE_CLASS, attr_to_print=NodeAttr.DIAMETER)
#     dr_gt.show_graph()
#     a =1
# dr_pred = DrawerEvalSourceGraphs(longit, SourceGraph.PRED)
# dr_pred.show_graph()
# for pat in get_patients_list():
#     if pat != "M_G_": continue
#     ev = LoaderSimpleFromJson(f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json")
#     longit = LongitClassification(ev, patient_name=pat, patient_dates=get_patient_dates(pat))
#     #longit = Longit(ev)
#     longit.add_cc_attribute()
#     #dr_gt = DrawerLabels(longit, attr_to_print=NodeAttr.CC_INDEX)
#     #dr_gt.show_graph()
#     #ev = LoaderSimpleFromJson(
#     #    f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/results/gt_segmentation_gw4_2/{pat}/gw4_2_intra=11_d=17.json")
#     # longit = LongitClassification(ev, patient_name=pat)
#     #longit = Longit(ev)
#     #longit.add_cc_attribute()
#     #dr_gt = DrawerLabels(longit, attr_to_print=NodeAttr.CC_INDEX)
#     dr_gt= DrawerClassification(longit, attr_to_show=NodeAttr.CHANGES, attr_to_print=NodeAttr.LABEL)
#     dr_gt.show_graph(f"/cs/usr/bennydv/Desktop/gt_class_lungs/{pat}.jpg")
#     a = 1
#



for pat in get_patients_list():

    # # RunDrawingFromJsons(gt_path=f"/cs/casmip/bennydv/liver_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json",
    # #                     pred_path=f"/cs/casmip/bennydv/liver_pipeline/lesions_matching/results/gt_segmentation_gw2_steps/{pat}/gw_v2_init_d=20.json",
    # #                     eval_type=EvaluationType.SIMPLE,
    # #                     only_edge_detection=True, path_type=PathType.BACKWARD)
    # RunDrawingFromJsons(gt_path=f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json",
    #                     pred_path=f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/results/gt_segmentation_gw2/{pat}/gw_v2_init_d=13.json",
    #                     eval_type=EvaluationType.SIMPLE,
    #                    only_edge_detection=True, path_type=PathType.BACKWARD)
    # RunDrawingFromJsons(gt_path=f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json",
    #                     pred_path=f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/results/gt_segmentation_gw2/{pat}/gw2_distance_13.json",
    #                     eval_type=EvaluationType.SIMPLE,
    #                     only_edge_detection=True, path_type=PathType.BACKWARD)
    a = 1



# ev = LoaderEval_acceptCCEdges(ld2, ld)
# dr = DrawerLabelsSkipEdges(Longit(ev, patient_name=p, patient_dates=get_patient_dates(p)), NodeAttr.LABEL)
# dr.show_graph()
#dd = DrawerLabels(l)
#dd.show_graph()
#
#     path = f"/cs/casmip/bennydv/liver_pipeline/gt_data_testset_cropped/size_filtered/labeled_no_reg/{p}"
#     pat_dates = get_patient_dates(p, testset=True)
#     nifti_scans = [load_nifti_data(f"{path}/{name.segm_name_gt(d)}")[1] for d in pat_dates]
#     labeled_scans = [f.get_fdata() for f in nifti_scans]
#     #scans_resolution = [np.product(f.header.get_zooms()) for f in nifti_scans]
#
#
#     def extract_diam(scan, lb, resolution):
#         label_vol = np.sum(scan == lb) * resolution
#         diam = round((6*label_vol/np.pi)**(1/3))
#         return diam
#
#     def extract_z(scan, lb, extra):
#         z = np.median(np.where(scan == lb)[2]) + 1
#         return int(z)
#
#     l.add_node_attribute_from_scan(labeled_scans,
#                                    attr_name=NodeAttr.SLICE,
#                                    extracting_function=extract_z,
#                                    extra_arg=None)
#
#
#     dr = DrawerLabels(l, NodeAttr.SLICE)
#     dr.show_graph()

# dr = DrawerBiggestComponents(l, max_components=20, show_diameter=True, same_color_cc=True)
# dr.show_graph()

