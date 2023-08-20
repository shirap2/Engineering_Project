import numpy as np
from common_packages.MatchingGroupwisePackage import LongitLongitudinalLesionsSimple, LongitudinalLesion
from common_packages.MatchingPairsPackage import PwMatching
from common_packages.BaseClasses import Longit
from common_packages.LongGraphPackage import LoaderSimple
import json
import networkx as nx

EARLY_LAYER = 'early_layer'
EARLY_LABELS = 'early_labels'
LATER_LAYER = 'later_layer'
LATER_LABELS = 'later_labels'


class AllPairsMatchingAlgo:

    def __init__(self, pat_name: str, lesion_segm_pairs: list, time_indices: list = None, max_dilation: int = 5,
                 registration_folder: str = None, predicted_lesions: bool = False):
        """
        :param pat_name: patient name
        :param lesion_segm_pairs: a list: [[tumors_t1_reg_t2, tumors_t2], []] of N-1 pairs of np.arrays. Each pair is
         a registered labeled lesion segmentation and fix labeled lesion segmentation (of the same dimensions (shape)!).
         The class assumes that the voxel volume is 1x1x1 mm^3.
        :param time_indices: a list of N integers being the time indices of the scans. If None, set as [0,...,N-1].
        :param max_dilation: the number of iterations of the algorithm
        :param registration_folder: the path to the folder containing patients folders
        """
        self.MOVING = 0
        self.FIX = 1
        self.pat_name = pat_name
        #convert lesion_segm_pairs to np.unint8
        self.lesion_segm_pairs = [(lesions_pair[self.MOVING].astype(np.uint8), lesions_pair[self.FIX].astype(np.uint8)) for lesions_pair in lesion_segm_pairs]

        self.segm_shape = self.lesion_segm_pairs[0][self.FIX].shape
        if time_indices is None:
            self.num_of_timepoints = len(self.lesion_segm_pairs) + 1
            self.time_indices = list(range(self.num_of_timepoints))
        else:
            self.time_indices = time_indices
            self.num_of_timepoints = len(self.time_indices)
        self.lesion_list = self.get_lesions_names()
        self.max_dilation = max_dilation
        self.longit = None
        self.predicted_lesions = predicted_lesions
        if self.predicted_lesions:
            self.tumors_name = name.segm_name_pred
        else:
            self.tumors_name = name.segm_name_gt
        self.registration_folder = registration_folder

    def run(self):
        # consecutive matching
        loader = self.find_consecutive_edges()
        self.longit = LongitLongitudinalLesionsSimple(loader, patient_dates=[f"{i}" for i in range(self.num_of_timepoints)])
        self.longit.add_cc_attribute()
        self.longit.create_long_lesions_list()
        # non-consecutive matching
        self.find_skip_edges()

    def get_lesions_names(self):
        """Return the dictionary that maps time_points (aka 'layers') to the labels of the lesions in the scan at that
        time point"""

        layers2labels = dict()
        for i, pairs in enumerate(self.lesion_segm_pairs):
            # use the moving image! Some label of little lesion may get deleted in the registarion
            moved_image = pairs[self.MOVING]
            lbs = np.unique(moved_image)
            ly = self.time_indices[i]
            layers2labels.update({ly: [int(lb) for lb in lbs if lb > 0]})
        last_image = self.lesion_segm_pairs[-1][self.FIX]
        lbs = np.unique(last_image)
        layers2labels.update({self.time_indices[-1]: [int(lb) for lb in lbs if lb > 0]})
        return layers2labels

    def find_consecutive_edges(self):
        edges_greedy = []
        for pair_ind, lesions_pair in enumerate(self.lesion_segm_pairs):
            bl_layer = pair_ind
            fu_layer = pair_ind + 1
            edges_current = PwMatching.match_2_cases_v5(baseline_moved_labeled=lesions_pair[self.MOVING],
                                                        followup_labeled=lesions_pair[self.FIX],
                                                        voxelspacing=(1, 1, 1),
                                                        max_dilate_param=self.max_dilation)
            for edge in edges_current:
                n0 = int(edge[0]); n1 = int(edge[1])
                if n0 in self.lesion_list[bl_layer] and n1 in self.lesion_list[fu_layer]:
                    edges_greedy += [(f"{n0}_{bl_layer}", f"{n1}_{fu_layer}")]

        # prepare nodes:
        nodes = [f"{lb}_{layer}" for layer, labels in self.lesion_list.items() for lb in labels]
        ld = LoaderSimple(labels_list=nodes, edges_list=edges_greedy)
        return ld

    def lls_are_unmatchable(self, ll_pair, ll_graph: nx.Graph) -> bool:
        """
        Given a pair of longitudinal lesion (ll) and the longitudinal lesions graph, return True if the two lls are
        unmatchable: i.e.
            (1): they share a layer
            (2): they are one layer apart (-edge should have been found by the consecutive match)
            (3): they are more than 1.5 year apart.

        lls come from pw matching graph. Thus, ll's layers are continuous
        if first_layer(ll2) - last_layer(ll1) = 1, their distance match may be a consecutive edge.
        Assuming that the pw matching is very good, avoid adding consecutive edges in this step. Make the lls incompatible
        """
        ll1, ll2 = ll_pair
        ll1_layers_p = ll1.get_layers_presence()
        ll2_layers_p = ll2.get_layers_presence()

        lls_share_layers = np.any(ll1_layers_p[ll2_layers_p])
        if lls_share_layers:
            return True

        early_ll1 = ll_graph.nodes[ll1][EARLY_LAYER]
        later_ll1 = ll_graph.nodes[ll1][LATER_LAYER]
        early_ll2 = ll_graph.nodes[ll2][EARLY_LAYER]
        later_ll2 = ll_graph.nodes[ll2][LATER_LAYER]

        if later_ll1 < early_ll2:
            extreme_first = later_ll1
            extreme_last = early_ll2

        else:
            extreme_first = later_ll2
            extreme_last = early_ll1

        if extreme_last - extreme_first <= 1:
            return True

        patient_dates = get_patient_dates(self.pat_name)
        first_date = patient_dates[extreme_first]
        last_date = patient_dates[extreme_last]
        date_diff = (datetime.strptime(last_date, '%d_%m_%Y') - datetime.strptime(first_date, '%d_%m_%Y')).days
        if date_diff > 365 * 1.5:  # discard matching between two dates more than 1.5 year apart
            return True
        return False

    @staticmethod
    def add_extremes_attributes(graph: nx.Graph) -> nx.Graph:
        """
        :param graph: the ll_graph
        :return: ll_graph, adding to graph nodes (ll) their first and last layers and labels.
        """
        attr_dict = dict()
        for ll in graph.nodes():
            early_layer = ll.get_extreme_layer(earliest=True)
            later_layer = ll.get_extreme_layer(earliest=False)
            early_labels = ll.get_labels_in_extreme_layer(earliest=True)
            later_labels = ll.get_labels_in_extreme_layer(earliest=False)
            attr_dict.update({ll: {EARLY_LAYER: early_layer,
                                   EARLY_LABELS: early_labels,
                                   LATER_LAYER: later_layer,
                                   LATER_LABELS: later_labels}})

        nx.set_node_attributes(graph, attr_dict)
        return graph

    @staticmethod
    def load_candidate_labels_mask(labels_image_path: str, labels_list: list, ) -> np.array:
        """
        :param labels_image_path: the path of the labeled image
        :param labels_list: the labels of the image we are trying to match
        :return: the loaded image with the labels of labels_list only
        """
        image = load_nifti_data(labels_image_path)[0].astype(np.uint8)
        labeled_image = np.zeros_like(image)
        candidate_label_mask = np.isin(image, labels_list)
        labeled_image[candidate_label_mask] = image[candidate_label_mask]
        return labeled_image

    def find_skip_edges(self):
        """Add to the pairwise longit graph the skip edges"""
        not_complete_ll = [ll for ll in self.longit.get_long_lesions_list() if not ll.check_completeness()]
        ll_graph = nx.complete_graph(not_complete_ll)
        ll_graph = self.add_extremes_attributes(ll_graph)
        incompatible_lls = [ll_pair for ll_pair in ll_graph.edges() if self.lls_are_unmatchable(ll_pair, ll_graph)]
        ll_graph.remove_edges_from(incompatible_lls)
        if len(ll_graph.edges) == 0:
            return

        """
        Create a dictionary: {(layer_i, layer_j): ({lesions in layer_i}, {lesions in layer_j})}, such that the lesion pairs
        are candidates to be skip edges vertices
        """
        layers_pairs2labels_pairs = dict()

        for ll_pair in ll_graph.edges:
            ll1, ll2 = ll_pair
            early_ll1 = ll_graph.nodes[ll1][EARLY_LAYER]
            early_ll2 = ll_graph.nodes[ll2][EARLY_LAYER]

            if early_ll1 < early_ll2:
                ll_first = ll1; ll_last = ll2
            else:
                ll_first = ll2; ll_last = ll1

            layer_moving = ll_graph.nodes[ll_first][LATER_LAYER]
            labels_moving = ll_graph.nodes[ll_first][LATER_LABELS]
            layer_fix = ll_graph.nodes[ll_last][EARLY_LAYER]
            labels_fix = ll_graph.nodes[ll_last][EARLY_LABELS]
            layers_pair = (layer_moving, layer_fix)

            if layers_pair not in layers_pairs2labels_pairs:
                layers_pairs2labels_pairs.update({layers_pair: (set(), set())})

            for lb in labels_moving:
                layers_pairs2labels_pairs[layers_pair][self.MOVING].add(lb)
            for lb in labels_fix:
                layers_pairs2labels_pairs[layers_pair][self.FIX].add(lb)

        skip_edges_greedy = []
        for layers_pair, labels_pairs in layers_pairs2labels_pairs.items():
            init_layer, end_layer = layers_pair
            init_labels, end_labels = labels_pairs
            patient_dates = get_patient_dates(self.pat_name)
            init_date = patient_dates[init_layer]
            end_date = patient_dates[end_layer]

            init_layer_image_path = f"{self.registration_folder}/{self.pat_name}/{self.tumors_name(init_date, end_date)}"
            init_labeled_image = self.load_candidate_labels_mask(init_layer_image_path, list(init_labels))

            end_layer_image_path = f"{self.registration_folder}/{self.pat_name}/{self.tumors_name(end_date)}"
            end_labeled_image = self.load_candidate_labels_mask(end_layer_image_path, list(end_labels))

            skip_edges_current = PwMatching.match_2_cases_v5(baseline_moved_labeled=init_labeled_image,
                                                  followup_labeled=end_labeled_image, voxelspacing=(1,1,1),
                                                  max_dilate_param=self.max_dilation)

            skip_edges_greedy += [(f"{int(e[0])}_{init_layer}", f"{int(e[1])}_{end_layer}") for e in skip_edges_current]
        skip_edges = skip_edges_greedy
        loader = LoaderSimple(labels_list=list(self.longit.get_graph().nodes),
                              edges_list=list(self.longit.get_graph().edges) + skip_edges)
        self.longit = Longit(loader)

    def save_matching_graph(self, graph_path):
        """
        The matching is saved as {'nodes': vertices, 'edges': edges}, where both vertices and edges use the notation {lb}_{layer}
        """
        edges = list(self.longit.get_graph().edges())
        lab_list = list(self.longit.get_graph().nodes())
        graph_dict = {'nodes': lab_list, 'edges': edges}
        with open(graph_path, 'w') as f:
            json.dump(graph_dict, f)


def load_pairs(pat, reg_folder, predicted_lesion):
    pat_path = f"{reg_folder}/{pat}"
    dates = get_patient_dates(pat)
    if predicted_lesion:
        segm_name = name.segm_name_pred
    else:
        segm_name = name.segm_name_gt
    pairs_series = [(load_nifti_data(f"{pat_path}/{segm_name(dates[i], dates[i+1])}")[0],
                     load_nifti_data(f"{pat_path}/{segm_name(dates[i+1])}")[0])
                    for i in range(len(dates) - 1)]

    return pairs_series


def pat_matching(pat):
    # reg_folder = f"{path.PROJ_NAME}/registration/results/pairwise_affine_all"
    reg_folder = f"/cs/casmip/bennydv/brain_pipeline/pred_data/size_filtered/labeled_pairwise"
    lesion_segm_series = load_pairs(pat, reg_folder, predicted_lesion=True)
    print(pat)
    os.makedirs(f"{path.LESIONS_MATCH_RESULTS_DIR}/pred_segmentation_gw13/{pat}", exist_ok=True)
    m = AllPairsMatchingAlgo(pat_name=pat, lesion_segm_pairs=lesion_segm_series, max_dilation=10,
                             registration_folder=reg_folder, predicted_lesions=True)
    m.run()
    m.save_matching_graph(f"{path.LESIONS_MATCH_RESULTS_DIR}/pred_segmentation_gw13/{pat}/gw13.json")
    print(f"{pat} done")


if __name__ == "__main__":
    from general_utils import *
    from config import Name
    from multiprocessing import Pool
    name = Name()
    pats = [pat for pat in get_patients_list()]
    #pats = ["C_A_", "Z_Aa_"]
    # import pandas as pd
    # df = pd.read_csv('/cs/usr/bennydv/Desktop/freesurfer_segm.csv', index_col=0)
    # fs_cases = df.index.tolist()
    # for pat in pats:
    #     for date in get_patient_dates(pat):
    #         case_name = f"{pat}_{date.replace('_','-')}"
    #         if case_name not in fs_cases:
    #             print(f"{case_name} dn't have freesurfer")

        #pat_matching(pat)

    with Pool(10) as p:
      p.map(pat_matching, pats)
