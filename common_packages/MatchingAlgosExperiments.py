import os

import numpy as np
import pandas as pd
from skimage.segmentation import expand_labels
from scipy import ndimage
from common_packages.old_classes.Old_LongGraphPackage import LoaderSimplePairwise
from common_packages.MatchingGroupwisePackage import LongitLongitudinalLesionsSimple, LongitudinalLesion
from common_packages.MatchingPairsPackage import PwMatching
from common_packages.BaseClasses import Longit, EdgeAttr, EdgesDetect, EdgesInSkipPath
from common_packages.LongGraphPackage import LoaderSimple, LoaderSimpleFromJson, LoaderEval_SkipEdgeHandlerSoft
from datetime import datetime
import json
import networkx as nx


class LesionAttr:
    REG_VOLUME = 'volume_cal'


class Lesion:
    def __init__(self, label=None, layer=None, lb_layer=None):
        if label is None or layer is None:
            label, layer = lb_layer.split('_')
        self._label = int(label)
        self._layer = int(layer)
        self._volume = None
        self._id = None

    def label(self):
        return self._label

    def layer(self):
        return self._layer

    def name(self):
        return f"{self._label}_{self._layer}"

    def set_id(self, id_num: int):
        self._id = id_num

    def get_id(self):
        if self._id is not None:
            return self._id
        else:
            raise ValueError(f"{self.name()} has no id")

    def set(self, attr, value):
        if attr == LesionAttr.REG_VOLUME:
            if self._volume is not None:
                raise ValueError(f"{self.name()}: {attr} has already a value!")
            self._volume = value
        else:
            raise ValueError(f"Unknown attribute {attr}")

    def get(self, attr):
        if attr == LesionAttr.REG_VOLUME:
            return self._volume
        else:
            raise ValueError(f"{self.name()}: Unknown attribute {attr}")

    def __hash__(self):
        return self._label + self._layer*1000


class LesionList(list):
    """A container of elements of type Lesion"""
    def __init__(self):
        super().__init__()
        self._name2les = dict()
        self._ind2les = dict()
        self._lock = False

    def append(self, les: Lesion):
        if self._lock:
            raise ValueError("Cannot append after locking!")
        curr_id = len(self) + 1
        les.set_id(curr_id)
        super().append(les)
        return curr_id

    def lock_list(self):
        """When the list is complete, must be locked to search it"""
        if self._lock:
            return
        self._lock = True
        self._name2les = {les.name(): les for les in self}
        self._ind2les = {les.get_id(): les for les in self}

    def find_by_name(self, name: str) -> Lesion:
        if not self._lock:
            raise ValueError("LesionList must be locked first: use lock_list")
        return self._name2les[name]

    def find_by_id(self, id_: int) -> Lesion:
        if not self._lock:
            raise ValueError("LesionList must be locked first: use lock_list")
        return self._ind2les[id_]

    def get_all_ids(self):
        if not self._lock:
            raise ValueError("LesionList must be locked first: use lock_list")
        return list(self._ind2les.keys())

    def get_all_names(self):
        if not self._lock:
            raise ValueError("LesionList must be locked first: use lock_list")
        return list(self._name2les.keys())


class LoaderPairwiseFromDf(LoaderSimplePairwise):
    """Load matches from pandas dataframes. The dataframe cols and index are {lb}_{layer}.
    The matrix is an upper triangular such that only forward matching are shown. Cell elements are the number of iterations
    of the algo in which the match was found"""
    def __init__(self, df_path=None, df=None, greedy=True, pat=None):
        self.pat = pat
        if df_path is not None:
            self.df = pd.read_excel(df_path, index_col=0)
        elif df is not None:
            self.df = df
        else:
            raise ValueError("Initialization error")

        lesion_list = LesionList()
        _ = [lesion_list.append(Lesion(lb_layer=l)) for l in self.df.index]
        lesion_list.lock_list()

        self.lesion_list = lesion_list
        if greedy:
           self.pw_edges = self.get_confidence_pairwise_matches()
           #self.pw_edges = self.get_greedy_pairwise_matches()
        else:
            self.pw_edges = self.get_pairwise_matches()

        # self.label_list = lesion_list.get_all_names()
        # self.edges_list = pw_edges
        super().__init__(labels_list=self.lesion_list.get_all_names(), edges_list=self.pw_edges)

    def get_dilation_table(self):
        return self.df

    def get_pairwise_matches(self):
        """
        for each curr_les, check  the next_layer_lesions: a list of candidate Lesions that can be matched with curr_les
        :return: a list of edges: [[curr_les, Lesion],..]. Edge condition: some match has been found,
        i.e: df[curr_les, Lesion] > 0
        """
        pw_edges = []
        les2layer = {les: les.layer() for les in self.lesion_list}
        for curr_les in self.lesion_list:
            layer = curr_les.layer()
            next_layer_les = [next_les for next_les, ly in les2layer.items() if ly == layer + 1]
            if len(next_layer_les) == 0:
                continue
            pw_edges += [(curr_les.name(), match_les.name()) for match_les in next_layer_les
                         if self.df.loc[curr_les.name(), match_les.name()] > 0]
        return pw_edges

    def get_greedy_pairwise_matches(self):
        """
        """
        pw_edges = []
        les2layer = {les: les.layer() for les in self.lesion_list}
        layers_list = list(set(les2layer.values()))
        for curr_layer, next_layer in zip(layers_list[:-1], layers_list[1:]):
            curr_layer_les = [curr_les.name() for curr_les, ly in les2layer.items() if ly == curr_layer]
            next_layer_les = [next_les.name() for next_les, ly in les2layer.items() if ly == next_layer]

            if len(curr_layer_les) == 0 or len(next_layer_les) == 0:
                continue
            pw_df = self.df.loc[curr_layer_les, next_layer_les].copy()
            pw_series = pw_df.stack()
            max_confidence = pw_df.max().max()
            for conf in range(max_confidence, 0, -1):
                current_edges = pw_series[(pw_series == conf)].index.tolist()
                if len(current_edges) == 0:
                    continue
                pw_edges += current_edges
                pw_df = self.delete_matched_lesions(pw_df, current_edges)
                pw_series = pw_df.stack()
        return pw_edges

    def get_confidence_pairwise_matches(self, save=False):
        """
        :return:
        """
        CONFIDENCE = 'confidence'
        edges2confidence = dict()
        for les in self.lesion_list.get_all_names():
            les_df = self.df.loc[les, (self.df.loc[les, :]>0)]
            les_confidence = les_df.values.tolist()
            les_matches = les_df.index.tolist()
            edges2confidence.update({(les, m_les): m_conf for m_les, m_conf in zip(les_matches, les_confidence)})

        ld = LoaderSimple(labels_list=self.lesion_list.get_all_names(), edges_list=list(edges2confidence.keys()))
        longit = Longit(ld)
        longit.add_edge_attribute_from_dict(attr_dict=edges2confidence, attr_name=CONFIDENCE)
        graph = longit.get_graph()

        skip_edges = [e for e, is_skip in nx.get_edge_attributes(graph, EdgeAttr.IS_SKIP).items() if is_skip]
        edges_consec2confidence = {e: conf for e, conf in edges2confidence.items() if e not in skip_edges}

        ld_p = LoaderSimplePairwise(labels_list=self.lesion_list.get_all_names(), edges_list=list(edges2confidence.keys()))
        longit_p = Longit(ld_p)
        graph_p = longit_p.make_graph_directed(longit_p.get_graph())

        for s_e in skip_edges:
            source = s_e[0]
            target = s_e[1]
            consecutive_paths = list(nx.all_simple_edge_paths(graph_p, source, target, cutoff=longit.get_num_of_layers()))
            if len(consecutive_paths) == 0:
                continue
            s_e_confidence = graph.edges[s_e][CONFIDENCE]
            consecutive_paths_edges = {e for p in consecutive_paths for e in p}
            # skip_edges_inside_ll.append(s_e)
            for path_e in consecutive_paths_edges:
                # add s_e confidence
                # edges_consec2confidence[path_e] += s_e_confidence
                # OR take maximum
                edges_consec2confidence[path_e] = max([s_e_confidence, edges_consec2confidence[path_e]])

        longit_p.add_edge_attribute_from_dict(attr_dict=edges_consec2confidence, attr_name=CONFIDENCE)
        graph_consecutive = longit_p.make_graph_directed(longit_p.get_graph())

        for node in graph_consecutive.nodes():
            # get adjacent edges, divided in IN and OUT
            in_edges = graph_consecutive.in_edges(node)
            self.delete_suboptimal_edges(graph=graph_consecutive, edges2confidence=edges_consec2confidence,
                                         edges_to_check=in_edges)

            out_edges = graph_consecutive.out_edges(node)
            self.delete_suboptimal_edges(graph=graph_consecutive, edges2confidence=edges_consec2confidence,
                                         edges_to_check=out_edges)

        if save:
            save_path = f"/cs/casmip/bennydv/liver_pipeline/lesions_matching/results/gt_segmentation_gw8/{self.pat}/edge_conf_8_1.json"
            print(save_path)
            with open(save_path, 'w') as f:
                dict_edges = nx.get_edge_attributes(graph_consecutive, CONFIDENCE)
                tuples = list(zip(dict_edges.keys(), dict_edges.values()))
                json.dump(tuples, f)

        return list(graph_consecutive.edges())

    @staticmethod
    def delete_suboptimal_edges(graph: nx.DiGraph, edges2confidence: dict, edges_to_check: list):
        """
        :param graph: a nx graph
        :param edges2confidence: a dictionary {edge: confidence (int)}
        :param edges_to_check: a list of edges.
        :return: delete from graph all the edges in edges_to_check whose confidence is less than the maximum
        """
        if len(edges_to_check) == 0:
            return
        max_conf = max([edges2confidence[e] for e in edges_to_check])
        suboptimal_edges = [e for e in edges_to_check if edges2confidence[e] != max_conf]
        graph.remove_edges_from(suboptimal_edges)

    @staticmethod
    def delete_matched_lesions(df, edges):
        """Replace the values in the columns/rows of the matched lesions (appearing in edges) with -1.
        In this way we perform a greedy match choice"""
        rows_names = list({e[0] for e in edges})
        cols_names = list({e[1] for e in edges})
        df.loc[rows_names, :] = -1
        df.loc[:, cols_names] = -1
        return df


class MatchingAlgo:
    """
    This class implements a multi scan version of pairwise match_2_cases_v5
    """
    def __init__(self, pat_name: str, lesion_segm_series: list = None, time_indices: list = None, max_dilation: int = 5):
        """
        :param pat_name: patient name
        :param lesion_segm_series: a list of N registered labeled lesion segmentations (of the same size!).
            The class assumes that the voxel volume_cal is 1x1x1 mm^3
        :param time_indices: a list of N integers being the time indices of the scans. If None, set as list indices.
        :param max_dilation: the number of iterations of the algorithm
        """
        if lesion_segm_series is None:
            return
        self.pat_name = pat_name
        self.lesion_segm_series = [lesions.astype(np.uint8) for lesions in lesion_segm_series]

        self.segm_shape = self.lesion_segm_series[0].shape
        if time_indices is None:
            self.num_of_timepoints = len(self.lesion_segm_series)
            self.time_indices = list(range(self.num_of_timepoints))
        else:
            self.time_indices = time_indices
            self.num_of_timepoints = len(self.time_indices)
        self.max_dilation = max_dilation
        self.lesion_list, self.lesion_id_segm_series = self.assign_lesions_data()

        num_of_lesions = len(self.lesion_list)
        self.adjacency_matrix = np.zeros((num_of_lesions, num_of_lesions), dtype=np.int8)
        self.dilation_table = None
        self.longit = None

    @staticmethod
    def lls_are_unmatchable(ll1: LongitudinalLesion, ll2: LongitudinalLesion) -> bool:
        """
        lls come from pw matching graph. Thus, ll's layers are continuous
        if first_layer(ll2) - last_layer(ll1) = 1, their distance match may be a consecutive edge.
        Assuming that the pw matching is very good, avoid adding consecutive edges in this step. Make the lls incompatible
        """
        ll1_layers_p = ll1.get_layers_presence()
        ll2_layers_p = ll2.get_layers_presence()

        lls_share_layers = np.any(ll1_layers_p[ll2_layers_p])
        if lls_share_layers:
            return True

        early1 = ll1.get_extreme_layer(earliest=True)
        early2 = ll2.get_extreme_layer(earliest=True)

        if early1 < early2:
            return not (early2 - ll1.get_extreme_layer(earliest=False) > 1)
        if early2 < early1:
            return not (early1 - ll2.get_extreme_layer(earliest=False) > 1)

    def lls_extreems_overlap(self, ll1: LongitudinalLesion, ll2: LongitudinalLesion):
        """Given two compatible lls, find skip edges between the latest lesions of the early ll and the earliest lesions
        of the late ll."""
        early0 = ll1.get_extreme_layer(earliest=True)
        early1 = ll2.get_extreme_layer(earliest=True)

        if early0 < early1:
            labels_prev = ll1.get_labels_in_extreme_layer(earliest=False)
            layer_prev = ll1.get_extreme_layer(earliest=False)
            labels_curr = ll2.get_labels_in_extreme_layer(earliest=True)
            layer_curr = early1
        else:
            labels_curr = ll1.get_labels_in_extreme_layer(earliest=True)
            layer_curr = early0
            labels_prev = ll2.get_labels_in_extreme_layer(earliest=False)
            layer_prev = ll2.get_extreme_layer(earliest=False)

        init_date = get_patient_dates(self.pat_name)[layer_prev]
        end_date = get_patient_dates(self.pat_name)[layer_curr]
        date_diff = (datetime.strptime(end_date, '%d_%m_%Y') - datetime.strptime(init_date, '%d_%m_%Y')).days
        if date_diff > 365 * 1.5:  # discard matching between two dates more than 1.5 year apart
            return []

        prev_lesions = [f"{lb_prev}_{layer_prev}" for lb_prev in labels_prev]
        curr_lesions = [f"{lb_curr}_{layer_curr}" for lb_curr in labels_curr]

        skip_edges = []
        for pr_les in prev_lesions:
            skip_edges += [(pr_les, cur_les) for cur_les in curr_lesions
                           if self.dilation_table.loc[pr_les, cur_les] > 0]

        return skip_edges

    @staticmethod
    def get_label_list(labeled_segmentation: np.array):
        """Get an np.array: a labeled segmentation. Return a list of the labels (without background 0 label), present
        in the matrix"""
        labels = np.unique(labeled_segmentation)
        les_labels = labels[labels > 0]
        return list(les_labels)

    def assign_lesions_data(self) -> (LesionList, list):
        """Taking the segmentations in self.lesion_segm_series, do:
            (1): store all lesions with their data in a container LesionList
            (2): replace lesions labels in self.lesion_segm_series, with lesion id (each lesion has a unique id in the
                series)
            :return: (1) LesionList, (2) the series of segmentations with lesions id (list of 3D np.arrays)
        """

        lesion_list = LesionList()
        lesion_id_series = []
        for ind, lesion_matrix in enumerate(self.lesion_segm_series):
            lesions_id_matrix = np.zeros_like(lesion_matrix).astype(np.int32)

            labels_list = self.get_label_list(lesion_matrix)
            labels_volumes = ndimage.sum(input=lesion_matrix>0, labels=lesion_matrix, index=labels_list)
            time_ind = self.time_indices[ind]
            for lb_ind, lb in enumerate(labels_list):
                les_vol = labels_volumes[lb_ind]
                les = Lesion(label=lb, layer=time_ind)
                les.set(attr=LesionAttr.REG_VOLUME, value=les_vol)
                les_id = lesion_list.append(les)
                lesions_id_matrix[lesion_matrix==lb] = les_id
            lesion_id_series.append(lesions_id_matrix)
        lesion_list.lock_list()
        return lesion_list, lesion_id_series

    def reduce_series_dimension(self):
        """
        Crop x,y,z of all the images in order to remove the regions with no labels (in none of the images)
        """
        segm_series_full_size = self.lesion_id_segm_series.copy()
        x_min, x_max, y_min, y_max, z_min, z_max = 10000, 0, 10000, 0, 10000, 0
        for segm in segm_series_full_size:
            x, y, z = np.where(segm > 0)
            if len(x) == 0:
                continue
            x_min = min([np.min(x), x_min])
            x_max = max([np.max(x), x_max])
            y_min = min([np.min(y), y_min])
            y_max = max([np.max(y), y_max])
            z_min = min([np.min(z), z_min])
            z_max = max([np.max(z), z_max])

        x_min = max([x_min - self.max_dilation, 0])
        x_max = min([x_max + self.max_dilation, self.segm_shape[0]])
        y_min = max([y_min - self.max_dilation, 0])
        y_max = min([y_max + self.max_dilation, self.segm_shape[1]])
        z_min = max([z_min - self.max_dilation, 0])
        z_max = min([z_max + self.max_dilation, self.segm_shape[2]])
        #print(self.segm_shape)
        segm_series_cropped = [segm[x_min:x_max, y_min:y_max, z_min:z_max] for segm in segm_series_full_size]
        self.segm_shape = segm_series_cropped[0].shape
        #print(self.segm_shape)
        return segm_series_cropped

    def dilation_overlap(self, dilation_table_path) -> pd.DataFrame:
        """
        Iteratively dilate all lesions segmentation and prepare a table, in which: Table[les1,les2]= # iterations in
        which les1 and les2 have an overlap of at least 10% of their dilated volume_cal.
        :param dilation_table_path: path where the excel table should be saved
        :return: df (pd.DataFrame) of the table.
        """
        expanded_labels = [[]]*(self.max_dilation + 1)
        #labels_series = self.lesion_id_segm_series
        labels_series = self.reduce_series_dimension()
        for dil in range(1, self.max_dilation+1):
            #print(f"iter: {dil}")
            # if dil == 1:
            #     dilated_label = [PwMatching.expand_labels(lesions, 1, voxelspacing=[1,1,1]) for lesions in labels_series]
            # else:
            #     dilated_label = [PwMatching.expand_labels(lesions, 1, voxelspacing=[1,1,1]) for lesions in expanded_labels[dil-1]]
            dilated_label = [expand_labels(lesions, dil) for lesions in labels_series]
            expanded_labels[dil] = dilated_label

            labels_tensor = np.stack(dilated_label, axis=-1) # 4D tensor
            # x,y,z,t = np.where(labels_tensor_full>0)
            # labels_tensor = labels_tensor_full[np.min(x):np.max(x), np.min(y):np.max(y), np.min(z):np.max(z), 0:self.num_of_timepoints]
            labels_overlapping = np.reshape(labels_tensor, (np.product(self.segm_shape), self.num_of_timepoints)) # 2D tensor. rows = matches; cols = layers

            #lesion_flatted = labels_overlapping[labels_overlapping > 0]
            lesion_curr_vol = ndimage.sum(labels_overlapping>0, labels_overlapping, self.lesion_list.get_all_ids())
            lesion_curr_vol_matrix = np.repeat(np.expand_dims(lesion_curr_vol, axis=1), len(self.lesion_list), axis=1)
            vol_matrix_thresholded = 0.1 * np.min([lesion_curr_vol_matrix, lesion_curr_vol_matrix.T], axis=0)

            intersections, vols = np.unique(labels_overlapping, axis=0, return_counts=True)
            self.vol_matrix = np.zeros_like(self.adjacency_matrix).astype(np.int32)
            ##
            # non_zero_inter_id, non_zero_layers = np.nonzero(intersections)
            # non_zero_labels = intersections[non_zero_inter_id, non_zero_layers]
            # non_single_inter = [(non_zero_labels[i], non_zero_layers[i], non_zero_inter_id[i])
            #                     for i, inter_id in enumerate(non_zero_inter_id) if non_zero_inter_id.count(inter_id) > 1]

            for inters_id, inters in enumerate(intersections):
                non_zero_layers = np.nonzero(inters)
                non_zero_labels = inters[non_zero_layers]
                if len(non_zero_labels) < 2:
                    continue
                les1 = non_zero_labels[:-1] - 1
                les2 = non_zero_labels[1:] - 1
                self.vol_matrix[les1, les2] += vols[inters_id]
            self.adjacency_matrix += (self.vol_matrix > vol_matrix_thresholded).astype(np.uint8)
        ordered_names = [self.lesion_list.find_by_id(les_id).name() for les_id in self.lesion_list.get_all_ids()]
        df = pd.DataFrame(data=self.adjacency_matrix, index=ordered_names, columns=ordered_names)
        df.to_excel(f"{dilation_table_path}")
        return df

    def run(self, dilation_table_path: str, load_dilation_table=True, pat=None):
        if not load_dilation_table:
            self.dilation_table = self.dilation_overlap(dilation_table_path)
            loader = LoaderPairwiseFromDf(df=self.dilation_table, pat=pat)
        else:
            loader = LoaderPairwiseFromDf(df_path=dilation_table_path, pat=pat)
            self.dilation_table = loader.get_dilation_table()
        self.longit = LongitLongitudinalLesionsSimple(loader, patient_dates=[f"{i}" for i in range(self.num_of_timepoints)])
        self.longit.add_cc_attribute()
        self.longit.create_long_lesions_list()
        self.find_skip_edges()

    def find_skip_edges(self):
        """Add to the pairwise longit graph the skip edges"""
        not_complete_ll = [ll for ll in self.longit.get_long_lesions_list() if not ll.check_completeness()]
        ll_graph = nx.complete_graph(not_complete_ll)

        incompatible_lls = [ll_pair for ll_pair in ll_graph.edges() if self.lls_are_unmatchable(ll_pair[0], ll_pair[1])]
        ll_graph.remove_edges_from(incompatible_lls)
        if len(ll_graph.edges) == 0:
            return

        skip_edges = []
        for ll_pair in ll_graph.edges:
            skip_edges += self.lls_extreems_overlap(*ll_pair)
        skip_edges_greedy = self.choose_high_overlap_skip_edges(skip_edges)
        print(f"deleted: {set(skip_edges) - set(skip_edges_greedy)}")

        loader = LoaderSimple(labels_list=list(self.longit.get_graph().nodes),
                              edges_list=list(self.longit.get_graph().edges) + skip_edges_greedy)
        self.longit = Longit(loader)


    def choose_high_overlap_skip_edges(self, skip_edges):
        skip_edges_extremes_prev = set([e[0] for e in skip_edges])
        skip_edges_extremes_next = set([e[1] for e in skip_edges])
        extremes_max_overlap_back = {les: max([self.dilation_table.loc[pr_les, les] for pr_les, curr_les in skip_edges if curr_les == les])
                                     for les in skip_edges_extremes_next}
        extremes_max_overlap_forw = {les: max([self.dilation_table.loc[les, next_les] for curr_les, next_les in skip_edges if curr_les == les])
                                     for les in skip_edges_extremes_prev}

        return [e for e in skip_edges if extremes_max_overlap_forw[e[0]] == extremes_max_overlap_back[e[1]]]


    def save_matching_graph(self, graph_path):
        """
        The matching is saved as {'nodes': vertices, 'edges': edges}, where both vertices and edges use the notation {lb}_{layer}
        """
        edges = list(self.longit.get_graph().edges())
        lab_list = list(self.longit.get_graph().nodes())
        graph_dict = {'nodes': lab_list, 'edges': edges}
        with open(graph_path, 'w') as f:
            json.dump(graph_dict, f)


class MatchingAlgoTest(MatchingAlgo):
    """
       This class TESTS a multi scan version of pairwise match_2_cases_v5
       """

    def __init__(self, test_df: pd.DataFrame, max_dilation: int = 5):
        super().__init__(pat_name='', lesion_segm_series=[np.zeros((1,1,1))])
        """
        :param pat_name: patient name
        :param lesion_segm_series: a list of N registered labeled lesion segmentations (of the same size!).
            The class assumes that the voxel volume_cal is 1x1x1 mm^3
        :param time_indices: a list of N integers being the time indices of the scans. If None, set as list indices.
        :param max_dilation: the number of iterations of the algorithm
        """
        les_names = test_df.index.tolist()
        self.lesion_list = LesionList()
        for les in les_names:
            self.lesion_list.append(Lesion(lb_layer=les))

        max_layer = max([les.layer() for les in self.lesion_list])
        self.num_of_timepoints = max_layer + 1
        self.time_indices = list(range(self.num_of_timepoints))
        self.max_dilation = max_dilation

        num_of_lesions = len(self.lesion_list)
        self.adjacency_matrix = np.zeros((num_of_lesions, num_of_lesions), dtype=np.int8)
        self.dilation_table = None
        self.longit = None


class AnalyzeMatchingEdgeConfidence:
    def __init__(self, max_dilation):
        self.max_dilation = max_dilation
        matching_algo = 'gt_segmentation_gw8'
        OPTION = 'gw8_1'
        patients = get_patients_list()
        rows = patients
        cols_tp = [f"{conf}-TP" for conf in range(1, self.max_dilation+1)]
        cols_fp = [f"{conf}-FP" for conf in range(1, self.max_dilation+1)]
        cols = np.ravel(np.stack([cols_tp, cols_fp]).T)
        empty_data = np.zeros((len(rows), len(cols)))
        edge_confidence_table = pd.DataFrame(data=empty_data, index=rows, columns=cols)
        skip_edge_confidence_table = edge_confidence_table.copy()

        for pat in patients:
            print(pat)
            matching_path = f"{path.LESIONS_MATCH_RESULTS_DIR}/{matching_algo}/{pat}"
            dilation_table = pd.read_excel(f"{matching_path}/dilation_table.xlsx", index_col=0)
            if OPTION.endswith('1'):
                with open(f"{matching_path}/edge_conf_{OPTION.replace('gw','')}.json", 'r') as f:
                    edge_conf_list = json.load(f)
                    edge_conf = {tuple(item[0]): item[1] for item in edge_conf_list}

            ld = LoaderSimpleFromJson(f"{matching_path}/{OPTION}.json")
            ld_gt = LoaderSimpleFromJson(f"{path.LESIONS_MATCH_GT_ORIGINAL}/{pat}glong_gt.json")
            eval = LoaderEval_SkipEdgeHandlerSoft(gt_loader=ld_gt, pred_loader=ld, patient_name=pat, patient_dates=get_patient_dates(pat))
            eval_edges = eval.get_edges_attributes()
            pred_edges = ld.get_edges()

            for edge in pred_edges:
                edge_det = eval_edges[edge][EdgeAttr.DETECTION]
                edge_is_skip = eval_edges[edge][EdgeAttr.IS_SKIP]
                edge_is_fp_skip_on_path = eval_edges[edge][EdgeAttr.IS_SKIP_EDGE_PATH]
                if OPTION.endswith('1') and not edge_is_skip:
                    edge_confidence = edge_conf[edge]
                else:
                    edge_confidence = dilation_table.loc[edge[0], edge[1]]

                if edge_is_skip:
                    if edge_det == EdgesDetect.TP or edge_is_fp_skip_on_path == EdgesInSkipPath.PRED:
                        det = 'TP'
                    else:
                        det = 'FP'
                    skip_edge_confidence_table.loc[pat,f"{edge_confidence}-{det}"] += 1
                else:
                    if edge_det == EdgesDetect.TP:
                        det = 'TP'
                    else:
                        det = 'FP'
                    edge_confidence_table.loc[pat,f"{edge_confidence}-{det}"] += 1

        with pd.ExcelWriter(f"{path.LESIONS_MATCH_RESULTS_DIR}/{matching_algo}/edge_confidence_{OPTION}.xlsx") as writer:
            edge_confidence_table.to_excel(writer, sheet_name='consec. edges')
            skip_edge_confidence_table.to_excel(writer, sheet_name='skip edges')


class PairsMatchingAlgo(MatchingAlgo):
    """
    This class implements a 2 scan version of pairwise match_2_cases_v5
    """

    def __init__(self, pat_name: str, lesion_segm_pairs: list, time_indices: list = None, max_dilation: int = 5,
                 registration_folder: str = None):
        """
        :param pat_name: patient name
        :param lesion_segm_pairs: a list: [[tumors_t1_reg_t2, tumors_t2], []] of N-1 pairs of registered labeled lesion
            segmentations and fix scan (of the same dimension!). The class assumes that the voxel volume_cal is 1x1x1 mm^3.
        :param time_indices: a list of N integers being the time indices of the scans. If None, set as list indices.
        :param max_dilation: the number of iterations of the algorithm
        :param registration_folder: the path to the folder containing patients folders
        """
        super().__init__(pat_name) # does nothing
        self.MOVING = 0
        self.FIX = 1
        self.pat_name = pat_name
        self.lesion_segm_pairs = [(lesions_pair[self.MOVING].astype(np.uint8), lesions_pair[self.FIX].astype(np.uint8)) for lesions_pair in lesion_segm_pairs]

        self.segm_shape = self.lesion_segm_pairs[0][self.FIX].shape
        if time_indices is None:
            self.num_of_timepoints = len(self.lesion_segm_pairs) + 1
            self.time_indices = list(range(self.num_of_timepoints))
        else:
            self.time_indices = time_indices
            self.num_of_timepoints = len(self.time_indices)
        self.max_dilation = max_dilation
        self.lesion_list, self.lesion_id_segm_series = self.assign_lesions_data()

        num_of_lesions = len(self.lesion_list)
        self.adjacency_matrix = np.zeros((num_of_lesions, num_of_lesions), dtype=np.int8)
        self.dilation_table = None
        self.longit = None
        self.registration_folder = registration_folder

    @staticmethod
    def assign_lesion_data_one_image(lesion_matrix: np.array, time_ind: int, lesion_list: LesionList):
        lesions_id_matrix = np.zeros_like(lesion_matrix).astype(np.int32)
        lb2id = dict()
        labels_list = MatchingAlgo.get_label_list(lesion_matrix)
        for lb in labels_list:
            les = Lesion(label=lb, layer=time_ind)
            les_id = lesion_list.append(les)
            lesions_id_matrix[lesion_matrix == lb] = les_id
            lb2id.update({lb: les_id})
        return lesions_id_matrix, lb2id

    @staticmethod
    def apply_lesion_data_one_image(lesion_matrix: np.array, label2id: dict):
        lesions_id_matrix = np.zeros_like(lesion_matrix).astype(np.int32)
        for lb, id in label2id.items():
            lesions_id_matrix[lesion_matrix == lb] = id
        return lesions_id_matrix

    def assign_lesions_data(self) -> (LesionList, list):
        """Taking the segmentations in self.lesion_segm_pairs, do:
            (1): store all lesions with their data in a container LesionList
            (2): replace lesions labels in self.lesion_segm_series, with lesion id (each lesion has a unique id in the
                series)
            :return: (1) LesionList, (2) the series of segmentations with lesions id (list of 3D np.arrays)
        """

        lesion_list = LesionList()
        lesion_id_series = [[None, None] for i in range(self.num_of_timepoints - 1)]

        for ind, lesion_pairs_matrix in enumerate(self.lesion_segm_pairs):
            lesion_matrix_moved = lesion_pairs_matrix[self.MOVING]
            time_ind = self.time_indices[ind]
            lesions_id_matrix_moved, lb2id = self.assign_lesion_data_one_image(lesion_matrix_moved, time_ind, lesion_list)
            lesion_id_series[time_ind][self.MOVING] = lesions_id_matrix_moved
            if ind>0:
                prev_time_ind = self.time_indices[ind-1]
                lesion_matrix_fix = self.lesion_segm_pairs[prev_time_ind][self.FIX]
                lesions_id_matrix_fix = self.apply_lesion_data_one_image(lesion_matrix_fix, lb2id)
                lesion_id_series[prev_time_ind][self.FIX] = lesions_id_matrix_fix

        lesion_matrix_fix = self.lesion_segm_pairs[-1][self.FIX]
        last_time_ind = self.time_indices[-1]
        lesions_id_matrix_fix, _ = self.assign_lesion_data_one_image(lesion_matrix_fix, last_time_ind, lesion_list)
        lesion_id_series[-1][self.FIX] = lesions_id_matrix_fix

        lesion_list.lock_list()
        return lesion_list, lesion_id_series

    def dilation_overlap(self, dilation_table_path) -> pd.DataFrame:
        """
        Iteratively dilate all lesions segmentation and prepare a table, in which: Table[les1,les2]= # iterations in
        which les1 and les2 have an overlap of at least 10% of their dilated volume_cal.
        :param dilation_table_path: path where the excel table should be saved
        :return: df (pd.DataFrame) of the table.
        """
        expanded_labels = [[]] * (self.max_dilation + 1)
        # labels_series = self.lesion_id_segm_series
        labels_series = self.lesion_id_segm_series
        for dil in range(1, self.max_dilation + 1):
            dilated_label = [[expand_labels(lesions_pair[self.MOVING], dil), expand_labels(lesions_pair[self.FIX], dil)]
                             for lesions_pair in labels_series]

            expanded_labels[dil] = dilated_label

            labels_tensor = np.array([np.stack(dilated_pair, axis=-1) for dilated_pair in dilated_label])  # a 5D tensor dim0 = #of pairs, dim1-3 = x,y,z dim4= 2
            labels_overlapping = np.reshape(labels_tensor, ((self.num_of_timepoints-1)*np.product(self.segm_shape), 2))  # 2D tensor. dim0 = matches; dim1 = layers (bl, fu)

            lesion_curr_vol = ndimage.sum(labels_overlapping > 0, labels_overlapping, self.lesion_list.get_all_ids())
            lesion_curr_vol_matrix = np.repeat(np.expand_dims(lesion_curr_vol, axis=1), len(self.lesion_list), axis=1)
            vol_matrix_thresholded = 0.1 * np.min([lesion_curr_vol_matrix, lesion_curr_vol_matrix.T], axis=0)

            intersections, vols = np.unique(labels_overlapping, axis=0, return_counts=True)
            self.vol_matrix = np.zeros_like(self.adjacency_matrix).astype(np.int32)

            for inters_id, inters in enumerate(intersections):
                non_zero_layers = np.nonzero(inters)
                non_zero_labels = inters[non_zero_layers]
                if len(non_zero_labels) < 2:
                    continue
                les1 = non_zero_labels[:-1] - 1 # matrix rows/cols are 0...N-1, ids are 1...N
                les2 = non_zero_labels[1:] - 1
                self.vol_matrix[les1, les2] += vols[inters_id]
            self.adjacency_matrix += (self.vol_matrix > vol_matrix_thresholded).astype(np.uint8)
        ordered_names = [self.lesion_list.find_by_id(les_id).name() for les_id in self.lesion_list.get_all_ids()]
        df = pd.DataFrame(data=self.adjacency_matrix, index=ordered_names, columns=ordered_names)
        df.to_excel(f"{dilation_table_path}")
        return df

    def find_skip_edges(self):
        """Add to the pairwise longit graph the skip edges"""
        not_complete_ll = [ll for ll in self.longit.get_long_lesions_list() if not ll.check_completeness()]
        ll_graph = nx.complete_graph(not_complete_ll)

        incompatible_lls = [ll_pair for ll_pair in ll_graph.edges() if self.lls_are_unmatchable(ll_pair[0], ll_pair[1])]
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
            early0 = ll1.get_extreme_layer(earliest=True)
            early1 = ll2.get_extreme_layer(earliest=True)

            if early0 < early1:
                labels_moving = ll1.get_labels_in_extreme_layer(earliest=False)
                layer_moving = ll1.get_extreme_layer(earliest=False)
                layer_fix = early1
                labels_fix = ll2.get_labels_in_extreme_layer(earliest=True)
            else:
                layer_fix = early0
                labels_fix = ll1.get_labels_in_extreme_layer(earliest=True)
                labels_moving = ll2.get_labels_in_extreme_layer(earliest=False)
                layer_moving = ll2.get_extreme_layer(earliest=False)
            layers_pair = (layer_moving, layer_fix)

            if layers_pair not in layers_pairs2labels_pairs:
                layers_pairs2labels_pairs.update({layers_pair: (set(), set())})

            for lb in labels_moving:
                layers_pairs2labels_pairs[layers_pair][self.MOVING].add(lb)
            for lb in labels_fix:
                layers_pairs2labels_pairs[layers_pair][self.FIX].add(lb)

        skip_edges_greedy = []
        patient_dates = get_patient_dates(self.pat_name)
        for layers_pair, labels_pairs in layers_pairs2labels_pairs.items():
            init_layer, end_layer = layers_pair
            init_date = patient_dates[init_layer]
            end_date = patient_dates[end_layer]
            date_diff = (datetime.strptime(end_date, '%d_%m_%Y') - datetime.strptime(init_date, '%d_%m_%Y')).days
            if date_diff > 365 * 1.5: # discard matching between two dates more than 1.5 year apart
                continue
            init_labels, end_labels = labels_pairs

            init_layer_image_path = f"{self.registration_folder}/{self.pat_name}/{name.segm_name_gt(init_date, end_date)}"
            init_image, _ = load_nifti_data(init_layer_image_path)
            init_image = init_image.astype(np.uint8)
            init_labeled_image = np.zeros_like(init_image)
            init_candidate_label_mask = np.isin(init_image, list(init_labels))
            init_labeled_image[init_candidate_label_mask] = init_image[init_candidate_label_mask]

            end_layer_image_path = f"{self.registration_folder}/{self.pat_name}/{name.segm_name_gt(end_date)}"
            end_image, _ = load_nifti_data(end_layer_image_path)
            end_image = end_image.astype(np.uint8)
            end_labeled_image = np.zeros_like(end_image)
            end_candidate_label_mask = np.isin(end_image, list(end_labels))
            end_labeled_image[end_candidate_label_mask] = end_image[end_candidate_label_mask]

            skip_edges_current = PwMatching.match_2_cases_v5(baseline_moved_labeled=init_labeled_image,
                                                  followup_labeled=end_labeled_image, voxelspacing=(1,1,1),
                                                  max_dilate_param=self.max_dilation)

            skip_edges_greedy += [(f"{int(e[0])}_{init_layer}", f"{int(e[1])}_{end_layer}") for e in skip_edges_current]
        skip_edges = skip_edges_greedy
        loader = LoaderSimple(labels_list=list(self.longit.get_graph().nodes),
                              edges_list=list(self.longit.get_graph().edges) + skip_edges)
        self.longit = Longit(loader)


def load_scans(pat):
    dateN = get_patient_dates(pat)[-1]
    pat_paths = [f"{path.GT_FILTERED_LABELED_GROUPWISE}/{pat}/{name.segm_name_gt(date1, dateN)}"
                 for date1 in get_patient_dates(pat)[:-1]]
    pat_paths += [f"{path.GT_FILTERED_LABELED_GROUPWISE}/{pat}/{name.segm_name_gt(dateN)}"]

    lesion_segm_series = [load_nifti_data(p)[0] for p in pat_paths]
    return lesion_segm_series

def load_scans_series(pat, reg_folder):
    """Segmentations from groupwise registration"""
    pat_path = f"{reg_folder}/{pat}/series_reg_tumors.nii.gz"
    loaded_data, _ = load_nifti_data(pat_path)
    lesion_segm_series = [loaded_data[:,:,:,i] for i in range(loaded_data.shape[-1])]
    return lesion_segm_series

def load_pairs(pat, reg_folder):
    pat_path = f"{reg_folder}/{pat}"
    dates = get_patient_dates(pat)
    pairs_series = [(load_nifti_data(f"{pat_path}/{name.segm_name_gt(dates[i], dates[i+1])}")[0],
                     load_nifti_data(f"{pat_path}/{name.segm_name_gt(dates[i+1])}")[0])
                    for i in range(len(dates) - 1)]

    return pairs_series


def pat_matching(pat):
    #reg_folder = f"{path.PROJ_NAME}/registration_results_exp/results/pairwise_affine_all"
    reg_folder = f"{path.PROJ_NAME}/registration_results_exp/results/affine_param_map"
    lesion_segm_series = load_scans_series(pat, reg_folder)
    #lesion_segm_series = load_pairs(pat, reg_folder)
    #lesion_segm_series = load_scans(pat)
    # print(f"{pat}: starting algo")
    # t = time()
    print(pat)
    os.makedirs(f"{path.LESIONS_MATCH_RESULTS_DIR}/gt_segmentation_gw12/{pat}", exist_ok=True)
    m = PairsMatchingAlgo(pat_name=pat, lesion_segm_pairs=lesion_segm_series, max_dilation=7,
                          registration_folder=reg_folder)
    #m = MatchingAlgo(pat_name=pat, lesion_segm_series=lesion_segm_series, max_dilation=7)
    m.run(dilation_table_path=f"{path.LESIONS_MATCH_RESULTS_DIR}/gt_segmentation_gw12/{pat}/dilation_table.xlsx",
          load_dilation_table=True, pat=pat)
    m.save_matching_graph(f"{path.LESIONS_MATCH_RESULTS_DIR}/gt_segmentation_gw12/{pat}/gw12.json")
    print(f"{pat} done")
    # print(f"{time() - t}")

def launch():
    path = Path()
    name = Name()

    # for pat in get_patients_list():
    #    pass
    # if pat == 'B_B_S_': continue
    #pats = [pat for pat in get_patients_list() if pat != 'B_B_S_']
    pats = [pat for pat in get_patients_list()]
    #pats = ["C_A_", "Z_Aa_"]
    for pat in pats:
        pat_matching(pat)
    #with Pool(1) as p:
    #   p.map(pat_matching, pats)

def pat_matching_test():
    df = pd.read_excel(f"/cs/usr/bennydv/Desktop/dilation_table_test.xlsx")
    m = MatchingAlgoTest(test_df=df, max_dilation=7)
    m.run(dilation_table_path=f"/cs/usr/bennydv/Desktop/dilation_table_test.xlsx",
          load_dilation_table=True, pat='')
    #m.save_matching_graph(f"{path.LESIONS_MATCH_RESULTS_DIR}/gt_segmentation_gw8/{pat}/gw8_1.json")
    print(m.longit.get_graph().edges())
    a = 1


def compose_tables():
    pw_table = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/results/gt_segmentation_gw11"
    gw_table = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/results/gt_segmentation_gw9"
    current_dir = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/results/gt_segmentation_gw12"
    for pat in get_patients_list():
        os.makedirs(f"{current_dir}/{pat}", exist_ok=True)
        pw_df = pd.read_excel(f"{pw_table}/{pat}/dilation_table.xlsx", index_col=0)
        gw_df = pd.read_excel(f"{gw_table}/{pat}/dilation_table.xlsx", index_col=0)
        df = pw_df.copy()
        for row_l in df.index:
            for col_l in df.columns:
                start_layer = int(row_l.split('_')[1])
                end_layer = int(col_l.split('_')[1])
                if end_layer - start_layer > 1:
                    if row_l in gw_df.index and col_l in gw_df.columns:
                        df.loc[row_l, col_l] = gw_df.loc[row_l, col_l]
        df.to_excel(f"{current_dir}/{pat}/dilation_table.xlsx")



if __name__ == "__main__":
    from general_utils import *
    from config import *
    from time import time
    from multiprocessing import Pool
    import shutil

    pat_matching_test()
    #compose_tables()
    #launch()
    #a = AnalyzeMatchingEdgeConfidence(max_dilation=7)
    # dir = "/cs/casmip/bennydv/liver_pipeline/lesions_matching/results/gt_segmentation_gw6/"
    # for pat in get_patients_list():
    #     os.makedirs(f"{dir}/{pat}", exist_ok=True)
    #     shutil.move(src=f"{dir}/{pat}.xlsx", dst=f"{dir}/{pat}/dilation_table.xlsx")


