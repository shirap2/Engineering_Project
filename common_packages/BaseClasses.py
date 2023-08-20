import copy
import itertools

import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import json
import os
import inspect
import networkx as nx
from typing import List, Dict, Tuple
import numpy as np
import collections
from skimage.morphology import label
from scipy import ndimage

import pandas as pd
from collections.abc import Iterable



class EvaluationType:
    SIMPLE = 'simple'
    SKIP_EDGE_HANDLES_CHANGES = 'skip_edge_handles_changes'
    SKIP_EDGE_HANDLES = 'skip_edge_handles'
    ACCEPT_CC_EDGES = 'accept_cc_edges'
    NODES_PATHS = 'nodes_paths'



class SourceGraph:
    GT = "GT"
    PRED = "PRED"


class PathType:
    UNDIRECTED = 'undirected'
    FORWARD = 'forward'
    BACKWARD = 'backward'


class StatNames:
    """Store the names of the statistics to be calculated on the evaluation graph"""
    def __init__(self):
        self.num_gt_lesions = 'num gt lesions'
        self.num_lesions_classes = 'num lesions classes'
        self.num_correct_lesions_classes = 'num correct lesions classes'
        self.num_gt_edges = 'num gt edges'
        self.num_pred_edges = 'num pred edges'
        self.num_tp_edges = 'num TP edges'
        self.num_fp_edges = 'num FP edges'
        self.num_fn_edges = 'num FN edges'
        self.num_tp_cc = 'num TP cc'
        self.num_fp_cc = 'num FP cc'
        self.num_fn_cc = 'num FN cc'


class StatNamesPredLesions(StatNames):
    """Expand parent with predicted lesions statistics"""
    def __init__(self):
        super().__init__()
        self.num_pred_lesions = 'num pred lesions'
        self.num_tp_lesions = 'num TP lesions'
        self.num_fp_lesions = 'num FP lesions'
        self.num_fn_lesions = 'num FN lesions'


class StatNamesSkipEdgeHandler(StatNamesPredLesions):
    """Expand parent with skip edge on path"""
    def __init__(self):
        super().__init__()
        self.num_gt_skip_edges = 'num gt skip edges'
        self.num_pred_skip_edges = 'num pred skip edges'
        self.num_tp_skip_edges = 'num TP skip edges'
        self.num_fp_skip_edges = 'num FP skip edges'
        self.num_fn_skip_edges = 'num FN skip edges'
        self.num_fp_skip_edges_path = 'num FP skip edges on path'
        self.num_fn_skip_edges_path = 'num FN skip edges on path'


class StatNamesAcceptCCEdges(StatNamesPredLesions):
    def __init__(self):
        super().__init__()
        self.num_fp_edges_in_cc = 'num FP edges in cc'
        self.num_fn_edges_in_cc = 'num FN edges in cc'


class StatNamesNodesCC(StatNamesPredLesions):
    def __init__(self, path_type: str):
        if path_type not in [PathType.UNDIRECTED, PathType.FORWARD, PathType.BACKWARD]:
            raise ValueError(f"{path_type} unknown!")
        super().__init__()
        self.mean_node_path_precision = f'mean node path {path_type} precision'
        self.std_node_path_precision = f'std node path {path_type} precision'
        self.mean_node_path_recall = f'mean node path {path_type} recall'
        self.std_node_path_recall = f'std node path {path_type} recall'
        self.num_node_paths = f"num node paths"


class StatNamesTwoMethods:
    def __init__(self):
        self.num_gt_lesions = 'num gt lesions'
        self.num_gt_consecutive_edges = 'num gt edges consecutive'
        self.num_gt_skip_edges = 'num gt edges skip'
        self.num_pw_consecutive_edges = 'num pw edges consecutive'
        self.num_gw_consecutive_edges = 'num gw edges consecutive'
        self.num_gw_skip_edges = 'num gw edges skip'

        self.num_tp_edges_both = 'num TP edges, both methods'
        self.num_fp_edges_both = 'num FP edges, both methods'
        self.num_fn_edges_both = 'num FN edges, both methods'
        self.num_tp_edges_pw = 'num TP edges, pw only'
        self.num_fp_edges_pw = 'num FP edges, pw only'
        self.num_fn_edges_pw = 'num FN edges, pw only'
        self.num_tp_edges_gw = 'num TP edges, gw only'
        self.num_fp_edges_gw = 'num FP edges, gw only'
        self.num_fn_edges_gw = 'num FN edges, gw only'
        self.num_tp_edges_gw_skip = 'num TP skip edges, gw only'
        self.num_fp_edges_gw_skip = 'num FP skip edges, gw only'
        self.num_fn_edges_gw_skip = 'num FN skip edges, gw only'



class NodeAttr:
    LAYER = 'layer'
    LABEL = 'label'
    COLOR = 'color'
    DIAMETER = 'extr_diameter'
    CAL_DIAMETER = 'cal_diameter'
    SLICE = 'slice'
    IS_PLACEHOLDER = 'is_placeholder'
    CLASSIFICATION = 'class'
    PRESENCE_CLASS = 'presence'
    EVOLUTION_CLASS = 'evolution'
    DETECTION = 'detection'
    EVAL_CLASSIFICATION = 'corr_class'
    CC_DRAW_INDEX = 'cc_draw_index'
    CC_INDEX = 'cc_index'
    SOURCE_GRAPHS_CC_INDICES = 'source_cc_indices'
    PATH_CORRECTNESS = 'path_correctness'
    PATH_CLASSIFICATION = 'path_class'
    CC_PATTERNS = 'cc_patterns'
    CHANGES = 'changes'


    @staticmethod
    def is_attr(attr):
        members = inspect.getmembers(NodeAttr, lambda a: not (inspect.isroutine(a)))
        attributes = [a[1] for a in members if not (a[0].startswith('__') and a[0].endswith('__'))]
        return attr in attributes


class EdgeAttr:
    IS_SKIP = 'is_skipedge'
    IS_SKIP_EDGE_PATH = 'is_skipedge_path'
    COLOR = 'color'
    CONNECTION_STYLE = 'connectionstyle'
    DETECTION = 'edge_detection'
    METHOD_DETECTION = 'edge method detection'

    @staticmethod
    def is_attr(attr):
        members = inspect.getmembers(EdgeAttr, lambda a: not (inspect.isroutine(a)))
        attributes = [a[1] for a in members if not (a[0].startswith('__') and a[0].endswith('__'))]
        return attr in attributes


class NodesClass:
    EXISTING_EXISTING = 'ex-ex'
    EXISTING_DISAPPEARED = 'ex-dis'
    NEW_EXISTING = 'new-ex'
    NEW_DISAPPEARED = 'new-dis'

    AS_VECTOR = {EXISTING_EXISTING: np.array([0, 0]),
                 EXISTING_DISAPPEARED: np.array([0, 1]),
                 NEW_EXISTING: np.array([1, 0]),
                 NEW_DISAPPEARED: np.array([1, 1])}


class NodesDetect:
    TP = 'tp'
    FP = 'fp'
    FN = 'fn'


class NodesClassEval:
    TWO_CORRECT = '2_corr'
    ONE_CORRECT = '1_corr'
    WRONG = '0_corr'
    UNSHARED = 'unshared'


class NodesSourceCC:
    GT = 0
    PRED = 1
    UNSHARED = None


class NodesPathCorrectness:
    PRECISION = 0
    RECALL = 1
    UNSHARED = None


class EdgesDetect:
    TP = 'tp'
    FP = 'fp'
    FN = 'fn'
    UNSHARED = 'unshared'
    FP_IN_SAME_CC = 'fp_in_cc'
    FN_IN_SAME_CC = 'fn_in_cc'


class EdgesMethodsDetect:
    BOTH = 'both'
    PW_MATCHING = 'pw_matching'
    GW_MATCHING = 'gw_matching'
    UNSHARED = 'unshared'


class EdgesInSkipPath:
    FALSE = 'false'
    GT = 'gt'
    PRED = 'pred'

class NodesPresence:
    LONE = 'lone'
    EXISTING = 'existing'
    NEW = 'new'
    DISAPPEARING = 'disap'

class NodesEvolution:
    LINEAR = 'linear'
    MERGED = 'merged'
    SPLITTING = 'splitting'
    COMPLEX = 'complex'

class NodesChanges:
    LONE = 'lone'
    UNIQUE = 'unique'
    NEW = 'new'
    DISAPPEARED = 'disap'
    MERGED = 'merge'
    SPLITTING = 'split'
    MERGED_DISAP = 'merge_disap'
    SPLITTING_NEW = 'split_new'
    COMPLEX = 'complex'

    @staticmethod
    def is_attr(attr):
        members = inspect.getmembers(NodesChanges, lambda a: not (inspect.isroutine(a)))
        attributes = [a[1] for a in members if not (a[0].startswith('__') and a[0].endswith('__'))]
        return attr in attributes


class PathClass:
    NONE = 'none'
    LONE = 'lone_path'
    LINEAR = 'linear_path'
    MERGING = 'merging_path'
    SPLITTING = 'splitting_path'
    COMPLEX = 'complex_path'

class CcPatterns:
    SINGLE = 'single_p'
    LINEAR = 'linear_p'
    MERGING = 'merge_p'
    SPLITTING = 'split_p'
    COMPLEX = 'complex_p'


class Colors:
    WHITE = (1.0, 1.0, 1.0)
    GRAY = (0.8, 0.8, 0.8)
    BLACK = (0.0, 0.0, 0.0)
    GREEN = (0, 1, 0)
    RED = (1, 0, 0)
    BLUE = (0, 0, 1)
    DARK_GRAY = (0.5, 0.5, 0.5)
    DARK_GREEN = (16 / 255, 125 / 255, 8 / 255)
    ORANGE = (200 / 255, 81 / 255, 0)
    DARK_ORANGE = (102/255, 51/255, 0)
    LIGHT_GREEN = (124/255, 162/255, 21/255)
    VIOLET = (1, 0, 81 / 255)
    DARK_VIOLET = (82/255, 39/255, 81/255)

    d_classification = {NodesClass.EXISTING_EXISTING: tuple([0, 0.59, 1]),
                       NodesClass.NEW_DISAPPEARED: tuple([0, 1, 0.78]),
                       NodesClass.NEW_EXISTING: tuple([0.78, 0, 1]),
                       NodesClass.EXISTING_DISAPPEARED: tuple([1, 0.59, 0])}

    d_node_det = {NodesDetect.TP: DARK_GREEN,
                 NodesDetect.FN: ORANGE,
                 NodesDetect.FP: VIOLET}

    d_eval_classification = {NodesClassEval.WRONG: RED,
                            NodesClassEval.ONE_CORRECT: ORANGE,
                            NodesClassEval.TWO_CORRECT: GREEN,
                            NodesClassEval.UNSHARED: GRAY}

    d_edge_det = {EdgesDetect.TP: DARK_GREEN,
                  EdgesDetect.FN: ORANGE,
                  EdgesDetect.FP: VIOLET,
                  EdgesDetect.UNSHARED: GRAY,
                  EdgesDetect.FN_IN_SAME_CC: DARK_ORANGE,
                  EdgesDetect.FP_IN_SAME_CC: LIGHT_GREEN}

    d_edge_method_det = {EdgesMethodsDetect.BOTH: DARK_GREEN,
                         EdgesMethodsDetect.PW_MATCHING: ORANGE,
                         EdgesMethodsDetect.GW_MATCHING: VIOLET,
                         EdgesMethodsDetect.UNSHARED: GRAY}

    d_skip_edge_p = {EdgesInSkipPath.FALSE: GRAY,
                     EdgesInSkipPath.PRED: LIGHT_GREEN,
                     EdgesInSkipPath.GT: DARK_ORANGE}

    @staticmethod
    def select_dict(attr_name):
        name2dict = {
            NodeAttr.CLASSIFICATION: Colors.d_classification,
            NodeAttr.DETECTION: Colors.d_node_det,
            NodeAttr.EVAL_CLASSIFICATION: Colors.d_eval_classification,
            EdgeAttr.DETECTION: Colors.d_edge_det,
            EdgeAttr.IS_SKIP_EDGE_PATH: Colors.d_skip_edge_p,
            EdgeAttr.METHOD_DETECTION: Colors.d_edge_method_det
        }
        if attr_name not in name2dict:
            raise ValueError(f"Attr name:{attr_name} not found")
        return name2dict[attr_name]

    @staticmethod
    def choose(attr_name: str, attr_value: str):
        d = Colors.select_dict(attr_name)
        if attr_value not in d:
            raise ValueError(f"Attr value:{attr_value} not found in {attr_name}")
        return d[attr_value]

    @staticmethod
    def itk_colors(label: int, src: str = '/cs/casmip/bennydv/lungs_pipeline/longitudinal/label_descriptions.txt'):
        class default_dict(collections.UserDict):
            def __init__(self, default_factory=None, *args, **kwargs):
                super().__init__(*args, **kwargs)
                if not callable(default_factory) and default_factory is not None:
                    raise TypeError('first argument must be callable or None')
                self.default_factory = default_factory

            def __missing__(self, key):
                if self.default_factory is None:
                    raise KeyError(key)
                if key not in self:
                    self[key] = self.default_factory()
                return self[key]

        with open(src) as f:
            colors = f.readlines()
        colors = colors[15:]
        colors = [[c for c in l.split(' ') if c != ''] for l in colors]
        d = default_dict(lambda: (0, 0, 0),
                         ((int(l[0]), (int(l[1]) / 255, int(l[2]) / 255, int(l[3]) / 255)) for l in colors))
        return d[label]

    @staticmethod
    def is_attr(attr):
        members = inspect.getmembers(CcPatterns, lambda a: not (inspect.isroutine(a)))
        attributes = [a[1] for a in members if not (a[0].startswith('__') and a[0].endswith('__'))]
        return attr in attributes


class ColorsClass:
    d_presence = {NodesPresence.EXISTING: tuple([0, 0.59, 1]),
                       NodesPresence.LONE: tuple([0, 1, 0.78]),
                       NodesPresence.NEW: tuple([0.78, 0, 1]),
                       NodesPresence.DISAPPEARING: tuple([1, 0.59, 0])}
    d_evolution = {NodesEvolution.LINEAR: Colors.DARK_GRAY,
                   NodesEvolution.MERGED: Colors.DARK_GREEN,
                   NodesEvolution.SPLITTING: Colors.ORANGE,
                   NodesEvolution.COMPLEX: Colors.DARK_VIOLET}

    d_changes = {NodesChanges.LONE: tuple([0, 1, 0.78]),
                 NodesChanges.NEW: tuple([0.78, 0, 1]),
                 NodesChanges.DISAPPEARED: tuple([1, 0.59, 0]),
                 NodesChanges.UNIQUE: tuple([0, 0.59, 1]),
                 NodesChanges.MERGED: Colors.DARK_GREEN,
                 NodesChanges.MERGED_DISAP: Colors.GREEN,
                 NodesChanges.SPLITTING: Colors.ORANGE,
                 NodesChanges.SPLITTING_NEW: Colors.DARK_ORANGE,
                 NodesChanges.COMPLEX: Colors.VIOLET}

    d_detection = {NodesDetect.TP: Colors.DARK_GREEN,
                 NodesDetect.FN: Colors.ORANGE,
                 NodesDetect.FP: Colors.VIOLET}

    p_class = {PathClass.NONE: Colors.GRAY,
               PathClass.LONE: tuple([0, 1, 0.78]),
               PathClass.LINEAR: Colors.BLUE,
               PathClass.MERGING: Colors.DARK_GREEN,
               PathClass.SPLITTING: Colors.ORANGE,
               PathClass.COMPLEX: Colors.DARK_VIOLET}
    c_patterns = {
               CcPatterns.SINGLE: tuple([0, 1, 0.78]),
               CcPatterns.LINEAR: Colors.BLUE,
               CcPatterns.MERGING: Colors.DARK_GREEN,
               CcPatterns.SPLITTING: Colors.ORANGE,
               CcPatterns.COMPLEX: Colors.VIOLET}


    @staticmethod
    def select_dict(attr_name):
        name2dict = {
            NodeAttr.PRESENCE_CLASS: ColorsClass.d_presence,
            NodeAttr.EVOLUTION_CLASS: ColorsClass.d_evolution,
            NodeAttr.PATH_CLASSIFICATION: ColorsClass.p_class,
            NodeAttr.DETECTION: ColorsClass.d_detection,
            NodeAttr.CHANGES: ColorsClass.d_changes,
            NodeAttr.CC_PATTERNS: ColorsClass.c_patterns
        }
        if attr_name not in name2dict:
            raise ValueError(f"Attr name:{attr_name} not found")
        return name2dict[attr_name]

    @staticmethod
    def choose(attr_name: str, attr_value: str):
        d = ColorsClass.select_dict(attr_name)
        if attr_value not in d:
            raise ValueError(f"Attr value:{attr_value} not found in {attr_name}")
        return d[attr_value]


class DeleteLayer:
    """This class delete a nodes at the given layer and nodes' adjacent edges. It connects with new edges nodes that were
    previously connected through the deleted layer"""
    def __init__(self, labels_list, edges_list):
        self._lb_list = labels_list.copy()
        self._ed_list = edges_list.copy()

    def apply(self, layers_to_delete):
        """Gets a list or an int: layers_to_delete and outputs the new label and edges list"""
        BACK = 'back'
        FORW = 'forward'

        if not isinstance(layers_to_delete, list):
            layers_to_delete = [layers_to_delete]

        labels_list = self._lb_list
        edges_list = self._ed_list
        deleted_labels = [l for l in labels_list if self.node2layer(l) in layers_to_delete]
        # go over layer to delete one-by-one, to add and remove edges recursively
        for ly_to_delete in layers_to_delete:
            deleted_labels_ly = [l for l in deleted_labels if self.node2layer(l) == ly_to_delete]

            # each label to be deleted : its adjacent edges
            deleted_labels2adjacents_edges = {l: [e for e in edges_list if (e[0] == l) or (e[1] == l)] for l in
                                              deleted_labels_ly}

            # each label to be deleted : its adjacent edges going to previous layers and those to next layers
            deleted_labels2adjacents_edges_groups = {l: {BACK: [e for e in adj_e if
                                                                (self.node2layer(e[0]) <= ly_to_delete) and (
                                                                            self.node2layer(e[1]) <= ly_to_delete)],
                                                         FORW: [e for e in adj_e if
                                                                (self.node2layer(e[0]) >= ly_to_delete) and (
                                                                            self.node2layer(e[1]) >= ly_to_delete)]}
                                                     for l, adj_e in deleted_labels2adjacents_edges.items()}

            # if B is to be deleted and the original graph was A-B-C, delete A-B, B-C (adjacent edges) and add A-C
            edges_to_add_for_label = [
                self.all_combinations(group1=e_groups[BACK], group2=e_groups[FORW], l_to_delete=l) for l, e_groups
                in deleted_labels2adjacents_edges_groups.items()]

            edges_to_add = [e for e_for_label in edges_to_add_for_label for e in e_for_label]
            edges_to_remove = [e for e_for_label in deleted_labels2adjacents_edges.values() for e in e_for_label]

            _ = [edges_list.remove(e) for e in edges_to_remove]
            edges_list += edges_to_add

        _ = [labels_list.remove(l) for l in deleted_labels]
        return labels_list, edges_list

    @staticmethod
    def all_combinations(group1, group2, l_to_delete):
        """make all possible pairs between nodes in group1 and group2"""
        if len(group1) == 0 or len(group2) == 0:
            return []
        # create empty list to store the
        # combinations
        unique_combinations = []

        nodes_group1 = []
        nodes_group2 = []
        for e in group1:
            if e[0] == l_to_delete:
                nodes_group1.append(e[1])
            else:
                nodes_group1.append(e[0])
        for e in group2:
            if e[0] == l_to_delete:
                nodes_group2.append(e[1])
            else:
                nodes_group2.append(e[0])

        # Getting all permutations of list_1
        # with length of list_2
        if len(nodes_group1) >= len(nodes_group2):
            permut = itertools.permutations(nodes_group1, len(nodes_group2))

            # zip() is called to pair each permutation
            # and shorter list element into combination
            for comb in permut:
                zipped = zip(comb, nodes_group2)
                unique_combinations += list(zipped)
        else:
            permut = itertools.permutations(nodes_group2, len(nodes_group1))

            # zip() is called to pair each permutation
            # and shorter list element into combination
            for comb in permut:
                zipped = zip(nodes_group1, comb)
                unique_combinations += list(zipped)
        return unique_combinations

    @staticmethod
    def node2layer(vert: str):
        """node (string) ='lb_layer' --> layer (int) """
        return int(float((vert.split('_')[1])))


class Loader:
    """
    This class loads a longitudinal graph. It stores its nodes as {lb_layer} and its edges as {lb_layer, lb_layer}. The nodes
    may have some attribute.
    """

    def __init__(self):
        self._nodes = None
        self._edges = None

    def get_nodes(self):
        """Returns the nodes ids only as a list"""
        nodes_attr = self.get_nodes_attributes()
        return list(nodes_attr.keys())

    def get_nodes_attributes(self):
        """Returns a dictionary {node_id: {attr_name: attr}}"""
        if isinstance(self._nodes, dict):
            return self._nodes
        else:
            raise ValueError("Benny: nodes must be stored as a dictionary")

    def get_edges(self):
        """Returns the edges ids only as a list"""
        edges_attr = self.get_edges_attributes()
        return list(edges_attr.keys())

    def get_edges_attributes(self):
        """Returns a dictionary {edge_id: {attr_name: attr}}"""
        if isinstance(self._edges, dict):
            return self._edges
        else:
            raise ValueError("Benny: edges must be stored as a dictionary")


class Longit:
    def __init__(self, loader: Loader, patient_name=None, patient_dates=None):
        """
        This class transforms any view of longitudinal graph (loaded through its loader) to a networkx graph.
        :param loader: a class that loads a specific type of longitudinal graph. In this class, there are lesions with a set of
        attributes, and edges between them
        :param patient_name: (optional) the name of the patient
        :param patient_dates: (optional) list of string with scans dates
        """

        self._graph = nx.Graph()
        # add nodes
        self._graph.add_nodes_from(loader.get_nodes())
        # add edges:
        n_nodes = self._graph.number_of_nodes()
        self._graph.add_edges_from(loader.get_edges())
        n_nodes_after_adding_edges = self._graph.number_of_nodes()
        if n_nodes_after_adding_edges > n_nodes:
            raise ValueError("Adding edges new nodes were added!")

        # add basic attributes:
        nx.set_node_attributes(self._graph, loader.get_nodes_attributes())
        nx.set_edge_attributes(self._graph, loader.get_edges_attributes())

        if patient_dates is None:
            self._num_of_layers = int(np.max(list(nx.get_node_attributes(self._graph, NodeAttr.LAYER).values())) + 1)
        else:
            self._num_of_layers = len(patient_dates)
        self._patient_name = patient_name
        self._patient_dates = patient_dates

    def get_graph(self) -> nx.Graph:
        return self._graph

    def get_num_of_layers(self):
        return self._num_of_layers

    def get_patient_name(self):
        return self._patient_name

    def get_patient_dates(self):
        return self._patient_dates

    def save(self, file_path: str):
        """
        Saves the Longit graph as {"name": , "dates": , "nodes": {node: {attr}}, "edges": [node1, node2]}
        :param file_path: a string the destination
        """
        longit_dict = {"name": self.get_patient_name(),
                       "dates": self.get_patient_dates(),
                       "nodes": {n: self._graph.nodes[n] for n in self._graph.nodes()},
                       "edges": {e: self._graph.edges[e] for e in self._graph.edges()}}

        if os.path.exists(file_path):
            print(f"WARNING: a file is stored in {file_path}!")
        with open(file_path, 'w') as f:
            json.dump(longit_dict, f)

    def classify_nodes(self):
        """
        This function classifies the vertices, giving each vertex two classifications, one according to the previous layer and one
         according to the next layer.
         Classes according to previous:
        NEW: no edges towards previous layer
        EXISTING: at least one edge towards previous layer
         Classes according to next:
        DISAPPEARING: no edges towards next layer
        EXISTING: at least one edge towards next layer
        The function updates dictionary: vert2class: {vert: [class_prev, class_next]}
        """

        nodes2class = dict()
        for node in self._graph.nodes():
            node_layer = self._graph.nodes[node][NodeAttr.LAYER]
            is_first_layer = node_layer == 0
            is_last_layer = node_layer == self._num_of_layers - 1

            # get the vert neighbours:
            neighbours = [v for v in self._graph[node]]
            has_next = len([v for v in neighbours if self._graph.nodes[v][NodeAttr.LAYER] > node_layer]) > 0
            has_prev = len([v for v in neighbours if self._graph.nodes[v][NodeAttr.LAYER] < node_layer]) > 0

            if is_first_layer:
                if has_next:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.EXISTING_EXISTING}})
                else:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.EXISTING_DISAPPEARED}})
            elif is_last_layer:
                if has_prev:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.EXISTING_EXISTING}})
                else:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.NEW_EXISTING}})
            else:
                if has_next and has_prev:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.EXISTING_EXISTING}})
                elif has_next and not has_prev:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.NEW_EXISTING}})
                elif not has_next and has_prev:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.EXISTING_DISAPPEARED}})
                else:
                    nodes2class.update({node: {NodeAttr.CLASSIFICATION: NodesClass.NEW_DISAPPEARED}})

        nx.set_node_attributes(self._graph, nodes2class)

    def nodes_have_attribute(self, attr_name: str):
        """Check that the graph nodes have the attributes attr_name """
        graph = self.get_graph()
        for n in graph.nodes():
            if attr_name not in graph.nodes[n]:
                raise ValueError(f"There is one node in current graph that don`t have the {attr_name} attribute")

    def edges_have_attribute(self, attr_name: str):
        """Check that the graph edges have the attributes attr_name """
        graph = self.get_graph()
        for e in graph.edges():
            if attr_name not in graph.edges[e]:
                raise ValueError(f"There is one edge in current graph that don`t have the {attr_name} attribute")

    def add_node_attribute_from_scan(self, labeled_scans_list: List[np.array], attr_name: str, extracting_function, extra_arg=None):
        """Given a list of loaded labeled_scans, ordered by time point, update the attribute of each node, extracting the attribute
        through the extracting function.
        :param labeled_scans_list: a list of np.array, whose length is as num_of_layer. Each np.array should be a label matrix
        :param attr_name: the name of the attribute
        :param extracting_function: a function that gets as first argument an np.array and a second argument a label. It gives the
                attribute relative to the label
        :param extra_arg: (default None), an extra argument for the extracting function. If not None, must be a list of length num_of_layer
        """
        assert len(labeled_scans_list) == self._num_of_layers
        assert callable(extracting_function)
        if extra_arg is None:
            extra_arg = [None] * self._num_of_layers
        else:
            assert len(extra_arg) == self._num_of_layers

        node2attr = dict()
        for node in self._graph.nodes:
            # check that the node's label exists in the scan:
            node_label = self._graph.nodes[node][NodeAttr.LABEL]
            node_layer = self._graph.nodes[node][NodeAttr.LAYER]
            assert np.sum(labeled_scans_list[node_layer] == node_label) > 0
            node2attr.update({node: {attr_name:
                                     extracting_function(labeled_scans_list[node_layer], node_label, extra_arg[node_layer])}})
        nx.set_node_attributes(self._graph, node2attr)

    def add_node_attribute_from_dict(self, attr_dict, attr_name):
        """:param: attr_dict {node_id: attr_value}
        :param: attr_name, string
        This method upadtes the nodes attribute with a new attribute named attr_name, with values attr_dict"""
        input_nodes = set(attr_dict.keys())
        graph_nodes = set(self._graph.nodes)
        if len(graph_nodes - input_nodes) > 0:
            raise ValueError("Some graph nodes got no attribute!")
        nx.set_node_attributes(self._graph, {node: {attr_name: attr_value} for node, attr_value in attr_dict.items()})

    def add_edge_attribute_from_dict(self, attr_dict, attr_name):
        """:param: attr_dict {edge_id: attr_value}
        :param: attr_name, string
        This method upadtes the edge attribute with a new attribute named attr_name, with values attr_dict"""
        input_edges = set(attr_dict.keys())
        graph_edges = set(self._graph.edges)
        if len(graph_edges - input_edges) > 0:
            raise ValueError("Some graph edges got no attribute!")
        nx.set_edge_attributes(self._graph, {edge: {attr_name: attr_value} for edge, attr_value in attr_dict.items()})

    def add_cc_attribute(self):
        """To each node, add an attribute representing the connected component index"""
        cc_set = nx.connected_components(self._graph)
        node2cc_index = dict()
        for cc_index, cc in enumerate(cc_set):
            node2cc_index.update({n: cc_index + 1 for n in cc})
        nx.set_node_attributes(self._graph, {node: {NodeAttr.CC_INDEX: cc_idx} for node, cc_idx in node2cc_index.items()})

    @staticmethod
    def make_graph_directed(graph, backwards=False) -> nx.DiGraph:
        """Make graph a directed graph (only forward edges allowed)"""
        dir_graph = nx.DiGraph(graph)
        graph_nodes = dir_graph.nodes()
        if backwards:
            wrong_dir_edges = [e for e in dir_graph.edges() if
                               graph_nodes[e[0]][NodeAttr.LAYER] < graph_nodes[e[1]][NodeAttr.LAYER]]
        else:
            wrong_dir_edges = [e for e in dir_graph.edges() if
                               graph_nodes[e[0]][NodeAttr.LAYER] > graph_nodes[e[1]][NodeAttr.LAYER]]
        dir_graph.remove_edges_from(wrong_dir_edges)
        return dir_graph

    @staticmethod
    def edgeview2dict(edge_data_view, nodes_attr):
        """transform input edge_data_view is [(node0, node1, {attr})] in {(node0, node1): {attr}}. Use the nodes_attr
        dictionary to find the nodes layer and save the edges in 'forward' order"""
        edge_dict = dict()
        for ed in edge_data_view:
            layer0 = nodes_attr[ed[0]][NodeAttr.LAYER]
            layer1 = nodes_attr[ed[1]][NodeAttr.LAYER]
            if layer0 < layer1:
                edge_dict.update({tuple([ed[0], ed[1]]): ed[2]})
            elif layer0 > layer1:
                edge_dict.update({tuple([ed[1], ed[0]]): ed[2]})
            else:
                raise ValueError(f"Edge {ed} connecting the same layer")
        return edge_dict


class Drawer:
    """The base class for displaying a longit graph"""
    def __init__(self, longit: Longit):
        longit.nodes_have_attribute(NodeAttr.LAYER)
        longit.nodes_have_attribute(NodeAttr.LABEL)
        self._cnt = 0
        self._base_graph = longit.get_graph()
        self._num_of_layers = longit.get_num_of_layers()

        pat_name = longit.get_patient_name()
        if pat_name is None:
            self._patient_name = ""
        else:
            self._patient_name = pat_name

        pat_dates = longit.get_patient_dates()
        if pat_dates is None:
            self._patient_dates = [f"t{i}" for i in range(self._num_of_layers)]
        else:
            self._patient_dates = pat_dates
        nx.set_node_attributes(self._base_graph, values=False, name=NodeAttr.IS_PLACEHOLDER)

    def set_nodes_drawing_attributes(self):
        """Add to each node the color attribute GRAY"""
        nx.set_node_attributes(self._base_graph, values=Colors.GRAY, name=NodeAttr.COLOR)

    def set_edges_drawing_attributes(self):
        """Add to each node the color attribute BLACK and set the connection style"""
        nx.set_edge_attributes(self._base_graph, values=Colors.BLACK, name=EdgeAttr.COLOR)
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)

        connectivity = dict()
        for edge, is_skip in edge_is_skip.items():
            if not is_skip:
                connectivity.update({edge: {EdgeAttr.CONNECTION_STYLE: 'arc3'}})
            else:
                connectivity.update({edge: {EdgeAttr.CONNECTION_STYLE: 'arc3,rad=1'}})
        nx.set_edge_attributes(self._base_graph, connectivity)

    def set_graph_layout(self):
        """Stack graph's connected components one upon the other and fill the blanks with placeholders"""
        cc_subgraphs = [self._base_graph.subgraph(cc) for cc in nx.connected_components(self._base_graph)]
        if len(cc_subgraphs) == 0:
            return
        cc_graph = self.fill_with_placeholders(cc_subgraphs[0])
        for i in range(1, len(cc_subgraphs)):
            curr_cc_graph = self.fill_with_placeholders(cc_subgraphs[i])
            cc_graph = nx.compose(cc_graph, curr_cc_graph)
        self._base_graph = cc_graph

    def fill_with_placeholders(self, sub_graph_init: nx.Graph):
        """For each layer, fill the layer with placeholder nodes, such that all the layers will have the same number of
         nodes"""

        sub_graph = nx.Graph(sub_graph_init)
        # count how many nodes in each layer
        nodes_layers = list(nx.get_node_attributes(sub_graph, name=NodeAttr.LAYER).values())
        num_nodes_in_layer = [nodes_layers.count(layer) for layer in range(self._num_of_layers)]
        max_num_nodes_in_layer = max(num_nodes_in_layer)

        # extract the names of the attributes of a node
        node_attributes_names = sub_graph.nodes[list(sub_graph.nodes())[0]].keys()

        # fill with placeholders
        for layer in range(self._num_of_layers):
            for i in range(max_num_nodes_in_layer - num_nodes_in_layer[layer]):
                place_holder, ph_attr = self.create_placeholder(attributes=node_attributes_names, layer=layer)
                sub_graph.add_node(place_holder)
                nx.set_node_attributes(sub_graph, {place_holder: ph_attr})
        return sub_graph

    def create_placeholder(self, attributes: List, layer: int):
        """a placeholder is an empty node that has all the attributes as the other nodes zeroed, expect for
        the layer"""
        self._cnt += 1
        ph_attributes = {att: '' for att in attributes}
        ph_attributes[NodeAttr.IS_PLACEHOLDER] = True
        ph_attributes[NodeAttr.LAYER] = layer
        self.assign_placeholder_special_attr(ph_attributes)
        return f"{1000 + self._cnt}_{layer}", ph_attributes

    def assign_placeholder_special_attr(self, ph_attr):
        ph_attr[NodeAttr.LABEL] = ''
        ph_attr[NodeAttr.COLOR] = Colors.WHITE

    def show_graph(self, save_path=None):
        """This function sets the nodes and edges attributes, rearrange the view and call the drawing function"""
        if len(self._base_graph) == 0:
            return
        self.set_nodes_drawing_attributes()
        self.set_edges_drawing_attributes()
        self.set_graph_layout()

        num_of_vert_in_layer = nx.number_of_nodes(self._base_graph) / self._num_of_layers

        # To make arched edge the graph must be directed. Update _base_graph being a directed graph
        self._base_graph = Longit.make_graph_directed(self._base_graph)
        pos = nx.drawing.layout.multipartite_layout(self._base_graph, subset_key=NodeAttr.LAYER)
        #plt.figure(figsize=(self._num_of_layers * 2, num_of_vert_in_layer / 4 + 2))
        self.write_dates(pos)
        self.draw(pos)
        self.add_legend(nodes_position=pos)
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    def write_dates(self, nodes_position):
        """Prints the layers' dates at the bottom of the layers"""
        # x position of layers:
        nodes_pos_x = [pos[0] for pos in nodes_position.values()]
        layer_pos_x = np.unique(nodes_pos_x)
        nodes_pos_y = [pos[1] for pos in nodes_position.values()]
        lower_node = np.min(nodes_pos_y)

        text_positions = [(pos_x, lower_node - 0.2) for pos_x in layer_pos_x]
        for layer_idx in range(self._num_of_layers):
            current_text_pos = text_positions[layer_idx]
            plt.text(current_text_pos[0], current_text_pos[1], self._patient_dates[layer_idx],
                     horizontalalignment='center')

    def set_nodes_labels(self):
        return nx.get_node_attributes(self._base_graph, self.attr_to_print_on_nodes())

    def draw(self, pos):
        """This function prints the title of the figure and the graph"""
        plt.title(self._patient_name, fontsize=12)
        nx.draw_networkx_nodes(G=self._base_graph,
                               pos=pos,
                               node_color=list(nx.get_node_attributes(self._base_graph, NodeAttr.COLOR).values()))
        nx.draw_networkx_labels(G=self._base_graph,
                                pos=pos,
                                labels=self.set_nodes_labels())
        is_skip_edge = nx.get_edge_attributes(self._base_graph, EdgeAttr.IS_SKIP)
        nx.draw_networkx_edges(G=self._base_graph,
                               pos=pos,
                               edgelist=[e for e, is_skip in is_skip_edge.items() if not is_skip],
                               edge_color=[c for e, c in
                                           nx.get_edge_attributes(self._base_graph, EdgeAttr.COLOR).items() if
                                           not is_skip_edge[e]],
                               connectionstyle='arc3')
        nx.draw_networkx_edges(G=self._base_graph,
                               pos=pos,
                               edgelist=[e for e, is_skip in is_skip_edge.items() if is_skip],
                               edge_color=[c for e, c in
                                           nx.get_edge_attributes(self._base_graph, EdgeAttr.COLOR).items() if
                                           is_skip_edge[e]],
                               connectionstyle='arc3, rad=-0.1')

    def attr_to_print_on_nodes(self):
        """select the attibute to show on graphs' nodes"""
        return NodeAttr.LABEL

    def add_legend(self, nodes_position, attr_name=None, **kwarg):
        if attr_name is None:
            return
        if 'color_class' in kwarg.keys():
            color_class = kwarg['color_class']
            color_dict = color_class.select_dict(attr_name)
        else:
            color_dict = Colors.select_dict(attr_name)
        num_texts = len(color_dict)
        nodes_pos_x = [pos[0] for pos in nodes_position.values()]
        right_layer = np.unique(nodes_pos_x)[-1]
        # nodes_pos_y = [pos[1] for pos in nodes_position.values()]
        # upper_node = np.max(nodes_pos_y)
        text_positions = [(right_layer-i*0.1, 1.15) for i in range(num_texts)]
        for i, (text, color) in enumerate(color_dict.items()):
            plt.text(text_positions[i][0], text_positions[i][1],
                     f"{text}", color=color, fontweight='bold', horizontalalignment='center')


def get_labeled_segmentation(img, connectivity=1, size_filtering=20):
    """
    :param img: the image to label
    :param connectivity: (default 1)
    :param size_filtering: eliminate all the labels that have less than 20 voxels
    :return: a labeled image
    """
    label_img = label(img, connectivity=connectivity)
    cc_num = label_img.max()
    cc_areas = ndimage.sum(img, label_img, range(cc_num + 1))
    area_mask = (cc_areas < size_filtering)
    label_img[area_mask[label_img]] = 0

    return label_img.astype(np.int8)
