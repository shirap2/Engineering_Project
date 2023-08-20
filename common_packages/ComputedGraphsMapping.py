import numpy as np
import networkx as nx
import json


NO_MAPPING = 0


class GtPredMapping:
    """Single layer mapping"""
    def __init__(self, patient_name, date):
        """
        match is a list of lists. Each sub list is a pair: [gt_lb, pred_lb]
        """
        self.patient_name = patient_name
        self.date = date
        self.match = list()
        self.pred_mapping = None

    def add_gt2pred(self, gt2pred):
        """
        :param gt2pred: a tuple: (gt_lb, pred_lb) or (gt_lb, list of pred_lb)
        It updates the match list
        """
        gt_lb = gt2pred[0]
        if isinstance(gt2pred[1], list):
            for pred_lb in gt2pred[1]:
                self.match.append([gt_lb, pred_lb])
        else:
            self.match.append([gt_lb, gt2pred[1]])

    def add_unmatched_gt(self, gt_lb):
        self.match.append([gt_lb, NO_MAPPING])

    def add_unmatched_pred(self, pred_lb):
        self.match.append([NO_MAPPING, pred_lb])

    def get_pred_mapping(self):
        return self.pred_mapping

    def get_all_gt_lb(self):
        return sorted(list(set([edge[0] for edge in self.match if edge[0] != NO_MAPPING])))

    def get_all_pred_lb(self):
        return sorted(list(set([edge[1] for edge in self.match if edge[1] != NO_MAPPING])))

    def get_connected_matches(self):
        return [edge for edge in self.match if edge[0] != NO_MAPPING and edge[1] != NO_MAPPING]

    def get_matches_for_gt_lb(self, gt_lb):
        pred_lbs = [edge[1] for edge in self.match if edge[0]==gt_lb]
        return pred_lbs

    def get_matches_for_pred_lb(self, pred_lb):
        gt_lbs = [edge[0] for edge in self.match if edge[1]==pred_lb]
        return gt_lbs

    def get_match(self):
        return self.match

    def calculate(self, gt_segm, pred_segm, return_dictionary=False):
        """
        :param gt_segm: a gt lesion labeled segmentation (after small object removal)
        :param pred_segm: the corresponding lesion segmentation prediction (labeled, before mapping)
        :return: a list. If return_dictionary is true: list elements are dictionaries, whose keys are the labels in the gt
        image that have a pred label corresponding to them. Else: lst elements are matches (list).
        """
        if gt_segm.shape != pred_segm.shape:
            raise ValueError("gt and pred segmentation have different shapes!")

        labels_in_gt = np.unique(gt_segm)
        labels_in_gt = labels_in_gt[labels_in_gt > 0]

        for lb_gt in labels_in_gt:
            current_gt_tumor = (gt_segm == lb_gt)
            touched_pred_lb = np.unique(current_gt_tumor * pred_segm)
            touched_pred_lb = list(touched_pred_lb[touched_pred_lb > 0])  # eliminate the 0-label if it's present
            if len(touched_pred_lb) > 0:
                self.add_gt2pred((lb_gt, touched_pred_lb))

            else:
                self.add_unmatched_gt(lb_gt)

        labels_in_pred = np.unique(pred_segm)
        labels_in_pred = set(labels_in_pred[labels_in_pred > 0])

        all_pred_lb_in_match = set(self.get_all_pred_lb())
        unmatched_pred_lbs = labels_in_pred - all_pred_lb_in_match
        for unm_pred_lb in unmatched_pred_lbs:
            self.add_unmatched_pred(unm_pred_lb)

        assert sorted(list(labels_in_pred)) == self.get_all_pred_lb()
        assert sorted(list(labels_in_gt)) == self.get_all_gt_lb()

        if return_dictionary:
            self.pred_mapping = self.label_mapping()
            return self.pred_mapping
        else:
            return self.match

    def label_mapping(self):
        """
        Create a bipartite nx graph from the matching. The function creates a mapping (dictionary) assigning each predicted label the correspondent gt label
        """
        matching_graph = BiPartiteGraph(self)
        pred_gt_cc = list(nx.connected_components(matching_graph.get_graph()))
        nodes_attr = nx.get_node_attributes(matching_graph.get_graph(), 'bipartite')

        pred_mapping = dict()
        cnt = 2000
        for cc in pred_gt_cc:
            gt_nodes = [BiPartiteGraph.get_label(n) for n in cc if nodes_attr[n] == 'gt']
            pred_nodes = [BiPartiteGraph.get_label(n) for n in cc if nodes_attr[n] == 'pred']
            if len(cc) == 2:
                assert (len(gt_nodes) == 1 and len(pred_nodes) == 1)
                pred_mapping.update({pred_nodes[0]: gt_nodes[0]})
            elif len(cc) == 1:
                if len(pred_nodes) == 1:
                    pred_mapping.update({pred_nodes[0]: f"{cnt}"})
                    cnt += 1
            else:
                print(f"{self.patient_name}, {self.date}: pred_nodes: {pred_nodes} are matched with {gt_nodes}")

        return pred_mapping


class BiPartiteGraph:
    def __init__(self, gt_pred_matching: GtPredMapping):
        """
        Given a GtPredMatching graph, create a bi-partite networkx graph. This new graph has all the nx functions,
        such as connected components
        """
        bigraph = nx.Graph()
        gt_nodes = [f'{v}_gt' for v in gt_pred_matching.get_all_gt_lb()]
        pred_nodes = [f'{v}_pred' for v in gt_pred_matching.get_all_pred_lb()]
        edges = [(f'{int(e[0])}_gt', f'{int(e[1])}_pred') for e in gt_pred_matching.get_connected_matches()]
        bigraph.add_nodes_from(gt_nodes, bipartite='gt')
        bigraph.add_nodes_from(pred_nodes, bipartite='pred')
        bigraph.add_edges_from(edges)

        self.bigraph = bigraph

    @staticmethod
    def get_label(vert: str):
        return vert.replace("_gt", "").replace("_pred", "")

    def get_graph(self):
        return self.bigraph


class PatientMatchingBipartiteGraph:
    def __init__(self, mapping):
        """
        Given a mapping list (list of gt_labels, pred_labels, not one-to-one),
        create a bi-partite networkx graph. This new graph has all the nx functions,
        such as connected components
        """
        bigraph = nx.Graph()
        gt_nodes = [f'{v[0]}_gt' for v in mapping if not self.is_null(v[0])]
        pred_nodes = [f'{v[1]}_pred' for v in mapping if not self.is_null(v[1])]
        edges = [(f'{v[0]}_gt', f'{v[1]}_pred') for v in mapping if not (self.is_null(v[0]) or self.is_null(v[1]))]
        bigraph.add_nodes_from(gt_nodes, bipartite='gt')
        bigraph.add_nodes_from(pred_nodes, bipartite='pred')
        bigraph.add_edges_from(edges)

        self.bigraph = bigraph

    @staticmethod
    def get_label(vert: str):
        """
        get vert = f"{label}_{layer}_{gt/pred}"
        """
        return vert.replace("_gt", "").replace("_pred", "")

    @staticmethod
    def is_null(lb: str):
        """lb: f"{label}_{layer}". Return True if {label} == NO_MAPPING"""
        return int(lb.split('_')[0]) == NO_MAPPING

    def get_graph(self):
        return self.bigraph

    def rename_nodes(self):
        """This function assignes to each node a new name as follows, (relying on connected components):
        if is 1to0 or 1to1 or 1toMany cc, new_name=gt_label
        if is 0to1 cc, new_name=2000+pred_label
        if is Many2Many cc, new_name= minimum label of gt"""
        cc_list = nx.connected_components(self.bigraph)
        nodes2names = dict()
        for cc in cc_list:
            gt_nodes =[n for n in cc if self.bigraph.nodes[n]['bipartite'] == 'gt']
            num_gt_nodes = len(gt_nodes)
            if num_gt_nodes == 0:
                assert len(cc) == 1
                n_pred = cc.pop()
                n = self.get_label(n_pred)
                lb, layer = n.split('_')
                nodes2names.update({n_pred: f"{int(lb) + 2000}_{layer}"})
            elif num_gt_nodes == 1:
                n_gt = gt_nodes[0]
                n = self.get_label(n_gt)
                nodes2names.update({n_x: n for n_x in cc})
            else:
                min_gt_lb = min([int(self.get_label(n_gt).split('_')[0]) for n_gt in gt_nodes])
                layer = int(self.get_label(gt_nodes[0]).split('_')[1])
                n = f"{min_gt_lb}_{layer}"
                nodes2names.update({n_x: n for n_x in cc})
        nx.set_node_attributes(self.bigraph, nodes2names, name='mapping')
        return nodes2names

    def get_mapping(self, is_gt):
        if is_gt:
            type_label = 'gt'
        else:
            type_label = 'pred'
        nodes2name = self.rename_nodes()
        return {self.get_label(node): name for node, name in nodes2name.items() if node.endswith(type_label)}

class MapComputedGraphs:
    def __init__(self, patient_list=None):
        self._patient_list = patient_list
        self.mapping = None

    def get_matches_for_pred_node(self, pred_node):
        if self.mapping is None:
            raise ValueError("Mapping is None - Load a mapping!")
        gt_lbs = [edge[0] for edge in self.mapping if edge[1] == pred_node]
        return gt_lbs

    def get_matches_for_gt_node(self, gt_node):
        if self.mapping is None:
            raise ValueError("Mapping is None - Load a mapping!")
        pr_lbs = [edge[1] for edge in self.mapping if edge[0] == gt_node]
        return pr_lbs

    def is_pred_node_tp(self, pred_node):
        """Given a predicted node {lb}_{layer}, return True if it's True Positive."""
        gt_match = self.get_matches_for_pred_node(pred_node)
        gt_label = int(gt_match[0].split('_')[0])
        if gt_label == NO_MAPPING:
            return False
        return True

    def load_mapping(self, mapping_path):
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)
        return self.mapping

    def get_patient_gt_and_computed_segmentations(self, patient_name):
        """Return triplets of date, loaded gt and loaded computed segmentations. The segmentations are sorted by date"""
        raise ValueError("Abstract Function")

    def get_mapping_path(self, patient_name):
        """Return a path where to save the mapping"""
        raise ValueError("Abstract Function")

    @staticmethod
    def save_mapping_dict(path, patient_mappings):
        """path: desired location of the json that will be created.
        patient_mappings: a list of dictionaries. Each dictionary is one layer matching"""
        layer_mappings = dict()
        for time_ind, time_ind_mapping in enumerate(patient_mappings):
            layer_mappings.update({f"{pr_n}_{time_ind}": f"{gt_n}_{time_ind}" for pr_n, gt_n in time_ind_mapping.items()})
        with open(path, 'w') as f:
            json.dump(layer_mappings, f)

    @staticmethod
    def save_mapping(path, patient_mappings):
        """path: desired location of the json that will be created.
        patient_mappings: a list of lists. Each sub list is one layer matching"""
        layer_mappings = list()
        for time_ind, time_ind_mapping in enumerate(patient_mappings):
            for pr_n, gt_n in time_ind_mapping:
                layer_mappings.append([f"{pr_n}_{time_ind}", f"{gt_n}_{time_ind}"])
        with open(path, 'w') as f:
            json.dump(layer_mappings, f)


    def run(self):
        for patient in self._patient_list:
            patient_mappings = list()
            for date, gt_segmentation, comp_segmentation in self.get_patient_gt_and_computed_segmentations(patient):
                mapping = GtPredMapping(patient, date)
                pred_mapping = mapping.calculate(gt_segmentation, comp_segmentation)
                patient_mappings.append(pred_mapping)
            MapComputedGraphs.save_mapping(path=self.get_mapping_path(patient), patient_mappings=patient_mappings)






