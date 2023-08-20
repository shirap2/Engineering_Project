import copy

import networkx as nx
import numpy as np

from common_packages.MatchingGroupwisePackage import *
from common_packages.MatchingPairsPackage import *

class GwMatching_v3(GwMatching_v2):
    """This class will use as matching initialization the predicted pairwise matching graph"""
    def __init__(self, pw_graph_path, label2data=None, tensor=None, dist_threshold=23, intra_ll_threshold=17):
        super().__init__(label2data, tensor, dist_threshold, intra_ll_threshold)
        self._pw_graph_path = pw_graph_path

    def initial_match(self):
        """The longitudinal lesions are pw registration connected components"""
        matching_tensor = np.reshape(self.tensor, (np.product(self.tensor.shape[0:-1]),self.n_layers)) # dim (x*y*z, n_img)
        matching_unique = np.unique(matching_tensor, axis=0) # dim: (num of different connections, n_img)
        labels_layers_matrix = GwMatching.get_labels_layers_matrix(matching_unique) # matrix with {lb_Layer}

        unique_entries = np.unique(labels_layers_matrix)
        all_lesion_instances = [l for l in unique_entries if not GwMatching.iszero(l)]

        #edges = self.get_edges_from_mat(labels_layers_matrix)

        pw_loader = LoaderSimpleFromJson(self._pw_graph_path)

        # unshared lesions: keep only nodes that appear in gw registration
        pw_lesions = pw_loader.get_nodes()
        pw_edges = pw_loader.get_edges()
        lesions_in_pw_only = set(pw_lesions) - set(all_lesion_instances)
        if len(lesions_in_pw_only) > 0:
            removed_edges = []
            les2edge = {n: [] for n in pw_lesions}
            _ = [les2edge[n].append(e) for e in pw_edges for n in e]
            for pw_les in lesions_in_pw_only:
                for adj_e in les2edge[pw_les]:
                    removed_edges.append(adj_e)
            if len(removed_edges) > 0:
                print(f" removed #{len(removed_edges)} edges")
                _ = [pw_edges.remove(e) for e in removed_edges]

        pw_loader_corrected = LoaderSimple(labels_list=all_lesion_instances, edges_list=pw_edges)
        self.longit = LongitLongitudinalLesions(pw_loader_corrected, thresh=self.intra_threshold)
        self.longit.add_cc_attribute()
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.CENTROID] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.CENTROID)
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.VOL_VOXEL] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.VOL_VOXEL)

        self.longit.create_long_lesions_list()

    @staticmethod
    def cls_are_incompatible(cl1, cl2):
        """
        Since cls are from pw matching, cl layers are continuous
        if first_layer(cl2) - last_layer(cl1) = 1, their distance match may be a consecutive edge.
        Assuming that the pw matching is very good, avoid adding consecutive edges in this step. Make the cls incompatible
        """
        are_in_same_layers = GwMatching_v2.cls_are_incompatible(cl1, cl2)
        if are_in_same_layers:
            return True
        # check consequntiality
        cl1_layers_p = cl1.get_layers_presence()
        cl2_layers_p = cl2.get_layers_presence()
        layers = np.array([l for l in range(len(cl1_layers_p))]).astype(int)
        first_cl1_l = layers[cl1_layers_p][0]
        last_cl1_l = layers[cl1_layers_p][-1]
        first_cl2_l = layers[cl2_layers_p][0]
        last_cl2_l = layers[cl2_layers_p][-1]
        if first_cl1_l - last_cl2_l == 1 or first_cl2_l - last_cl1_l == 1:
            return True
        return False

    def sequential_edges_correction(self):
        self.longit.sequential_edges_correction(dont_touch_consecutive=True)


class GwGwMatching_v4(GwMatching_v3):
    def __init__(self, label2data=None, tensor=None, dist_threshold=23, intra_ll_threshold=17, dilation=7):
        super().__init__(None, label2data, tensor, dist_threshold, intra_ll_threshold)
        self._dilation = dilation

    def initial_match(self):
        """The longitudinal lesions are pw registration connected components"""
        layer2scan_pairs = {(l-1, l): [self.tensor[:,:,:,l-1], self.tensor[:,:,:,l]] for l in range(1, self.n_layers)}
        layer2voxel_spacing = {l: [1, 1, 1] for l in range(1, self.n_layers)}
        pw = PwMatching(layer2scan_pairs=layer2scan_pairs, layer2voxel_spacing=layer2voxel_spacing, dilate_param=self._dilation)
        loader = pw.run()

        self.longit = LongitLongitudinalLesions(loader, patient_dates=[f"{i}" for i in range(self.n_layers)] ,thresh=self.intra_threshold)
        self.longit.add_cc_attribute()
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.CENTROID] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.CENTROID)
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.VOL_VOXEL] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.VOL_VOXEL)

        self.longit.create_long_lesions_list()

class GwMatching_v4_1(GwGwMatching_v4):
    def __init__(self, label2data=None, tensor=None, dist_threshold=23, intra_ll_threshold=17, dilation=7):
        super().__init__(label2data, tensor, dist_threshold, intra_ll_threshold, dilation)
        self.INFINITY = 1000

    def distance_match(self):
        disappearing_ll = [ll for ll in self.longit.get_long_lesions_list() if not ll.check_completeness()]
        # _ = [ll.calculate_long_centroid() for ll in disappearing_ll]
        dist_graph = nx.complete_graph(disappearing_ll)

        incompatible_lls = [edge for edge in dist_graph.edges() if self.cls_are_incompatible(edge[0], edge[1])]
        dist_graph.remove_edges_from(incompatible_lls)
        if len(dist_graph.edges) == 0:
            return

        edges_dist_attr = {edge: self.cl_dilation_distance(edge) for edge in dist_graph.edges}
        nx.set_edge_attributes(dist_graph, edges_dist_attr)
        far_lls = [edge for edge, dil_dist in
                   nx.get_edge_attributes(dist_graph, MatchingDistance.DILATION_DISTANCE).items() if dil_dist == self.INFINITY]
        dist_graph.remove_edges_from(far_lls)
        if len(dist_graph.edges) == 0:
            return

        cc_edges = []
        for e_matches in nx.get_edge_attributes(dist_graph, MatchingDistance.MATCHING).values():
            cc_edges += list(e_matches.keys())

        loader = LoaderSimple(labels_list=list(self.longit.get_graph().nodes), edges_list=list(self.longit.get_graph().edges) + cc_edges)
        self.longit = Longit(loader)

    def cl_dilation_distance(self, edge):
        no_matches = False
        ll0: LongitudinalLesion = edge[0]
        ll1: LongitudinalLesion = edge[1]
        early0 = ll0.get_extreme_layer(earliest=True)
        early1 = ll1.get_extreme_layer(earliest=True)

        if early0 < early1:
            labels_prev = ll0.get_labels_in_extreme_layer(earliest=False)
            layer_prev = ll0.get_extreme_layer(earliest=False)
            labels_curr = ll1.get_labels_in_extreme_layer(earliest=True)
            layer_curr = early1
        else:
            labels_curr = ll0.get_labels_in_extreme_layer(earliest=True)
            layer_curr = early0
            labels_prev = ll1.get_labels_in_extreme_layer(earliest=False)
            layer_prev = ll1.get_extreme_layer(earliest=False)

        segm_prev = self.tensor[:, :, :, layer_prev].astype(int)
        mask_prev = segm_prev.copy()
        mask_prev[np.isin(segm_prev, labels_prev, invert=True)] = 0

        segm_curr = self.tensor[:, :, :, layer_curr].astype(int)
        mask_curr = segm_curr.copy()
        mask_curr[np.isin(segm_curr, labels_curr, invert=True)] = 0

        matches = PwMatching.match_2_cases_v5(mask_prev, mask_curr, voxelspacing=[1,1,1], max_dilate_param=self._dilation, return_iteration_indicator=True)
        if len(matches) == 0:
            no_matches = True
        if no_matches:
            return {MatchingDistance.DILATION_DISTANCE: self.INFINITY, MatchingDistance.MATCHING: {}}
        return {MatchingDistance.DILATION_DISTANCE: np.min([m[0] for m in matches]),
                MatchingDistance.MATCHING: {(f"{int(m[1][0])}_{layer_prev}", f"{int(m[1][1])}_{layer_curr}"): m[0] for m in matches}}


class GwMatching_v4_2(GwMatching_v4_1):
    def __init__(self, label2data=None, tensor=None, dist_threshold=23, intra_ll_threshold=17, dilation=5):
        super().__init__(label2data, tensor, dist_threshold, intra_ll_threshold, dilation)

    def distance_match(self):
        disappearing_ll = [ll for ll in self.longit.get_long_lesions_list() if not ll.check_completeness()]
        _ = [ll.calculate_long_centroid() for ll in disappearing_ll]
        dist_graph = nx.complete_graph(disappearing_ll)

        incompatible_lls = [edge for edge in dist_graph.edges() if self.cls_are_incompatible(edge[0], edge[1])]
        dist_graph.remove_edges_from(incompatible_lls)
        if len(dist_graph.edges) == 0:
            return

        distances = {edge: self.cl_distance(edge[0], edge[1]) for edge in dist_graph.edges}
        nx.set_edge_attributes(dist_graph, distances, name=MatchingDistance.CENTROID_DISTANCE)
        far_centr_lls = [edge for edge, dist in
                         nx.get_edge_attributes(dist_graph, name=MatchingDistance.CENTROID_DISTANCE).items()
                         if dist > self.dist_threshold]

        dist_graph.remove_edges_from(far_centr_lls)
        if len(dist_graph.edges) == 0:
            return

        edges_dist_attr = {edge: self.cl_dilation_distance(edge) for edge in dist_graph.edges}
        nx.set_edge_attributes(dist_graph, edges_dist_attr)
        far_lls = [edge for edge, dil_dist in
                   nx.get_edge_attributes(dist_graph, MatchingDistance.DILATION_DISTANCE).items()
                   if dil_dist == self.INFINITY]
        dist_graph.remove_edges_from(far_lls)
        if len(dist_graph.edges) == 0:
            return

        ll_unified_list = []
        working_dist_graph = copy.deepcopy(dist_graph)
        num_ll = len(dist_graph)
        GwMatching_v2.keep_best_cliques(ll_clustered_list=ll_unified_list, ll_graph=working_dist_graph, num_ll=num_ll,
                                        dist_metrics=MatchingDistance.DILATION_DISTANCE)
        for ll_unified in ll_unified_list:
            if len(ll_unified) == 1:
                continue
            for ll in ll_unified:
                ll_connections = dist_graph.edges(ll)
                edges_out_clique = [e for e in ll_connections if (e[0] not in ll_unified) or (e[1] not in ll_unified)]
                dist_graph.remove_edges_from(edges_out_clique)

        cc_edges = []
        for e_matches in nx.get_edge_attributes(dist_graph, MatchingDistance.MATCHING).values():
            cc_edges += list(e_matches.keys())

        loader = LoaderSimple(labels_list=list(self.longit.get_graph().nodes),
                              edges_list=list(self.longit.get_graph().edges) + cc_edges)
        self.longit = Longit(loader)


class GwGwMatching_v5(GwMatching_v3):
    def __init__(self, label2data=None, tensor=None, dist_threshold=23, intra_ll_threshold=17):
        super().__init__(None, label2data, tensor, dist_threshold, intra_ll_threshold)

    def initial_match(self):
        """The longitudinal lesions are pw registration connected components"""
        layer2scan_pairs = {}
        for prev_l in range(0, self.n_layers - 1):
            for curr_l in range(prev_l+1, self.n_layers):
                layer2scan_pairs.update({(prev_l, curr_l): (self.tensor[:, :, :, prev_l], self.tensor[:, :, :, curr_l])})

        layer2voxel_spacing = {l: [1, 1, 1] for l in range(1, self.n_layers)}
        pw = PwMatching(layer2scan_pairs=layer2scan_pairs, layer2voxel_spacing=layer2voxel_spacing, dilate_param=7)
        loader = pw.run()

        self.longit = LongitLongitudinalLesions(loader, thresh=self.intra_threshold)
        self.longit.add_cc_attribute()
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.CENTROID] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.CENTROID)
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.VOL_VOXEL] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.VOL_VOXEL)

        self.longit.create_long_lesions_list()
        self.sequential_edges_correction()
