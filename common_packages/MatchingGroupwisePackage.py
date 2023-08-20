import networkx as nx
import os

from skimage.measure import regionprops
import itertools
import copy
from common_packages.LongGraphPackage import *


class NodeAttrMatching(NodeAttr):
    CENTROID = "centroid"
    VOL_VOXEL = "vol_voxels"

class MatchingDistance:
    DILATION_DISTANCE = "dil_distance"
    CENTROID_DISTANCE = "cent_distance"
    MATCHING = "matching"

NODES = "nodes"
EDGES = "edges"
LABELS = "labels"
CENTROID = "centroid"
VOL_VOXEL = "voxels"

CENTROID_IDX = 0
VOL_IDX = 1


def vert2layer(vert):
    """vert (string) ='lb_layer' --> layer (int) """
    return int(float((vert.split('_')[1])))

def vert2lb(vert):
    """vert (string) ='lb_layer' --> lb (int) """
    return int(float((vert.split('_')[0])))


class LongitudinalLesion:
    def __init__(self, long_lesion_graph: nx.DiGraph(), n_layers: int):
        """Longitudinal lesion holds the graph of a connected component in a matching graph with n_layers.
        long_lesion_graph is a subgraph of the main graph"""
        self._lesion_graph = long_lesion_graph
        self._tot_num_of_layers = n_layers
        nodes_layers = np.unique(list(nx.get_node_attributes(self._lesion_graph, NodeAttr.LAYER).values()))
        self._layers_presence = np.array([layer in nodes_layers for layer in range(self._tot_num_of_layers)])
        self._is_complete = self.check_completeness()
        self._long_centroid = None
        self._identifier = self.set_identifier()

    def __hash__(self):
        return self._identifier

    def __eq__(self, other):
        return self.__hash__() == hash(other)

    def set_str_identifier(self):
        """Longitudinal lesion id is a string containing the cc of the initial graph cc"""
        cc_indices = np.unique(list(nx.get_node_attributes(self._lesion_graph, NodeAttr.CC_INDEX).values()))
        init_identifier = ""
        for cc_ind in cc_indices:
            init_identifier += f"{cc_ind}_"
        return init_identifier[:-1]

    def set_identifier(self):
        """Longitudinal lesion id is a string containing the cc of the initial graph cc"""
        cc_indices = np.unique(list(nx.get_node_attributes(self._lesion_graph, NodeAttr.CC_INDEX).values()))
        init_identifier = 0
        for cc_ind in cc_indices:
            init_identifier += cc_ind
            init_identifier *= 1000
        return int(init_identifier/1000)

    def get_long_centroid(self):
        return self._long_centroid

    def get_graph(self):
        return self._lesion_graph

    def get_layers_presence(self, layers_id=False):
        """Get an np.array of length: patient's #timepoints, where:
        (if layers_id == False)True represent presence of lesion in that layer
        else: layers_id in which the lesions of the ll are found"""
        if layers_id:
            return np.arange(self._tot_num_of_layers)[self._layers_presence]
        return self._layers_presence

    def get_extreme_layer(self, earliest: bool) -> int:
        """(For ll dwelling in consecutive layers):
           Return the earliest/latest layer of the ll"""
        layer_presence_ids = self.get_layers_presence(layers_id=True)
        if len(layer_presence_ids) > 0:
            is_continuous = np.all(layer_presence_ids[1:] - layer_presence_ids[:-1] == 1)
            assert is_continuous  # check if the ll occupies consecutive layers
        if earliest:
            extreme_layer = layer_presence_ids[0]
        else:
            extreme_layer = layer_presence_ids[-1]
        return extreme_layer

    def get_labels_in_extreme_layer(self, earliest: bool):
        """For ll in consecutive layers:
        Get a list of the lesions in lb format dwelling in the earliest/latest layer of the ll"""
        extreme_layer = self.get_extreme_layer(earliest)
        les2layer = nx.get_node_attributes(self._lesion_graph, NodeAttr.LAYER)
        les2labels = nx.get_node_attributes(self._lesion_graph, NodeAttr.LABEL)
        return [les2labels[les] for les, layer in les2layer.items() if layer == extreme_layer]

    def check_completeness(self):
        "Check if the longitudinal lesion is present in all the layers"
        return np.all(self._layers_presence)

    def calculate_long_centroid(self):
        lesion_instances_centroids_tuples = nx.get_node_attributes(self._lesion_graph, NodeAttrMatching.CENTROID)
        lesion_instances_centroids = [np.array(c) for c in lesion_instances_centroids_tuples.values()]
        mean_centroid = np.mean(lesion_instances_centroids, axis=0)
        #assert mean_centroid.shape == (3, )
        self._long_centroid = mean_centroid


class LongitLongitudinalLesions(Longit):
    """
    Expantion of Longit. This class holds a list of LongitudinalLesion that make up the graph. Its methods operate on
    the LongitudinalLesions
    """
    def __init__(self, loader: Loader, thresh: int, patient_name=None, patient_dates=None):
        super().__init__(loader, patient_name, patient_dates)
        self._long_lesions_list = list()
        self._num_long_lesions = None
        self._thresh = thresh

    def update_longitudinal_centroids(self):
        for long_lesion in self._long_lesions_list:
            long_lesion.calculate_long_centroid()

    @staticmethod
    def centroid_distance(c1, c2):
        c1 = np.array(c1)
        c2 = np.array(c2)
        return np.linalg.norm(c1 - c2, ord=2)

    def sort_edges_by_length(self, edge):
        node0 = edge[0]
        node1 = edge[1]
        layer0 = self._graph.nodes[node0][NodeAttr.LAYER]
        layer1 = self._graph.nodes[node1][NodeAttr.LAYER]
        return abs(layer1 - layer0)

    def create_long_lesions_list(self):
        """
        Update the field 'long_lesion_list'. This field contains a list of the LongitudinalLesions of the Longit graph
        """
        self.nodes_have_attribute(NodeAttr.CC_INDEX)
        nodes2cc_indices = nx.get_node_attributes(self._graph, name=NodeAttr.CC_INDEX)
        cc_indices = list(np.unique(list(nodes2cc_indices.values())))
        self._num_long_lesions = len(cc_indices)
        for current_cc_ind in cc_indices:
            cc_subgraph = self._graph.subgraph([n for n, cc_ind in nodes2cc_indices.items() if cc_ind == current_cc_ind])
            cc_directed_subgraph = Longit.make_graph_directed(cc_subgraph)
            long_lesion = LongitudinalLesion(cc_directed_subgraph, self._num_of_layers)
            self._long_lesions_list.append(long_lesion)

    def sequential_edges_correction_v0(self):
        """
        Go over all the skip edges in each long_lesion. Delete the skip edge if it connects two already connected nodes.
        Switch the skip edge (a_layer{i}, b_layer{i+N}) with a shorter edge (a_layer{i}, c_layer{i+M}), M < N with
        the target node c_layer{i+M} in a mid layer (0<M<N) and distance(centroid(source node), centroid(target node) < threshold
        """
        for long_lesion in self._long_lesions_list:
            long_lesion_graph = long_lesion.get_graph()
            #initial_long_lesion_graph = long_lesion_graph.copy()
            nodes2layers = nx.get_node_attributes(long_lesion_graph, name=NodeAttr.LAYER)
            edges2is_skip = nx.get_edge_attributes(long_lesion_graph, name=EdgeAttr.IS_SKIP)
            skip_edges = [e for e, is_skip in edges2is_skip.items() if is_skip]
            for s_edge in skip_edges:
                node0 = s_edge[0]
                node1 = s_edge[1]
                node0_layer = nodes2layers[node0]
                node1_layer = nodes2layers[node1]

                long_lesion_graph.remove_edge(*s_edge)

                if nx.has_path(long_lesion_graph, source=node0, target=node1):
                    # there is another directed path between the two nodes, do not add back the edge
                    continue
                self.add_edge_with_skip_edge_attr(long_lesion_graph, *s_edge)

                # check if no lesion is found in timepoints between node0_layer and node1_layer
                mid_layers_presence = [pres for layer, pres in enumerate(long_lesion.get_layers_presence())
                                       if (layer > node0_layer) and (layer < node1_layer)]
                if not any(mid_layers_presence):
                    continue

                for mid_layer, nodes_presence in enumerate(long_lesion.get_layers_presence()):
                    if (mid_layer <= node0_layer) or (mid_layer >= node1_layer):
                        # the mid_layer is not a mid layer
                        continue
                    if not nodes_presence:
                        # there are no nodes in the mid_layer
                        continue

                    node0_centroid = long_lesion_graph.nodes[node0][NodeAttrMatching.CENTROID]
                    nodes_in_mid_layer = [n for n, layer in nodes2layers.items() if layer == mid_layer]
                    candidate_nodes2centroids = {n: long_lesion_graph.nodes[n][NodeAttrMatching.CENTROID]
                                                 for n in nodes_in_mid_layer
                                                 if nx.has_path(long_lesion_graph, source=n, target=node1)}
                    if len(candidate_nodes2centroids) == 0:
                        # no candidate target node
                        continue

                    distances = [self.centroid_distance(node0_centroid, n_centroid)
                                 for n_centroid in candidate_nodes2centroids.values()]

                    shortest_dist_arg = np.argmin(distances)
                    shortest_dist = distances[shortest_dist_arg]
                    node_best_candidate = list(candidate_nodes2centroids.keys())[shortest_dist_arg]
                    if shortest_dist < self._thresh:
                        long_lesion_graph.remove_edge(*s_edge)
                        self.add_edge_with_skip_edge_attr(long_lesion_graph, node0, node_best_candidate)
                        break
        self.update_main_graph()

    def sequential_edges_correction(self, dont_touch_consecutive=False):
        """
        For each long lesion graph, (where an edge between two nodes means: >0 overlap)
        (1) check the possibility of adding new edges between nodes of consecutive layers (edge iff centroid distance is
        below threshold
        (2) delete improper skip edges: go over all the long lesions 'overlap' skip edges and delete those that
        have a parallel path between their vertices (passing through shorter edges)
        """
        for long_lesion in self._long_lesions_list:
            long_lesion_graph = long_lesion.get_graph()
            overlapping_edges = Longit.edgeview2dict(long_lesion_graph.edges(data=True),
                                                     long_lesion_graph.nodes(data=True))

            self.add_sequential_edges(long_lesion, dont_touch_consecutive)
            self.remove_improper_skip_edges(long_lesion, overlapping_edges)

        self.update_main_graph()

    def remove_improper_skip_edges(self, long_lesion: LongitudinalLesion, overlapping_edges: Dict):
        """
        Delete improper skip edges: go over all the long lesions 'overlap' skip edges and delete those that
        have a parallel path between their vertices (passing through shorter edges)
        :param long_lesion: the current longitudinal lesion
        :param overlapping_edges: a dict {edge: edge_attributes}, where edge_attributes = {attr_name: attr_value}
        """
        long_lesion_graph = long_lesion.get_graph()
        overlapping_skip_edges = [e for e, attr in overlapping_edges.items() if attr[EdgeAttr.IS_SKIP]]
        # sort overlapping skip edges by length:
        overlapping_skip_edges_sorted = sorted(overlapping_skip_edges, key=self.sort_edges_by_length, reverse=True)

        # delete all s.e. that have a parallel path, starting by the longest edges.
        for e in overlapping_skip_edges_sorted:
            long_lesion_graph.remove_edge(*e)
            start_node = e[0]
            target_node = e[1]
            if not nx.has_path(long_lesion_graph, start_node, target_node):
                self.add_edge_with_skip_edge_attr(long_lesion_graph, *e)

    def add_sequential_edges(self, long_lesion: LongitudinalLesion, dont_touch_consecutive=False):
        """
        Get a long_lesion, whose edges connect overlapping nodes only.
        Add new edges between nodes of sequential layers iff centroid distance is
        below threshold.
        dont_touch_consecutive: (default False). If True, do not try to add new edges between layer i and layer i+1. Add
        new edges only if layers gap is more than 1. (add only skip edges)
        ** This function updates the long_lesion graph **
        """
        long_lesion_graph = long_lesion.get_graph()
        overlapping_edges = list(long_lesion_graph.edges())
        nodes2layers = nx.get_node_attributes(long_lesion_graph, name=NodeAttr.LAYER)
        long_lesion_layers = [layer for layer, is_present in enumerate(long_lesion.get_layers_presence())
                              if is_present]

        # check all the possible edges between nodes of two sequential layers
        for layer_prev, layer_next in zip(long_lesion_layers, long_lesion_layers[1:]):
            if dont_touch_consecutive and (layer_next - layer_prev == 1):
                continue
            nodes_layer_prev = [n for n, ly in nodes2layers.items() if ly == layer_prev]
            nodes_layer_next = [n for n, ly in nodes2layers.items() if ly == layer_next]

            for e in itertools.product(nodes_layer_prev, nodes_layer_next):
                if e not in overlapping_edges:
                    start_node = e[0]
                    target_node = e[1]
                    start_centroid = long_lesion_graph.nodes[start_node][NodeAttrMatching.CENTROID]
                    target_centroid = long_lesion_graph.nodes[target_node][NodeAttrMatching.CENTROID]
                    if self.centroid_distance(start_centroid, target_centroid) < self._thresh:
                        self.add_edge_with_skip_edge_attr(long_lesion_graph, *e)

    def add_edge_with_skip_edge_attr(self, les_graph: nx.DiGraph, start_node, target_node):
        start_layer = self._graph.nodes[start_node][NodeAttr.LAYER]
        target_layer = self._graph.nodes[target_node][NodeAttr.LAYER]
        is_skip_e = (target_layer>start_layer+1)
        les_graph.add_edge(start_node, target_node)
        les_graph.edges[(start_node, target_node)][EdgeAttr.IS_SKIP] = is_skip_e

    def update_main_graph(self):
        """
        Switch the current graph with a new graph, with the edges of longitudinal_lesions
        """
        updated_graph = nx.union_all([ll.get_graph() for ll in self._long_lesions_list])
        self._graph = nx.Graph(updated_graph) #make it undirected

    def get_long_lesions_list(self):
        return self._long_lesions_list

    def unify_long_lesions(self, list_of_long_lesions: List[LongitudinalLesion], list_of_edges=None):
        if len(list_of_long_lesions) < 2:
            return
        unified_graph = nx.union_all([ll.get_graph() for ll in list_of_long_lesions])

        if list_of_edges is not None:
            num_nodes = len(unified_graph)
            unified_graph.add_edges_from(list_of_edges)
            if len(unified_graph) > num_nodes:
                raise ValueError("New nodes were added by adding edges!")

        unified_ll = LongitudinalLesion(long_lesion_graph=unified_graph, n_layers=self.get_num_of_layers())
        for ll in list_of_long_lesions:
            self._long_lesions_list.remove(ll)
        self._long_lesions_list.append(unified_ll)


class LongitLongitudinalLesionsSimple(Longit):
    """
        Expantion of Longit. This class holds a list of LongitudinalLesion that make up the graph. Its methods operate on
        the LongitudinalLesions
        (a LongitudinalLesion is a connected component on the lesion matching graph)
        """

    def __init__(self, loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(loader, patient_name, patient_dates)
        self._long_lesions_list = list()
        self._num_long_lesions = None

    def create_long_lesions_list(self):
        """
        Update the field 'long_lesion_list'. This field contains a list of the LongitudinalLesions of the Longit graph
        """
        self.nodes_have_attribute(NodeAttr.CC_INDEX)
        nodes2cc_indices = nx.get_node_attributes(self._graph, name=NodeAttr.CC_INDEX)
        cc_indices = list(np.unique(list(nodes2cc_indices.values())))
        self._num_long_lesions = len(cc_indices)
        for current_cc_ind in cc_indices:
            cc_subgraph = self._graph.subgraph([n for n, cc_ind in nodes2cc_indices.items() if cc_ind == current_cc_ind])
            cc_directed_subgraph = Longit.make_graph_directed(cc_subgraph)
            long_lesion = LongitudinalLesion(cc_directed_subgraph, self._num_of_layers)
            self._long_lesions_list.append(long_lesion)

    def get_long_lesions_list(self):
        return self._long_lesions_list

class GwMatching:
    def __init__(self, label2data=None, tensor=None, dist_threshold=20):
        """
        :param label2data: is a list of dictionaries. Each dictionary is a layer. The dictionary index inside the list is the layer
        index. The layer is: {lb : ((centroid), volume)
        :param tensor: is a numpy array of dim: (x_img, y_img, z_img, n_layers)
        """
        self.cl_list = list()
        self.lesions_data = label2data
        self.tensor = tensor
        self.dist_threshold = dist_threshold
        self.n_layers = self.tensor.shape[-1]

    def run(self):
        self.initial_match()
        self.distance_match()
        return self.get_longit()

    def save_matching_graph(self, graph_path):
        """
        The matching is saved as {'nodes': vertices, 'edges': edges}, where both vertices and edges use the notation {lb}_{layer}
        """
        edges = self.get_match_list()
        lab_list = self.get_labels_list()
        vertices = [f"{lb}_{layer_idx}" for layer_idx, layer_lab_list in enumerate(lab_list) for lb in layer_lab_list]
        graph_dict = {'nodes': vertices, 'edges': edges}
        with open(graph_path, 'w') as f:
            json.dump(graph_dict, f)

    @staticmethod
    def load_matching_graph(graph_path=None, complete_cc=True, loaded_graph=None):
        """
        The matching is saved as {'nodes': vertices, 'edges': edges}, where both vertices and edges use the notation {lb}_{layer}
        The load function gives the labels_list [[labels of layer 1], [labels of layer 2], ...], in format {lb} and the edges
        in format {lb}_{layer}. If complete_cc is True, it also adds all the edges between consequent layers in a connected component.
        This function can also be inputted a loaded json (used for debugging).
        """
        raise ValueError("Deprecated function!")


        if graph_path is not None:
            with open(graph_path, 'r') as f:
                graph_dict = json.load(f)
        elif loaded_graph is not None:
            graph_dict = loaded_graph
        else:
            raise ValueError("Bad inputs!")

        last_layer = np.max([vert2layer(v) for v in graph_dict['nodes']])
        labels_list = [[] for i in range(last_layer + 1)]
        for v in graph_dict['nodes']:
            labels_list[vert2layer(v)].append(vert2lb(v))
        if not complete_cc:
            return labels_list, graph_dict['edges']
        else:
            edges = graph_dict['edges']
            complete_edges = []
            nx_graph = nx.Graph()
            layers = [l for l in range(last_layer + 1)]
            for layer in layers:
                nx_graph.add_nodes_from([LongitudinalGraph.lb2vert(lb, layer) for lb in labels_list[layer]], layer=layer)
            nx_graph.add_edges_from(edges)
            ccs = nx.connected_components(nx_graph) # take the loaded graph connected components
            for cc in ccs:
                cc_layers = list(set(LongitudinalGraph.vert2layer(v) for v in cc))
                for i in range(len(cc_layers) - 1):
                    layer1 = cc_layers[i]
                    layer2 = cc_layers[i+1]
                    vert_layer1 = [v for v in cc if LongitudinalGraph.vert2layer(v) == layer1]
                    vert_layer2 = [v for v in cc if LongitudinalGraph.vert2layer(v) == layer2]
                    # create all the possible edges between two layers of the same connected component
                    for e in itertools.product(vert_layer1, vert_layer2):
                        complete_edges.append(list(e))
            if len(complete_edges) > len(edges):
                print(f"Added {len(complete_edges) - len(edges)} edges")
            return labels_list, complete_edges

    def get_labels_list(self):
        """
        Get all all the labels in the format: list of lists [[labels of layer 1], [labels of layer 2], ...]
        """
        labels_list = [[] for i in range(self.n_layers)]
        for cl in self.cl_list:
            for vert in cl.match:
                labels_list[vert2layer(vert)].append(vert2lb(vert))
        # check that in empty layers there is 0:
        labels_list_copy = labels_list.copy()
        for i, layer_list in enumerate(labels_list_copy):
            if len(layer_list) == 0:
                labels_list[i].append(0)
        return labels_list

    def get_match_list(self):
        """
        Get all all the "edges" in the format: list of lists [[vert1, vert2], [vert3, vert4]..].
        The vert are written in the format {lb}_{layer}
        """
        match_list = list()
        for cl in self.cl_list:
            layers_idx_presence = np.arange(self.n_layers)[cl.layers_presence]
            if len(layers_idx_presence) == 1:  # if all labels of cl are in one layer, skip
                continue

            for layer_idx in range(len(layers_idx_presence)-1):
                layer0 = layers_idx_presence[layer_idx]
                layer1 = layers_idx_presence[layer_idx+1]
                layer0_vert = [vert for vert in cl.match if vert2layer(vert) == layer0]
                layer1_vert = [vert for vert in cl.match if vert2layer(vert) == layer1]
                # create all the pairs between the two layers
                pair_match = list(itertools.product(layer0_vert, layer1_vert))
                match_list += pair_match

        return match_list

    @staticmethod
    def get_labels_layers_matrix(labels_matrix):
        """
        Get a matrix of integers (labels_matrix). Returns a matrix of stings (string_matrix) of the same size, where:
        string_matrix[i,j] = f"{label_matrix[i,j]}_{j}" i.e. each number is replaced by the number and the number of column
        """
        n_matches = labels_matrix.shape[0]
        labels_layers_matrix = []
        for m in range(n_matches):
            labels_layers_matrix.append([f"{int(el)}_{int(col_num)}" for col_num, el in enumerate(labels_matrix[m,:])])
        return labels_layers_matrix

    def unify_matches(self, labels_layers_matrix):
        """
        Get a labels_layers_matrix (each entry is {label}_{layer}). Matrix rows are matches, matrix columns are layers
        Return a list of matches, such that: (1) there is no '0' label. (2) there are no sub groups: for example: if the
        match: [1_0, 1_1, 1_2] is present, the function will delete the match [1_0, 1_1], that is a subset of the first.
        (3) Matches with a common lesion will be unified: [1_0, 1_1] and [1_1, 1_2] will be unified to [1_0, 1_1, 1_2]
        """
        # create a list of matches, without '0' labels
        matches_list = [[el for el in match if not self.iszero(el)] for match in labels_layers_matrix]
        edges_list = [[match[lesion_idx], match[lesion_idx+1]] for match in matches_list for lesion_idx in range(len(match)-1)]
        vert_list = [m for match in matches_list for m in match]
        connectivity_graph = nx.Graph()
        connectivity_graph.add_nodes_from(vert_list)
        connectivity_graph.add_edges_from(edges_list)
        con_comp_generator = nx.connected_components(connectivity_graph)
        con_comp = [sorted(list(cc), key=lambda u:vert2layer(u)) for cc in con_comp_generator]
        return con_comp

    @staticmethod
    def iszero(lab_layer):
        """True: if the label is a placeholder"""
        return lab_layer.startswith("0")

    def initial_match(self):
        matching_tensor = np.reshape(self.tensor, (np.product(self.tensor.shape[0:-1]),self.n_layers)) # dim (x*y*z, n_img)
        matching_unique = np.unique(matching_tensor, axis=0) # dim: (num of different connections, n_img)
        labels_layers_matrix = self.get_labels_layers_matrix(matching_unique)
        match_list = self.unify_matches(labels_layers_matrix)
        for match in match_list:
            cl = ConnectedLesion(match, self.lesions_data, self.n_layers)
            self.cl_list.append(cl)

    @staticmethod
    def cl_distance(cl1, cl2):
        distance = np.sum((cl1.centroid - cl2.centroid) ** 2) ** 0.5
        return distance

    @staticmethod
    def cls_are_incompatible(cl1, cl2):
        """
        True if there is at least one layer in which cl1 and cl2 have one lesion.
        """
        return np.any(cl1.layers_presence[cl2.layers_presence])

    def distance_match(self):
        """
        Build a graph whose vertices are long lesions. Draw edges between two compatible long lesions
        whose centroids are close enough. Unify all long lesions in the graph connected component
        ** incompatible long lesions may be connected! **:
        Ex: A--B--C. A and C are incompatible, but A--B and B--C. A and C get unified!
        """

        not_complete_cl = [cl for cl in self.cl_list if not cl.is_complete]
        dist_graph = nx.complete_graph(len(not_complete_cl))

        incompatible_cls = [edge for edge in dist_graph.edges if self.cls_are_incompatible(not_complete_cl[edge[0]], not_complete_cl[edge[1]])]
        dist_graph.remove_edges_from(incompatible_cls)
        if len(dist_graph.edges) == 0:
            return
        distances = {edge: self.cl_distance(not_complete_cl[edge[0]], not_complete_cl[edge[1]]) for edge in dist_graph.edges}
        nx.set_edge_attributes(dist_graph, distances, name='distance')
        far_cls = [edge for edge, dist in nx.get_edge_attributes(dist_graph, 'distance').items() if dist > self.dist_threshold]
        dist_graph.remove_edges_from(far_cls)
        if len(dist_graph.edges) == 0:
            return
        con_comp_generator = nx.connected_components(dist_graph)
        for connected_cls in con_comp_generator:
            if len(connected_cls) > 1:
                print(f"merged:{len(connected_cls)} cc")
                connected_cls = list(connected_cls)
                base_cl_pntr = not_complete_cl[connected_cls[0]]
                base_cl = copy.deepcopy(not_complete_cl[connected_cls[0]])
                self.cl_list.remove(base_cl_pntr)
                for cl_idx in range(1, len(connected_cls)):
                    cl = not_complete_cl[connected_cls[cl_idx]]
                    base_cl.update_connected_lesion(cl.match)
                    self.cl_list.remove(cl)
                self.cl_list.append(base_cl)

    def get_longit(self):
        edges = self.get_match_list()
        lab_list = self.get_labels_list()
        vertices = [f"{lb}_{layer_idx}" for layer_idx, layer_lab_list in enumerate(lab_list) for lb in layer_lab_list]
        loader = LoaderSimple(labels_list=vertices, edges_list=edges)
        return Longit(loader=loader)


class GwMatching_v2:
    def __init__(self, label2data=None, tensor=None, dist_threshold=23, intra_ll_threshold=17):
        self.lesions_data = label2data
        self.tensor = tensor
        if intra_ll_threshold is None:
            self.intra_threshold = dist_threshold
        else:
            self.intra_threshold = intra_ll_threshold
        self.dist_threshold = dist_threshold
        self.n_layers = self.tensor.shape[-1]
        self.longit = None

        self.lesion_attr = dict()
        for layer in range(self.n_layers):
            for lb in self.lesions_data[layer].keys():
                self.lesion_attr.update({f"{lb}_{layer}": {NodeAttrMatching.CENTROID: self.lesions_data[layer][lb][CENTROID_IDX],
                                                           NodeAttrMatching.VOL_VOXEL: self.lesions_data[layer][lb][VOL_IDX]}})

    def save_matching_graph(self, graph_path):
        """
        The matching is saved as {'nodes': vertices, 'edges': edges}, where both vertices and edges use the notation {lb}_{layer}
        """
        edges = list(self.longit.get_graph().edges())
        lab_list = list(self.longit.get_graph().nodes())
        graph_dict = {'nodes': lab_list, 'edges': edges}
        with open(graph_path, 'w') as f:
            json.dump(graph_dict, f)

    def get_matching_graph(self):
        return self.longit.get_graph()

    def get_longit(self):
        return self.longit

    def run(self):
        self.initial_match()
        self.distance_match()
        return self.get_longit()

    def initial_match(self):
        matching_tensor = np.reshape(self.tensor, (np.product(self.tensor.shape[0:-1]),self.n_layers)) # dim (x*y*z, n_img)
        matching_unique = np.unique(matching_tensor, axis=0) # dim: (num of different connections, n_img)
        labels_layers_matrix = GwMatching.get_labels_layers_matrix(matching_unique) # matrix with {lb_Layer}

        unique_entries = np.unique(labels_layers_matrix)
        all_lesion_instances = [l for l in unique_entries if not GwMatching.iszero(l)]
        edges = self.get_edges_from_mat(labels_layers_matrix)
        loader = LoaderSimple(labels_list=all_lesion_instances, edges_list=edges)
        self.longit = LongitLongitudinalLesions(loader, thresh=self.intra_threshold)
        self.longit.add_cc_attribute()
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.CENTROID] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.CENTROID)
        self.longit.add_node_attribute_from_dict(attr_dict={node: attr[NodeAttrMatching.VOL_VOXEL] for node, attr in self.lesion_attr.items()},
                                                 attr_name=NodeAttrMatching.VOL_VOXEL)

        self.longit.create_long_lesions_list()
        self.sequential_edges_correction()

    def sequential_edges_correction(self):
        self.longit.sequential_edges_correction()

    def distance_match_cc(self):
        """
        Build a graph whose vertices are long lesions. Draw edges between two compatible long lesions
        whose centroids are close enough. Unify all long lesions in the graph connected component
        ** incompatible long lesions may be connected! **:
        Ex: A--B--C. A and C are incompatible, but A--B and B--C. A and C get unified!
        """
        incomplete_ll = [ll for ll in self.longit.get_long_lesions_list() if not ll.check_completeness()]
        _ = [ll.calculate_long_centroid() for ll in incomplete_ll]
        dist_graph = nx.complete_graph(incomplete_ll)

        incompatible_lls = [edge for edge in dist_graph.edges() if self.cls_are_incompatible(edge[0], edge[1])]
        dist_graph.remove_edges_from(incompatible_lls)
        if len(dist_graph.edges) == 0:
            return
        distances = {edge: self.cl_distance(edge[0], edge[1]) for edge in dist_graph.edges}
        nx.set_edge_attributes(dist_graph, distances, name=MatchingDistance.CENTROID_DISTANCE)
        far_lls = [edge for edge, dist in
                   nx.get_edge_attributes(dist_graph, MatchingDistance.CENTROID_DISTANCE).items() if dist > self.dist_threshold]
        dist_graph.remove_edges_from(far_lls)
        if len(dist_graph.edges) == 0:
            return
        con_comp_generator = nx.connected_components(dist_graph)
        for connected_lls in con_comp_generator:
            if len(connected_lls) == 1:
                continue
            self.longit.unify_long_lesions(list(connected_lls))

        for ll in self.longit.get_long_lesions_list():
            self.longit.add_consecutive_edges(ll)
        self.longit.update_main_graph()

    def distance_match(self):
        """
        Build a graph whose vertices are long lesions. Draw edges between two compatible long lesions
        whose centroids are close enough.
        """
        disappearing_ll = [ll for ll in self.longit.get_long_lesions_list() if not ll.check_completeness()]
        _ = [ll.calculate_long_centroid() for ll in disappearing_ll]
        dist_graph = nx.complete_graph(disappearing_ll)

        incompatible_lls = [edge for edge in dist_graph.edges() if self.cls_are_incompatible(edge[0], edge[1])]
        dist_graph.remove_edges_from(incompatible_lls)
        if len(dist_graph.edges) == 0:
            return
        distances = {edge: self.cl_distance(edge[0], edge[1]) for edge in dist_graph.edges}
        nx.set_edge_attributes(dist_graph, distances, name=MatchingDistance.CENTROID_DISTANCE)
        far_lls = [edge for edge, dist in
                   nx.get_edge_attributes(dist_graph, MatchingDistance.CENTROID_DISTANCE).items() if dist > self.dist_threshold]
        dist_graph.remove_edges_from(far_lls)
        if len(dist_graph.edges) == 0:
            return

        ll_unified_list = []
        num_ll = len(dist_graph)
        GwMatching_v2.keep_best_cliques(ll_clustered_list=ll_unified_list, ll_graph=dist_graph, num_ll=num_ll)
        for ll_unified in ll_unified_list:
            if len(ll_unified) == 1:
                continue
            self.longit.unify_long_lesions(ll_unified)

        self.sequential_edges_correction()
        #for ll in self.longit.get_long_lesions_list():
        #    self.longit.add_sequential_edges(ll)
        #self.longit.update_main_graph()

    @staticmethod
    def keep_best_cliques(ll_graph: nx.Graph, ll_clustered_list: List, num_ll: int, dist_metrics=MatchingDistance.DILATION_DISTANCE):
        """
        For each iteration:
            - calculate the maximum cliques in the graph
            - find the 'lone' lls and add them to the output list. (|cliques| = 1)
            - find the best clique, add it to the output list and delete from the graph (in the next iteration, new cliques will be formed)

        :param ll_graph: a graph whose nodes are longitudinal lesions and its edges the possible connections between
            them. If the algorithm decided what to do about a ll, it gets removed from the graph
        :param ll_clustered_list: a list of list of longitudinal lesions. Each sublist represent a cluster of
                longitudinal lesions that can be merged
        :param num_ll: number of nodes (longitudinal lesions) in the initial ll_graph
        :return: None. Update the initially empty ll_clustered_list with clusters of ll.
        """
        if sum([len(cluster) for cluster in ll_clustered_list]) == num_ll:
            return
        ll_cliques = list(nx.find_cliques(ll_graph))
        cliques_id2dist = dict()
        for clique_id, clique in enumerate(ll_cliques):
            if len(clique) == 1:
                ll_graph.remove_nodes_from(clique)
                ll_clustered_list.append(clique)
                continue
            clique_graph = nx.subgraph(ll_graph, clique)
            clique_dist = np.mean(list(nx.get_edge_attributes(clique_graph, dist_metrics).values()))
            cliques_id2dist.update({clique_id: clique_dist})

        if len(cliques_id2dist) == 0:
            return
        current_clique_ids = list(cliques_id2dist.keys())
        clique_ids_sort_by_shortest_dist = sorted(current_clique_ids, key=lambda i: cliques_id2dist[i])
        best_clique = ll_cliques[clique_ids_sort_by_shortest_dist[0]]
        ll_graph.remove_nodes_from(best_clique)
        ll_clustered_list.append(best_clique)
        GwMatching_v2.keep_best_cliques(ll_graph, ll_clustered_list, num_ll)

    def get_edges_from_mat(self, labels_layers_matrix):
        edges_list = list()
        for mat_row in labels_layers_matrix:
            mat_row_lesions = [l for l in mat_row if not GwMatching.iszero(l)]
            edges_list += self.get_lesions_pairs(mat_row_lesions)
        return edges_list

    @staticmethod
    def get_lesions_pairs(les_list):
        list_len = len(les_list)
        start = [i for i in range(list_len - 1)]
        return [[les_list[start_id], les_list[start_id + 1]] for start_id in start]

    @staticmethod
    def cl_distance(cl1, cl2):
        distance = np.sum((cl1.get_long_centroid() - cl2.get_long_centroid()) ** 2) ** 0.5
        return distance

    @staticmethod
    def cls_are_incompatible(cl1, cl2):
        """
        True if there is at least one layer in which cl1 and cl2 have one lesion.
        """
        return np.any(cl1.get_layers_presence()[cl2.get_layers_presence()])


class OptionTest:
    GW_V1_INIT = 'gw_v1_init'
    GW_V1_FULL = 'gw_v1_full'
    GW_V2_INIT = 'gw_v2_init'
    GW_V2_FULL = 'gw_v2_full'

class GwMatching_v1_init(GwMatching):
    def __init__(self, label2data, tensor, dist_threshold):
        super().__init__(label2data, tensor, dist_threshold)
    def distance_match(self):
        return

class GwMatching_v2_init(GwMatching_v2):
    def __init__(self, label2data, tensor, dist_threshold):
        super().__init__(label2data, tensor, dist_threshold)
    def distance_match(self):
        return


class GwMatchingTestFactory:

    def __init__(self, option):
        if option == OptionTest.GW_V1_INIT:
            self.Algo = GwMatching_v1_init
        elif option == OptionTest.GW_V1_FULL:
            self.Algo = GwMatching
        elif option == OptionTest.GW_V2_INIT:
            self.Algo = GwMatching_v2_init
        elif option == OptionTest.GW_V2_FULL:
            self.Algo = GwMatching_v2
        else:
            raise ValueError("!")

    def run(self, label2data, tensor, dist_threshold, save_path):
        algo = self.Algo(label2data, tensor, dist_threshold)
        algo.run()
        algo.save_matching_graph(save_path)


class ConnectedLesion:
    def __init__(self, match, lesion_data, n_layers):
        match_layers = [vert2layer(v) for v in match]
        self.lesion_data = lesion_data
        self.layers_presence = np.array([layer in match_layers for layer in range(n_layers)])
        self.n_layers = n_layers
        self.is_complete = self.check_completeness()
        self.match = match
        self.centroid = self.get_lesions_mean_param(param="centroid")
        self.volume = self.get_lesions_mean_param(param="volume")

    def check_completeness(self):
        return np.all(self.layers_presence)

    def get_lesions_mean_param(self, param="centroid"):
        if self.lesion_data is None:
            return None
        if param == "centroid":
            param_ind = CENTROID_IDX
        elif param == "volume":
            param_ind = VOL_IDX
        else:
            raise ValueError("")

        param_list = list()
        for vert in self.match:
            layer = vert2layer(vert)
            lb = vert2lb(vert)
            try:
                param_case = self.lesion_data[layer][lb][param_ind]
            except:
                a = 1
            param_list.append(param_case)
        mean_param = np.mean(param_list, axis=0)
        return mean_param

    def update_connected_lesion(self, new_lesions):
        """
        Get a list of lesions ("{label}_{layer}") to add to the current CL. Check if it's possible to add lesions to the
        CL, check in which layers is possible to add lesions and then (if there are addable lesions) add them to the CL,
        updating the CL centroid and volume.
        """


        # if self.is_complete:
        #     return
        #
        # new_lesions_copy = new_lesions.copy()
        # for new_les in new_lesions_copy:
        #     if self.layers_presence[vert2layer(new_les)]:
        #         new_lesions.remove(new_les)

        new_match = self.match + new_lesions
        self.match = sorted(new_match, key=lambda u: vert2layer(u))
        self.centroid = self.get_lesions_mean_param(param="centroid")
        self.volume = self.get_lesions_mean_param(param="volume")
        match_layers = [vert2layer(v) for v in self.match]
        self.layers_presence = np.array([layer in match_layers for layer in range(self.n_layers)])
        self.is_complete = self.check_completeness()


from skimage.draw import disk


class ImagesCreator:

    def __init__(self, show_im):
        CENTROID = 'cent'
        RADIUS = 'rad'
        self.max_label = 10
        self.CENTROID = CENTROID
        self.RADIUS = RADIUS
        self.show_im = show_im
        self.shape = (50, 50)
        self.n_layers = 3

        # self.setup = [{1: {CENTROID: (35, 10), RADIUS: 2}, 2: {CENTROID: (20, 10), RADIUS: 2}},
        #               {1: {CENTROID: (30, 10), RADIUS: 2}, 3: {CENTROID: (17, 10), RADIUS: 2}},
        #               {1: {CENTROID: (25, 15), RADIUS: 2}},
        #               {2: {CENTROID: (30, 10), RADIUS: 2}, 3: {CENTROID: (17, 13), RADIUS: 2}}]
        self.setup = [{1: {CENTROID: (40, 10), RADIUS: 6}, 2: {CENTROID: (10, 10), RADIUS: 5}},
                     {3: {CENTROID: (35, 10), RADIUS: 7}, 4: {CENTROID: (20, 10), RADIUS: 5}},
                     {5: {CENTROID: (25, 15), RADIUS: 12}},
                     ]#{2: {CENTROID: (35, 10), RADIUS: 5}, 3: {CENTROID: (15, 10), RADIUS: 5}}]
        # self.setup = [{1: {CENTROID: (40, 10), RADIUS: 5}},
        #               {1: {CENTROID: (25, 10), RADIUS: 5}},
        #               {1: {CENTROID: (25, 15), RADIUS: 15}},
        #               {2: {CENTROID: (35, 10), RADIUS: 5}}]
        #self.setup = [{1: {CENTROID: (5, 5), RADIUS: 1}}, {}, {1: {CENTROID: (5, 5), RADIUS: 2}}, {}]

        assert len(self.setup) == self.n_layers
        blank_image = np.zeros(self.shape)
        self.image_series = [blank_image.copy() for i in range(self.n_layers)]

        self.clrmap = self.def_colormap()

    def image_series_as_tensor(self):
        tensor = np.zeros((self.shape[0], self.shape[1], self.n_layers))
        for l in range(self.n_layers):
            tensor[:,:,l] = self.image_series[l]
        return tensor

    def def_colormap(self):
        color_list = [Colors.BLACK] + [Colors.itk_colors(l) for l in range(1, self.max_label + 1)]
        return color_list

    def add_disk(self, image, label, centroid, radius):
        rr, cc = disk(centroid, radius, shape=self.shape)
        image[rr, cc] = label

    def create(self):
        for layer in range(self.n_layers):
            layer_setup = self.setup[layer]

            for lb, attr in layer_setup.items():
                self.add_disk(image=self.image_series[layer], label=lb, centroid=attr[self.CENTROID], radius=attr[self.RADIUS])
            self.image_series[layer] = self.image_series[layer].astype(int)

    def show(self):
        if not self.show_im:
            return
        f, a = plt.subplots(1, self.n_layers)
        for layer in range(self.n_layers):
            image = self.image_series[layer]
            image_clr = np.zeros((self.shape[0], self.shape[1], 3))
            for lb in range(1, self.max_label + 1):
                image_clr[image==lb, :] = self.clrmap[lb]
            a[layer].imshow(image_clr)
            a[layer].set_axis_off()
            a[layer].set_title(f"t_{layer}")
        plt.show()

    def run(self, get_image_list=False):
        "Create image series and return them as a tensor"
        self.create()
        self.show()
        if get_image_list:
            return self.image_series
        return self.image_series_as_tensor()

class GwMatchingTester:
    def __init__(self, algo_class, images_creator, show_im=True):
        self.Algo = algo_class
        self.ImagesCreator = images_creator
        self.show_im = show_im

    @staticmethod
    def extract_lesions_data(segmentation_series, voxel_volumes=None):
        """
        :param segmentation_series: a tensor of dim (image_dim1, image_dim2, .. , n_layers)
        :param voxel_volumes: a list/nparray of length n_layer. Each element is the average voxel volume in the correspondent layer
        :return: label2data is a list of dictionaries. Each dictionary is a layer. The dictionary index inside the list is the layer
        index. The layer is: {lb : ((centroid), volume)
        """

        def round_tuple(tup, n_digit):
            return tuple([round(el, n_digit) for el in tup])

        n_layers = segmentation_series.shape[-1]
        if voxel_volumes is None:
            voxel_volumes = [1]*n_layers
        label2data = list()

        for layer_idx in range(n_layers):
            curr_orig_img = segmentation_series[:, :, layer_idx].astype(int)

            img_prop = regionprops(curr_orig_img)
            layer_labels = [lb_prop.label for lb_prop in img_prop]
            centroids = [round_tuple(lb_prop.centroid, 2) for lb_prop in img_prop]

            volumes = [np.sum(curr_orig_img == lb) * voxel_volumes[layer_idx] for lb in layer_labels]

            layer_dict = dict()
            for lb_idx, lb in enumerate(layer_labels):
                layer_dict.update({lb: (centroids[lb_idx], round(volumes[lb_idx], 2))})
            label2data.append(layer_dict)
        return label2data

    def run(self):
        im_creator = self.ImagesCreator(self.show_im)
        images_series = im_creator.run()
        lesion_data = self.extract_lesions_data(images_series)
        algo = self.Algo(label2data=lesion_data, tensor=images_series, dist_threshold=0)
        match_longit = algo.run()
        drawer = DrawerLabels(match_longit)
        drawer.show_graph()

if __name__ == "__main__":
    #i = ImagesCreator(show_im=True)
    #i.run()
    tester = GwMatchingTester(algo_class=GwMatching_v2, images_creator=ImagesCreator, show_im=True)
    tester.run()
