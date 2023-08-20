from common_packages.BaseClasses import *
from common_packages.LongGraphPackage import *

class DrawerBiggestComponents(DrawerLabels):
    def __init__(self, longit: Longit, max_components=20, show_diameter=False, same_color_cc=False,
                 diam_attr=NodeAttr.DIAMETER):
        """
        This class show the graphs' connected components sorted from the biggest to the smallest
        :param longit: a Longit class, containing the longitudinal graph. This graph nodes must have the diameter attribute
        :param max_components: how many connected components to show
        :param show_diameter: if True, instead of node's label, show lesion's diameter
        :param same_color_cc: if True, show the node of the same connected_component with the same color
        """
        self._max_components = max_components
        self._show_diameter = show_diameter
        self._same_color_cc = same_color_cc
        self._diam_attr = diam_attr
        assert self._diam_attr in [NodeAttr.DIAMETER or NodeAttr.CAL_DIAMETER]
        longit.nodes_have_attribute(self._diam_attr)
        super().__init__(longit)

    def attr_to_print_on_nodes(self):
        if self._show_diameter:
            return self._diam_attr
        else:
            return NodeAttr.LABEL

    def set_graph_layout(self):
        cc_subgraphs = [self._base_graph.subgraph(cc) for cc in nx.connected_components(self._base_graph)]
        cc_subgraphs_sorted = self.sort_subgraphs(cc_subgraphs)
        self.set_connected_component_attribute(cc_subgraphs_sorted)
        cc_graph = self.fill_with_placeholders(cc_subgraphs_sorted[0])
        for i in range(1, min(len(cc_subgraphs), self._max_components)):
            curr_cc_graph = self.fill_with_placeholders(cc_subgraphs_sorted[i])
            cc_graph = nx.compose(cc_graph, curr_cc_graph)
        self._base_graph = cc_graph

    def sort_subgraphs(self, subgraphs_collection):
        """Sort according to sum of diameters in subgraph"""

        def get_sum_max_diam(graph):
            return np.sum(list(nx.get_node_attributes(graph, NodeAttr.DIAMETER).values()))

        subgraphs = sorted(subgraphs_collection, key=get_sum_max_diam, reverse=True)
        return subgraphs

    def assign_placeholder_special_attr(self, ph_attr):
        super().assign_placeholder_special_attr(ph_attr)
        ph_attr[self._diam_attr] = ''
        ph_attr[NodeAttr.CC_DRAW_INDEX] = 0

    def set_connected_component_attribute(self, cc_sorted):
        """Add to each node of a cc the attribute CC_INDEX. If we need to color the same cc with the same color, replace
        the previous coloring"""
        for index, cc in enumerate(cc_sorted):
            nx.set_node_attributes(cc, name=NodeAttr.CC_DRAW_INDEX, values=(index + 1))

            if self._same_color_cc:
                colors = {node: {NodeAttr.COLOR: Colors.itk_colors(index + 1)} for node in cc.nodes()}
                nx.set_node_attributes(cc, colors)


class DrawerClassificationOld(Drawer):
    """Display the nodes classification (new, existing, disappeared)"""

    def __init__(self, longit: Longit):
        super().__init__(longit)
        self._longit = longit

    def set_nodes_drawing_attributes(self):
        self._longit.classify_nodes()

        classes = nx.get_node_attributes(self._base_graph, name=NodeAttr.CLASSIFICATION)
        colors = {node: {NodeAttr.COLOR: Colors.choose(NodeAttr.CLASSIFICATION, node_class)} for node, node_class in
                  classes.items()}
        nx.set_node_attributes(self._base_graph, colors)

    def add_legend(self, nodes_position, attr_name=NodeAttr.CLASSIFICATION, **kwarg):
        super().add_legend(nodes_position, attr_name)

class DrawerPathScore(DrawerEval):
    def __init__(self, longit: Longit, path_det_attr: int):
        self._path_det_attr = path_det_attr
        longit.nodes_have_attribute(attr_name=NodeAttr.PATH_CORRECTNESS)
        super().__init__(longit, attr=NodeAttr.DETECTION)

    def set_nodes_drawing_attributes(self):
        path_detection = nx.get_node_attributes(self._base_graph, name=NodeAttr.PATH_CORRECTNESS)
        node2clr = dict()
        for node, det in path_detection.items():
            if det == NodesPathCorrectness.UNSHARED:
                clr = Colors.GRAY
            else:
                if det[self._path_det_attr] == 1:
                    clr = colormap.RdYlGn(0.99)
                else:
                    clr = colormap.RdYlGn(det[self._path_det_attr])
            node2clr.update({node: {NodeAttr.COLOR: clr}})
        nx.set_node_attributes(self._base_graph, node2clr)

    def attr_to_print_on_nodes(self):
        return NodeAttr.PATH_CORRECTNESS


class DrawerEvalSourceGraphs(DrawerEval):
    """Draw the GT or the Pred graph"""
    def __init__(self, longit: Longit, source_gr: str):
        super().__init__(copy.deepcopy(longit), attr=NodeAttr.DETECTION)
        assert source_gr in [SourceGraph.GT, SourceGraph.PRED]
        self._source_gr_type = source_gr

    def set_nodes_drawing_attributes(self):
        det_values = nx.get_node_attributes(self._base_graph, name=NodeAttr.DETECTION)
        labels = nx.get_node_attributes(self._base_graph, name=NodeAttr.LABEL)
        new_attr = dict()
        if self._source_gr_type == SourceGraph.GT:
            det_class = [NodesDetect.TP, NodesDetect.FN]
        elif self._source_gr_type == SourceGraph.PRED:
            det_class = [NodesDetect.TP, NodesDetect.FP]
        else:
            raise ValueError(f"Bad {self._source_gr_type}")

        for node, d_attr in det_values.items():
            if d_attr in det_class:
                new_attr.update({node: {NodeAttr.COLOR: Colors.DARK_GRAY, NodeAttr.LABEL: labels[node]}})
            else:
                new_attr.update({node: {NodeAttr.COLOR: Colors.WHITE, NodeAttr.LABEL: ''}})
        nx.set_node_attributes(self._base_graph, new_attr)

    def set_edges_drawing_attributes(self):
        super().set_edges_drawing_attributes()

        nodes_detection = nx.get_node_attributes(self._base_graph, name=NodeAttr.DETECTION)
        edges_detection = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.DETECTION)
        edges_colors = dict()

        for edge in edges_detection.keys():
            if self.is_edge_in_source_graph(edge, self._source_gr_type, nodes_detection, edges_detection):
                edges_colors.update({edge: {EdgeAttr.COLOR: Colors.BLACK}})
            else:
                edges_colors.update({edge: {EdgeAttr.COLOR: Colors.WHITE}})
        nx.set_edge_attributes(self._base_graph, edges_colors)

    @staticmethod
    def is_edge_in_source_graph(edge, source_gr_type, nodes_detection, edges_detection):
        if source_gr_type == SourceGraph.GT:
            det_class = NodesDetect.FN
            other_graph = NodesDetect.FP
            other_graph_shared_edges = [EdgesDetect.FP, EdgesDetect.FP_IN_SAME_CC]
        elif source_gr_type == SourceGraph.PRED:
            det_class = NodesDetect.FP
            other_graph = NodesDetect.FN
            other_graph_shared_edges = [EdgesDetect.FN, EdgesDetect.FN_IN_SAME_CC]
        else:
            raise ValueError("")

        node0 = edge[0]
        node1 = edge[1]
        if nodes_detection[node0] == det_class or nodes_detection[node1] == det_class:
            edge_in_source_graph = True
        elif nodes_detection[node0] == other_graph or nodes_detection[node1] == other_graph:
            edge_in_source_graph = False
        else:  # both nodes are TP
            if edges_detection[edge] in other_graph_shared_edges:
                edge_in_source_graph = False
            else:
                edge_in_source_graph = True
        return edge_in_source_graph

    def draw(self, pos):
        super().draw(pos)
        plt.title(f"{self._source_gr_type} {self._patient_name}", fontsize=12)


class DrawerEval_SkipEdgeHandler(DrawerEval):
    """Display the edges detection evaluation graph, coloring the edges in the skip_edge_path in a different color"""
    def __init__(self, longit: Longit, attr: str):
        longit.edges_have_attribute(EdgeAttr.IS_SKIP_EDGE_PATH)
        super().__init__(longit, attr)

    def set_edges_drawing_attributes(self):
        super().set_edges_drawing_attributes()
        if not self._is_node_eval:
            is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)
            attr_values = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP_EDGE_PATH)
            color_values = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.COLOR)
            # treat the redeemed edges as TP:
            color_values_updated = dict()
            for e, color in color_values.items():
                if not is_skip[e]:
                    continue
                if attr_values[e] == EdgesInSkipPath.FALSE:
                    color_values_updated.update({e: {EdgeAttr.COLOR: color}})
                else:
                    color_values_updated.update({e: {EdgeAttr.COLOR: Colors.choose(EdgeAttr.IS_SKIP_EDGE_PATH, attr_values[e])}})
            nx.set_edge_attributes(self._base_graph, color_values_updated)
