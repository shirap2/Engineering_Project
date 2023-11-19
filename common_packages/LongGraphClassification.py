import json

from common_packages.BaseClasses import *
from common_packages.ComputedGraphsMapping import NO_MAPPING, MapComputedGraphs

class LongitClassification_v0(Longit):
    def __init__(self, loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(loader, patient_name, patient_dates)
        self._forward_graph = self.make_graph_directed(self._graph)

    def classify_nodes(self):
        """
        This function classifies the vertices, giving each vertex two classifications, one according to the PRESENCE one
         to the EVOLUTION:
        PRESENCE:
        Lone, new, disappearing, existing
        EVOLUTION:
        Linear, merged, splitting, complex
        The function updates dictionary: vert2class: {vert: [class_presence, class_evolution]}
        """

        nodes2class = dict()
        for node in self._graph.nodes():
            in_deg = self._forward_graph.in_degree(node)
            out_deg = self._forward_graph.out_degree(node)
            # presence
            if in_deg == 0:
                if out_deg == 0:
                    presence = NodesPresence.LONE
                else:
                    presence = NodesPresence.NEW
            else:
                if out_deg == 0:
                    presence = NodesPresence.DISAPPEARING
                else:
                    presence = NodesPresence.EXISTING

            # evolution
            if in_deg <= 1:
                if out_deg <= 1:
                    evolution = NodesEvolution.LINEAR
                else:
                    evolution = NodesEvolution.SPLITTING
            else:
                if out_deg <= 1:
                    evolution = NodesEvolution.MERGED
                else:
                    evolution = NodesEvolution.COMPLEX

            nodes2class.update({node: {NodeAttr.PRESENCE_CLASS: presence, NodeAttr.EVOLUTION_CLASS: evolution}})

        nx.set_node_attributes(self._graph, nodes2class)

    def classify_forward_paths(self):
        if not self.nodes_have_attribute(attr_name=NodeAttr.PRESENCE_CLASS) or \
                not self.nodes_have_attribute(attr_name=NodeAttr.EVOLUTION_CLASS):
            self.classify_nodes()

        self.add_cc_attribute()

        node2pres = nx.get_node_attributes(self._graph, NodeAttr.PRESENCE_CLASS)
        node2evol = nx.get_node_attributes(self._graph, NodeAttr.EVOLUTION_CLASS)
        nodes2path_class = dict()

        new_nodes = [n for n, classification in node2pres.items()
                     if classification == NodesPresence.NEW]
        disap_nodes = [n for n, classification in node2pres.items()
                       if classification == NodesPresence.DISAPPEARING]
        lone_nodes = [n for n, classification in node2pres.items() if
                      classification == NodesPresence.LONE]

        for node in self._graph.nodes():
            if node in lone_nodes:
                path_class = PathClass.LONE
            elif node in new_nodes:
                node_cc = self._graph.nodes[node][NodeAttr.CC_INDEX]
                nodes_end_paths = [n for n in disap_nodes if node_cc == self._graph.nodes[n][NodeAttr.CC_INDEX]]
                nodes_on_paths = self.get_nodes_on_fw_paths(source=node, target_list=nodes_end_paths)
                nodes_on_paths_evolution = [node2evol[n] for n in nodes_on_paths]

                num_complex = nodes_on_paths_evolution.count(NodesEvolution.COMPLEX)
                num_merged = nodes_on_paths_evolution.count(NodesEvolution.MERGED)
                num_split = nodes_on_paths_evolution.count(NodesEvolution.SPLITTING)

                if num_complex > 0 or (num_merged > 0 and num_split > 0):
                    path_class = PathClass.COMPLEX
                elif num_merged == 0 and num_split == 0:
                    path_class = PathClass.LINEAR
                elif num_merged > 0:
                    path_class = PathClass.MERGING
                elif num_split > 0:
                    path_class = PathClass.SPLITTING
                else:
                    raise ValueError("Error in path classification")
            else:
                path_class = PathClass.NONE
            nodes2path_class.update({node: {NodeAttr.PATH_CLASSIFICATION: path_class}})

        nx.set_node_attributes(self._graph, nodes2path_class)

    def get_nodes_on_fw_paths(self, source, target_list: List):
        nodes_on_path = set()
        for target in target_list:
            paths = nx.all_simple_paths(self._forward_graph, source=source, target=target, cutoff=self._num_of_layers)
            _ = [nodes_on_path.add(n) for p in paths for n in p]
        assert len(nodes_on_path) > 1
        return nodes_on_path


class LongitClassification(Longit):
    def __init__(self, loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(loader, patient_name, patient_dates) 
        self._forward_graph = self.make_graph_directed(self._graph)

    def classify_nodes(self):
        """
        classify nodes Changes, according to d_in, d_out. First layer has d_in=1, last, d_out=1
        """
        nodes2class = dict()
        for node in self._graph.nodes():
            in_deg = self._forward_graph.in_degree(node)
            out_deg = self._forward_graph.out_degree(node)
            if self._graph.nodes[node][NodeAttr.LAYER] == 0:
                in_deg = 1
            if self._graph.nodes[node][NodeAttr.LAYER] == self.get_num_of_layers() - 1:
                out_deg = 1

            if in_deg == 0:
                if out_deg == 0:
                    change = NodesChanges.LONE
                elif out_deg == 1:
                    change = NodesChanges.NEW
                else:
                    change = NodesChanges.SPLITTING_NEW
            elif in_deg == 1:
                if out_deg == 0:
                    change = NodesChanges.DISAPPEARED
                elif out_deg == 1:
                    change = NodesChanges.UNIQUE
                else:
                    change = NodesChanges.SPLITTING
            else:
                if out_deg == 0:
                    change = NodesChanges.MERGED_DISAP
                elif out_deg == 1:
                    change = NodesChanges.MERGED
                else:
                    change = NodesChanges.COMPLEX

            nodes2class.update({node: {NodeAttr.CHANGES: change}})

        nx.set_node_attributes(self._graph, nodes2class)

    def classify_cc(self):
        if not self.nodes_have_attribute(attr_name=NodeAttr.CHANGES):
            self.classify_nodes()
        self.add_cc_attribute()

        node2changes = nx.get_node_attributes(self._graph, NodeAttr.CHANGES)
        node2cc = nx.get_node_attributes(self._graph, NodeAttr.CC_INDEX)
        cc_set = set(node2cc.values())
        cc2nodes = {cc_id : [n for n, cc in node2cc.items() if cc == cc_id] for cc_id in cc_set}
        nodes2cc_class = dict()

        for cc_id, node_list in cc2nodes.items():
            if len(node_list) == 1:
                cc_class = CcPatterns.SINGLE
            else:
                node_changes = {node2changes[n] for n in node_list}
                linear_changes = [NodesChanges.UNIQUE, NodesChanges.NEW, NodesChanges.DISAPPEARED]
                if all([n_c in linear_changes for n_c in node_changes]):
                    cc_class = CcPatterns.LINEAR
                elif all([(n_c in linear_changes) or (n_c == NodesChanges.MERGED) or (n_c == NodesChanges.MERGED_DISAP)
                          for n_c in node_changes]):
                    cc_class = CcPatterns.MERGING
                elif all([(n_c in linear_changes) or (n_c == NodesChanges.SPLITTING) or (n_c == NodesChanges.SPLITTING_NEW)
                          for n_c in node_changes]):
                    cc_class = CcPatterns.SPLITTING
                else:
                    cc_class = CcPatterns.COMPLEX
            nodes2cc_class.update({node: {NodeAttr.CC_PATTERNS: cc_class} for node in node_list})
        nx.set_node_attributes(self._graph, nodes2cc_class)

    def classify_nodes_detection(self, mapping_path):
        """
        Each node is classified as TP or FP in relation to the presence/absence in GT.
        :param mapping_path: the path of the mapping file (json). Mapping is a list of [gt_node, pred_node]
        """
        mapping = MapComputedGraphs()
        mapping.load_mapping(mapping_path)
        nodes2detect = dict()
        for node in self._graph.nodes():
            is_tp = mapping.is_pred_node_tp(node)
            if is_tp:
                nodes2detect.update({node: {NodeAttr.DETECTION: NodesDetect.TP}})
            else:
                nodes2detect.update({node: {NodeAttr.DETECTION: NodesDetect.FP}})
        nx.set_node_attributes(self._graph, nodes2detect)


class DrawerClassification_v0(Drawer):
    """Display the nodes classification (new, existing, disappeared)"""

    def __init__(self, longit: LongitClassification_v0, attr_to_show, attr_to_print=None):
        super().__init__(longit)
        self._longit = longit
        assert attr_to_show in [NodeAttr.PRESENCE_CLASS, NodeAttr.EVOLUTION_CLASS, NodeAttr.PATH_CLASSIFICATION]
        self._attr_to_show = attr_to_show
        self._attr_to_print = attr_to_print

    def set_nodes_drawing_attributes(self):
        self._longit.classify_nodes()
        if self._attr_to_show == NodeAttr.PATH_CLASSIFICATION:
            self._longit.classify_forward_paths()

        classes = nx.get_node_attributes(self._base_graph, name=self._attr_to_show)
        colors = {node: {NodeAttr.COLOR: ColorsClass.choose(self._attr_to_show, node_class)} for node, node_class in
                  classes.items()}
        nx.set_node_attributes(self._base_graph, colors)

    def add_legend(self, nodes_position, attr_name=None, **kwarg):
        super().add_legend(nodes_position, attr_name=self._attr_to_show, color_class=ColorsClass)

    def set_nodes_labels(self):
        if self._attr_to_print is None:
            super().set_nodes_labels()
        else:
            self._longit.nodes_have_attribute(attr_name=self._attr_to_print)
            return nx.get_node_attributes(self._base_graph, self._attr_to_print)


class DrawerClassification(Drawer):
    """Display the nodes classification (lesion indivdual changes or lesion change patterns)"""

    def __init__(self, longit: LongitClassification, attr_to_show, attr_to_print=None, **kwargs):
        super().__init__(longit)
        self._longit = longit
        assert attr_to_show in [NodeAttr.CC_PATTERNS, NodeAttr.CHANGES, NodeAttr.DETECTION]
        self._attr_to_show = attr_to_show
        self._attr_to_print = attr_to_print
        if attr_to_show == NodeAttr.DETECTION:
            if "mapping_path" not in kwargs:
                raise ValueError("BENNY: Give the DrawClassification initializer the parameter: mapping_path!")
            self.mapping_path = kwargs["mapping_path"]

    def set_nodes_drawing_attributes(self):
        self._longit.classify_nodes()
        if self._attr_to_show == NodeAttr.CC_PATTERNS:
            self._longit.classify_cc()
        elif self._attr_to_show == NodeAttr.DETECTION:
            self._longit.classify_nodes_detection(self.mapping_path)

        classes = nx.get_node_attributes(self._base_graph, name=self._attr_to_show)
        colors = {node: {NodeAttr.COLOR: ColorsClass.choose(self._attr_to_show, node_class)} for node, node_class in
                  classes.items()}
        nx.set_node_attributes(self._base_graph, colors)

    def add_legend(self, nodes_position, attr_name=None, **kwarg):
        super().add_legend(nodes_position, attr_name=self._attr_to_show, color_class=ColorsClass)

    def set_nodes_labels(self):
        if self._attr_to_print is None:
            return super().set_nodes_labels()
        else:
            self._longit.nodes_have_attribute(attr_name=self._attr_to_print)
            return nx.get_node_attributes(self._base_graph, self._attr_to_print)
