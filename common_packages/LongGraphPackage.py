from common_packages.BaseClasses import *
from common_packages.LongGraphClassification import LongitClassification
from common_packages.ComputedGraphsMapping import MapComputedGraphs, PatientMatchingBipartiteGraph


class LoaderSimple(Loader):

    """
    This class loads a graph whose nodes don't have any attribute. label_list is a list of str lb_layer, and edges_list, couples
    of lb_layers
    """

    def __init__(self, labels_list: List, edges_list: List, layers_to_delete=None):
        super().__init__()
        if layers_to_delete is not None:
            remover = DeleteLayer(labels_list, edges_list)
            labels_list, edges_list = remover.apply(layers_to_delete)

        # if label_mapping is not None:
        #     label_list_unmapped = labels_list.copy()
        #     edges_list_unmapped = edges_list.copy()
        #     labels_list = [label_mapping[l] for l in label_list_unmapped]
        #     edges_list = [[label_mapping[l0], label_mapping[l1]] for l0, l1 in edges_list_unmapped]

        self._nodes = {l: {NodeAttr.LABEL: self.node2lb(l), NodeAttr.LAYER: self.node2layer(l)} for l in labels_list if self.node2lb(l) > 0}
        # Order edges such that e[0]<e[1]
        edges_list_ordered = list()
        for e in edges_list:
            node0 = e[0]
            node1 = e[1]
            if self.node2layer(node0) == self.node2layer(node1):
                raise ValueError(f"Edge {e} has two nodes in the same layer!")
            elif self.node2layer(node0) > self.node2layer(node1):
                edges_list_ordered.append([node1, node0])
            else:
                edges_list_ordered.append([node0, node1])

        self._edges = {tuple(e): {EdgeAttr.IS_SKIP: (self.node2layer(e[0]) + 1 < self.node2layer(e[1]))} for e in
                       edges_list_ordered}

    @staticmethod
    def node2lb(vert: str):
        """node (string) ='lb_layer' --> lb (int) """
        return int(float((vert.split('_')[0])))

    @staticmethod
    def node2layer(vert: str):
        """node (string) ='lb_layer' --> layer (int) """
        return int(float((vert.split('_')[1])))


class LoaderSimpleWithMapping(LoaderSimple):
    def __init__(self, label_list, edges_list, mapping_path: str, is_gt: bool):
        """
        :param label_list:
        :param edges_list:
        :param mapping_path: A string containing a path to a mapping json. The mapping file is a list of lists. Each
        sublist is a pair of [gt_label, pred_label] (*_label format: f"{label}_{layer}"). Unmatched labels are coupled with
        f"0_{layer}". **The mapping is not one-to-one. It can be even many-to-many**
        :param is_gt:
        """
        map_obj = MapComputedGraphs()
        mapping_gt_pred = map_obj.load_mapping(mapping_path)
        # make it a bipartite graph:
        bi_graph = PatientMatchingBipartiteGraph(mapping_gt_pred)
        mapping = bi_graph.get_mapping(is_gt)
        # patch for lungs ground truth: sometimes there are placeholder!
        label_list_ = label_list.copy()
        label_list = [lb for lb in label_list_ if not lb.startswith('0')]
        mapped_label_list = list({mapping[lb] for lb in label_list})
        mapped_edges_sets = {(mapping[e[0]],mapping[e[1]]) for e in edges_list}
        mapped_edges_list = [list(e) for e in mapped_edges_sets]
        super().__init__(labels_list=mapped_label_list, edges_list=mapped_edges_list)


class LoaderSimpleFromJson(LoaderSimple):
    """Load from a json in which {'nodes': list of str lb_layer, 'edges': couples
    of lb_layers} """
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        labels_list = data['nodes']
        edges_list = data['edges']
        super().__init__(labels_list, edges_list)


class LoaderEval(Loader):
    """Base class that evaluates nodes and edges of a gt and pred graphs, given as Loaders"""
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None,
                 LongitClass=Longit):
        super().__init__()
        self._patient_name = patient_name
        self._patient_dates = patient_dates
        self._gt_longit = LongitClass(gt_loader, patient_name, patient_dates)
        self._pred_longit = LongitClass(pred_loader, patient_name, patient_dates)

        self._num_of_layers = self._gt_longit.get_num_of_layers()

        merged_nodes_attributes = self.evaluate_nodes()
        merged_edges_attributes = self.evaluate_edges(merged_nodes_attributes)

        self._nodes = merged_nodes_attributes
        self._edges = merged_edges_attributes

        self._stat_names = StatNamesPredLesions()

    def evaluate_nodes(self):
        """
        This method merges the nodes and the attributes of the gt and pred graphs, adding the detection attribute and the
        the classification evaluation attribute to the TP nodes.
        inputs (self) (1) gt_longit: the Longit of the gt graph
                      (2) pred_longit: the Longit of the pred graph
        :return: merged_nodes_attributes {node_id: {attr_name: attr_value}} of the the union of the graphs
        """
        self._gt_longit.classify_nodes()
        self._pred_longit.classify_nodes()

        self._gt_longit.nodes_have_attribute(NodeAttr.CLASSIFICATION)
        gt_nodes = dict(self._gt_longit.get_graph().nodes(data=True))

        self._pred_longit.nodes_have_attribute(NodeAttr.CLASSIFICATION)
        pred_nodes = dict(self._pred_longit.get_graph().nodes(data=True))

        merged_nodes_ids = set(gt_nodes.keys()) | set(pred_nodes.keys())
        merged_nodes_attributes = {**gt_nodes, **pred_nodes}
        for node in merged_nodes_ids:
            if (node in gt_nodes) and (node in pred_nodes):
                # check the classification:
                gt_class = self._gt_longit.get_graph().nodes[node][NodeAttr.CLASSIFICATION]
                pred_class = self._pred_longit.get_graph().nodes[node][NodeAttr.CLASSIFICATION]
                layer = self._gt_longit.get_graph().nodes[node][NodeAttr.LAYER]

                merged_nodes_attributes[node].update({NodeAttr.EVAL_CLASSIFICATION: self.eval_classification(gt_class, pred_class, layer),
                                                      NodeAttr.DETECTION: NodesDetect.TP})

            elif node in gt_nodes:
                merged_nodes_attributes[node].update({NodeAttr.DETECTION: NodesDetect.FN,
                                                      NodeAttr.EVAL_CLASSIFICATION: NodesClassEval.UNSHARED})

            elif node in pred_nodes:
                merged_nodes_attributes[node].update({NodeAttr.DETECTION: NodesDetect.FP,
                                                      NodeAttr.EVAL_CLASSIFICATION: NodesClassEval.UNSHARED})

            else:
                raise ValueError("Node must be or in gt or in pred")

        return merged_nodes_attributes

    def eval_classification(self, gt_class, pred_class, node_layer):
        """Given the classification of the same lesion in gt and pred, return the classification correctness
        If the node is in the first or last timepoint, only one class will be considered"""
        #raise ValueError("This evaluation method is deprecated! Doesn't fit the MICCAI 2023 paper definitions")
        is_first_layer = node_layer == 0
        is_last_layer = node_layer == self._num_of_layers - 1
        correct_classes = 2 - np.sum(np.abs(NodesClass.AS_VECTOR[gt_class] - NodesClass.AS_VECTOR[pred_class]))

        if correct_classes == 2:
            if is_first_layer or is_last_layer:
                return NodesClassEval.ONE_CORRECT
            return NodesClassEval.TWO_CORRECT
        elif correct_classes == 1:
            if is_first_layer or is_last_layer:
                return NodesClassEval.WRONG
            return NodesClassEval.ONE_CORRECT
        else:
            return NodesClassEval.WRONG

    def evaluate_edges(self, merged_nodes_attributes: Dict):
        """
        This method merges the edges and the attributes of the gt and pred graphs, adding the detection attribute to edges
        between 2 TP vertices.
        The input graphs nodes must be assigned the detection attribute
        input (self) gt_longit: the Longit of the gt graph
                     pred_longit: the Longit of the pred graph
        :param merged_nodes_attributes: {node_id: {attr_name: attr_value}} of the union of the graphs
        :return merged_edges_attributes: {edge_id: {attr_name: attr_value}} of the union of the graphs
        """

        # check that the merged_nodes_attributes contains the DETECTION attribute for all the nodes
        assert np.all([NodeAttr.DETECTION in attr.keys() for attr in merged_nodes_attributes.values()])

        gt_edges = Longit.edgeview2dict(self._gt_longit.get_graph().edges(data=True), merged_nodes_attributes)
        pred_edges = Longit.edgeview2dict(self._pred_longit.get_graph().edges(data=True), merged_nodes_attributes)

        merged_edges_ids = set(gt_edges.keys()) | set(pred_edges.keys())
        merged_edges_attributes = {**gt_edges, **pred_edges}

        for edge in merged_edges_ids:
            node0 = edge[0]
            node1 = edge[1]
            # both nodes are TP
            if merged_nodes_attributes[node0][NodeAttr.DETECTION] == NodesDetect.TP and \
                    merged_nodes_attributes[node1][NodeAttr.DETECTION] == NodesDetect.TP:

                if (edge in gt_edges) and (edge in pred_edges):
                    merged_edges_attributes[edge].update({EdgeAttr.DETECTION: EdgesDetect.TP})
                elif edge in gt_edges:
                    merged_edges_attributes[edge].update({EdgeAttr.DETECTION: EdgesDetect.FN})
                elif edge in pred_edges:
                    merged_edges_attributes[edge].update({EdgeAttr.DETECTION: EdgesDetect.FP})
                else:
                    raise ValueError("Edge must be or in gt or in pred")
            # at least one of the nodes is not TP
            else:
                merged_edges_attributes[edge].update({EdgeAttr.DETECTION: EdgesDetect.UNSHARED})

        return merged_edges_attributes

    def get_patient_name(self):
        return self._patient_name

    def get_patient_dates(self):
        return self._patient_dates

    def get_num_of_layers(self):
        return self._num_of_layers

    @staticmethod
    def get_if(object_dict, attr_name, attr_values, excluded_values=False):
        """get all the objects (edges or nodes) with the attr_name set to one of attr_values"""
        if isinstance(attr_values, str) or not isinstance(attr_values, Iterable):
            attr_values = [attr_values]
        else:
            attr_values = list(attr_values)
        chosen_object = dict()
        for obj, attr in object_dict.items():
            try:
                curr_val = object_dict[obj][attr_name]
            except KeyError:
                raise KeyError(f"No {attr_name} in node {obj} of current graph!")
            if excluded_values:
                if curr_val not in attr_values:
                    chosen_object.update({obj: attr})
            else:
                if curr_val in attr_values:
                    chosen_object.update({obj: attr})
        return chosen_object

    def get_num_tp_lesions(self):
        return len(self.get_if(self._nodes, attr_name=NodeAttr.DETECTION, attr_values=NodesDetect.TP))

    def get_num_fp_lesions(self):
        return len(self.get_if(self._nodes, attr_name=NodeAttr.DETECTION, attr_values=NodesDetect.FP))

    def get_num_fn_lesions(self):
        return len(self.get_if(self._nodes, attr_name=NodeAttr.DETECTION, attr_values=NodesDetect.FN))

    def get_num_gt_lesions(self):
        return self.get_num_tp_lesions() + self.get_num_fn_lesions()

    def get_num_pred_lesions(self):
        return self.get_num_tp_lesions() + self.get_num_fp_lesions()

    def get_num_lesions_classes(self):
        """Count how many TP lesions classes. Mid-layers have two class, external layers have one"""
        tp_lesions = self.get_if(self._nodes, attr_name=NodeAttr.DETECTION, attr_values=NodesDetect.TP)
        layers = [l for l in range(self._num_of_layers)]
        tp_lesions_mid_layers = self.get_if(tp_lesions, attr_name=NodeAttr.LAYER, attr_values=layers[1:-1])
        num_tp_lesions_mid_layers = len(tp_lesions_mid_layers)
        num_tp_lesions_first_last_layers = len(tp_lesions) - num_tp_lesions_mid_layers
        return num_tp_lesions_first_last_layers + 2*num_tp_lesions_mid_layers

    def get_num_correct_lesions_classes(self):
        one_corr_class = len(self.get_if(self._nodes, attr_name=NodeAttr.EVAL_CLASSIFICATION,
                                         attr_values=NodesClassEval.ONE_CORRECT))
        two_corr_class = len(self.get_if(self._nodes, attr_name=NodeAttr.EVAL_CLASSIFICATION,
                                         attr_values=NodesClassEval.TWO_CORRECT))
        return one_corr_class + 2*two_corr_class

    def get_num_tp_edges(self):
        return len(self.get_if(self._edges, attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.TP))

    def get_num_fp_edges(self):
        return len(self.get_if(self._edges, attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.FP))

    def get_num_fn_edges(self):
        return len(self.get_if(self._edges, attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.FN))

    def get_num_gt_edges(self):
        """return the number of gt edges between tp nodes"""
        return self.get_num_tp_edges() + self.get_num_fn_edges()

    def get_num_pred_edges(self):
        """return the number of predicted edges between tp nodes"""
        return self.get_num_tp_edges() + self.get_num_fp_edges()

    def get_stats(self, pred_lesions):
        """calculate the statistics, and return them as a dictionary: {stat_name: stat_value}"""
        stat_dict = {
            self._stat_names.num_gt_lesions: self.get_num_gt_lesions()}

        if pred_lesions:
            stat_dict.update({
                self._stat_names.num_gt_lesions: self.get_num_gt_lesions(),
                self._stat_names.num_pred_lesions: self.get_num_pred_lesions(),
                self._stat_names.num_tp_lesions: self.get_num_tp_lesions(),
                self._stat_names.num_fp_lesions: self.get_num_fp_lesions(),
                self._stat_names.num_fn_lesions: self.get_num_fn_lesions(),
                })

        stat_dict.update({self._stat_names.num_lesions_classes: self.get_num_lesions_classes(),
                          self._stat_names.num_correct_lesions_classes: self.get_num_correct_lesions_classes(),
                          self._stat_names.num_gt_edges: self.get_num_gt_edges(),
                          self._stat_names.num_pred_edges: self.get_num_pred_edges(),
                          self._stat_names.num_tp_edges: self.get_num_tp_edges(),
                          self._stat_names.num_fp_edges: self.get_num_fp_edges(),
                          self._stat_names.num_fn_edges: self.get_num_fn_edges(),
                         })
        return stat_dict

    def get_stat_as_dataframe(self, pred_lesions):
        """Calculate the stats and return a pandas dataframe made of one line: line's name is patient's name,
        and the columns contain stats names and values"""
        if self._patient_name is None:
            raise ValueError("To get a table, patient_name must be set")
        stat = {name: [val] for name, val in self.get_stats(pred_lesions).items()}
        return pd.DataFrame(data=stat, index=[self._patient_name])

    def get_summary(self, tot_fun, pred_lesions):
        """
        :param tot_fun: an external function that given the name of a stat gets the sum of all the patient values for that stat.
        :param pred_lesions: a bool: if True, add to the summary nodes detection stat.
        """
        assert callable(tot_fun)
        summary = dict()
        if pred_lesions:
            summary.update({
                'nodes precision':  self.precision(tot_fun(self._stat_names.num_tp_lesions), tot_fun(self._stat_names.num_pred_lesions)),
                'nodes recall':     self.recall(tot_fun(self._stat_names.num_tp_lesions), tot_fun(self._stat_names.num_gt_lesions))
            })

        summary.update({
            'nodes correct classification %': self.correctness(tot_fun(self._stat_names.num_correct_lesions_classes), tot_fun(self._stat_names.num_lesions_classes)),
            'edges precision': self.precision(tot_fun(self._stat_names.num_tp_edges), tot_fun(self._stat_names.num_pred_edges)),
            'edges recall': self.recall(tot_fun(self._stat_names.num_tp_edges), tot_fun(self._stat_names.num_gt_edges))
        })
        return summary

    @staticmethod
    def precision(num_of_tp: int, num_of_predicted: int):
        assert num_of_tp <= num_of_predicted
        if num_of_predicted == 0:
            return 1
        else:
            return round(num_of_tp / num_of_predicted, 2)

    @staticmethod
    def recall(num_of_tp: int, num_of_gt: int):
        assert num_of_tp <= num_of_gt
        if num_of_gt == 0:
            return 1
        else:
            return round(num_of_tp / num_of_gt, 2)

    def correctness(self, num_of_correct: int, num_of_all: int):
        """Return correct elements percentage"""
        assert num_of_correct <= num_of_all
        if num_of_all == 0:
            return 0
        else:
            return round(num_of_correct / num_of_all, 2) * 100

    @staticmethod
    def is_forward_path(path, nodes_attr, reverse=False):
        """Get path (list of nodes), and their attributes. returns True if the nodes' layer are sorted from the smallest
        to the largest i.e., if the path is in chronological order"""
        path_layers = [nodes_attr[n][NodeAttr.LAYER] for n in path]
        if not reverse:
            nodes_are_ordered = [path_layers[i - 1] < path_layers[i] for i in range(1, len(path_layers))]
        else:
            nodes_are_ordered = [path_layers[i - 1] > path_layers[i] for i in range(1, len(path_layers))]
        return all(nodes_are_ordered)


class LoaderEvalChanges(LoaderEval):
    """Same as Loader Eval, just different vertices labels (classes)"""
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates, LongitClass=LongitClassification)

    def evaluate_nodes(self):
        """
        This method merges the nodes and the attributes of the gt and pred graphs, adding the detection attribute and the
        the classification evaluation attribute to the TP nodes.
        inputs (self) (1) gt_longit: the Longit of the gt graph
                      (2) pred_longit: the Longit of the pred graph
        :return: merged_nodes_attributes {node_id: {attr_name: attr_value}} of the the union of the graphs
        """
        self._gt_longit.classify_nodes()
        self._pred_longit.classify_nodes()

        self._gt_longit.nodes_have_attribute(NodeAttr.CHANGES)
        gt_nodes = dict(self._gt_longit.get_graph().nodes(data=True))

        self._pred_longit.nodes_have_attribute(NodeAttr.CHANGES)
        pred_nodes = dict(self._pred_longit.get_graph().nodes(data=True))

        merged_nodes_ids = set(gt_nodes.keys()) | set(pred_nodes.keys())
        merged_nodes_attributes = {**gt_nodes, **pred_nodes}
        for node in merged_nodes_ids:
            if (node in gt_nodes) and (node in pred_nodes):
                # check the classification:
                gt_class = self._gt_longit.get_graph().nodes[node][NodeAttr.CHANGES]
                pred_class = self._pred_longit.get_graph().nodes[node][NodeAttr.CHANGES]
                layer = self._gt_longit.get_graph().nodes[node][NodeAttr.LAYER]

                merged_nodes_attributes[node].update({NodeAttr.EVAL_CLASSIFICATION: self.eval_classification(gt_class, pred_class, layer),
                                                      NodeAttr.DETECTION: NodesDetect.TP})

            elif node in gt_nodes:
                merged_nodes_attributes[node].update({NodeAttr.DETECTION: NodesDetect.FN,
                                                      NodeAttr.EVAL_CLASSIFICATION: NodesClassEval.UNSHARED})

            elif node in pred_nodes:
                merged_nodes_attributes[node].update({NodeAttr.DETECTION: NodesDetect.FP,
                                                      NodeAttr.EVAL_CLASSIFICATION: NodesClassEval.UNSHARED})

            else:
                raise ValueError("Node must be or in gt or in pred")

        return merged_nodes_attributes

    def eval_classification(self, gt_class, pred_class, node_layer):
        """Given the classification of the same lesion in gt and pred, return the classification correctness"""
        if pred_class == gt_class:
            return NodesClassEval.ONE_CORRECT
        else:
            return NodesClassEval.WRONG

    def get_num_lesions_classes(self):
        """Count how many TP lesions classes"""
        tp_lesions = self.get_if(self._nodes, attr_name=NodeAttr.DETECTION, attr_values=NodesDetect.TP)
        return len(tp_lesions)

    def get_num_correct_lesions_classes(self):
        one_corr_class = len(self.get_if(self._nodes, attr_name=NodeAttr.EVAL_CLASSIFICATION,
                                         attr_values=NodesClassEval.ONE_CORRECT))
        return one_corr_class


class LoaderEval_SkipEdgeHandler(LoaderEvalChanges):
    """This class expands the Evaluator, adding the IS_SKIP_EDGE_PATH attribute"""

    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None, unshared_parallel_path=True):
        self._unshared_parallel_path = unshared_parallel_path
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates)
        self._stat_names = StatNamesSkipEdgeHandler()

    def evaluate_edges(self, merged_nodes_attributes: Dict):
        """
        This method adds to the parent method the skip edges handling. If there is a skip edge in one of the graphs,
        check if in the other graph there is a path between the two skip edges vertices.
        The input graphs nodes must be assigned the detection attribute
        :param gt_longit: the Longit of the gt graph
        :param pred_longit: the Longit of the pred graph
        :param merged_nodes_attributes: {node_id: {attr_name: attr_value}} of the union of the graphs
        :return merged_edges_attributes: {edge_id: {attr_name: attr_value}} of the union of the graphs
        """

        merged_edges_attributes = super().evaluate_edges(merged_nodes_attributes)
        gt_graph = self._gt_longit.get_graph()
        pred_graph = self._pred_longit.get_graph()
        # add skip_edge_path attribute:
        for attr in merged_edges_attributes.values():
            attr.update({EdgeAttr.IS_SKIP_EDGE_PATH: EdgesInSkipPath.FALSE})

        # try to "redeem" the skip edges between two TP vertices classified by the base method as FP/FN
        for edge, attr in merged_edges_attributes.items():
            if attr[EdgeAttr.IS_SKIP] and \
                    (attr[EdgeAttr.DETECTION] == EdgesDetect.FN or attr[EdgeAttr.DETECTION] == EdgesDetect.FP):
                # the skip edge must be or FP or FN (between two TP vertices)
                node0 = edge[0]
                node1 = edge[1]
                skip_edge_in_gt = (attr[EdgeAttr.DETECTION] == EdgesDetect.FN)

                if skip_edge_in_gt:
                    # find the paths between the two vertices in the other graph
                    paths_undirected = list(nx.all_simple_paths(pred_graph, source=node0, target=node1))
                else:
                    paths_undirected = list(nx.all_simple_paths(gt_graph, source=node0, target=node1))

                paths = [p for p in paths_undirected if self.is_forward_path(p, merged_nodes_attributes)]
                if len(paths) == 0:
                    continue

                # update the attribute of all the edges in the path
                """
                Check the parallel path. Examples:
                (1): D, F are TP, E is FP. GT: D-->F; Pred: D->E->F. Edges are in skip path. (D, E), (E, F) are UNSHARED
                (2): A, B, C are TP. GT: A-->C; Pred: A->B->C. (A, B), (B, C) are FP
                    self._unshared_parallel_path==TRUE: Exclude edges from skip path.
                    else                              : Edges are in skip path. 
                """
                unshared_paths = 0
                for path in paths:
                    path_edges = [(path[i], path[i+1]) for i, _ in enumerate(path[:-1])]

                    if self._unshared_parallel_path:
                        is_path_unshared = all([merged_edges_attributes[e][EdgeAttr.DETECTION] == EdgesDetect.UNSHARED
                                                for e in path_edges])
                    else:
                        is_path_unshared = True
                    unshared_paths += int(is_path_unshared)
                    # Commented: the edges parallel to the skip edge don't get any attribute
                    # if is_path_unshared:
                    #     for e in path_edges:
                    #         if skip_edge_in_gt:
                    #             merged_edges_attributes[e][EdgeAttr.IS_SKIP_EDGE_PATH] = EdgesInSkipPath.PRED
                    #         else:
                    #             merged_edges_attributes[e][EdgeAttr.IS_SKIP_EDGE_PATH] = EdgesInSkipPath.GT
                # update the attribute of the skip edge
                if unshared_paths > 0:
                    if skip_edge_in_gt:
                        merged_edges_attributes[edge][EdgeAttr.IS_SKIP_EDGE_PATH] = EdgesInSkipPath.GT
                    else:
                        merged_edges_attributes[edge][EdgeAttr.IS_SKIP_EDGE_PATH] = EdgesInSkipPath.PRED

        return merged_edges_attributes

    def get_skip_edges(self):
        return self.get_if(self._edges, attr_name=EdgeAttr.IS_SKIP, attr_values=True)

    def get_num_tp_skip_edges(self):
        return len(self.get_if(self.get_skip_edges(), attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.TP))

    def get_num_fp_skip_edges(self):
        return len(self.get_if(self.get_skip_edges(), attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.FP))

    def get_num_fn_skip_edges(self):
        return len(self.get_if(self.get_skip_edges(), attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.FN))

    def get_num_pred_skip_edges(self):
        return self.get_num_tp_skip_edges() + self.get_num_fp_skip_edges()

    def get_num_gt_skip_edges(self):
        return self.get_num_tp_skip_edges() + self.get_num_fn_skip_edges()

    def get_num_fp_skip_edges_parallel_to_path(self):
        """Get the number of FP predicted skip edges whose vertices are connected in GT graph"""
        return len(self.get_if(self.get_skip_edges(), attr_name=EdgeAttr.IS_SKIP_EDGE_PATH, attr_values=EdgesInSkipPath.PRED))

    def get_num_fn_skip_edges_parallel_to_path(self):
        """Get the number of FN GT edges whose vertices are connected in Pred graph"""
        return len(self.get_if(self.get_skip_edges(), attr_name=EdgeAttr.IS_SKIP_EDGE_PATH, attr_values=EdgesInSkipPath.GT))

    def get_stats(self, pred_lesions):
        stat_dict = super().get_stats(pred_lesions)
        stat_dict.update({
            self._stat_names.num_gt_skip_edges: self.get_num_gt_skip_edges(),
            self._stat_names.num_pred_skip_edges: self.get_num_pred_skip_edges(),
            self._stat_names.num_tp_skip_edges: self.get_num_tp_skip_edges(),
            self._stat_names.num_fn_skip_edges: self.get_num_fn_skip_edges(),
            self._stat_names.num_fp_skip_edges: self.get_num_fp_skip_edges(),
            self._stat_names.num_fn_skip_edges_path: self.get_num_fn_skip_edges_parallel_to_path(),
            self._stat_names.num_fp_skip_edges_path: self.get_num_fp_skip_edges_parallel_to_path()
        })
        return stat_dict

    def get_summary(self, tot_fun, pred_lesions):
        summary = super().get_summary(tot_fun, pred_lesions)
        num_tp_with_fp_skip_edge_on_path = tot_fun(self._stat_names.num_tp_edges) + tot_fun(self._stat_names.num_fp_skip_edges_path)
        num_tp_with_fn_skip_edge_on_path = tot_fun(self._stat_names.num_tp_edges) + tot_fun(self._stat_names.num_fn_skip_edges_path)

        num_tp_skip_edges = tot_fun(self._stat_names.num_tp_skip_edges)


        summary.update({
            'edges precision with s.e. paths':  self.precision(num_of_tp=num_tp_with_fp_skip_edge_on_path,
                                                               num_of_predicted=tot_fun(self._stat_names.num_pred_edges)),
            'edges recall with s.e. paths':     self.recall(num_of_tp=num_tp_with_fn_skip_edge_on_path,
                                                            num_of_gt=tot_fun(self._stat_names.num_gt_edges)),
            'edge s.e. precision (on path)':    self.precision(num_of_tp=num_tp_skip_edges + tot_fun(self._stat_names.num_fp_skip_edges_path),
                                                               num_of_predicted=tot_fun(self._stat_names.num_pred_skip_edges)),
            'edge s.e. recall (on path)':    self.recall(
                                                            num_of_tp=num_tp_skip_edges + tot_fun(self._stat_names.num_fn_skip_edges_path),
                                                            num_of_gt=tot_fun(self._stat_names.num_gt_skip_edges))

        })
        return summary


class LoaderEval_SkipEdgeHandlerSoft(LoaderEval_SkipEdgeHandler):
    """
    SkipEdgeHandlerSoft considers TP skip edges also skip edges parallel to a path made of TP-detected nodes
    """
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates, unshared_parallel_path=False)


class DrawerLabels(Drawer):
    """Displays the Longit graph with the nodes' color as the ITKSNAP label color. With the parameter attr_to_show you
    can decide what text to print on the nodes. The default is label number"""
    def __init__(self, longit: Longit, attr_to_print=None):
        self._attr_to_print = attr_to_print
        if self._attr_to_print is not None:
            longit.nodes_have_attribute(self._attr_to_print)
        super().__init__(longit)

    def set_nodes_drawing_attributes(self):
        labels = nx.get_node_attributes(self._base_graph, name=NodeAttr.LABEL)
        colors = {node: {NodeAttr.COLOR: Colors.itk_colors(node_label)} for node, node_label in
                  labels.items()}
        nx.set_node_attributes(self._base_graph, colors)

    def attr_to_print_on_nodes(self):
        if self._attr_to_print is None:
            return NodeAttr.LABEL
        else:
            return self._attr_to_print


class DrawerEval(Drawer):
    def __init__(self, longit: Longit, attr: str):
        """
        A general drawer for evaluation graphs. According to the input attribute, shows node classification correctness,
        edge detection or node detection
        """
        self._is_node_eval = True
        self._attr = attr
        # needed for all teh graphs
        longit.nodes_have_attribute(NodeAttr.DETECTION)
        if attr == NodeAttr.EVAL_CLASSIFICATION:
            longit.nodes_have_attribute(attr)
        if attr == EdgeAttr.DETECTION:
            longit.edges_have_attribute(attr)
            self._is_node_eval = False
        super().__init__(longit)

    def set_nodes_drawing_attributes(self):
        """
        Set the base colors to the nodes: TP are highlighted.
        If the graph shows nodes evaluation, change the node colors
        """
        det_values = nx.get_node_attributes(self._base_graph, name=NodeAttr.DETECTION)
        base_colors = dict()
        for node, d_attr in det_values.items():
            if d_attr == NodesDetect.TP:
                clr = Colors.DARK_GRAY
            else:
                clr = Colors.GRAY
            base_colors.update({node: {NodeAttr.COLOR: clr}})
        nx.set_node_attributes(self._base_graph, base_colors)

        if self._is_node_eval:
            attr_values = nx.get_node_attributes(self._base_graph, name=self._attr)
            colors = {node: {NodeAttr.COLOR: Colors.choose(self._attr, node_attr)} for node, node_attr in
                      attr_values.items()}
            nx.set_node_attributes(self._base_graph, colors)

    def set_edges_drawing_attributes(self):
        super().set_edges_drawing_attributes()
        if not self._is_node_eval:
            attr_values = nx.get_edge_attributes(self._base_graph, name=self._attr)
            colors = {edge: {EdgeAttr.COLOR: Colors.choose(self._attr, edge_attr)} for edge, edge_attr in
                      attr_values.items()}
            nx.set_edge_attributes(self._base_graph, colors)

    def add_legend(self, nodes_position, attr_name=None, **kwarg):
        super().add_legend(nodes_position, self._attr)


class DrawerLabelsSkipEdges(DrawerLabels):
    """Display only the connected components that contain skip edges"""
    def __init__(self, longit: Longit, attr_to_print=None, label_edges=None):
        self._label_edges = label_edges
        super().__init__(longit, attr_to_print)

    def set_graph_layout(self):
        cc_subgraphs_all = [self._base_graph.subgraph(cc) for cc in nx.connected_components(self._base_graph)]
        cc_subgraphs = [cc for cc in cc_subgraphs_all if True in nx.get_edge_attributes(cc, EdgeAttr.IS_SKIP).values()]
        cc_graph = self.fill_with_placeholders(cc_subgraphs[0])
        for i in range(1, len(cc_subgraphs)):
            curr_cc_graph = self.fill_with_placeholders(cc_subgraphs[i])
            cc_graph = nx.compose(cc_graph, curr_cc_graph)
        self._base_graph = cc_graph

    def draw(self, pos):
        super().draw(pos)
        if self._label_edges is not None:
            nx.draw_networkx_edge_labels(G=self._base_graph,
                                         pos=pos,
                                         edge_labels={e: f"{lb}" for e, lb in nx.get_edge_attributes(self._base_graph, self._label_edges).items()
                                                      if self._base_graph.edges[e][EdgeAttr.IS_SKIP]})




if __name__ == "__main__":
    #cl = DeleteLayerTester()
    #dr2 = LongitLabelsDrawer(l2)
    #dr2.show_graph()
    # gr = l.get_graph()
    h = 1