from common_packages.LongGraphPackage import *
from common_packages.old_classes.Other_Drawers import *

class LoaderSimplePairwise(LoaderSimple):
    """Load the graph, discarding the skip edges"""
    def __init__(self, labels_list: List, edges_list: List):
        super().__init__(labels_list, edges_list)
        all_edges = self._edges
        self._edges = {e: attr for e, attr in all_edges.items() if not attr[EdgeAttr.IS_SKIP]}


class LoaderFromLongit(Loader):
    """Load a Longit from another Longit"""

    def __init__(self, longit_to_load):
        super().__init__()
        self._nodes = dict(longit_to_load.get_graph().nodes(data=True))
        self._edges = dict(longit_to_load.get_graph().edges(data=True))
        # self.patient_name = longit_to_load.get_patient_name()
        # self.patient_dates = longit_to_load.get_patient_dates()


class LoaderTwoMethods(LoaderEval):
    def __init__(self, pw_loader: Loader, gw_loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(pw_loader, gw_loader, patient_name, patient_dates)

    def evaluate_edges(self, merged_nodes_attributes: Dict):
        """
        Give the edges the DetectionMethod attribute: if the edge was detected in pw only, gw only, both
        """
        # check that the merged_nodes_attributes contains the DETECTION attribute for all the nodes
        assert np.all([NodeAttr.DETECTION in attr.keys() for attr in merged_nodes_attributes.values()])

        pw_edges = Longit.edgeview2dict(self._gt_longit.get_graph().edges(data=True), merged_nodes_attributes)
        gw_edges = Longit.edgeview2dict(self._pred_longit.get_graph().edges(data=True), merged_nodes_attributes)

        merged_edges_ids = set(pw_edges.keys()) | set(gw_edges.keys())
        merged_edges_attributes = {**pw_edges, **gw_edges}

        for edge in merged_edges_ids:
            node0 = edge[0]
            node1 = edge[1]
            # both nodes are TP
            if merged_nodes_attributes[node0][NodeAttr.DETECTION] == NodesDetect.TP and \
                    merged_nodes_attributes[node1][NodeAttr.DETECTION] == NodesDetect.TP:

                if (edge in pw_edges) and (edge in gw_edges):
                    merged_edges_attributes[edge].update({EdgeAttr.METHOD_DETECTION: EdgesMethodsDetect.BOTH})
                elif edge in pw_edges:
                    merged_edges_attributes[edge].update({EdgeAttr.METHOD_DETECTION: EdgesMethodsDetect.PW_MATCHING})
                elif edge in gw_edges:
                    merged_edges_attributes[edge].update({EdgeAttr.METHOD_DETECTION: EdgesMethodsDetect.GW_MATCHING})
                else:
                    raise ValueError("Edge must be or in pw or in gw")
            # at least one of the nodes is not TP
            else:
                merged_edges_attributes[edge].update({EdgeAttr.METHOD_DETECTION: EdgesMethodsDetect.UNSHARED})

        return merged_edges_attributes


class LoaderEval_TwoMethods(LoaderEval):
    """
    This class evaluates the predicted graph (with both methods) against the gt.
    It changes the calculation of the statistics
    """
    def __init__(self, gt_loader: Loader, pred_loader: LoaderTwoMethods, patient_name=None, patient_dates=None):
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates)
        self._stat_names = StatNamesTwoMethods()

    def get_det_class_edges(self, det_class):
        return self.get_if(self._edges, attr_name=EdgeAttr.DETECTION, attr_values=det_class)

    def get_num_edges_sub_class(self, det_class, sub_class_name, sub_class_value):
        return len(self.get_if(self.get_det_class_edges(det_class), attr_name=sub_class_name, attr_values=sub_class_value))

    def get_num_edges_method_sub_class(self, method, sub_class_name, sub_class_value):
        return len(
            self.get_if(self.get_if(self.get_det_class_edges(det_class=[EdgesDetect.TP, EdgesDetect.FP]),
                        attr_name=EdgeAttr.METHOD_DETECTION, attr_values=[EdgesMethodsDetect.BOTH, method]),
                        attr_name=sub_class_name, attr_values=sub_class_value)
            )

    def get_stats(self, pred_lesions):
        """calculate the statistics, and return them as a dictionary: {stat_name: stat_value}"""
        num_gt_cons = self.get_num_edges_sub_class(det_class=[EdgesDetect.TP, EdgesDetect.FN], sub_class_name=EdgeAttr.IS_SKIP, sub_class_value=False)
        num_gt_skip = self.get_num_edges_sub_class(det_class=[EdgesDetect.TP, EdgesDetect.FN], sub_class_name=EdgeAttr.IS_SKIP, sub_class_value=True)
        stat_dict = {
                    self._stat_names.num_gt_lesions: self.get_num_gt_lesions(),
                    self._stat_names.num_gt_consecutive_edges: num_gt_cons,
                    self._stat_names.num_gt_skip_edges: num_gt_skip,
                    self._stat_names.num_pw_consecutive_edges: self.get_num_edges_method_sub_class(method=EdgesMethodsDetect.PW_MATCHING, sub_class_name=EdgeAttr.IS_SKIP, sub_class_value=False),
                    self._stat_names.num_gw_consecutive_edges: self.get_num_edges_method_sub_class(method=EdgesMethodsDetect.GW_MATCHING, sub_class_name=EdgeAttr.IS_SKIP, sub_class_value=False),
                    self._stat_names.num_gw_skip_edges: self.get_num_edges_method_sub_class(method=EdgesMethodsDetect.GW_MATCHING, sub_class_name=EdgeAttr.IS_SKIP, sub_class_value=True),

                    self._stat_names.num_tp_edges_both: self.get_num_edges_sub_class(det_class=EdgesDetect.TP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=EdgesMethodsDetect.BOTH),
                    self._stat_names.num_fp_edges_both: self.get_num_edges_sub_class(det_class=EdgesDetect.FP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=EdgesMethodsDetect.BOTH),
                    #self._stat_names.num_fn_edges_both: self.get_num_edges_sub_class(det_class=EdgesDetect.FN, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=EdgesMethodsDetect.BOTH),
                    self._stat_names.num_fn_edges_both:  num_gt_cons - self.get_num_edges_sub_class(det_class=EdgesDetect.TP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=EdgesMethodsDetect.BOTH),

                    self._stat_names.num_tp_edges_pw: self.get_num_edges_sub_class(det_class=EdgesDetect.TP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value= [EdgesMethodsDetect.PW_MATCHING]),
                    self._stat_names.num_fp_edges_pw: self.get_num_edges_sub_class(det_class=EdgesDetect.FP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value= [EdgesMethodsDetect.PW_MATCHING]),
                    #self._stat_names.num_fn_edges_pw: self.get_num_edges_sub_class(det_class=EdgesDetect.FN, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=EdgesMethodsDetect.PW_MATCHING),
                    self._stat_names.num_fn_edges_pw: num_gt_cons - self.get_num_edges_sub_class(det_class=EdgesDetect.TP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=[EdgesMethodsDetect.BOTH, EdgesMethodsDetect.PW_MATCHING]),

                    self._stat_names.num_tp_edges_gw: self.get_num_edges_sub_class(det_class=EdgesDetect.TP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=[EdgesMethodsDetect.GW_MATCHING]),
                    self._stat_names.num_fp_edges_gw: self.get_num_edges_sub_class(det_class=EdgesDetect.FP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=[EdgesMethodsDetect.GW_MATCHING]),
                    #self._stat_names.num_fn_edges_gw: self.get_num_edges_sub_class(det_class=EdgesDetect.FN, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=EdgesMethodsDetect.GW_MATCHING),
                    self._stat_names.num_fn_edges_gw: num_gt_cons + num_gt_skip -
                                                      self.get_num_edges_sub_class(det_class=EdgesDetect.TP, sub_class_name=EdgeAttr.METHOD_DETECTION, sub_class_value=[EdgesMethodsDetect.BOTH, EdgesMethodsDetect.GW_MATCHING]),

                    self._stat_names.num_tp_edges_gw_skip: self.get_num_edges_sub_class(det_class=EdgesDetect.TP,
                                                                                        sub_class_name=EdgeAttr.IS_SKIP,
                                                                                        sub_class_value=True),
                    self._stat_names.num_fp_edges_gw_skip: self.get_num_edges_sub_class(det_class=EdgesDetect.FP,
                                                                                        sub_class_name=EdgeAttr.IS_SKIP,
                                                                                        sub_class_value=True),
                    self._stat_names.num_fn_edges_gw_skip: self.get_num_edges_sub_class(det_class=EdgesDetect.FN,
                                                                                        sub_class_name=EdgeAttr.IS_SKIP,
                                                                                        sub_class_value=True),

                          }
        return stat_dict

    def get_summary(self, tot_fun, pred_lesions=False):
        """
        :param tot_fun: an external function that given the name of a stat gets the sum of all the patient values for that stat.
        :param pred_lesions: a bool: if True, add to the summary nodes detection stat.
        """
        assert callable(tot_fun)
        summary = dict()
        summary.update({
            'both method edges precision': self.precision(tot_fun(self._stat_names.num_tp_edges_both),
                                                          tot_fun(self._stat_names.num_fp_edges_both) + tot_fun(self._stat_names.num_tp_edges_both)),
            'both method edges recall': self.recall(tot_fun(self._stat_names.num_tp_edges_both),
                                                          tot_fun(self._stat_names.num_fn_edges_both) + tot_fun(self._stat_names.num_tp_edges_both)),
            'pw only method edges precision': self.precision(tot_fun(self._stat_names.num_tp_edges_pw),
                                                          tot_fun(self._stat_names.num_fp_edges_pw) + tot_fun(
                                                              self._stat_names.num_tp_edges_pw)),
            'pw only method edges recall': self.precision(tot_fun(self._stat_names.num_tp_edges_pw),
                                                             tot_fun(self._stat_names.num_fn_edges_pw) + tot_fun(
                                                                 self._stat_names.num_tp_edges_pw)),
            'gw only method edges precision (all)': self.precision(tot_fun(self._stat_names.num_tp_edges_gw),
                                                             tot_fun(self._stat_names.num_fp_edges_gw) + tot_fun(
                                                                 self._stat_names.num_tp_edges_gw)),
            'gw only method edges recall (all)': self.precision(tot_fun(self._stat_names.num_tp_edges_gw),
                                                          tot_fun(self._stat_names.num_fn_edges_gw) + tot_fun(
                                                              self._stat_names.num_tp_edges_gw)),
            'gw only method edges precision (skip)': self.precision(tot_fun(self._stat_names.num_tp_edges_gw_skip),
                                                                   tot_fun(self._stat_names.num_fp_edges_gw_skip) + tot_fun(
                                                                       self._stat_names.num_tp_edges_gw_skip)),
            'gw only method edges recall (skip)': self.precision(tot_fun(self._stat_names.num_tp_edges_gw_skip),
                                                                tot_fun(self._stat_names.num_fn_edges_gw_skip) + tot_fun(
                                                                    self._stat_names.num_tp_edges_gw_skip)),

        })
        return summary


class LoaderEval_acceptCCEdges(LoaderEval):
    """
    This class expands the parent class by labeling FP/FN edges whose vertices belong to the same connect component in
    the graph in which the edge doesn't appear.
    """
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates)
        self._stat_names = StatNamesAcceptCCEdges()

    def evaluate_nodes(self):
        """Add to parent method the calculation of the connected component index for each graph NodeAttr.CC_INDEX,
        and a new attribute for the merged graph containing both graph's ids NodesSourceCC.UNSHARED"""
        merged_nodes_attributes = super().evaluate_nodes()
        self._gt_longit.add_cc_attribute()
        self._pred_longit.add_cc_attribute()
        for node in merged_nodes_attributes.keys():
            if merged_nodes_attributes[node][NodeAttr.DETECTION] == NodesDetect.TP:
                gt_cc_id = self._gt_longit.get_graph().nodes[node][NodeAttr.CC_INDEX]
                pred_cc_id = self._pred_longit.get_graph().nodes[node][NodeAttr.CC_INDEX]
                source_cc_ids = [None, None]
                source_cc_ids[NodesSourceCC.GT] = gt_cc_id
                source_cc_ids[NodesSourceCC.PRED] = pred_cc_id
                merged_nodes_attributes[node].update({NodeAttr.SOURCE_GRAPHS_CC_INDICES: source_cc_ids})
            else:
                merged_nodes_attributes[node].update({NodeAttr.SOURCE_GRAPHS_CC_INDICES: NodesSourceCC.UNSHARED})

        return merged_nodes_attributes

    def evaluate_edges(self, merged_nodes_attributes: Dict):
        """Give a further edge detection class after the parent method evaluation, changing the FP edges, whose vertices
        belong to the same GT connected component to EdgesDetect.FP_IN_SAME_CC and the FN edges connecting two vertices
         of the same cc in PRED graph to EdgesDetect.FN_IN_SAME_CC"""
        merged_edges_attributes = super().evaluate_edges(merged_nodes_attributes)
        merged_edges_attributes_new = merged_edges_attributes.copy()
        for edge, attr in merged_edges_attributes.items():
            if attr[EdgeAttr.DETECTION] == EdgesDetect.TP:
                continue
            if attr[EdgeAttr.DETECTION] == EdgesDetect.FP and self.is_edge_in_same_cc(edge, NodesSourceCC.GT, merged_nodes_attributes):
                merged_edges_attributes_new[edge][EdgeAttr.DETECTION] = EdgesDetect.FP_IN_SAME_CC
            elif attr[EdgeAttr.DETECTION] == EdgesDetect.FN and self.is_edge_in_same_cc(edge, NodesSourceCC.PRED, merged_nodes_attributes):
                merged_edges_attributes_new[edge][EdgeAttr.DETECTION] = EdgesDetect.FN_IN_SAME_CC

        return merged_edges_attributes_new

    @staticmethod
    def is_edge_in_same_cc(edge, graph_ind: int, merged_nodes_attributes):
        """Return True if the $edge's vertices belong to the same connected component in source graph $graph_ind"""
        assert graph_ind in [NodesSourceCC.GT, NodesSourceCC.PRED]
        node0 = edge[0]
        node1 = edge[1]
        node0_cc_ind = merged_nodes_attributes[node0][NodeAttr.SOURCE_GRAPHS_CC_INDICES][graph_ind]
        node1_cc_ind = merged_nodes_attributes[node1][NodeAttr.SOURCE_GRAPHS_CC_INDICES][graph_ind]
        return node0_cc_ind == node1_cc_ind

    def get_num_fn_edges_in_cc(self):
        return len(self.get_if(self._edges, attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.FN_IN_SAME_CC))

    def get_num_fp_edges_in_cc(self):
        return len(self.get_if(self._edges, attr_name=EdgeAttr.DETECTION, attr_values=EdgesDetect.FP_IN_SAME_CC))

    def get_num_gt_edges(self):
        return super().get_num_gt_edges() + self.get_num_fn_edges_in_cc()

    def get_num_pred_edges(self):
        return super().get_num_pred_edges() + self.get_num_fp_edges_in_cc()

    def get_stats(self, pred_lesions):
        stat_dict = super().get_stats(pred_lesions)
        stat_dict.update({
            self._stat_names.num_fp_edges_in_cc: self.get_num_fp_edges_in_cc(),
            self._stat_names.num_fn_edges_in_cc: self.get_num_fn_edges_in_cc()
        })
        return stat_dict

    def get_summary(self, tot_fun, pred_lesions):
        summary = super().get_summary(tot_fun, pred_lesions)
        num_tp_with_fp_in_cc = tot_fun(self._stat_names.num_tp_edges) + tot_fun(self._stat_names.num_fp_edges_in_cc)
        num_tp_with_fn_in_cc = tot_fun(self._stat_names.num_tp_edges) + tot_fun(self._stat_names.num_fn_edges_in_cc)

        summary.update({
            'edges precision in cc': self.precision(num_of_tp=num_tp_with_fp_in_cc,
                                                    num_of_predicted=tot_fun(self._stat_names.num_pred_edges)),
            'edges recall in cc': self.recall(num_of_tp=num_tp_with_fn_in_cc,
                                              num_of_gt=tot_fun(self._stat_names.num_gt_edges))
        })
        return summary


class LoaderEval_NodesCCPaths(LoaderEval):
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None):
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates)
        self._num_of_node_paths = None

    def evaluate_nodes(self):
        merged_nodes_attributes = super().evaluate_nodes()
        self._gt_longit.add_cc_attribute()
        self._pred_longit.add_cc_attribute()

        for node in merged_nodes_attributes.keys():
            if merged_nodes_attributes[node][NodeAttr.DETECTION] == NodesDetect.TP:

                gt_connected_nodes = self.get_connected_nodes(curr_node=node, longit=self._gt_longit,
                                                              merged_attr=merged_nodes_attributes)

                pred_connected_nodes = self.get_connected_nodes(curr_node=node, longit=self._pred_longit,
                                                                merged_attr=merged_nodes_attributes)

                num_tp_connected_nodes = len(gt_connected_nodes & pred_connected_nodes)
                num_gt_connected_nodes = len(gt_connected_nodes)
                num_pred_connected_nodes = len(pred_connected_nodes)

                node_connection = [None, None]
                node_connection[NodesPathCorrectness.PRECISION] = LoaderEval.precision(num_tp_connected_nodes,
                                                                                       num_pred_connected_nodes)
                node_connection[NodesPathCorrectness.RECALL] = LoaderEval.recall(num_tp_connected_nodes,
                                                                                 num_gt_connected_nodes)
                merged_nodes_attributes[node].update({NodeAttr.PATH_CORRECTNESS: node_connection})
            else:
                merged_nodes_attributes[node].update({NodeAttr.PATH_CORRECTNESS: NodesPathCorrectness.UNSHARED})

        return merged_nodes_attributes

    def get_connected_nodes(self, curr_node, longit, merged_attr):
        """
        Return a set of the connected nodes to the curr_node. The connection is calculated according to self._path_type
        :param curr_node: the node you want to check the connections
        :param longit: the Longit graph the nodes belong to.
        :param merged_attr: attributes of the evaluation between GT and Pred graph
        """

        nodes_cc = nx.get_node_attributes(longit.get_graph(), name=NodeAttr.CC_INDEX)
        connected_nodes = set()
        for n in longit.get_graph().nodes():
            if (n == curr_node) or (merged_attr[n][NodeAttr.DETECTION] != NodesDetect.TP):
                # add to connected nodes only nodes TP nodes, excluding the source node
                continue
            # this condition saves time
            if nodes_cc[curr_node] != nodes_cc[n]:
                continue

            # define the connection according to the path_type
            if self._path_type == PathType.UNDIRECTED:
                connected_nodes.add(n)
            else:
                # forward or backward paths can be maximum num of layers long
                paths = nx.all_simple_paths(longit.get_graph(), source=curr_node, target=n,
                                            cutoff=longit.get_num_of_layers())
                for p in paths:
                    if (self._path_type == PathType.FORWARD and
                            LoaderEval.is_forward_path(p, merged_attr, reverse=False)):
                        connected_nodes.add(n)
                        break
                    if (self._path_type == PathType.BACKWARD and
                            LoaderEval.is_forward_path(p, merged_attr, reverse=True)):
                        connected_nodes.add(n)
                        break
        return connected_nodes

    def get_node_path_stats(self):

        nodes_stat = self.get_if(self._nodes, attr_name=NodeAttr.DETECTION,
                                 attr_values=NodesDetect.TP)

        path_stats = np.array([v[NodeAttr.PATH_CORRECTNESS] for v in nodes_stat.values()])
        return path_stats

    def get_node_average_stats(self, stat_index: int):
        path_stats = self.get_node_path_stats()
        return np.mean(path_stats, axis=0)[stat_index]

    def get_node_std_stats(self, stat_index: int):
        path_stats = self.get_node_path_stats()
        return np.std(path_stats, axis=0)[stat_index]

    def get_num_node_paths(self):
        """Return the number of nodes for which the path statistics was calculated"""
        raise ValueError("Not implemented. Check in child classes")

    def get_stats(self, pred_lesions):
        stat_dict = super().get_stats(pred_lesions)
        stat_dict.update({
            self._stat_names.mean_node_path_precision: self.get_node_average_stats(NodesPathCorrectness.PRECISION),
            self._stat_names.std_node_path_precision: self.get_node_std_stats(NodesPathCorrectness.PRECISION),
            self._stat_names.mean_node_path_recall: self.get_node_average_stats(NodesPathCorrectness.RECALL),
            self._stat_names.std_node_path_recall: self.get_node_std_stats(NodesPathCorrectness.RECALL),
            self._stat_names.num_node_paths: self.get_num_node_paths()
        })
        return stat_dict

    def get_summary(self, tot_fun, pred_lesions, sum_prod_fun=None):

        summary = super().get_summary(tot_fun, pred_lesions)

        summary.update({
            'mean path precision': sum_prod_fun(self._stat_names.mean_node_path_precision,
                                                self._stat_names.num_node_paths)
                                   / tot_fun(self._stat_names.num_node_paths),
            'mean path recall': sum_prod_fun(self._stat_names.mean_node_path_recall, self._stat_names.num_node_paths)
                                / tot_fun(self._stat_names.num_node_paths),
        })
        return summary


class LoaderEval_NodesCCPaths_undirected(LoaderEval_NodesCCPaths):
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None):
        self._path_type = PathType.UNDIRECTED
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates)
        self._stat_names = StatNamesNodesCC(PathType.UNDIRECTED)


class LoaderEval_NodesCCPaths_forward(LoaderEval_NodesCCPaths):
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None):
        self._path_type = PathType.FORWARD
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates)
        self._stat_names = StatNamesNodesCC(PathType.FORWARD)
        self._num_of_node_paths = None

    def get_node_path_stats(self):
        """Do not take the last layer nodes stat. It will be always perfect, because there are no fw paths"""
        nodes_stat = self.get_if(self._nodes, attr_name=NodeAttr.DETECTION,
                                 attr_values=NodesDetect.TP)

        nodes_nolast_stat = self.get_if(nodes_stat, attr_name=NodeAttr.LAYER, attr_values=self.get_num_of_layers() - 1,
                                        excluded_values=True)
        path_stats = np.array([v[NodeAttr.PATH_CORRECTNESS] for v in nodes_nolast_stat.values()])
        return path_stats


class LoaderEval_NodesCCPaths_backward(LoaderEval_NodesCCPaths):
    def __init__(self, gt_loader: Loader, pred_loader: Loader, patient_name=None, patient_dates=None):
        self._path_type = PathType.BACKWARD
        super().__init__(gt_loader, pred_loader, patient_name, patient_dates)
        self._stat_names = StatNamesNodesCC(PathType.BACKWARD)
        self._num_of_node_paths = None

    def get_node_path_stats(self):
        path_stats = np.array(list(self.get_node_path_stats_dict().values()))
        return path_stats

    def get_node_path_stats_dict(self):
        """Do not take the first layer nodes stat. It will be always perfect, because there are no bw paths"""
        nodes_stat = self.get_if(self._nodes, attr_name=NodeAttr.DETECTION,
                                 attr_values=NodesDetect.TP)

        nodes_nofirst_stat = self.get_if(nodes_stat, attr_name=NodeAttr.LAYER, attr_values=0, excluded_values=True)
        path_stats_dict = {v: nodes_nofirst_stat[v][NodeAttr.PATH_CORRECTNESS] for v in nodes_nofirst_stat.keys()}
        return path_stats_dict

    def get_num_node_paths(self):
        nodes_stat = self.get_if(self._nodes, attr_name=NodeAttr.DETECTION,
                                 attr_values=NodesDetect.TP)

        nodes_nofirst_stat = self.get_if(nodes_stat, attr_name=NodeAttr.LAYER, attr_values=0, excluded_values=True)
        return len(nodes_nofirst_stat)


class LoaderEval_NodesCCPaths_Factory():
    def __init__(self):
        pass

    @staticmethod
    def get(path_type):
        if path_type == PathType.UNDIRECTED:
            return LoaderEval_NodesCCPaths_undirected
        elif path_type == PathType.FORWARD:
            return LoaderEval_NodesCCPaths_forward
        elif path_type == PathType.BACKWARD:
            return LoaderEval_NodesCCPaths_backward
        else:
            raise ValueError("path type not defined")


class RunDrawingFromJsons:
    """This class displays the GT, PRED, edge detection, lesion detection and classification correctness graphs"""
    def __init__(self,
                 gt_path: str,
                 pred_path: str,
                 eval_type: str,
                 path_type=None,
                 only_edge_detection=False,
                 pat_name=None,
                 pat_dates=None):

        if eval_type == EvaluationType.SKIP_EDGE_HANDLES:
            loader_eval = LoaderEval_SkipEdgeHandler
            drawer = DrawerEval_SkipEdgeHandler
        elif eval_type == EvaluationType.NODES_PATHS:
            loader_eval = LoaderEval_NodesCCPaths_Factory.get(path_type)
            drawer = DrawerEval
        elif eval_type == EvaluationType.ACCEPT_CC_EDGES:
            loader_eval = LoaderEval_acceptCCEdges
            drawer = DrawerEval
        elif eval_type == EvaluationType.SIMPLE:
            loader_eval = LoaderEval
            drawer = DrawerEval
        else:
            raise ValueError(f"{eval_type} is unknown")

        ev = loader_eval(LoaderSimpleFromJson(gt_path), LoaderSimpleFromJson(pred_path))
        longit = Longit(ev, patient_name=pat_name, patient_dates=pat_dates)
        # dr_gt = DrawerEvalSourceGraphs(longit, SourceGraph.GT)
        # dr_gt.show_graph()
        # dr_pred = DrawerEvalSourceGraphs(longit, SourceGraph.PRED)
        # dr_pred.show_graph()

        dr_det = drawer(longit, EdgeAttr.DETECTION)
        dr_det.show_graph()
        if not only_edge_detection:
            dr_n_det = drawer(longit, NodeAttr.DETECTION)
            dr_n_det.show_graph()
            dr_class = drawer(longit, NodeAttr.EVAL_CLASSIFICATION)
            dr_class.show_graph()

        if eval_type == EvaluationType.NODES_PATHS:
            dr_p_prec = DrawerPathScore(longit, path_det_attr=NodesPathCorrectness.PRECISION)
            dr_p_prec.show_graph()
            dr_p_recall = DrawerPathScore(longit, path_det_attr=NodesPathCorrectness.RECALL)
            dr_p_recall.show_graph()


# from lesions_matching.graph_compare_trials import json_creator_same_cc_edges, json_creator, json_creator_same_cc, many_lones, simple_ex

# class DeleteLayerTester:
#     def __init__(self):
#         gt_graph, _ = json_creator()
#         l0 = LoaderSimple(gt_graph["nodes"], gt_graph["edges"],)
#         l1 = LoaderSimple(gt_graph["nodes"], gt_graph["edges"], layers_to_delete=[0,2])
#         d0 = Drawer(Longit(l0))
#         d0.show_graph()
#         d1 = Drawer(Longit(l1))
#         d1.show_graph()


if __name__ == "__main__":
    #cl = DeleteLayerTester()
    a = 1