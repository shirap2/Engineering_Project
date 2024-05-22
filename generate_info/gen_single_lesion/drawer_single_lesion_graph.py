import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from common_packages.BaseClasses import Longit, NodeAttr, EdgeAttr, Colors, Drawer, Loader
from volume.volume_calculation import get_edges_to_node_dict, get_edges_from_node_dict
from enum import Enum



class BottomEdgeDesign(Enum):
    ONLY_TOTAL_CHANGE = 'total'
    ADDITIONAL_TOTAL_CHANGE = 'additional_total'
    SPLIT_MERGE_CHANGE = 'split_merge_change'
    NONE = 'none'


colors_dict = {
    0: "lightskyblue",
    1: "lightgoldenrodyellow",
    2: "lightcyan",
    3: "lightpink",
    4: "lightseagreen",
    5: "lavender",
    6: "lightsteelblue",
    7: "lightcoral",
    8: "lightblue",
    9: "lightyellow",
    10: "lightgray",
    11: "lightseagreen",
    12: "lightcoral",
    13: "lightpink",
    14: "lightblue",
    15: "lightgoldenrodyellow",
    16: "lightcyan",
    17: "lavender",
    18: "lightsteelblue",
    19: "lightgray"
}


def get_time(node: str):
    return int(node.split('_')[1])

def get_node_volume(node_str: str, longitudinal_volumes_array):
    idx, time = node_str.split('_')
    if int(idx) in longitudinal_volumes_array[int(time)]:
        return round(longitudinal_volumes_array[int(time)][int(idx)], 2), True
    return 0, False

def edit_volume_percentage_data_to_str_and_color(vol_percentage_diff_per_edge: dict):
    edited_dict = dict()
    color_dict = dict()
    for edge, percentage in vol_percentage_diff_per_edge.items():
        if percentage is not None:
            if percentage == "+inf":
                color = 'red'

            else:
                color = 'green'
                diff_is_positive = (percentage > 0)
                if math.isinf(percentage):
                    percentage = ""
                else:
                    percentage = str(round(percentage)) + "%"

                if diff_is_positive:
                    percentage = "+" + percentage
                    color = 'red'

        edited_dict.update({edge: percentage})
        color_dict.update({edge: color})

    return edited_dict, color_dict


def get_edge_label_color(edge_labels: dict):
    color_dict = dict()
    for edge, vol_percent_str in edge_labels.items():
        sign = vol_percent_str[0]
        if sign == '+':
            color_dict.update({edge: 'red'})
        elif sign == '-':
            color_dict.update({edge: 'green'})
    return color_dict


def get_nodes_for_graph(nodes_list, range_min, range_max):
    nodes_to_return = []
    for n in nodes_list:
        node_time = n.split("_")[1]
        if int(node_time) > int(range_max) or int(node_time) < int(range_min):
            continue
        nodes_to_return.append(n)
    return nodes_to_return


class PatientData:
    def __init__(self, longit: Longit, ld: Loader, components: list,
                 longitudinal_volumes_array: list, percentage_diff_per_edge_dict):
        self.lg = longit
        self.ld = ld
        self.components = components
        self.longitudinal_volumes_array = longitudinal_volumes_array
        self.percentage_diff_per_edge_dict = percentage_diff_per_edge_dict
        self.edges_to_node_dict = get_edges_to_node_dict(self.ld)  # {node : [edges to node]}
        self.edges_from_node_dict = get_edges_from_node_dict(self.ld)  # {node : [edges from node]}
        self.num_of_scans = len(self.lg.get_patient_dates())
        self.doesnt_appear_per_time_dict = self.get_doesnt_appear_per_time_dict()
        self.total_edges_without_doesnt_appear = self.get_total_edges_without_doesnt_appear()

    def get_doesnt_appear_per_time_dict(self):
        is_place_holder_dict = nx.get_node_attributes(self.lg.get_graph(), NodeAttr.IS_PLACEHOLDER)
        doesnt_appear_per_time_dict = dict()

        for t in range(self.num_of_scans):
            doesnt_appear_per_time_dict[t] = []

        for node, is_place_holder in is_place_holder_dict.items():
            if not is_place_holder:
                _, is_existing = get_node_volume(node, self.longitudinal_volumes_array)
                if not is_existing:
                    # if get_time(node) < self.num_of_scans:
                    doesnt_appear_per_time_dict[get_time(node)].append(node)
        return doesnt_appear_per_time_dict

    def get_total_edges_without_doesnt_appear(self):
        edges = []
        time = 0
        while (time < self.num_of_scans):
            if len(self.doesnt_appear_per_time_dict[time]) != 0:
                time += 1
                continue
            else:
                # no doesnt appear - time is a good src, search for next
                next = time + 1
                while next < self.num_of_scans and self.doesnt_appear_per_time_dict[next] != 0:
                    next += 1

                if next < self.num_of_scans:
                    # found a good next
                    edges.append((time, next))
                time = next
        print(edges)
        return edges



class GraphDisplay:
    def __init__(self, cc_idx: int, first_time_stamp: int, last_time_stamp: int):
        self.cc_idx = cc_idx
        self.first_time_stamp = first_time_stamp
        self.last_time_stamp = last_time_stamp


class DrawerLabelsAndLabeledEdges(Drawer):
    """Displays the Longit graph with the nodes' color as the ITKSNAP label color. With the parameter attr_to_show you
    can decide what text to print on the nodes. The default is label number"""
    def __init__(self, patient_data: PatientData, graph_display: GraphDisplay, attr_to_print=None):

        self.first_time_stamp = graph_display.first_time_stamp
        self.last_time_stamp = graph_display.last_time_stamp

        self._attr_to_print = attr_to_print
        if self._attr_to_print is not None:
            patient_data.lg.nodes_have_attribute(self._attr_to_print)

        self.ld = patient_data.ld

        G = patient_data.lg.get_graph()
        self._is_graph_empty = False
        if graph_display.cc_idx >= len(patient_data.components):
            self._is_graph_empty = True
            return
        nodes_to_put = get_nodes_for_graph(patient_data.components[graph_display.cc_idx],
                                           graph_display.first_time_stamp, graph_display.last_time_stamp - 1)
        if len(nodes_to_put) == 0:
            self._is_graph_empty = True
            return
        self.cc_idx = graph_display.cc_idx
        subgraph = G.subgraph([n for n in G if n in nodes_to_put])
        self._base_graph = subgraph

        patient_data.lg.nodes_have_attribute(NodeAttr.LAYER)
        patient_data.lg.nodes_have_attribute(NodeAttr.LABEL)
        self._cnt = 0
        self._num_of_layers = patient_data.lg.get_num_of_layers()

        pat_name = patient_data.lg.get_patient_name()
        if pat_name is None:
            self._patient_name = ""
        else:
            self._patient_name = pat_name

        pat_dates = patient_data.lg.get_patient_dates()
        if pat_dates is None:
            self._patient_dates = [f"t{i}" for i in range(self._num_of_layers)]
        else:
            self._patient_dates = pat_dates
        nx.set_node_attributes(self._base_graph, values=False, name=NodeAttr.IS_PLACEHOLDER)

        self.longitudinal_volumes_array = patient_data.longitudinal_volumes_array
        self.percentage_diff_per_edge_dict = patient_data.percentage_diff_per_edge_dict
        self.bottom_arrow_design = BottomEdgeDesign.ADDITIONAL_TOTAL_CHANGE

        self.should_print_label_on_edge = dict()
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)
        for edge in edge_is_skip.keys():
            head, tail = edge
            if int(head.split("_")[1]) > int(tail.split("_")[1]):
                edge = tail, head
            self.should_print_label_on_edge[edge] = True

        self.edges_to_node_dict = patient_data.edges_to_node_dict  # {node : [edges to node]}
        self.edges_from_node_dict = patient_data.edges_from_node_dict  # {node : [edges from node]}

        if self.bottom_arrow_design == BottomEdgeDesign.SPLIT_MERGE_CHANGE:
            self.process_merge_and_split_bottom_arrows_edges_labels(self.percentage_diff_per_edge_dict)
        elif self.bottom_arrow_design == BottomEdgeDesign.ONLY_TOTAL_CHANGE:
            self.process_none_bottom_arrows_edges_labels()
            self.process_total_bottom_arrows_edges_labels()
        elif self.bottom_arrow_design == BottomEdgeDesign.ADDITIONAL_TOTAL_CHANGE:
            self.process_none_bottom_arrows_edges_labels()
            self.process_total_bottom_arrows_edges_labels()
        elif self.bottom_arrow_design == BottomEdgeDesign.NONE:
            self.process_none_bottom_arrows_edges_labels()

        self.skipping_edges_for_doesnt_appear_dict = self.find_skipping_edges_for_doesnt_appear()
        self.nodes_volume_labels = self.set_nodes_volume_labels()

    def get_prev_appeared_nodes(self, node):
        prev_nodes = [src for src, _ in self.edges_to_node_dict[node]]
        for prev in prev_nodes:
            src_vol, _ = self.get_node_volume(prev)
            if src_vol == 0:
                # remove src
                prev_nodes.remove(prev)
                prev_nodes += self.get_prev_appeared_nodes(prev)
        return prev_nodes

    def get_next_appeared_nodes(self, node):
        next_nodes = [dest for _, dest in self.edges_from_node_dict[node]]
        for next in next_nodes:
            _, dest_vol = self.get_node_volume(next)
            if dest_vol == 0:
                # remove src
                next_nodes.remove(next)
                next_nodes += self.get_prev_appeared_nodes(next)
        return next_nodes

    def find_skipping_edges_for_doesnt_appear(self):
        skipping_edges_for_doesnt_appear_dict = dict()

        is_place_holder_dict = nx.get_node_attributes(self._base_graph, NodeAttr.IS_PLACEHOLDER)
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)

        # self.edges_to_node_dict = get_edges_to_node_dict(self.ld)  # {node : [edges to node]}
        # self.edges_from_node_dict = get_edges_from_node_dict(self.ld)  # {node : [edges from node]}

        for node, is_place_holder in is_place_holder_dict.items():
            if not is_place_holder:
                vol, is_existing = self.get_node_volume(node)
                if not is_existing and node not in skipping_edges_for_doesnt_appear_dict:
                    skipping_edges = []
                    prev_nodes = self.get_prev_appeared_nodes(node)
                    next_nodes = self.get_next_appeared_nodes(node)
                    for prev in prev_nodes:
                        for next in next_nodes:
                            skipping_edges.append((prev, next))
                    skipping_edges_for_doesnt_appear_dict[node] = skipping_edges
        return skipping_edges_for_doesnt_appear_dict


    def process_total_bottom_arrows_edges_labels(self):

        total_edges_to_add_dict = {}  # {(t1, t2): label}
        total_vol_list = [0 for _ in range(self._num_of_layers)]  # total_vol_list[i] is the total vol in layer i

        # fill in total_vol_list
        is_place_holder_dict = nx.get_node_attributes(self._base_graph, NodeAttr.IS_PLACEHOLDER)
        for node, is_place_holder in is_place_holder_dict.items():
            layer = get_time(node)
            if not is_place_holder:
                vol, is_existing = self.get_node_volume(node)
                if is_existing:
                    total_vol_list[layer - self.first_time_stamp] += vol

        for i in range(0, self._num_of_layers - 1):
            percentage_diff = "+inf"
            if total_vol_list[i] != 0:
                percentage_diff = ((total_vol_list[i + 1]/total_vol_list[i]) - 1) * 100
            time_stamp = i + self.first_time_stamp
            total_edges_to_add_dict[(time_stamp, time_stamp + 1)] = percentage_diff

        self.summing_edges_to_add_dict = total_edges_to_add_dict

    def process_none_bottom_arrows_edges_labels(self):
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)
        for edge in edge_is_skip.keys():
            head, tail = edge
            if int(head.split("_")[1]) > int(tail.split("_")[1]):
                edge = tail, head
                head, tail = edge

            if tail in self.edges_to_node_dict:
                if len(self.edges_to_node_dict[tail]) > 1:  # merge
                    self.should_print_label_on_edge[edge] = False

            if head in self.edges_from_node_dict:
                if len(self.edges_from_node_dict[head]) > 1:  # split
                    self.should_print_label_on_edge[edge] = False


    def process_merge_and_split_bottom_arrows_edges_labels(self, percentage_diff_per_edge_dict):
        total_edges_to_add_dict = {}  # {(t1, t2): label}

        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)
        for edge in edge_is_skip.keys():

            head, tail = edge
            if int(head.split("_")[1]) > int(tail.split("_")[1]):
                edge = tail, head
                head, tail = edge

            if tail in self.edges_to_node_dict:
                if len(self.edges_to_node_dict[tail]) > 1:  # merge
                    self.should_print_label_on_edge[edge] = False

                    t2 = int(tail.split("_")[1])
                    t1 = t2 - 1
                    total_edges_to_add_dict[(t1, t2)] = percentage_diff_per_edge_dict[edge]

            if head in self.edges_from_node_dict:
                if len(self.edges_from_node_dict[head]) > 1:  # split
                    self.should_print_label_on_edge[edge] = False

                    t1 = int(head.split("_")[1])
                    t2 = t1 + 1
                    total_edges_to_add_dict[(t1, t2)] = percentage_diff_per_edge_dict[edge]
        self.summing_edges_to_add_dict = total_edges_to_add_dict


    def set_nodes_drawing_attributes(self):
        """Add to each node the color attribute GRAY"""
        nx.set_node_attributes(self._base_graph, values=Colors.GRAY, name=NodeAttr.COLOR)


    def set_nodes_drawing_attributes(self):
        labels = nx.get_node_attributes(self._base_graph, name=NodeAttr.LABEL)
        highest_time = max(int(key.split('_')[1]) for key in labels.keys())
        last_node_color_key = max(labels, key=lambda k: int(k.split('_')[1]))
        # color_to_apply = Colors.itk_colors(labels[last_node_color_key])
        color_to_apply = colors_dict[self.cc_idx % 20]
        # colors = {node: {NodeAttr.COLOR: Colors.itk_colors(node_label)} for node, node_label in labels.items()}
        colors = {node: {NodeAttr.COLOR: color_to_apply} for node in labels.keys()}
        # last_node_colour =
        # same_colors = {node: {colors(node_label)} for node, node_label in
        #           labels.items()}
        nx.set_node_attributes(self._base_graph, colors)

    def get_node_volume(self, node_str: str):
        return get_node_volume(node_str, self.longitudinal_volumes_array)

    def attr_to_print_on_nodes(self):
        if self._attr_to_print is None:
            return NodeAttr.LABEL
        else:
            return self._attr_to_print

    def set_edges_drawing_attributes(self):
        nx.set_edge_attributes(self._base_graph, values=Colors.BLACK, name=EdgeAttr.COLOR)

        # remove edges that we dont want to print their label
        self.percentage_diff_per_edge_dict = {edge: vol for edge, vol in self.percentage_diff_per_edge_dict.items() if
                                              self.should_print_label_on_edge.get(edge)}
        # self.percentage_diff_per_edge_dict = {edge: vol for edge, vol in self.percentage_diff_per_edge_dict.items()}

        percentage_diff_per_edge_dict, color_dict = edit_volume_percentage_data_to_str_and_color(
            self.percentage_diff_per_edge_dict)  # volume
        nx.set_edge_attributes(self._base_graph, percentage_diff_per_edge_dict, name='label')  # volume
        nx.set_edge_attributes(self._base_graph, color_dict, name='color')  # volume

    def add_volume_labels_to_skipping_edges(self):
        percentage_diff = "+inf"
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)
        for edge, is_skip in edge_is_skip.items():
            if is_skip:
                # direct edge
                node1, node2 = edge
                node1_time = node1.split("_")[1]
                node2_time = node2.split("_")[1]
                if node1_time > node2_time:
                    edge = (node2, node1)
                else:
                    edge = (node1, node2)

                src, dest = edge
                src_vol, _ = self.get_node_volume(src)
                dest_vol, _ = self.get_node_volume(dest)

                if src_vol != 0:
                    percentage_diff = ((dest_vol / src_vol) - 1) * 100
                self.percentage_diff_per_edge_dict[edge] = percentage_diff
                self.should_print_label_on_edge[edge] = True

    def add_edge_skipping_over_node(self, node):
        # prev_node = ""
        # next_node = ""
        #
        # # find edge into unseen node
        # edges_before = self.edges_to_node_dict[node]
        # # if len(edges_before) != 1:
        # #     print("Error: more than one / zero edges into 'doesnt appear' node")
        # # else:
        # node1, node2 = edges_before[0]
        # if get_time(node2) < get_time(node1):
        #     node2, node1 = edges_before[0]
        #
        # # don't add if the root of the edge is out of the display
        # if get_time(node1) >= self.start:
        #     prev_node = node1
        # ########### add check if this is also unseen
        #
        # # find edge from unseen node
        # edges_after = self.edges_from_node_dict[node]
        # # if len(edges_after) != 1:
        # #     print("Error: more than one / zero edges from 'doesnt appear' node")
        # # else:
        # node1, node2 = edges_after[0]
        # if get_time(node2) < get_time(node1):
        #     node2, node1 = edges_before[0]
        #
        # # don't add if the tail of the edge is out of the display
        # if get_time(node1) < self.end:
        #     next_node = node2
        # ########### add check if this is also unseen

        # find edge before and after
        prev_node = ""
        next_node = ""
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)
        node_time = node.split("_")[1]
        for edge, _ in edge_is_skip.items():
            node1, node2 = edge

            if node2 == node:
                node1_time = node1.split("_")[1]
                if node1_time > node_time:
                    if next_node != "":
                        print("Error in add_edge_skipping_over_node 1")
                    next_node = node1
                else:
                    if prev_node != "":
                        print("Error in add_edge_skipping_over_node 2")
                    prev_node = node1

            if node1 == node:
                node2_time = node2.split("_")[1]
                if node2_time > node_time:
                    if next_node != "":
                        print("Error in add_edge_skipping_over_node 3")
                    next_node = node2
                else:
                    if prev_node != "":
                        print("Error in add_edge_skipping_over_node 4")
                    prev_node = node2

        if (prev_node == "") or (next_node == ""):
            print(f"Error: cant find skipping arrow for {node}")
        else:
            src_vol, _ = self.get_node_volume(prev_node)
            dest_vol, _ = self.get_node_volume(next_node)

            if (src_vol == 0) or (dest_vol == 0):
                print(f"Error: cant find skipping arrow for {node} because neighbor doesnt appear")
            else:
                print(f"added {(prev_node, next_node)}")
                mutable_graph = nx.Graph(self._base_graph)
                mutable_graph.add_edge(prev_node, next_node)
                mutable_graph.edges[(prev_node, next_node)][EdgeAttr.IS_SKIP] = True
                edge_is_skip = nx.get_edge_attributes(mutable_graph, name=EdgeAttr.IS_SKIP)
                self._base_graph = mutable_graph

                self.add_volume_labels_to_skipping_edges()
                self.should_print_label_on_edge[(prev_node, next_node)] = True


    def add_volume_labels_to_skipping_edges(self):
        percentage_diff = "+inf"
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)
        for edge, is_skip in edge_is_skip.items():
            if is_skip:
                # direct edge
                node1, node2 = edge
                node1_time = node1.split("_")[1]
                node2_time = node2.split("_")[1]
                if node1_time > node2_time:
                    edge = (node2, node1)
                else:
                    edge = (node1, node2)
                
                src, dest = edge
                src_vol, _ = self.get_node_volume(src)
                dest_vol, _ = self.get_node_volume(dest)

                if src_vol != 0:
                    percentage_diff = ((dest_vol/src_vol) - 1) * 100
                self.percentage_diff_per_edge_dict[edge] = percentage_diff


    def set_nodes_volume_labels(self):
        is_place_holder_dict = nx.get_node_attributes(self._base_graph, NodeAttr.IS_PLACEHOLDER)
        nodes_volume_labels_dict = dict()
        for node, is_place_holder in is_place_holder_dict.items():
            if not is_place_holder:
                vol, is_existing = self.get_node_volume(node)
                if is_existing:
                    nodes_volume_labels_dict[node] = f'{vol}cm³'
                else:
                    nodes_volume_labels_dict[node] = "doesn't\nappear"

                    if node in self.skipping_edges_for_doesnt_appear_dict:
                        for src_skip_node, dest_skip_node in self.skipping_edges_for_doesnt_appear_dict[node]:
                            if get_time(src_skip_node) >= self.first_time_stamp and get_time(dest_skip_node) <= self.last_time_stamp:
                                mutable_graph = nx.Graph(self._base_graph)
                                mutable_graph.add_edge(src_skip_node, dest_skip_node)
                                mutable_graph.edges[(src_skip_node, dest_skip_node)][EdgeAttr.IS_SKIP] = True
                                edge_is_skip = nx.get_edge_attributes(mutable_graph, name=EdgeAttr.IS_SKIP)
                                self._base_graph = mutable_graph

                                self.add_volume_labels_to_skipping_edges()
                                self.should_print_label_on_edge[(src_skip_node, dest_skip_node)] = True

        return nodes_volume_labels_dict


    def get_lesion_idx(self):
        is_place_holder_dict = nx.get_node_attributes(self._base_graph, NodeAttr.IS_PLACEHOLDER)
        nodes_not_place_holders = [node for node, is_place_holder in is_place_holder_dict.items() if
                                   not is_place_holder]
        max_node = list(nodes_not_place_holders)[0]
        max_time_stamp = int(max_node.split("_")[1])
        for node in nodes_not_place_holders:
            time = int(node.split("_")[1])
            if time > max_time_stamp:
                max_node = node
                max_time_stamp = time

        max_nodes = [int(node.split("_")[0]) for node in nodes_not_place_holders if
                     int(node.split("_")[1]) == max_time_stamp]

        return max_nodes


    def color_edges_labels(self, pos):
        import copy
        pos_of_skip_edges = copy.deepcopy(pos)
        for node in pos:
            pos_of_skip_edges[node][1] = pos_of_skip_edges[node][1] + 0.7

        is_skip_edge = nx.get_edge_attributes(self._base_graph, EdgeAttr.IS_SKIP)

        edge_labels = nx.get_edge_attributes(self._base_graph, 'label')
        colors = get_edge_label_color(edge_labels)
        for edge, label in edge_labels.items():
            if edge in colors:  # added this bc of keyerror
                color = colors[edge]
                if not is_skip_edge[edge]:
                    nx.draw_networkx_edge_labels(G=self._base_graph, pos=pos, edge_labels={edge: label},
                                                 font_color=color)
                else:
                    nx.draw_networkx_edge_labels(G=self._base_graph, pos=pos_of_skip_edges, edge_labels={edge: label},
                                                 font_color=color)

    def draw_nodes_volume_labels(self, pos):
        nx.draw_networkx_labels(G=self._base_graph,
                                pos={k: (v[0], v[1] + 0.3) for k, v in pos.items()},
                                labels=self.nodes_volume_labels, font_size=10, font_color='black')

    def draw_volume_related_attributes_on_graph(self, pos):
        if not self.bottom_arrow_design == BottomEdgeDesign.ONLY_TOTAL_CHANGE:
            self.color_edges_labels(pos)
        self.draw_nodes_volume_labels(pos)


    def add_bottom_arrows(self, nodes_position):
        add_to_pos = dict()
        add_to_edges = list()
        add_to_colors = dict()
        add_to_labels = dict()

        nodes_pos_x = [pos[0] for pos in nodes_position.values()]
        layer_pos_x = np.unique(nodes_pos_x)
        nodes_pos_y = [pos[1] for pos in nodes_position.values()]
        lower_node = np.min(nodes_pos_y)

        arrow_positions = [[pos_x, lower_node - 0.01] for pos_x in layer_pos_x]

        attributes = self._base_graph.nodes[list(self._base_graph.nodes())[0]].keys()

        for (t1, t2), vol in self.summing_edges_to_add_dict.items():
            if t1 >= self.first_time_stamp and t2 <= self.last_time_stamp:
                node1_attributes = {att: '' for att in attributes}
                node2_attributes = {att: '' for att in attributes}

                node1 = f'2000_{t1}'
                node2 = f'2000_{t2}'

                self._base_graph.add_node(node1)
                self._base_graph.add_node(node2)

                arrow_pos1 = arrow_positions[t1 - self.first_time_stamp]
                arrow_pos2 = arrow_positions[t2 - self.first_time_stamp]

                add_to_pos[node1] = np.array(arrow_pos1)
                add_to_pos[node2] = np.array(arrow_pos2)

                node1_attributes[NodeAttr.IS_PLACEHOLDER] = False
                node1_attributes[NodeAttr.LAYER] = t1
                node1_attributes[NodeAttr.LABEL] = ''
                node1_attributes[NodeAttr.COLOR] = Colors.WHITE

                node2_attributes[NodeAttr.IS_PLACEHOLDER] = False
                node2_attributes[NodeAttr.LAYER] = t1
                node2_attributes[NodeAttr.LABEL] = ''
                node2_attributes[NodeAttr.COLOR] = Colors.WHITE

                nx.set_node_attributes(self._base_graph, {node1: node1_attributes})
                nx.set_node_attributes(self._base_graph, {node2: node2_attributes})

                edge = (node1, node2)
                percentage_diff_dict, color_dict = edit_volume_percentage_data_to_str_and_color({edge: vol})
                add_to_edges.append(edge)
                add_to_colors.update(color_dict)
                add_to_labels.update(percentage_diff_dict)

        return add_to_edges, add_to_pos, add_to_colors, add_to_labels

    def draw(self, pos):
        """This function prints the title of the figure and the graph"""
        plt.xlim([-1.5, 1.5])
        plt.ylim([-2, 2])
        # plt.title(self._patient_name, fontsize=12)
        if not self.bottom_arrow_design == BottomEdgeDesign.NONE:
            add_to_edges, add_to_pos, add_to_colors, add_to_labels = self.add_bottom_arrows(pos)
            # add add_to_pos to pos
            pos = {**pos, **add_to_pos}

        # set the nodes size to default (300) and only the white nodes (for the split&merge arrows)
        # are small (so to not intifear the arrows)
        colors = nx.get_node_attributes(self._base_graph, NodeAttr.COLOR)
        size_list = []
        for node, color in colors.items():
            if color == Colors.WHITE:
                size_list.append(0)
            else:
                size_list.append(300)

        nx.draw_networkx_nodes(G=self._base_graph,
                               pos=pos,
                               node_color=list(nx.get_node_attributes(self._base_graph, NodeAttr.COLOR).values()),
                               node_size=size_list)

        nx.draw_networkx_labels(G=self._base_graph,
                                pos=pos,
                                labels=self.set_nodes_labels())

        is_skip_edge = nx.get_edge_attributes(self._base_graph, EdgeAttr.IS_SKIP)

        if not self.bottom_arrow_design == BottomEdgeDesign.ONLY_TOTAL_CHANGE:
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
                                   connectionstyle='arc3, rad=-0.5')
        else:  # no color
            nx.draw_networkx_edges(G=self._base_graph,
                                   pos=pos,
                                   edgelist=[e for e, is_skip in is_skip_edge.items() if not is_skip],
                                   connectionstyle='arc3')
            nx.draw_networkx_edges(G=self._base_graph,
                                   pos=pos,
                                   edgelist=[e for e, is_skip in is_skip_edge.items() if is_skip],
                                   connectionstyle='arc3, rad=-0.5')

        if not self.bottom_arrow_design == BottomEdgeDesign.NONE:
            # add the summing edges in the split and merge cases
            for edge, label in add_to_labels.items():
                if edge in add_to_colors:
                    color = add_to_colors[edge]
                    nx.draw_networkx_edge_labels(G=self._base_graph, pos=pos, edge_labels={edge: label}, font_color=color)
            nx.draw_networkx_edges(self._base_graph, pos, edgelist=add_to_edges,
                                   arrowstyle='|-|', width=2.0, edge_color=[c for e, c in
                                               add_to_colors.items()], node_size=0)  # actual white node size is 5, set edge as if it is 20

        self.draw_volume_related_attributes_on_graph(pos)  # volume
        nx.spring_layout(self._base_graph, scale=6.0)


    def set_graph_layout(self):
        """ Stack graph's connected components one upon the other and fill the blanks with placeholders """
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
        num_nodes_in_layer = \
            [nodes_layers.count(layer + self.first_time_stamp) for layer in range(self._num_of_layers)]
        max_num_nodes_in_layer = max(num_nodes_in_layer) + 1

        # extract the names of the attributes of a node
        node_attributes_names = sub_graph.nodes[list(sub_graph.nodes())[0]].keys()

        # fill with placeholders
        for layer in range(self._num_of_layers):
            for i in range(max_num_nodes_in_layer - num_nodes_in_layer[layer]):
                place_holder, ph_attr = \
                    self.create_placeholder(attributes=node_attributes_names, layer=layer + self.first_time_stamp)
                sub_graph.add_node(place_holder)
                nx.set_node_attributes(sub_graph, {place_holder: ph_attr})
        return sub_graph


    def write_dates(self, nodes_position):
        """Prints the layers' dates at the bottom of the layers"""
        # x position of layers:
        nodes_pos_x = [pos[0] for pos in nodes_position.values()]
        layer_pos_x = np.unique(nodes_pos_x)
        nodes_pos_y = [pos[1] for pos in nodes_position.values()]
        lower_node = np.min(nodes_pos_y)

        # # lower date text
        # lower_node -= 0.2

        text_positions = [(pos_x, lower_node - 0.3) for pos_x in layer_pos_x]
        for layer_idx in range(self._num_of_layers):
            current_text_pos = text_positions[layer_idx]
            plt.text(current_text_pos[0], current_text_pos[1], self._patient_dates[layer_idx],
                     horizontalalignment='center')
