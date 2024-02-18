import math
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict
import numpy as np
from common_packages.BaseClasses import Longit, NodeAttr, EdgeAttr, Colors, Drawer, Loader

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


class DrawerLabelsAndLabeledEdges(Drawer):
    """Displays the Longit graph with the nodes' color as the ITKSNAP label color. With the parameter attr_to_show you
    can decide what text to print on the nodes. The default is label number"""

    def __init__(self, longit: Longit, cc_idx: int, ld: Loader, components: list,
                 longitudinal_volumes_array: list,
                 percentage_diff_per_edge_dict, start: int, end_of_patient_dates: int, attr_to_print=None):
        self._attr_to_print = attr_to_print
        if self._attr_to_print is not None:
            longit.nodes_have_attribute(self._attr_to_print)

        # self.partial_patient_path = partial_patient_path
        self.ld = ld
        G = longit.get_graph()

        self._is_graph_empty = False
        if cc_idx >= len(components):
            self._is_graph_empty = True
            return
        # subgraph = G.subgraph(components[cc_idx])
        nodes_to_put = get_nodes_for_graph(components[cc_idx], start, end_of_patient_dates - 1)
        if len(nodes_to_put) == 0:
            self._is_graph_empty = True
            return
        self.cc_idx = cc_idx
        self.starting_node_idx = start
        subgraph = G.subgraph([n for n in G if n in nodes_to_put])
        self._base_graph = subgraph

        longit.nodes_have_attribute(NodeAttr.LAYER)
        longit.nodes_have_attribute(NodeAttr.LABEL)
        self._cnt = 0
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

        self.longitudinal_volumes_array = longitudinal_volumes_array
        self.percentage_diff_per_edge_dict = percentage_diff_per_edge_dict

        self.nodes_volume_labels = self.set_nodes_volume_labels()

    def set_nodes_drawing_attributes(self):
        """Add to each node the color attribute GRAY"""
        nx.set_node_attributes(self._base_graph, values=Colors.GRAY, name=NodeAttr.COLOR)

    # def find_max_time_for_colour_funct()
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
        idx, time = node_str.split('_')
        if int(idx) in self.longitudinal_volumes_array[int(time)]:
            return round(self.longitudinal_volumes_array[int(time)][int(idx)], 2), True
            # return self.longitudinal_volumes_array[int(time)][int(idx)] # todo talia check
        return 0, False

    def attr_to_print_on_nodes(self):
        if self._attr_to_print is None:
            return NodeAttr.LABEL
        else:
            return self._attr_to_print

    def set_edges_drawing_attributes(self):

        nx.set_edge_attributes(self._base_graph, values=Colors.BLACK, name=EdgeAttr.COLOR)
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)

        # connectivity = dict()
        # for edge, is_skip in edge_is_skip.items():
        #     if not is_skip:
        #         connectivity.update({edge: {EdgeAttr.CONNECTION_STYLE: 'arc3'}})
        #     else:
        #         connectivity.update({edge: {EdgeAttr.CONNECTION_STYLE: 'arc3,rad=0.3'}})
        # nx.set_edge_attributes(self._base_graph, connectivity)

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

    def add_edge_skipping_over_node(self, node):
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
                        print("Error: LESION DOESNT APPEAR AND ISNT IN LINEAR FORMAT")
                    next_node = node1
                else:
                    if prev_node != "":
                        print("Error: LESION DOESNT APPEAR AND ISNT IN LINEAR FORMAT")
                    prev_node = node1

            if node1 == node:
                node2_time = node2.split("_")[1]
                if node2_time > node_time:
                    if next_node != "":
                        print("Error: LESION DOESNT APPEAR AND ISNT IN LINEAR FORMAT")
                    next_node = node2
                else:
                    if prev_node != "":
                        print("Error: LESION DOESNT APPEAR AND ISNT IN LINEAR FORMAT")
                    prev_node = node2
        if (prev_node == "") or (next_node == ""):
            print("Error: LESION DOESNT APPEAR AND ISNT IN LINEAR FORMAT")
        else:
            mutable_graph = nx.Graph(self._base_graph)
            mutable_graph.add_edge(prev_node, next_node)
            mutable_graph.edges[(prev_node, next_node)][EdgeAttr.IS_SKIP] = True
            edge_is_skip = nx.get_edge_attributes(mutable_graph, name=EdgeAttr.IS_SKIP)
            self._base_graph = mutable_graph

            self.add_volume_labels_to_skipping_edges()

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
                    self.add_edge_skipping_over_node(node)  # add edge skipping over this node

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

    def color_edges(self, pos):
        vol_edge_label_pos = dict()
        edge_is_skip = nx.get_edge_attributes(self._base_graph, name=EdgeAttr.IS_SKIP)

        # for edge, is_skip in edge_is_skip.items():
        #     if is_skip:
        #         v = pos[edge[0]]
        #         vol_edge_label_pos[edge[0]] = (v[0], v[1]+0.3)
        #     else:
        #         vol_edge_label_pos[edge[0]] = pos[edge[0]]

        #     k1, v1 = pos[edge[0]]
        #     if is_skip:
        #         vol_edge_label_pos[edge[0]] = [k1, v1+1]
        #     else:
        #         vol_edge_label_pos[edge[0]] = pos[edge[0]]

        # # print(vol_edge_label_pos)
        # for k, v in pos.items():
        #     print(k)
        #     print((v[0], v[1]+0.3))
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
        self.color_edges(pos)
        self.draw_nodes_volume_labels(pos)

    def draw(self, pos):
        """This function prints the title of the figure and the graph"""
        plt.xlim([-1.5, 1.5])
        plt.ylim([-2, 2])
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
                               connectionstyle='arc3, rad=-0.5')
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
            [nodes_layers.count(layer + self.starting_node_idx) for layer in range(self._num_of_layers)]
        max_num_nodes_in_layer = max(num_nodes_in_layer) + 1

        # extract the names of the attributes of a node
        node_attributes_names = sub_graph.nodes[list(sub_graph.nodes())[0]].keys()

        # fill with placeholders
        for layer in range(self._num_of_layers):
            for i in range(max_num_nodes_in_layer - num_nodes_in_layer[layer]):
                place_holder, ph_attr = \
                    self.create_placeholder(attributes=node_attributes_names, layer=layer + self.starting_node_idx)
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

        # lower date text
        lower_node -= 0.3

        text_positions = [(pos_x, lower_node - 0.2) for pos_x in layer_pos_x]
        for layer_idx in range(self._num_of_layers):
            current_text_pos = text_positions[layer_idx]
            plt.text(current_text_pos[0], current_text_pos[1], self._patient_dates[layer_idx],
                     horizontalalignment='center')
