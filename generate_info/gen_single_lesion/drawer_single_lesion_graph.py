import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict
import numpy as np
from common_packages.BaseClasses import Longit, NodeAttr, EdgeAttr, Colors, Drawer, Loader
from volume.volume_calculation import get_percentage_diff_per_edge_dict, generate_longitudinal_volumes_array

def edit_volume_percentage_data_to_str_and_color(vol_percentage_diff_per_edge: dict):
    edited_dict = dict()
    color_dict = dict()
    for edge, percentage in vol_percentage_diff_per_edge.items():
        color = 'green'
        diff_is_positive = (percentage > 0)
        percentage = str(round(percentage)) + "%"

        if diff_is_positive:
            percentage = "+" + percentage
            color = 'red'

        edited_dict.update({edge: percentage})
        color_dict.update({edge: color})
    
    return edited_dict, color_dict



def get_node_volume(node_str : str, partial_patient_path : str):
    idx, time = node_str.split('_')
    longitudinal_volumes_array = generate_longitudinal_volumes_array(partial_patient_path)
    return round(longitudinal_volumes_array[int(time)][int(idx)], 2)


def get_edge_label_color(edge_labels : dict):
    color_dict = dict()
    for edge, vol_percent_str in edge_labels.items():
        sign = vol_percent_str[0]
        if sign == '+':
            color_dict.update({edge : 'red'})
        elif sign == '-':
            color_dict.update({edge : 'green'})
    return color_dict


class DrawerLabelsAndLabeledEdges(Drawer):
    """Displays the Longit graph with the nodes' color as the ITKSNAP label color. With the parameter attr_to_show you
    can decide what text to print on the nodes. The default is label number"""
    def __init__(self, longit: Longit, cc_idx: int, partial_patient_path : str, ld: Loader , attr_to_print=None):
        self._attr_to_print = attr_to_print
        if self._attr_to_print is not None:
            longit.nodes_have_attribute(self._attr_to_print)

        self.partial_patient_path = partial_patient_path
        self.ld = ld
        G = longit.get_graph()
        components = list(nx.connected_components(G))

        self._is_graph_empty = False
        if cc_idx >= len(components):
            self._is_graph_empty = True
            return
        subgraph = G.subgraph(components[cc_idx])
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

    def set_nodes_drawing_attributes(self):
        """Add to each node the color attribute GRAY"""
        nx.set_node_attributes(self._base_graph, values=Colors.GRAY, name=NodeAttr.COLOR)


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
        
    def set_edges_drawing_attributes(self):
        """Add to each node the color attribute BLACK and set the connection style"""
        super().set_edges_drawing_attributes()
        percentage_diff_per_edge_dict = get_percentage_diff_per_edge_dict(self.ld, self.partial_patient_path)
        percentage_diff_per_edge_dict, color_dict = edit_volume_percentage_data_to_str_and_color(percentage_diff_per_edge_dict)
        nx.set_edge_attributes(self._base_graph, percentage_diff_per_edge_dict, name='label')
        nx.set_edge_attributes(self._base_graph, color_dict, name='color')

    def set_nodes_volume_labels(self):
        idx_dict = nx.get_node_attributes(self._base_graph, self.attr_to_print_on_nodes())
        # return {node : f'Lesion ID: {idx}\nVolume: {get_volume(node)}[cm³]' for node, idx in idx_dict.items()}
        # print(idx_dict)
        return {node : f'{get_node_volume(node, self.partial_patient_path)}' for node, idx in idx_dict.items()}
    
    def get_lesion_idx(self):
        current_node = list(self._base_graph.nodes)[0]
        max_time_stamp = int(current_node.split("_")[1])
        for node in self._base_graph.nodes:
            time = int(node.split("_")[1])
            if time > max_time_stamp:
                current_node = node

        return int(node.split("_")[0])

    
    def draw(self, pos):
        """This function prints the title of the figure and the graph"""

        # node_sizes = nx.get_node_attributes(self._base_graph, 'size')
        # scaling_factor = 0.5
        # font_sizes = {n: int(size * scaling_factor) for n, size in node_sizes.items()}
        plt.xlim([-2, 2])
        plt.ylim([-1, 1])
        plt.title(self._patient_name, fontsize=12)
        nx.draw_networkx_nodes(G=self._base_graph,
                               pos=pos,
                               node_color=list(nx.get_node_attributes(self._base_graph, NodeAttr.COLOR).values()))
        nx.draw_networkx_labels(G=self._base_graph,
                                pos=pos,
                                labels=self.set_nodes_labels())
        nodes_volume_labels = self.set_nodes_volume_labels()
        nx.draw_networkx_labels(G=self._base_graph,
                                pos={k: (v[0], v[1]+0.15) for k, v in pos.items()},
                                labels=nodes_volume_labels, font_size=10, font_color='black')
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
        edge_labels = nx.get_edge_attributes(self._base_graph, 'label')
        colors = get_edge_label_color(edge_labels)
        for edge, label in edge_labels.items():
            color = colors[edge]
            nx.draw_networkx_edge_labels(G=self._base_graph, pos=pos, edge_labels={edge: label}, font_color=color)



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
    
