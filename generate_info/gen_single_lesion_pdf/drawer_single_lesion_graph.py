import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict
import numpy as np
from common_packages.BaseClasses import Longit, NodeAttr, EdgeAttr, Colors, Drawer
from volume_calculation.volume_calculation import get_percentage_diff_per_edge_dict, generate_longitudinal_volumes_array

def edit_volume_percentage_data_to_str(vol_percentage_diff_per_edge: dict):
    edited_dict = dict()
    for edge, percentage in vol_percentage_diff_per_edge.items():

        diff_is_positive = (percentage > 0)
        percentage = str(round(percentage, 2)) + "%"

        if diff_is_positive:
            percentage = "+" + percentage

        edited_dict.update({edge: percentage})
    
    return edited_dict



def get_node_volume(node_str : str, partial_patient_path : str):
    idx, time = node_str.split('_')
    longitudinal_volumes_array = generate_longitudinal_volumes_array(partial_patient_path)
    return round(longitudinal_volumes_array[int(time)][int(idx)], 2)


def get_edge_label_color(edge_labels : dict):
    # color_dict = dict()
    # for edge, vol_percent_str in edge_labels.items():
    #     sign = vol_percent_str[0]
    #     if sign == '+':
    #         color_dict.update({edge : 'red'})
    #     elif sign == '-':
    #         color_dict.update({edge : 'green'})
    # return color_dict
    return 'red'


class DrawerLabelsAndLabeledEdges(Drawer):
    """Displays the Longit graph with the nodes' color as the ITKSNAP label color. With the parameter attr_to_show you
    can decide what text to print on the nodes. The default is label number"""
    def __init__(self, longit: Longit, partial_patient_path : str, ld , attr_to_print=None):
        self._attr_to_print = attr_to_print
        if self._attr_to_print is not None:
            longit.nodes_have_attribute(self._attr_to_print)
        super().__init__(longit)
        self.partial_patient_path = partial_patient_path
        self.ld = ld

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
        percentage_diff_per_edge_dict = edit_volume_percentage_data_to_str(percentage_diff_per_edge_dict)
        nx.set_edge_attributes(self._base_graph, percentage_diff_per_edge_dict, name='label')

    def set_nodes_volume_labels(self):
        idx_dict = nx.get_node_attributes(self._base_graph, self.attr_to_print_on_nodes())
        # return {node : f'Lesion ID: {idx}\nVolume: {get_volume(node)}[mm³]' for node, idx in idx_dict.items()}
        print(idx_dict)
        return {node : f'{get_node_volume(node, self.partial_patient_path)}[mm³]' for node, idx in idx_dict.items()}
    
    def draw(self, pos):
        """This function prints the title of the figure and the graph"""

        # node_sizes = nx.get_node_attributes(self._base_graph, 'size')
        # scaling_factor = 0.5
        # font_sizes = {n: int(size * scaling_factor) for n, size in node_sizes.items()}

        plt.title(self._patient_name, fontsize=12)
        nx.draw_networkx_nodes(G=self._base_graph,
                               pos=pos,
                               node_color=list(nx.get_node_attributes(self._base_graph, NodeAttr.COLOR).values()))
        nx.draw_networkx_labels(G=self._base_graph,
                                pos=pos,
                                labels=self.set_nodes_labels())
        nx.draw_networkx_labels(G=self._base_graph,
                                pos={k: (v[0], v[1]-0.07) for k, v in pos.items()},
                                labels=self.set_nodes_volume_labels(), font_size=10, font_color='black')
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
        nx.draw_networkx_edge_labels(G=self._base_graph, pos=pos, edge_labels=edge_labels, font_color=get_edge_label_color(edge_labels), font_size=10)

        
