import math
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict
import numpy as np
from common_packages.BaseClasses import Longit, NodeAttr, EdgeAttr, Colors, Drawer, Loader
from volume.volume_calculation import get_percentage_diff_per_edge_dict

def edit_volume_percentage_data_to_str_and_color(vol_percentage_diff_per_edge: dict):
    edited_dict = dict()
    color_dict = dict()
    for edge, percentage in vol_percentage_diff_per_edge.items():
        if percentage == "+":
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
    def __init__(self, longit: Longit, cc_idx: int, ld: Loader , components: list, longitudinal_volumes_array: list,
                  percentage_diff_per_edge_dict, attr_to_print=None):
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

        self.longitudinal_volumes_array = longitudinal_volumes_array
        self.percentage_diff_per_edge_dict = percentage_diff_per_edge_dict
        # get_percentage_diff_per_edge_dict(self.ld, self.partial_patient_path)

    def set_nodes_drawing_attributes(self):
        """Add to each node the color attribute GRAY"""
        nx.set_node_attributes(self._base_graph, values=Colors.GRAY, name=NodeAttr.COLOR)

    # def find_max_time_for_colour_funct()
    def set_nodes_drawing_attributes(self):
        labels = nx.get_node_attributes(self._base_graph, name=NodeAttr.LABEL)
        highest_time = max(int(key.split('_')[1]) for key in labels.keys())
        last_node_color_key = max(labels, key=lambda k: int(k.split('_')[1]))
        color_to_apply = Colors.itk_colors(labels[last_node_color_key])
        # colors = {node: {NodeAttr.COLOR: Colors.itk_colors(node_label)} for node, node_label in labels.items()}
        colors = {node: {NodeAttr.COLOR: color_to_apply} for node in labels.keys()}
        # last_node_colour =
        # same_colors = {node: {colors(node_label)} for node, node_label in
        #           labels.items()}
        nx.set_node_attributes(self._base_graph, colors)

    def get_node_volume(self, node_str : str):
        idx, time = node_str.split('_')
        if int(idx) in self.longitudinal_volumes_array[int(time)]:
            return round(self.longitudinal_volumes_array[int(time)][int(idx)], 2)
        return 0

    def attr_to_print_on_nodes(self):
        if self._attr_to_print is None:
            return NodeAttr.LABEL
        else:
            return self._attr_to_print
        
    def set_edges_drawing_attributes(self):
        super().set_edges_drawing_attributes()
        percentage_diff_per_edge_dict, color_dict = edit_volume_percentage_data_to_str_and_color(self.percentage_diff_per_edge_dict) # volume
        nx.set_edge_attributes(self._base_graph, percentage_diff_per_edge_dict, name='label') # volume
        nx.set_edge_attributes(self._base_graph, color_dict, name='color') # volume

    def set_nodes_volume_labels(self):
        is_place_holder_dict = nx.get_node_attributes(self._base_graph, NodeAttr.IS_PLACEHOLDER)
        nodes_volume_labels_dict = dict()
        for node, is_place_holder in is_place_holder_dict.items():
            if not is_place_holder:
                nodes_volume_labels_dict[node] = f'{self.get_node_volume(node)} cm³'
        return nodes_volume_labels_dict
    
    def get_lesion_idx(self):
        is_place_holder_dict = nx.get_node_attributes(self._base_graph, NodeAttr.IS_PLACEHOLDER)
        nodes_not_place_holders = [node for node, is_place_holder in is_place_holder_dict.items() if not is_place_holder]
        max_node = list(nodes_not_place_holders)[0]
        max_time_stamp = int(max_node.split("_")[1])
        for node in nodes_not_place_holders:
            time = int(node.split("_")[1])
            if time > max_time_stamp:
                max_node = node
                max_time_stamp = time

        max_nodes = [int(node.split("_")[0]) for node in nodes_not_place_holders if int(node.split("_")[1]) == max_time_stamp]

        return max_nodes
    
    def color_edges(self, pos):
        edge_labels = nx.get_edge_attributes(self._base_graph, 'label')
        colors = get_edge_label_color(edge_labels)
        for edge, label in edge_labels.items():
            if edge in colors: ## added this bc of keyerror
                color = colors[edge]
                nx.draw_networkx_edge_labels(G=self._base_graph, pos=pos, edge_labels={edge: label}, font_color=color)

    def draw_nodes_volume_labels(self, pos):
        nodes_volume_labels = self.set_nodes_volume_labels()
        nx.draw_networkx_labels(G=self._base_graph,
                                pos={k: (v[0], v[1]+0.3) for k, v in pos.items()},
                                labels=nodes_volume_labels, font_size=10, font_color='black')

    
    def draw_volume_related_attributes_on_graph(self, pos):
        self.color_edges(pos)
        self.draw_nodes_volume_labels(pos)

    def draw(self, pos):
        """This function prints the title of the figure and the graph"""
        plt.xlim([-1.5, 1.5])
        plt.ylim([-2, 2])
        Drawer.draw(self, pos)
        self.draw_volume_related_attributes_on_graph(pos) # volume
        nx.spring_layout(self._base_graph, scale=6.0)



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


    def write_dates(self, nodes_position):
        """Prints the layers' dates at the bottom of the layers"""
        # x position of layers:
        nodes_pos_x = [pos[0] for pos in nodes_position.values()]
        layer_pos_x = np.unique(nodes_pos_x)
        nodes_pos_y = [pos[1] for pos in nodes_position.values()]
        lower_node = np.min(nodes_pos_y)

        # lower date text
        lower_node -=0.3

        text_positions = [(pos_x, lower_node - 0.2) for pos_x in layer_pos_x]
        for layer_idx in range(self._num_of_layers):
            current_text_pos = text_positions[layer_idx]
            plt.text(current_text_pos[0], current_text_pos[1], self._patient_dates[layer_idx],
                     horizontalalignment='center')
    
