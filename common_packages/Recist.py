import networkx as nx
from common_packages.old_classes.Old_LongGraphPackage import DrawerBiggestComponents
from common_packages.LongGraphPackage import *
from common_packages.LesionsAnalysisPackage import *


class RecistSimulation:
    def __init__(self, lesion_data_path, pat2matching):
        """
        :param lesion_data_path: a path of an excel in which lesions data are stored. The file should be obtained by LesionAnalysisPackage
        :param pat2matching: a dictionary {pat_name: patient longitudinal matching graph json path
        """
        self._lesion_data_path = lesion_data_path
        self._pat2matching_path = pat2matching
        self._patient_list = list(pat2matching.keys())
        self._pat2longit = self.load_patients_longits()

    def load_patients_longits(self):
        """
        Create a dict in which each patient is assigned his loaded longit graph, with:
        extrapulated diameter (from volume) and with calliper diameter
        """
        pat2longit = dict()
        for pat in self._patient_list:
            df = pd.read_excel(self._lesion_data_path, index_col=0, sheet_name=pat)
            les2extr_diam = dict(df[LesionAttr.DIAMETER])
            les2cal_diam = dict(df[LesionAttr.CALLIPER_DIAMETER])
            l = LoaderSimpleFromJson(self._pat2matching_path[pat])
            longit = Longit(l)
            longit.add_node_attribute_from_dict(attr_dict=les2extr_diam, attr_name=NodeAttr.DIAMETER)
            longit.add_node_attribute_from_dict(attr_dict=les2cal_diam, attr_name=NodeAttr.CAL_DIAMETER)
            pat2longit.update({pat: longit})
        return pat2longit

    def show_all(self):
        for l in self._pat2longit.values():
            d = DrawerMax2Diam(l, NodeAttr.DIAMETER)
            d.show_graph()

    def max2recist(self, diam_type, longit):
        for l in range(1, longit._num_of_layers):
            bw_longit = longit.make_graph_directed(backwards=True)


class DrawerMax2Diam(DrawerBiggestComponents):
    def __init__(self, longit, diam_type):
        super().__init__(longit, diam_attr=diam_type, same_color_cc=False)

    def set_nodes_drawing_attributes(self):
        node2diam = nx.get_node_attributes(self._base_graph, name=self._diam_attr)
        node2layer = nx.get_node_attributes(self._base_graph, name=NodeAttr.LAYER)
        node2colors = dict()
        num_layers = self._num_of_layers
        for layer in range(num_layers):
            nodes_in_layer = [n for n, l in node2layer.items() if l==layer]
            diam_in_layer = [d for n, d in node2diam.items() if n in nodes_in_layer]
            nodes_ordered_in_size = [n for _, n in sorted(zip(diam_in_layer, nodes_in_layer), key=lambda pair: pair[0],
                                                          reverse=True)]
            #largest two get special color. Other the gray color.
            for i, n in enumerate(nodes_ordered_in_size):
                if i < 2:
                    node2colors.update({n: {NodeAttr.COLOR: Colors.LIGHT_GREEN}})
                else:
                    node2colors.update({n: {NodeAttr.COLOR: Colors.GRAY}})
        nx.set_node_attributes(self._base_graph, node2colors)

    def set_nodes_labels(self):
        node2diam = nx.get_node_attributes(self._base_graph, name=self._diam_attr)
        node2label = nx.get_node_attributes(self._base_graph, name=NodeAttr.LABEL)
        node2placeholder = nx.get_node_attributes(self._base_graph, name=NodeAttr.IS_PLACEHOLDER)
        n2text = dict()
        for n, d in node2diam.items():
            if node2placeholder[n]:
                n2text.update({n: ""})
            else:
                n2text.update({n: f"{node2label[n]}: {round(d,1)}"})
        return n2text
