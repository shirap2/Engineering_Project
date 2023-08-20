import networkx as nx
import pandas as pd

from common_packages.LongGraphPackage import *
from common_packages.LongGraphClassification import *

class ColNames:
    n_lone = "Lesions: #Lone"
    n_new = "Lesions: #New"
    n_dis = "Lesions: #Disappeared"
    n_unique = "Lesions: #Unique"
    n_merge = "Lesions: #Merge"
    n_merge_dis = "Lesions: #Merge-Disappeared"
    n_split = "Lesions: #Split"
    n_split_new = "Lesions: #Split-New"
    n_complex = "Lesions: #Complex"
    c_single = "CC: #Single"
    c_linear = "CC: #Linear"
    c_merge = "CC: #Merge"
    c_split = "CC: #Split"
    c_complex = "CC: #Complex"

    col2attr = {
        n_lone: NodesChanges.LONE,
        n_new: NodesChanges.NEW,
        n_dis: NodesChanges.DISAPPEARED,
        n_unique: NodesChanges.UNIQUE,
        n_merge: NodesChanges.MERGED,
        n_merge_dis: NodesChanges.MERGED_DISAP,
        n_split: NodesChanges.SPLITTING,
        n_split_new: NodesChanges.SPLITTING_NEW,
        n_complex: NodesChanges.COMPLEX,
        c_single: CcPatterns.SINGLE,
        c_linear: CcPatterns.LINEAR,
        c_merge: CcPatterns.MERGING,
        c_split: CcPatterns.SPLITTING,
        c_complex: CcPatterns.COMPLEX
    }

    attr2col = {val: key for key, val in col2attr.items()}


class ChangesReport:
    """Fill ColNames with values: How many Individual Lesion changes of each class, how many Patterns of lesion changes
    of each class"""
    def __init__(self, path_table_save, patient_list):
        self._path_to_save = path_table_save
        self._patients = patient_list
        self._df_data = {"patient": []}
        self._df_data.update({col_name: [] for col_name in ColNames.col2attr.keys()})

    def get_patient_json(self, pat_name):
        raise ValueError("Abstract function!")

    def run(self):
        writer = pd.ExcelWriter(self._path_to_save, engine='xlsxwriter')
        for patient in self._patients:
            self._df_data["patient"].append(patient)
            pat_json = self.get_patient_json(patient)
            loader = LoaderSimpleFromJson(pat_json)
            longit = LongitClassification(loader, patient_name=patient)
            longit.classify_nodes()
            longit.classify_cc()

            cl_graph = longit.get_graph()

            node_changes = list(nx.get_node_attributes(cl_graph, NodeAttr.CHANGES).values())
            changes_attr = [attr for attr in ColNames.col2attr.values() if NodesChanges.is_attr(attr)]
            for attr_name in changes_attr:
                self._df_data[ColNames.attr2col[attr_name]].append(node_changes.count(attr_name))

            nodes2cc = list(nx.get_node_attributes(cl_graph, NodeAttr.CC_INDEX).values())
            nodes2pattern = list(nx.get_node_attributes(cl_graph, NodeAttr.CC_PATTERNS).values())
            cc_patterns = list(dict(zip(nodes2cc, nodes2pattern)).values())
            patterns_attr = [attr for attr in ColNames.col2attr.values() if CcPatterns.is_attr(attr)]
            for attr_name in patterns_attr:
                self._df_data[ColNames.attr2col[attr_name]].append(cc_patterns.count(attr_name))

        df = pd.DataFrame(self._df_data)
        df.loc['Total'] = df.sum(numeric_only=True)
        self.write_to_excel(df, writer)
        writer.save()

    @staticmethod
    def write_to_excel(df, writer, sheet_name='-'):
        """Function to write the excel with a nice layout"""
        columns_order = list(df.columns)
        workbook = writer.book
        cell_format = workbook.add_format({'num_format': '#,##0.00'})
        cell_format.set_font_size(16)

        df.to_excel(writer, sheet_name=sheet_name, startrow=1, startcol=1, header=False, index=False)
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'font_size': 16,
            'valign': 'top',
            'border': 1})

        worksheet = writer.sheets[sheet_name]
        worksheet.freeze_panes(1, 1)

        for col_num, value in enumerate(columns_order):
            worksheet.write(0, col_num + 1, value, header_format)
        for row_num, value in enumerate(df.axes[0].astype(str)):
            worksheet.write(row_num + 1, 0, value, header_format)

        # Fix first column
        column_len = df.axes[0].astype(str).str.len().max() + df.axes[0].astype(str).str.len().max() * 0.5
        worksheet.set_column(0, 0, column_len, cell_format)

        # Fix all  the rest of the columns
        for i, col in enumerate(columns_order):
            # find length of column i
            column_len = df[col].astype(str).str.len().max()
            # Setting the length if the column header is larger
            # than the max column value length
            column_len = max(column_len, len(col))
            column_len += column_len * 0.5
            # set the column length
            worksheet.set_column(i + 1, i + 1, column_len, cell_format)


class ChangesConfusionMatrix:
    """
    Compute the distribution of lesion change and CC patterns in dataset.
    For the identical connected components (i.e., having the same nodes) compare cc classification between GT and pred graph
    """
    def __init__(self, path_table_save, patient_list, is_lesion_pred=False):
        self._path_to_save = path_table_save
        self._patients = patient_list
        self._les_pred = is_lesion_pred

    def get_patient_gt_json(self, pat_name):
        raise ValueError("Abstract function!")

    def get_patient_pred_json(self, pat_name):
        raise ValueError("Abstract function!")

    def get_mapping(self, pat_name):
        raise ValueError("Abstract function!")

    def run(self):
        writer = pd.ExcelWriter(self._path_to_save, engine='xlsxwriter')
        gt2pred_change = list()
        matching_cc = {CcPatterns.SINGLE: 0,
                       CcPatterns.LINEAR: 0,
                       CcPatterns.MERGING: 0,
                       CcPatterns.SPLITTING: 0,
                       CcPatterns.COMPLEX: 0,
                       "Identical cc wrong classification": 0,
                       "TP pred cc not identical": 0,
                       "TP gt cc not identical": 0,
                       "FP cc": 0,
                       "FN cc": 0}
        matching_cc_table = {'Compare cc': matching_cc.copy(), 'Predicted cc distribution': matching_cc.copy(),
                             'Gt cc distribution': matching_cc.copy()}

        # pred_cc = {CcPatterns.SINGLE: 0,
        #                CcPatterns.LINEAR: 0,
        #                CcPatterns.MERGING: 0,
        #                CcPatterns.SPLITTING: 0,
        #                CcPatterns.COMPLEX: 0,
        #                "Unmatched": None}
        # gt_cc = pred_cc.copy()
        # gt_tpv_cc = pred_cc.copy()

        for patient in self._patients:
            if self._les_pred:
                with open(self.get_patient_gt_json(patient), 'r') as f:
                    graph_gt = json.load(f)
                with open(self.get_patient_pred_json(patient), 'r') as f:
                    graph_pred = json.load(f)
                if len(graph_pred['nodes']) == 0:
                    continue
                loader_gt = LoaderSimpleWithMapping(label_list=graph_gt["nodes"], edges_list=graph_gt["edges"],
                                                    mapping_path=self.get_mapping(patient), is_gt=True)
                loader_pred = LoaderSimpleWithMapping(label_list=graph_pred["nodes"], edges_list=graph_pred["edges"],
                                                      mapping_path=self.get_mapping(patient), is_gt=False)
            else:
                loader_gt = LoaderSimpleFromJson(self.get_patient_gt_json(patient))
                loader_pred = LoaderSimpleFromJson(self.get_patient_pred_json(patient))

            longit_gt = LongitClassification(loader_gt, patient_name=patient)
            longit_gt.classify_nodes()
            longit_gt.classify_cc()
            longit_pred = LongitClassification(loader_pred, patient_name=patient)
            longit_pred.classify_nodes()
            longit_pred.classify_cc()

            gt_graph = longit_gt.get_graph()
            pred_graph = longit_pred.get_graph()

            node_changes_gt = nx.get_node_attributes(gt_graph, NodeAttr.CHANGES)
            node_changes_pred = nx.get_node_attributes(pred_graph, NodeAttr.CHANGES)
            for node, gt_class in node_changes_gt.items():
                if node not in node_changes_pred:
                    "FN"
                    continue
                pred_class = node_changes_pred[node]
                gt2pred_change.append([gt_class, pred_class])

            #self.cc_matching_patient_old(gt_graph, pred_graph, matching_cc, pred_cc, gt_cc, gt_tpv_cc)
            self.cc_matching_patient(gt_graph, pred_graph, matching_cc_table)

        df, df_perc = self.confusion_matrix(gt2pred_change)
        ChangesReport.write_to_excel(df, writer, "raw data")
        ChangesReport.write_to_excel(df_perc, writer, "percentage")
        #df_cc = self.create_cc_matching_summary_old(matching_cc, pred_cc, gt_cc, gt_tpv_cc)
        df_cc = self.create_cc_matching_summary(matching_cc_table)
        ChangesReport.write_to_excel(df_cc, writer, "cc matching")
        writer.save()

    @staticmethod
    def confusion_matrix(pairs: List):
        all_categories = [NodesChanges.LONE, NodesChanges.NEW, NodesChanges.DISAPPEARED, NodesChanges.UNIQUE,
                          NodesChanges.MERGED, NodesChanges.MERGED_DISAP, NodesChanges.SPLITTING, NodesChanges.SPLITTING_NEW,
                          NodesChanges.COMPLEX]

        n_categories = len(all_categories)
        df = pd.DataFrame(data=np.zeros((n_categories+1, n_categories+1)).astype(int), index=all_categories + ['Tot Pred'], columns=all_categories + ['Tot GT'])
        for gt_c, pred_c in pairs:
            df.loc[gt_c, pred_c] += 1
        df.loc[:, 'Tot GT'] = df.sum(axis=1)
        df.loc['Tot Pred'] = df.sum(axis=0)
        df_perc = df.copy()
        for gt_cat in all_categories:
            for pr_cat in all_categories:
                num_les = df.loc[gt_cat, pr_cat]
                perc = num_les/df.loc[gt_cat, 'Tot GT']*100
                if np.isnan(perc):
                    perc = 0
                df_perc.loc[gt_cat, pr_cat] = f"{int(num_les)}\n{int(perc)}%"
        return df, df_perc

    @staticmethod
    def get_cc2pattern(graph: LongitClassification):
        """Get a graph, return a dictionary: {cc_index: cc_pattern}"""
        nodes2cc = list(nx.get_node_attributes(graph, NodeAttr.CC_INDEX).values())
        nodes2pattern = list(nx.get_node_attributes(graph, NodeAttr.CC_PATTERNS).values())
        cc2pattern = dict(zip(nodes2cc, nodes2pattern))
        return cc2pattern

    @staticmethod
    def get_cc2detection_old(gt_graph: LongitClassification, pred_graph: LongitClassification):
        """
        Gets a GT and pred graphs
        return a dictionary: {"Identical": [(pred_cc_index, gt_cc_index), .. ]; "Pred": [pred_cc_index]}
        """
        gt_node2cc = nx.get_node_attributes(gt_graph, NodeAttr.CC_INDEX)
        pred_node2cc = nx.get_node_attributes(pred_graph, NodeAttr.CC_INDEX)

        cc2detection = {"Identical": [], "Pred": [], "FP": [], "FN": []}
        cc2detection["FN"] = list({cc_id for gt_node, cc_id in gt_node2cc.items() if gt_node not in pred_node2cc.keys()})

        for pred_cc_id in set(pred_node2cc.values()):
            # take all the nodes with the current pred_cc_id
            pred_nodes_in_cc = [n for n, cc_index in pred_node2cc.items() if cc_index == pred_cc_id]
            # select one node
            node_sample = pred_nodes_in_cc[0]
            # check its gt_cc_id
            if any([node_sample not in gt_node2cc.keys() for node_sample in pred_nodes_in_cc]):
                "There is a node that corresponds to lesion detection FP"
                #print(f"{node_sample} not in gt")
                cc2detection["FP"].append(pred_cc_id)
                continue
            sample_gt_cc_id = gt_node2cc[node_sample]
            # take all the nodes in gt with gt_cc_id
            gt_nodes_in_cc = [n for n, cc_index in gt_node2cc.items() if cc_index == sample_gt_cc_id]
            # compare the two nodes sets
            if set(pred_nodes_in_cc) == set(gt_nodes_in_cc):
                #DEBUG:
                #nx.get_node_attributes(pred_graph)
                cc2detection["Identical"].append((pred_cc_id, sample_gt_cc_id))
            else:
                cc2detection["Pred"].append(pred_cc_id)

        return cc2detection

    def cc_matching_patient_old(self, gt_graph, pred_graph, cc_matching_dict, pred_cc_dict, gt_cc_dict, gt_tpv_cc_dict):
        """Update the cc_matching_dict, pred_cc_dict, gt_cc_dict dictionaries, with the count of ccs devided in categories"""
        cc_pred2gt = self.get_cc2detection(gt_graph, pred_graph)
        cc_pred2pattern = self.get_cc2pattern(pred_graph)
        cc_gt2pattern = self.get_cc2pattern(gt_graph)
        for pattern in cc_pred2pattern.values():
            pred_cc_dict[pattern] += 1
        for cc_id, pattern in cc_gt2pattern.items():
            gt_cc_dict[pattern] += 1
            if self._les_pred and (cc_id not in cc_pred2gt['FN']):
                gt_tpv_cc_dict[pattern] += 1

        #num_identical = len(cc_pred2gt["Identical"])
        num_unmatched = len(cc_pred2gt["Pred"])
        for pr_cc_id, gt_cc_id in cc_pred2gt["Identical"]:
            if cc_pred2pattern[pr_cc_id] != cc_gt2pattern[gt_cc_id]:
                cc_matching_dict["Unmatched"] += 1
            else:
                cc_matching_dict[cc_pred2pattern[pr_cc_id]] += 1
        cc_matching_dict["Unmatched"] += num_unmatched

    @staticmethod
    def revert_dict(d: dict) -> dict:
        d_reverted = {v: set() for v in set(d.values())}
        for k, v in d.items():
            d_reverted[v].add(k)
        return d_reverted

    def get_cc2detection(self, gt_graph: LongitClassification, pred_graph: LongitClassification):
        """
        Gets a GT and pred graphs
        return a dictionary: {"Identical": [(pred_cc_index, gt_cc_index), .. ]; two ccs having the same vert
                              "TP non identical pred": [pred_cc_index], pred cc whose vert are TP but are not in the same gt cc
                              "TP non identical gt": [gt_cc_index], gt cc whose vert are TP but are not in the same pred cc
                              "FP": [pred_cc_index], pred cc in which at lest one vert is FP
                              "FN": [gt_cc_index]} gt cc in which at lest one vert is FN
        """
        gt_node2cc = nx.get_node_attributes(gt_graph, NodeAttr.CC_INDEX)
        pred_node2cc = nx.get_node_attributes(pred_graph, NodeAttr.CC_INDEX)
        cc2gt_nodes = self.revert_dict(gt_node2cc)
        cc2pred_nodes = self.revert_dict(pred_node2cc)

        cc_pred_nodes_list = list(cc2pred_nodes.values())
        all_pred_nodes = list(pred_node2cc.keys())
        all_gt_nodes = list(gt_node2cc.keys())

        cc2detection = {"Identical": [],
                        "TP non identical pred": [],
                        "TP non identical gt": [],
                        "FP": [],
                        "FN": []}

        for cc_gt_id, cc_gt_nodes in cc2gt_nodes.items():
            identical_cc_pred_id = None
            for cc_pred_id, cc_pred_nodes in cc2pred_nodes.items():
                if cc_gt_nodes == cc_pred_nodes:
                    identical_cc_pred_id = cc_pred_id
                    break
            if identical_cc_pred_id is not None:
                cc2detection["Identical"].append((identical_cc_pred_id, cc_gt_id))
            elif all([gt_n in all_pred_nodes for gt_n in cc_gt_nodes]):
                cc2detection["TP non identical gt"].append(cc_gt_id)
            else:
                cc2detection["FN"].append(cc_gt_id)

        identical_cc_pred = [cc_pair[0] for cc_pair in cc2detection["Identical"]]
        for cc_pred_id, cc_pred_nodes in cc2pred_nodes.items():
            if cc_pred_id in identical_cc_pred:
                continue
            elif all([pred_n in all_gt_nodes for pred_n in cc_pred_nodes]):
                cc2detection["TP non identical pred"].append(cc_pred_id)
            else:
                cc2detection["FP"].append(cc_pred_id)

        return cc2detection

    def cc_matching_patient(self, gt_graph, pred_graph, matching_table):
        """Update the matching table, devided in categories"""
        cc_detection = self.get_cc2detection(gt_graph, pred_graph)
        cc_pred2pattern = self.get_cc2pattern(pred_graph)
        cc_gt2pattern = self.get_cc2pattern(gt_graph)

        for cc_pred, cc_gt in cc_detection["Identical"]:
            pred_pattern = cc_pred2pattern[cc_pred]
            gt_pattern = cc_gt2pattern[cc_gt]
            matching_table['Predicted cc distribution'][pred_pattern] += 1
            matching_table['Gt cc distribution'][gt_pattern] += 1
            if gt_pattern == pred_pattern:
                matching_table['Compare cc'][gt_pattern] += 1
            else:
                matching_table['Compare cc']["Identical cc wrong classification"] += 1

        for cc_pred in cc_detection["TP non identical pred"]:
            pred_pattern = cc_pred2pattern[cc_pred]
            matching_table['Compare cc']["TP pred cc not identical"] += 1
            matching_table['Predicted cc distribution'][pred_pattern] += 1

        for cc_gt in cc_detection["TP non identical gt"]:
            gt_pattern = cc_gt2pattern[cc_gt]
            matching_table['Compare cc']["TP gt cc not identical"] += 1
            matching_table['Gt cc distribution'][gt_pattern] += 1

        for cc_pred in cc_detection["FP"]:
            pred_pattern = cc_pred2pattern[cc_pred]
            matching_table['Compare cc']["FP cc"] += 1
            matching_table['Predicted cc distribution'][pred_pattern] += 1

        for cc_gt in cc_detection["FN"]:
            gt_pattern = cc_gt2pattern[cc_gt]
            matching_table['Compare cc']["FN cc"] += 1
            matching_table['Gt cc distribution'][gt_pattern] += 1

    def create_cc_matching_summary_old(self, matching_cc, pred_cc, gt_cc, gt_tpv_cc):
        col_names = list(matching_cc.keys())
        row_names = ["Identical cc distribution ", "Predicted cc distribution", "Gt cc distribution"]
        if self._les_pred:
            row_names.append("Gt TP cc distribution")

        df = pd.DataFrame(data=np.zeros((len(row_names), len(col_names))), index=row_names, columns=col_names)
        df.loc[row_names[0], :] = list(matching_cc.values())
        df.loc[row_names[1], :] = list(pred_cc.values())
        df.loc[row_names[2], :] = list(gt_cc.values())
        if self._les_pred:
            df.loc[row_names[3], :] = list(gt_tpv_cc.values())

        return df

    def create_cc_matching_summary(self, matching_table):
        row_names = list(matching_table.keys())
        col_names = list(matching_table[row_names[0]].keys())

        df = pd.DataFrame(data=np.zeros((len(row_names), len(col_names))), index=row_names, columns=col_names)
        for row in row_names:
            df.loc[row, :] = list(matching_table[row].values())

        return df



