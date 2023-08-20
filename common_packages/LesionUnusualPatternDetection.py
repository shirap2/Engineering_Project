import glob

import networkx as nx
import pandas as pd

from common_packages.LesionsAnalysisPackage import *

class TableQueries:
    def __init__(self, lesion_table_path="/cs/casmip/bennydv/liver_pipeline/lesions_matching/lesion_matching_database_nnunet/lesions_data.xlsx",
                 diam=5, patterns=False):
        self.diam = diam
        self.lesion_table_path = lesion_table_path
        self.df_dict = pd.read_excel(lesion_table_path, sheet_name=None, index_col=0)
        self.analysis_df = pd.DataFrame()
        self.patterns = patterns
        if self.patterns:
            for pattern in ['single_p', 'linear_p', 'merge_p', 'split_p', 'complex_p']:
                self.analysis_df.loc['FP', pattern] = self.num_of_fp_lesion_with_pattern(pattern)
                self.analysis_df.loc['Tot', pattern] = self.num_of_lesion_with_pattern(pattern)
                if self.analysis_df.loc['Tot', pattern] != 0:
                    self.analysis_df.loc['%', pattern] = round(self.analysis_df.loc['FP', pattern] / self.analysis_df.loc['Tot', pattern], 2) * 100
                self.analysis_df.loc[f"FP>={self.diam}mm", pattern] = self.num_of_fp_lesion_with_pattern(pattern, diam=self.diam)
                self.analysis_df.loc[f"Tot>={self.diam}mm", pattern] = self.num_of_lesion_with_pattern(pattern, diam=self.diam)
                if self.analysis_df.loc[f"Tot>={self.diam}mm", pattern] != 0:
                    self.analysis_df.loc[f"%>={self.diam}mm", pattern] = round(self.analysis_df.loc[f"FP>={self.diam}mm", pattern] / self.analysis_df.loc[f"Tot>={self.diam}mm", pattern], 2) * 100
            self.analysis_df.loc['Tot', 'registratmion deletion'] = sum(
                [sum(df[LesionAttr.LESION_CHANGE].isnull()) for df in self.df_dict.values()])
            self.analysis_df['TOT'] = self.analysis_df.sum(axis=1)
        else:
            for change_label in ['lone', 'new', 'disap', 'unique', 'merge', 'merge_disap', 'split', 'split_new','complex']:
                self.analysis_df.loc['FP', change_label] = self.num_of_fp_lesion_with_change(change_label)
                self.analysis_df.loc['Tot', change_label] = self.num_of_lesion_with_change(change_label)
                if self.analysis_df.loc['Tot', change_label] != 0:
                    self.analysis_df.loc['%', change_label] = round(self.analysis_df.loc['FP', change_label]/self.analysis_df.loc['Tot', change_label],2)*100
                self.analysis_df.loc[f"FP>={self.diam}mm", change_label] = self.num_of_fp_lesion_with_change(change_label, diam=self.diam)
                self.analysis_df.loc[f"Tot>={self.diam}mm", change_label] = self.num_of_lesion_with_change(change_label, diam=self.diam)
                if self.analysis_df.loc[f"Tot>={self.diam}mm", change_label] != 0:
                    self.analysis_df.loc[f"%>={self.diam}mm", change_label] = round(self.analysis_df.loc[f"FP>={self.diam}mm", change_label]/self.analysis_df.loc[f"Tot>={self.diam}mm", change_label],2)*100
            self.analysis_df.loc['Tot', 'registratmion deletion'] = sum([sum(df[LesionAttr.LESION_CHANGE].isnull()) for df in self.df_dict.values()])
            self.analysis_df['TOT'] = self.analysis_df.sum(axis=1)

    def num_of_lesion_with_change(self, les_change: str, diam=0):
        return sum([sum((df[LesionAttr.LESION_CHANGE] == les_change) & (df[LesionAttr.DIAMETER] >= diam)) for df in self.df_dict.values()])

    def num_of_fp_lesion_with_change(self, les_change: str, diam=0):
        return sum([sum((df[LesionAttr.DETECTION] == 'fp') & (df[LesionAttr.LESION_CHANGE] == les_change) & (df[LesionAttr.DIAMETER] >= diam)) for df in self.df_dict.values()])

    def num_of_lesion_with_pattern(self, pattern: str, diam=0):
        return sum([sum((df[LesionAttr.PATTERN_CLASSIFICATION] == pattern) & (df[LesionAttr.DIAMETER] >= diam)) for df in self.df_dict.values()])

    def num_of_fp_lesion_with_pattern(self, pattern: str, diam=0):
        return sum([sum((df[LesionAttr.DETECTION] == 'fp') & (df[LesionAttr.PATTERN_CLASSIFICATION] == pattern) & (df[LesionAttr.DIAMETER] >= diam)) for df in self.df_dict.values()])


    def save_table(self):
        if self.patterns:
            file_name = "lesions_data_pattern_stats"
        else:
            file_name = "lesions_data_changes_stats"
        file_path = self.lesion_table_path.replace("lesions_data", file_name)
        print(f"Saving as table at: {file_path}")
        self.analysis_df.to_excel(file_path)


class DetectionQueries:
    def __init__(self, pipeline,
                 diam=5, patterns=False):
        lesion_table_path = f"/cs/casmip/bennydv/{pipeline}/lesions_matching/lesion_matching_database_nnunet/lesions_data.xlsx"
        self.lesion_table_path = lesion_table_path
        self.df_dict = pd.read_excel(lesion_table_path, sheet_name=None, index_col=0)
        self.patients = list(self.df_dict.keys())
        empty_cols = [0 for i in range(len(self.patients))]
        self.analysis_dict = {'#Lone les before' : empty_cols.copy(),
                              '#Lone les after' : empty_cols.copy(),
                              'Lone les: #FP': empty_cols.copy(),
                              'Lone les: #FN': empty_cols.copy(),
                              '#NC edges before': empty_cols.copy(),
                              '#NC edges after': empty_cols.copy(),
                              'NC edges: #FP': empty_cols.copy(),
                              'NC edges: #FN': empty_cols.copy()}

        for pat_id, pat in enumerate(self.patients):
            gt_matching_graph_path = f"/cs/casmip/bennydv/{pipeline}/lesions_matching/longitudinal_gt/original/{pat}glong_gt.json"
            longit_gt = LongitClassification(LoaderSimpleFromJson(gt_matching_graph_path))
            longit_gt.classify_nodes()

            longit_pred = LongitClassification(LoaderSimpleFromJson(f"/cs/casmip/bennydv/{pipeline}/lesions_matching/results/pred_segmentation_gw13/{pat}/gw13.json"))

            pat_df = self.df_dict[pat]
            pred_lone_les_df = pat_df.loc[pat_df[LesionAttr.LESION_CHANGE] == 'lone', :]
            pred_tp_les_df = pat_df.loc[pat_df[LesionAttr.DETECTION] == 'tp', :]
            gt_tp_labels = [lb.replace('\'', '') for lb in pred_tp_les_df.loc[:, 'gt_mapping'].tolist()]
            nc_edges = [(n1, n2) for n1, n2, data in longit_pred.get_graph().edges(data=True) if data[EdgeAttr.IS_SKIP]]
            longit_gt_directed_graph = longit_gt.make_graph_directed(longit_gt.get_graph())

            """Count Lone lesions"""
            num_lone_before_rev = len(pred_lone_les_df)
            num_lone_after_rev = len(pred_lone_les_df)
            self.analysis_dict['#Lone les before'][pat_id] += num_lone_before_rev
            if len(pred_lone_les_df) > 0:
                pred_fp_lone_les_df = pred_lone_les_df.loc[pred_lone_les_df[LesionAttr.DETECTION] == 'fp', :]
                pred_tp_lone_les_df = pred_lone_les_df.loc[pred_lone_les_df[LesionAttr.DETECTION] == 'tp', :]
                num_lone_after_rev -= len(pred_fp_lone_les_df)
                fn_neighbouring_lesions = set()
                for tp_lone_les in pred_tp_lone_les_df.index:
                    # find the mapped gt label
                    mapped_gt_labels = self.get_mapped_gt_lesions(pred_tp_lone_les_df, tp_lone_les)
                    current_les_fn_neighbouring_les = set()
                    for gt_label in mapped_gt_labels:
                        # find the unmapped neighbours of the gt label -- these are the fn discoverable through the lone predicted lesion
                        current_les_fn_neighbouring_les = current_les_fn_neighbouring_les | {n for n in longit_gt.get_graph().neighbors(gt_label) if n not in gt_tp_labels}
                    # if at least one fn neighbouring lesion was found - the label of the original lesion will not be 'lone' anymore
                    if len(current_les_fn_neighbouring_les) > 0:
                        num_lone_after_rev -= 1
                    fn_neighbouring_lesions = fn_neighbouring_lesions | current_les_fn_neighbouring_les

                self.analysis_dict['Lone les: #FP'][pat_id] += len(pred_fp_lone_les_df)
                self.analysis_dict['Lone les: #FN'][pat_id] += len(fn_neighbouring_lesions)
                self.analysis_dict['#Lone les after'][pat_id] += num_lone_after_rev

            """Count NC edges"""
            num_nc_before_rev = len(nc_edges)
            num_nc_after_rev = len(nc_edges)
            self.analysis_dict['#NC edges before'][pat_id] += num_nc_before_rev
            if len(nc_edges) > 0:
                fn_skipped_lesions = set()
                fp_vertices_lesions = set()
                for nc_e in nc_edges:
                    n1, n2 = nc_e
                    # extremes are TP nodes
                    if n1 in pred_tp_les_df.index and n2 in pred_tp_les_df.index:
                        mapped_gt_n1 = self.get_mapped_gt_lesions(pred_tp_les_df, n1)
                        mapped_gt_n2 = self.get_mapped_gt_lesions(pred_tp_les_df, n2)
                        nodes_in_paths = {n for gt_n1, gt_n2 in zip(mapped_gt_n1, mapped_gt_n2)
                                          for p in nx.all_simple_paths(longit_gt_directed_graph, gt_n1, gt_n2)
                                          for n in p}
                        skipped_nodes = nodes_in_paths - set(mapped_gt_n1) - set(mapped_gt_n2)
                        current_edges_fn_skipped_nodes = {n for n in skipped_nodes if n not in gt_tp_labels}
                        # if at least one fn skipped lesion was found - the skip edge will be a shorter edge:
                        # here we will assume that any shorter edge is a consecutive edge:
                        if len(current_edges_fn_skipped_nodes) > 0:
                            num_nc_after_rev -= 1
                        fn_skipped_lesions = fn_skipped_lesions | current_edges_fn_skipped_nodes

                    else:
                        fp_vertices_lesions = fp_vertices_lesions | {n for n in [n1, n2] if n not in pred_tp_les_df.index}
                        num_nc_after_rev -= 1

                self.analysis_dict['NC edges: #FP'][pat_id] += len(fp_vertices_lesions)
                self.analysis_dict['NC edges: #FN'][pat_id] += len(fn_skipped_lesions)
                self.analysis_dict['#NC edges after'][pat_id] += num_nc_after_rev

    def save_table(self):
        file_path = self.lesion_table_path.replace("lesions_data", 'suspicious_patterns_detection_new')
        print(f"Saving as table at: {file_path}")
        analysis_df = pd.DataFrame(data=self.analysis_dict, index=self.patients)
        analysis_df.to_excel(file_path)


    @staticmethod
    def get_mapped_gt_lesions(df, index) -> list:
        mapped_nodes_str = df.loc[index, 'gt_mapping']
        return mapped_nodes_str.replace('\'','').split(', ')


def number_of_skip_edges_vertices(pipeline):
    jsons = glob.glob(f"/cs/casmip/bennydv/{pipeline}/lesions_matching/results/pred_segmentation_gw13/*/gw13.json")
    num_se_vert = 0
    num_e_vert = 0
    for j in jsons:
        pat_name = j.split('/')[-2]
        if pat_name == 'B_B_S_' and pipeline == 'lungs_pipeline':
            continue
        longit_pred = LongitClassification(LoaderSimpleFromJson(j))
        longit_pred.classify_nodes()
        longit_pred.classify_cc()
        s_edges = [(n1, n2) for n1, n2, data in longit_pred.get_graph().edges(data=True) if data[EdgeAttr.IS_SKIP]]
        se_vert = set()
        for se in s_edges:
            se_vert = se_vert | {se[0]} | {se[1]}
        non_single_vert = {n for n, data in longit_pred.get_graph().nodes(data=True) if data[NodeAttr.CC_PATTERNS] != CcPatterns.SINGLE}
        e_vert = non_single_vert - se_vert
        pat_se_vert = len(se_vert)
        pat_e_vert = len(e_vert)
        print(f"{pat_name}: #se vert={pat_se_vert}. #e vert={pat_e_vert}")
        num_se_vert += pat_se_vert
        num_e_vert += pat_e_vert
    print(f"\n{pipeline}: #se vert={num_se_vert}. #e vert={num_e_vert}")


def calculate_p_values_table():

    def create_samples(n_fp, n_class):
        assert n_fp <= n_class
        sample = np.zeros(n_class)
        sample[0:n_fp] = 1
        return sample

    data = {
                # Class = [#FP, #All]
        # "Brain": {'Class1': [8, 14],
        #           'Class2': [33, 242]},
        # "Lungs": {'Class1': [23, 55],
        #           'Class2': [74, 732]},
        # "Liver": {'Class1': [10, 51],
        #           'Class2': [80, 579]},
        "Brain": {'Class1': [8+49, 14+68],
                  'Class2': [118,377]},
        "Lungs": {'Class1': [23+26, 55+84],
                  'Class2': [101,826]},
        "Liver": {'Class1': [10+48, 51+74],
                  'Class2': [136,717]},
        # "Brain": {'Class1': [49,68],
        #           'Class2': [118,377]},
        # "Lungs": {'Class1': [26,84],
        #           'Class2': [101,826]},
        # "Liver": {'Class1': [48,74],
        #           'Class2': [136,717]},
    }
    for organ, classes in data.items():
        class1 = create_samples(*classes['Class1'])
        class2 = create_samples(*classes['Class2'])
        print(f"{organ}")
        print(stats.ttest_ind(a=class1, b=class2, equal_var=False))

if __name__ == "__main__":
    import scipy.stats as stats
    a = DetectionQueries(pipeline='brain_pipeline')
    a.save_table()
    #number_of_skip_edges_vertices(pipeline='brain_pipeline')
    #calculate_p_values_table()
