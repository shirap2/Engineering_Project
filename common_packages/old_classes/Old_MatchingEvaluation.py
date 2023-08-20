from common_packages.MatchingEvaluationPackage import *
from common_packages.old_classes.Old_LongGraphPackage import *
from common_packages.MatchingAlgosExperiments import LoaderPairwiseFromDf

class ReportEvaluationPaths(ReportEvaluation):
    def __init__(self, pred_lesions: bool, path_type=PathType.BACKWARD):
        super().__init__(pred_lesions, evaluation_type=EvaluationType.NODES_PATHS, path_type=path_type)

    def report_patient(self, patient_name: str, gt_graph_path: str, pred_graph_path: str, *other_path):
        """Get evaluation statistics of a single patient"""
        with open(gt_graph_path, 'r') as f:
            graph_gt = json.load(f)
        with open(pred_graph_path, 'r') as f:
            graph_pred = json.load(f)

        ld1 = LoaderSimple(labels_list=graph_gt["nodes"], edges_list=graph_gt["edges"])
        ld2 = LoaderSimple(labels_list=graph_pred["nodes"], edges_list=graph_pred["edges"])
        ld_ev = LoaderEval_NodesCCPaths_backward(ld1, ld2, patient_name=patient_name)
        nodes2path_stats = ld_ev.get_node_path_stats_dict()
        df = pd.DataFrame(columns=['precision', 'recall'], index=list(nodes2path_stats.keys()),
                          data=list(nodes2path_stats.values()))
        return df

    def path_summary(self, list_pat_df):
        all_paths_precision = []
        all_paths_recall = []
        for df in list_pat_df:
            all_paths_precision += list(df['precision'])
            all_paths_recall += list(df['recall'])
        mean_prec = np.mean(all_paths_precision)
        std_prec = np.std(all_paths_precision)
        mean_recall = np.mean(all_paths_recall)
        std_recall = np.std(all_paths_recall)
        s_df = pd.DataFrame(data=[[mean_prec, std_prec, mean_recall, std_recall]], columns=["mean_prec", "std_prec", "mean_recall", "std_recall"], index=[''])
        return s_df


class ReportEvaluationTwoMethods(ReportEvaluation):
    def __init__(self):
        super().__init__(pred_lesions=False, evaluation_type=EvaluationType.SIMPLE, path_type=None)

    def report_patient(self, patient_name: str, gt_graph_path: str, pred_graph_path: str, *other_path):
        l_gt = LoaderSimpleFromJson(gt_graph_path)
        l_pw = LoaderSimpleFromJson(pred_graph_path)
        l_gw = LoaderSimpleFromJson(list(other_path)[0])

        l_two_graphs = LoaderTwoMethods(l_pw, l_gw)

        ld_ev = LoaderEval_TwoMethods(l_gt, l_two_graphs, patient_name=patient_name)

        if self._summary_function is None:
            self._summary_function = ld_ev.get_summary
        return ld_ev.get_stat_as_dataframe(self._pred_lesions)


class ReportEvaluationPairwise(ReportEvaluation):
    def __init__(self):
        super().__init__(pred_lesions=False, evaluation_type=EvaluationType.SIMPLE, path_type=None)

    def report_patient(self, patient_name: str, gt_graph_path: str, pred_graph_path: str, *other_path):
        l_gt = LoaderSimpleFromJson(gt_graph_path)
        l_gt = LoaderSimplePairwise(l_gt.get_nodes(), l_gt.get_edges())
        if pred_graph_path.endswith('json'):
            l_pr = LoaderSimpleFromJson(pred_graph_path)
            l_pr = LoaderSimplePairwise(l_pr.get_nodes(), l_pr.get_edges())
        else:
            l_pr = LoaderPairwiseFromDf(pred_graph_path, greedy=True)

        ld_ev = LoaderEvalChanges(gt_loader=l_gt, pred_loader=l_pr, patient_name=patient_name)

        if self._summary_function is None:
            self._summary_function = ld_ev.get_summary
        return ld_ev.get_stat_as_dataframe(self._pred_lesions)


#from lesions_matching.graph_compare_trials import json_creator_same_cc_edges, json_creator, json_creator_same_cc, many_lones, simple_ex


class ReportEvaluationTest(ReportEvaluation):
    def __init__(self, pred_lesions: bool, eval_type, path_type=None):
        super().__init__(pred_lesions, eval_type, path_type)

    def run(self):
        stat_excel_path = f"/cs/usr/bennydv/Desktop/matching_eval_test_paths.xlsx"
        writer = pd.ExcelWriter(stat_excel_path, engine='xlsxwriter')
        df = pd.DataFrame()
        gt_graph, pred_graph = simple_ex()
        gt_graph_path = "/cs/usr/bennydv/Desktop/gt_graph.json"
        pred_graph_path = "/cs/usr/bennydv/Desktop/pred_graph.json"

        with open(gt_graph_path, 'w') as f:
            json.dump(gt_graph, f)
        with open(pred_graph_path, 'w') as f:
            json.dump(pred_graph, f)

        RunDrawingFromJsons(gt_path=gt_graph_path, pred_path=pred_graph_path, eval_type=self._eval_type,
                            path_type=self._path_type)

        pat_df = self.report_patient(patient_name='patient',
                                     gt_graph_path=gt_graph_path,
                                     pred_graph_path=pred_graph_path)

        df = df.append(pat_df)

        for col in df.columns:
            df.loc["sum", col] = df[col].sum()

        self.write_to_excel(sheet_name="-", df=df, writer=writer)
        summary_df = self.summary(df)
        self.write_to_excel(sheet_name='summary', df=summary_df, writer=writer)
        writer.save()

if __name__ == "__main__":
    rep = ReportEvaluationTest(pred_lesions=True, eval_type=EvaluationType.NODES_PATHS, path_type=PathType.BACKWARD)
    rep.run()
