import numpy as np
from common_packages.MatchingEvaluationPackage import *
from general_utils import *

path = Path()
name = Name()

class LungsReportEvaluation(ReportEvaluation):
    def __init__(self, pred_lesions: bool, eval_type, path_type=None):
        super().__init__(pred_lesions, eval_type, path_type)

    def run(self):
        patients_list = get_patients_list()
        OPTION = "gw13"
        stat_excel_path = f"{path.LESIONS_MATCH_RESULTS_DIR}/pred_segmentation_gw13/se_{OPTION}.xlsx"
        writer = pd.ExcelWriter(stat_excel_path, engine='xlsxwriter')
        df = pd.DataFrame()
        for patient in patients_list:
            if patient == 'B_B_S_': continue
            gt_graph_path = f"{path.LESIONS_MATCH_GT_ORIGINAL}/{patient}glong_gt.json"
            pred_graph_path = f"{path.LESIONS_MATCH_RESULTS_DIR}/pred_segmentation_gw13/{patient}/{OPTION}.json"
            if self._pred_lesions:
                mapping_path = f"{path.PRED_FILTERED_LABELED_NO_REG}/{patient}/mapping_list.json"
            else:
                mapping_path = None
            pat_df = self.report_patient(patient,
                                         gt_graph_path,
                                         pred_graph_path,
                                         mapping_path)

            df = df.append(pat_df)
        for col in df.columns:
            df.loc["sum", col] = df[col].sum()

        self.write_to_excel(sheet_name="-", df=df, writer=writer)
        summary_df = self.summary(df)
        self.write_to_excel(sheet_name='summary', df=summary_df, writer=writer)
        writer.save()

if __name__ == '__main__':
    rep = LungsReportEvaluation(pred_lesions=True, eval_type=EvaluationType.SKIP_EDGE_HANDLES, path_type=PathType.BACKWARD)
    rep.run()