import json

from common_packages.LongGraphPackage import *
import pandas as pd
from collections.abc import Iterable


class ReportEvaluation:
    def __init__(self, pred_lesions: bool, evaluation_type: str = EvaluationType.SKIP_EDGE_HANDLES, path_type=None):
        """Class with methods for creating an excel table of the evaluation metrics. It creates both patient metrics and
        summary metrics
        :param pred_lesions: are the lesion to be evaluated computed or gt. (boolean)
        :param evaluation_type: is the type of evaluation you want to calculate. Must be one of the fields of EvaluationTypes"""

        self._pred_lesions = pred_lesions
        self._patients_df = None
        self._Evaluator = None
        self._summary_function = None
        self._path_type = None
        if evaluation_type == EvaluationType.SIMPLE:
            self._Evaluator = LoaderEvalChanges
        elif evaluation_type == EvaluationType.SKIP_EDGE_HANDLES:
            self._Evaluator = LoaderEval_SkipEdgeHandlerSoft
        elif evaluation_type == EvaluationType.ACCEPT_CC_EDGES:
            self._Evaluator = LoaderEval_acceptCCEdges
        elif evaluation_type == EvaluationType.NODES_PATHS:
            self._Evaluator = LoaderEval_NodesCCPaths_Factory.get(path_type)
            self._path_type = path_type
        else:
            raise ValueError("Unkonwn evaluator")
        self._eval_type = evaluation_type

    def write_to_excel(self, sheet_name, df, writer):
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

    def report_patient(self, patient_name: str, gt_graph_path: str, pred_graph_path: str, *other_path):
        print(f"Working on {patient_name}")
        """Get evaluation statistics of a single patient"""
        with open(gt_graph_path, 'r') as f:
            graph_gt = json.load(f)
        with open(pred_graph_path, 'r') as f:
            graph_pred = json.load(f)

        if self._pred_lesions:
            mapping_path = other_path[0]
            ld1 = LoaderSimpleWithMapping(label_list=graph_gt["nodes"], edges_list=graph_gt["edges"],
                                          mapping_path=mapping_path, is_gt=True)
            ld2 = LoaderSimpleWithMapping(label_list=graph_pred["nodes"], edges_list=graph_pred["edges"],
                                          mapping_path=mapping_path, is_gt=False)
        else:
            ld1 = LoaderSimple(labels_list=graph_gt["nodes"], edges_list=graph_gt["edges"])
            ld2 = LoaderSimple(labels_list=graph_pred["nodes"], edges_list=graph_pred["edges"])
        ld_ev = self._Evaluator(ld1, ld2, patient_name=patient_name)

        if self._summary_function is None:
            self._summary_function = ld_ev.get_summary
        return ld_ev.get_stat_as_dataframe(self._pred_lesions)

    def run(self):
        """Run the evaluation on all patients (this method must be overridden)
        SEE EXAMPLE IN lesions_matching folder
        """
        pass
    
    def summary(self, patients_df, name=None):
        """Get a dataframe whose indices are patient names and columns the statistics and their sum. 
        Make a new df with a summary"""
        assert callable(self._summary_function)
        self._patients_df = patients_df

        if self._eval_type == EvaluationType.NODES_PATHS:
            sumry = self._summary_function(tot_fun=self.tot, pred_lesions=self._pred_lesions, sum_prod_fun=self.sum_prod_fun)
        else:
            sumry = self._summary_function(tot_fun=self.tot, pred_lesions=self._pred_lesions)


        if name is not None:
            row_name = name
        else:
            row_name = ''
        df = pd.DataFrame(data={name: [val] for name, val in sumry.items()}, index=[row_name])
        return df

    def tot(self, stat_name):
        if 'sum' not in self._patients_df.index:
            raise ValueError("The 'sum' row was not calculated")
        return int(self._patients_df.loc['sum', stat_name])

    def sum_prod_fun(self, stat_name1, stat_name2):
        patients_name = list(self._patients_df.index)
        patients_name.remove('sum')
        sum_prod = 0
        for pat_name in patients_name:
            sum_prod += (self._patients_df.loc[pat_name, stat_name1] * self._patients_df.loc[pat_name, stat_name2])
        return sum_prod

if __name__ == "__main__":
    a = 1
