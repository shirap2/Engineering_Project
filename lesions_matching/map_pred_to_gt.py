from common_packages.ComputedGraphsMapping import *
from general_utils import *

class LungsMapComputedGraphs(MapComputedGraphs):
    def __init__(self):
        super().__init__(patient_list=get_patients_list())

    def get_patient_gt_and_computed_segmentations(self, patient_name):
        """Return triplets of date, loaded gt and loaded computed segmentations. The segmentations are sorted by date"""
        for date in get_patient_dates(patient_name):
            gt, _ = load_nifti_data(f"{path.GT_FILTERED_LABELED_NO_REG}/{patient_name}/{name.segm_name_gt(date)}", type_integer=True)
            pr, _ = load_nifti_data(f"{path.PRED_FILTERED_LABELED_NO_REG}/{patient_name}/{name.segm_name_pred(date)}", type_integer=True)
            yield date, gt, pr

    def get_mapping_path(self, patient_name):
        """Return a path where to save the mapping"""
        return f"{path.PRED_FILTERED_LABELED_NO_REG}/{patient_name}/mapping_list.json"

if __name__ == "__main__":
    lv = LungsMapComputedGraphs()
    lv.run()