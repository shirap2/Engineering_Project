import sys

sys.path.append("/cs/casmip/bennydv/lungs_pipeline/")
sys.path.append("/cs/casmip/bennydv/lungs_pipeline/common_packages/")
sys.path.append("/cs/casmip/bennydv/lungs_pipeline/lesions_matching/")

print(sys.path)

from common_packages.LesionChangesAnalysis import *
from common_packages.LesionsAnalysisPackage import *
from general_utils import *

path = Path()
name = Name()

# class TableCreatorLiver(TableCreator):
#     def __init__(self, table_path):
#         patient_list = get_patients_list()
#         super().__init__(table_path, patient_list)
#
#     def load_patient_series(self, patient_name):
#
#         patient_dir_reg = f'{path.GT_FILTERED_LABELED_GROUPWISE}/{patient_name}'
#         segmentation_reg_paths = glob.glob(f'{patient_dir_reg}/*lesion*')
#         segmentation_reg_paths_sorted = sorted(segmentation_reg_paths, key=sort_dates_paths)
#         reg_series = [load_nifti_data(file_name, type_integer=True)[0] for file_name in segmentation_reg_paths_sorted]
#
#         patient_dir_no_reg = f"{path.GT_FILTERED_LABELED_NO_REG}/{patient_name}"
#         segmentation_paths = glob.glob(f'{patient_dir_no_reg}/*lesion*')
#         segmentation_paths_sorted = sorted(segmentation_paths, key=sort_dates_paths)
#         no_reg_series = list()
#         voxel_dim_series = list()
#         for filename in segmentation_paths_sorted:
#             scan, nifti = load_nifti_data(filename, type_integer=True)
#             no_reg_series.append(scan)
#             voxel_dim_series.append(np.product(nifti.header.get_zooms()))
#
#         return reg_series, no_reg_series, voxel_dim_series


class LungsTableManager(TableManager):
    def __init__(self):
        table_path = "/cs/usr/bennydv/Desktop/bennydv/lungs_pipeline/lesions_matching/lesion_matching_database_correction"
        print(f"Creating tables in {table_path}")
        patient_list = get_patients_list()
        super().__init__(table_path, patient_list)
        self._matching_graph_paths = "/cs/casmip/bennydv/lungs_pipeline/lesions_matching/longitudinal_gt/original_corrected"
        self._matching_graph_name = "glong_gt.json"
        self._mapping_path = "/cs/casmip/bennydv/lungs_pipeline/pred_data/size_filtered/labeled_no_reg"
        self._mapping_name = "mapping_list.json"

    @staticmethod
    def local_load_series(patient_name):
        #patient_dir_reg = f'{path.PRED_FILTERED_LABELED_GROUPWISE}/{patient_name}'
        patient_dir_reg = f"/cs/casmip/bennydv/lungs_pipeline/gt_data/size_filtered/labeled_corrected/{patient_name}"
        segmentation_reg_paths = glob.glob(f'{patient_dir_reg}/*lesion*')
        segmentation_reg_paths_sorted = sorted(segmentation_reg_paths, key=sort_dates)
        reg_series = [load_nifti_data(file_name, type_integer=True)[0] for file_name in segmentation_reg_paths_sorted]

        #patient_dir_no_reg = f"{path.PRED_FILTERED_LABELED_NO_REG}/{patient_name}"
        patient_dir_no_reg = patient_dir_reg
        segmentation_paths = glob.glob(f'{patient_dir_no_reg}/*lesion*')
        segmentation_paths_sorted = sorted(segmentation_paths, key=sort_dates)
        no_reg_series = list()
        voxel_dim_series = list()
        for filename in segmentation_paths_sorted:
            scan, nifti = load_nifti_data(filename, type_integer=True)
            no_reg_series.append(scan)
            voxel_dim_series.append(np.product(nifti.header.get_zooms()))

        return reg_series, no_reg_series, voxel_dim_series

    @staticmethod
    def local_load_reg_series(patient_name):
        patient_dir_reg = f"/cs/casmip/bennydv/lungs_pipeline/gt_data/size_filtered/labeled_corrected/{patient_name}"
        #patient_dir_reg = f'{path.PRED_FILTERED_LABELED_GROUPWISE}/{patient_name}'
        segmentation_reg_paths = glob.glob(f'{patient_dir_reg}/*lesion*')
        segmentation_reg_paths_sorted = sorted(segmentation_reg_paths, key=sort_dates)
        reg_series = [load_nifti_data(file_name, type_integer=True)[0] for file_name in segmentation_reg_paths_sorted]

        return reg_series

    @staticmethod
    def local_load_orig_series(patient_name):
        patient_dir_no_reg = f"/cs/casmip/bennydv/lungs_pipeline/gt_data/size_filtered/labeled_corrected/{patient_name}"
        #patient_dir_no_reg = f"{path.PRED_FILTERED_LABELED_NO_REG}/{patient_name}"
        segmentation_paths = glob.glob(f'{patient_dir_no_reg}/*lesion*')
        segmentation_paths_sorted = sorted(segmentation_paths, key=sort_dates)
        no_reg_series = list()
        voxel_dim_series = list()
        for filename in segmentation_paths_sorted:
            scan, nifti = load_nifti_data(filename, type_integer=True)
            no_reg_series.append(scan)
            voxel_dim_series.append(nifti.header.get_zooms())

        return no_reg_series, voxel_dim_series


class LungsChangeAnalysis(ChangesReport):
    def __init__(self):
        super().__init__(path_table_save="/cs/usr/bennydv/Desktop/bennydv/lungs_pipeline/lesions_matching/lesion_matching_database/lesion_changes.xlsx",
                         patient_list=get_patients_list())

    def get_patient_json(self, pat_name):
        return f"{path.LESIONS_MATCH_GT_ORIGINAL}/{pat_name}glong_gt.json"

class LungsChangesConfusionMatrix(ChangesConfusionMatrix):
    def __init__(self):
        super().__init__(
            path_table_save="/cs/casmip/bennydv/lungs_pipeline/lesions_matching/results/pred_segmentation_gw13/conf_matrix_cc_.xlsx",
            patient_list=get_patients_list(),
            is_lesion_pred=True)

    def get_patient_gt_json(self, pat_name):
        return f"{path.LESIONS_MATCH_GT_ORIGINAL}/{pat_name}glong_gt.json"

    def get_patient_pred_json(self, pat_name):
        return f"/cs/casmip/bennydv/lungs_pipeline/lesions_matching/results/pred_segmentation_gw13/{pat_name}/gw13.json"

    def get_mapping(self, pat_name):
        return f"{path.PRED_FILTERED_LABELED_NO_REG}/{pat_name}/mapping_list.json"



if __name__ == "__main__":
    # tb = LungsTableManager()
    # tb.run()
    # tb.lesions_update()
    # tb.lesion_labels_update()
    #l = LungsChangeAnalysis()
    l = LungsChangesConfusionMatrix()
    l.run()