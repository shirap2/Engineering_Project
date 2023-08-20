from os.path import join
from copy import deepcopy
"""
This script contains the paths and names of all the directories in the project
"""

class Path:
    def __init__(self, longitudinal=True):
        self.LAB = "/cs/casmip"
        self.USER = join(self.LAB, "bennydv")
        self.PROJ_NAME = join(self.USER, "lungs_pipeline")

        self.DATABASE = join(self.PROJ_NAME, "cases_summary.csv")
        # ____________________________________
        # 1
        self.DATASET_CREATION = join(self.PROJ_NAME, "dataset_creation")

        # 1.1
        self.JSONS_DIR = join(self.DATASET_CREATION, "dataset_json")

        # 1.1...
        if not longitudinal:
            pass
            # self.PAIRS_JSON = join(self.JSONS_DIR, "pairs_toy.json")
            # self.FOR_LUNG_SEGM_JSON = join(self.JSONS_DIR, "lung_segmentation_data_toy.json")
        else:
            self.PAIRS_JSON = join(self.JSONS_DIR, "long_pairs.json")
            self.PAIRS_TO_LAST_JSON = join(self.JSONS_DIR, "long_pairs_to_last.json")
            self.SINGLES_JSON = join(self.JSONS_DIR, "long_singles.json")
            self.FOR_LUNG_SEGM_JSON = join(self.JSONS_DIR, "long_lung_segmentation_scans.json")

        self.FOR_LUNG_REG_JSON = join(self.JSONS_DIR, "pairs_for_registration.json")
        self.FOR_LESION_SEGM_JSON = join(self.JSONS_DIR, "pair_registered.json")
        self.PATIENTS_REPORT_JSON = join(self.JSONS_DIR, "patients_report.json")


        # 1.2
        self.SET_SPLIT_DIR = join(self.DATASET_CREATION, "train_test_val_split")

        # 1.2...
        if not longitudinal:
            self.LONGITUDINAL_CASES = join(self.SET_SPLIT_DIR, "toy_test.csv")
        else:
            self.LONGITUDINAL_CASES = join(self.SET_SPLIT_DIR, "long_cases.csv")

        # ____________________________________
        # 2
        self.LUNG_SEGMENTATION = join(self.PROJ_NAME, "lung_segmentation")

        # 2.1
        self.LUNG_NET_DIR = join(self.LUNG_SEGMENTATION, "lung_UNet_inference")

        # 2.1.1
        self.LUNG_NET_MODELS_DIR = join(self.LUNG_NET_DIR, "models")

        # 2.1.1...
        self.LUNG_NET_MODEL_WEIGHTS = join(self.LUNG_NET_MODELS_DIR, ".mdl_wts_32_patch_p3-333.h5")

        # 2.2
        self.LUNG_SEGM_RESULTS_DIR = join(self.LUNG_SEGMENTATION, "results")


        # _____________________________________
        #
        self.REGISTRATION = join(self.PROJ_NAME, "registration")
        #
        self.REG_PARAMS_DIR = join(self.REGISTRATION, "param_files")

        self.REG_PREPROC_RESULTS_DIR = join(self.REGISTRATION, "preproc_results")

        self.REG_RESULTS_DIR = join(self.REGISTRATION, "registration_results")

        self.AFFINE_PARAMS = join(self.REG_PARAMS_DIR, "par0011_affine.txt")
        self.ROI_BSPLINE_PARAMS = join(self.REG_PARAMS_DIR, "par0011_bspline1.txt")
        self.LUNGS_BSPLINE_PARAMS = join(self.REG_PARAMS_DIR, "par0011_bspline2.txt")

        # ------------------------------------
        self.GT_DATA = join(self.PROJ_NAME, "gt_data")
        self.PRED_DATA = join(self.PROJ_NAME, "pred_data")

        self.GT_ORIGINAL = join(self.GT_DATA, "original")
        self.GT_SIZE_FILTERED = join(self.GT_DATA, "size_filtered")
        self.GT_MAPPED = join(self.GT_DATA, "mapped")

        self.GT_ORIG_LABELED_GROUPWISE = join(self.GT_ORIGINAL, "labeled_groupwise")
        self.GT_ORIG_LABELED_PAIRWISE = join(self.GT_ORIGINAL, "labeled_pairwise")
        self.GT_ORIG_LABELED_NO_REG = join(self.GT_ORIGINAL, "labeled_no_reg")

        self.GT_FILTERED_LABELED_GROUPWISE = join(self.GT_SIZE_FILTERED, "labeled_groupwise")
        self.GT_FILTERED_LABELED_PAIRWISE = join(self.GT_SIZE_FILTERED, "labeled_pairwise")
        self.GT_FILTERED_LABELED_NO_REG = join(self.GT_SIZE_FILTERED, "labeled_no_reg")
        self.GT_FILTERED_LABELED_CORRECTED = join(self.GT_SIZE_FILTERED, "labeled_corrected")

        self.GT_MAPPED_LABELED_GROUPWISE = join(self.GT_MAPPED, "labeled_groupwise")
        self.GT_MAPPED_LABELED_PAIRWISE = join(self.GT_MAPPED, "labeled_pairwise")
        self.GT_MAPPED_LABELED_NO_REG = join(self.GT_MAPPED, "labeled_no_reg")

        self.PR_ORIGINAL = join(self.PRED_DATA, "original")
        self.PR_MAPPED = join(self.PRED_DATA, "mapped")

        self.PRED_SIZE_FILTERED = join(self.PRED_DATA, "size_filtered")
        self.PRED_FILTERED_LABELED_GROUPWISE = join(self.PRED_SIZE_FILTERED, "labeled_groupwise")
        self.PRED_FILTERED_LABELED_PAIRWISE = join(self.PRED_SIZE_FILTERED, "labeled_pairwise")
        self.PRED_FILTERED_LABELED_NO_REG = join(self.PRED_SIZE_FILTERED, "labeled_no_reg")

        self.PR_ORIG_LABELED_PAIRWISE = join(self.PR_ORIGINAL, "labeled_pairwise")
        self.PR_ORIG_LABELED_NO_REG = join(self.PR_ORIGINAL, "labeled_no_reg")
        self.PR_ORIG_LABELED_GROUPWISE = join(self.PR_ORIGINAL, "labeled_groupwise")

        self.PR_MAPPED_LABELED_PAIRWISE = join(self.PR_MAPPED, "labeled_pairwise")
        self.PR_MAPPED_LABELED_NO_REG = join(self.PR_MAPPED, "labeled_no_reg")
        self.PR_MAPPED_LABELED_GROUPWISE = join(self.PR_MAPPED, "labeled_groupwise")
        # _____________________________________
        # 4
        self.LESIONS_SEGMENTATION = join(self.PROJ_NAME, "lesions_segmentation")

        # 4.1
        self.LESIONS_NET_MODELS_DIR = join(self.LESIONS_SEGMENTATION, "models")

        # 4.1.1
        self.LESIONS_NET_MODEL_WEIGHTS = None

        # 4.2
        self.LESIONS_SEGM_RESULTS_DIR = join(self.LESIONS_SEGMENTATION, "results")

        # 4.2.1
        self.LESIONS_SEGM_RESULTS_MEASURES_DIR = join(self.LESIONS_SEGM_RESULTS_DIR, "measure_results")

        # 4.3
        self.LESIONS_TRAIN_TEST_SETS_DIR = join(self.LESIONS_SEGMENTATION, "train_test_sets")
        self.LESIONS_SETS = join(self.LESIONS_TRAIN_TEST_SETS_DIR, "sets")
        self.LESION_TRAIN_SET_CSV = join(self.LESIONS_SETS, "train.csv")
        self.LESION_VAL_SET_CSV = join(self.LESIONS_SETS, "validation.csv")
        self.LESION_TEST_SET_CSV = join(self.LESIONS_SETS, "test.csv")

        # _____________________________________
        # 5
        self.LESIONS_MATCHING = join(self.PROJ_NAME, "lesions_matching")

        # 5.1
        self.LESIONS_MATCH_RESULTS_DIR = join(self.LESIONS_MATCHING, "results")
        self.LESIONS_MATCH_LONGITUDINAL_RESULTS_DIR = join(self.LESIONS_MATCHING, "longitudinal_results")
        self.LESIONS_MATCH_GT_DIR = join(self.LESIONS_MATCHING, "longitudinal_gt")
        self.LESIONS_MATCH_GT_ORIGINAL = join(self.LESIONS_MATCH_GT_DIR, "original")
        self.LESIONS_MATCH_GT_CORRECTED = join(self.LESIONS_MATCH_GT_DIR, "original_corrected")
        self.LESIONS_MATCH_GT_MAPPED = join(self.LESIONS_MATCH_GT_DIR, "mapped")

        # 6
        self.LONGITUDINAL = join(self.PROJ_NAME, "longitudinal")
        self.BAD_LABELS_JSON = join(self.LONGITUDINAL, "bad_labels.json")
        # _______________________________________

    def longitudinal_report_path(self, scenario_name, is_volumetric):
        if is_volumetric:
            #file_name = f"vol_longitudinal_{scenario_name}.xlsx"
            file_name = f"les_longitudinal_{scenario_name}_vol.xlsx"
        else:
            file_name = f"les_longitudinal_{scenario_name}_labels.xlsx"
        return join(self.LONGITUDINAL, file_name)

    def choose_model(self, model_name, using_first_pair_m2=False, new_model=False):
        if model_name == "M0":
            self.LESIONS_NET_MODEL_WEIGHTS = join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-1677.h5')
        elif model_name == "M1":
            if new_model is True:
                self.LESIONS_NET_MODEL_WEIGHTS = join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-971.h5')
            else:
                self.LESIONS_NET_MODEL_WEIGHTS = join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-1490.h5')
        elif model_name == "M2": #simultaneous with prior (3-channel)
            if not using_first_pair_m2: #cascade: need also a 2-channel model for the first pair.
                if new_model is True:
                    self.LESIONS_NET_MODEL_WEIGHTS = [join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-641.h5'),
                                                      join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-495.h5')]
                else:
                    self.LESIONS_NET_MODEL_WEIGHTS = [join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-1490.h5'),
                                                      join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-585.h5')]
            else:
                if new_model is True:
                    self.LESIONS_NET_MODEL_WEIGHTS = join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-495.h5')
                else:
                    self.LESIONS_NET_MODEL_WEIGHTS = join(self.LESIONS_NET_MODELS_DIR, '.mdl_wts_32_patch-585.h5')

        else:
            print("ERROR: Wrong lesion net model name! Valid models name are: 'M0', 'M1', 'M2'!")


class Name:
    class Lsegm:
        RAW_LUNG_SEGM = "initial_lungs_seg.nii.gz"
        LUNG_SEGM_PRED = "lung_seg_pred.nii.gz"
        LUNG_SEGM_PRED_POST_PROC = "lung_seg_pp.nii.gz"
        LUNG_SEGM_PRED_ABM = "after_border_marching.nii.gz"
        ABM_ADDICTION = "ABM_diff_segmentation.nii.gz"
        ABM_ADDICTION_CROPPED = "ABM_diff_segmentation.nii.gz"
    class Reg:
        BL_LUNG_SEGM_CROPPED = "BL_lung.nii.gz"
        BL_SCAN_CROPPED = "BL_scan.nii.gz"
        BL_TUMORS_CROPPED = "BL_tumors.nii.gz"
        BL_LABELED_TUMORS_CROPPED = "BL_tumors_orig_labeled.nii.gz"
        BL_EXCLUDED_TUMORS_CROPPED = "BL_excluded_tumors.nii.gz"
        FU_LUNG_SEGM_CROPPED = "FU_lung.nii.gz"
        FU_SCAN_CROPPED = "FU_scan.nii.gz"
        FU_TUMORS_GT_CROPPED = "FU_tumors.nii.gz"
        FU_LABELED_TUMORS_GT_CROPPED = "FU_tumors_orig_labeled.nii.gz"
        FU_EXCLUDED_TUMORS_GT_CROPPED = "FU_excluded_tumors.nii.gz"

        BL_LUNG_SEGM_REGISTERED = "BL_Scan_Lung.nii.gz"
        BL_SCAN_REGISTERED = "BL_Scan_CT.nii.gz"
        BL_TUMORS_REGISTERED = "BL_Scan_Tumors.nii.gz"
        BL_LABELED_TUMORS_REGISTERED = "BL_Scan_Tumors_labeled.nii.gz"
        BL_EXCLUDED_TUMORS_REGISTERED = "BL_Excluded_Tumors.nii.gz"

        PREV_BL_LABELED_TUMORS_RESIZED = "prev_BL_tumors_lab_resized.nii.gz"
        PREV_BL_LABELED_TUMORS_REGISTERED = "prev_BL_tumors_registered.nii.gz"
        PREV_BL_LABELED_TUMORS_RESIZED_GT = "prev_BL_tumors_lab_resized_gt.nii.gz"
        PREV_BL_LABELED_TUMORS_REGISTERED_GT = "prev_BL_tumors_registered_gt.nii.gz"

    class Tsegm:
        BL_SCAN_CLIPPED = "bl_scan_clipped_normalized.nii.gz"
        BL_TUMORS_CROPPED = "bl_tumors_cropped.nii.gz"
        FU_SCAN_CLIPPED = "fu_scan_clipped_normalized.nii.gz"
        FU_LUNGS_SEGM = "liver.nii.gz"
        FU_TUMORS_GT = "tumors.nii.gz"
        FU_TUMORS_CROPPED_GT = "tumors_cropped.nii.gz"
        FU_TUMORS_LABELED_GT = "tumors_gt_lesions_labels.nii.gz"
        FU_TUMORS_PRED = "tumors_pred_label_cropped.nii.gz"
        FU_TUMORS_PRED_2 = "tumors_pred_th_label.nii.gz"
        FU_TUMORS_PRED_OUTSIDE_ROI = "tumors_pred_th_no_roi.nii.gz"
        FU_TUMORS_PRED_POST_PROC = "tumors_pred_pp.nii.gz"
        FU_TUMORS_PRED_POST_PROC_LABELED = "tumors_pred_pp_lesions_labels.nii.gz"
        TUMORS_TH_MEASURE = "tumors_measurements"
    class Match:
        BL_MATCH = "bl_matched.nii.gz"
        FU_MATCH = "fu_matched.nii.gz"
        CASE_MATCHING_JSON = "pair_matching.json"
        MATCHING_GRAPH_JPG = "pair_matching_graph.png"

        LONG_GT_GRAPH_JSON = "long_gt.json" # gt graph

        LONG_GRAPH_GT_JPG = "long_graph_gt.png" # graph on gt
        LONG_GRAPH_GT_JSON = "long_graph_gt.json"
        LONG_GRAPH_GT_MAPPED_JSON = "long_graph_mapped_gt.json"

        LONG_GRAPH_PRED_JPG = "long_graph_pred.png" # graph on pred
        LONG_GRAPH_PRED_JSON = "long_graph_pred.json"
        LONG_GRAPH_PRED_MAPPED_JSON = "long_graph_mapped_pred.json"

        LONG_GRAPH_GW_GT_JSON = "long_gw_gt_graph.json"
        LONG_GRAPH_GW_GT_JPG = "long_gw_gt_graph.json"

        LONG_GRAPH_GW_PRED_JSON = "gw_distance_13_match_tolast.json"
        LONG_GRAPH_GW_PRED_MAPPED_JSON = "gw_distance_13_match_tolast_mapped.json"

        LONG_DETECTION_GRAPH_JPG = "detection_graph.png"

    class Measures:
        MEASURE_FILE_NAME_TEMPLATE = "tumors_measurements_th"
        MEASURE_FILE_NAME_TEMPLATE_POST_PROC = "tumors_measurements_"
        MEASURE_FILE_TYPE = ".xlsx"

    @staticmethod
    def general_name(file_name, date1, date2=None, end_file='nii.gz'):
        if date2 is None:
            return f'{file_name}_{date1}.{end_file}'
        else:
            return f'{file_name}_{date1}_{date2}.{end_file}'

    def scan_name(self, date1, date2=None):
        return self.general_name('scan', date1, date2, 'nii.gz')

    def segm_name_gt(self, date1, date2=None):
        return self.general_name('lesions_gt', date1, date2, 'nii.gz')

    def segm_name_gt_corr(self, date1, date2=None):
        return self.general_name('lesions_gt_corr', date1, date2, 'nii.gz')

    def segm_name_pred(self, date1, date2=None):
        return self.general_name('lesions_pred', date1, date2, 'nii.gz')

    def lungs_name(self, date1, date2=None):
        return self.general_name('lungs', date1, date2, 'nii.gz')

    def match_name_gt(self, date1, date2):
        return self.general_name('pair_matching', date1, date2, 'json')

    def match_img_name_gt(self, date1, date2):
        return self.general_name('pair_matching', date1, date2, 'jpeg')

    def match_name_pred(self, date1, date2):
        return self.general_name('pair_matching_pred', date1, date2, 'json')

    def match_img_name_pred(self, date1, date2):
        return self.general_name('pair_matching_pred', date1, date2, 'jpeg')

    lsegm = Lsegm()
    reg = Reg()
    tsegm = Tsegm()
    match = Match()
    measures = Measures()



class Params:
    threshold = None
    duplicate_standalone = False
    simultaneous_with_prior = None
    model_name = None
    name = None

    def __init__(self, model_name, dup=False, use_prev_segm_gt=False):
        self.model_name = model_name
        self.prior_gt = None
        if model_name == "M0":
            self.threshold = 13
            self.simultaneous_with_prior = False
            self.name = "single_standalone"
        elif model_name == "M1":
            self.threshold = 11
            self.simultaneous_with_prior = False
            self.name = "simultaneous"
            if dup:
                self.duplicate_standalone = True
                self.name = "duplicate_standalone"
        elif model_name == "M2":
            self.threshold = 12
            self.simultaneous_with_prior = True
            self.name = "simultaneous_with_prior"
            self.prior_gt = use_prev_segm_gt
            if self.prior_gt:
                self.name = "simultaneous_with_prior_gt"

        else:
            print("Wrong model name, using default")
            self.threshold = 8

    def print_model_config(self):
        print(f"Model name = {self.model_name}")
        if self.simultaneous_with_prior:
            print("Simultaneous with prior")
            if self.prior_gt:
                print("Prior segmentation is gt")
        if self.duplicate_standalone:
            print("Duplicate standalone")