from xlsxwriter.utility import xl_col_to_name
from typing import List

import pandas as pd
import numpy as np
from scipy import ndimage
from skimage import measure
import nibabel as nib
import operator
import glob
from multiprocessing import Pool

from copy import deepcopy
import os
from time import time, gmtime
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from common_packages.BaseClasses import get_labeled_segmentation
import re


class Task:
    lungs = 'Lungs'
    liver = 'Liver'
    brain = 'Brain'

def get_date_longitudinal(img_name):
    """
    :param img_name: str, for example: lesion_gt_25_09_2014_03_08_2015.nii.gz ('registered lesion') or lesion_gt_25_09_2014.nii.gz
    :return: date1, date2 (if present), for example: 25_09_2014 [03_08_2015]
    """
    no_file_type = img_name.replace(".nii.gz", "")
    components = no_file_type.split('_')
    date1 = ""
    date2 = ""
    count = 0
    for c in components:
        if c.isdigit():
            if count < 3:
                date1+= f'{c}_'
            else:
                date2+= f'{c}_'
            count+=1
    if len(date1)>0:
        date1 = date1[:-1]
    if len(date2)>0:
        date2 = date2[:-1]
    return date1, date2


# class tumors_statistics():
#     def __init__(self, roi, gt, predictions, case_name, roi_is_gt=True, is_labeled=False, nnunet=False):
#         # Loading 3 niftis files
#         self.is_nnunet = nnunet
#         self.case_name = case_name
#         self.gt_nifti = nib.load(gt)
#         pred_nifti = nib.load(predictions)
#
#         # Getting voxel_volume
#         self.pix_dims = self.gt_nifti.header.get_zooms()
#         self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]
#         self.gt = self.gt_nifti.get_fdata()
#         self.gt = binary_fill_holes(self.gt, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(self.gt.dtype))
#         self.gt = remove_small_objects(self.gt.astype(bool), min_size=20).astype(self.gt.dtype)
#
#         # getting the 3 numpy arrays
#         if not nnunet:
#             roi_nifti = nib.load(roi)
#             self.roi = roi_nifti.get_fdata()
#             if roi_is_gt:
#                 self.roi = np.logical_or(self.roi, self.gt).astype(self.roi.dtype)
#
#             # self.roi = self.getLargestCC(self.roi)
#             self.roi = binary_fill_holes(self.roi, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).reshape([3, 3, 1]).astype(
#                 self.roi.dtype))
#
#             if roi_is_gt:
#                 self.gt = np.logical_and(self.roi, self.gt).astype(self.gt.dtype)
#
#         self.predictions = pred_nifti.get_fdata()
#         if is_labeled:
#             self.predictions = self.predictions > 0
#
#         # unique lesions for gt and predictions
#         self.unique_gt = self.CC(self.gt, min_size=20)
#         self.unique_predictions = self.CC(self.predictions, min_size=20)
#
#         self.num_of_lesions = self.unique_gt[1]
#         self.dice_score = self.dice(self.gt, self.predictions)
#
#     def calculate_statistics_by_diameter(self, diameter, oper=operator.gt, three_biggest=False,
#                                          calculate_ASSD=False, calculate_HD=False):
#
#         predict_lesions_touches = np.zeros(self.gt.shape)
#         gt_lesions_touches = np.zeros(self.gt.shape)
#
#         # calculate diameter for each lesion in GT
#         tumors_with_diameter_gt_unique, tumors_with_diameter_gt, tumors_with_diameter = \
#             self.mask_by_diameter(self.unique_gt, diameter, oper)
#
#         # unique_gt = nib.Nifti1Image(tumors_with_diameter_gt_unique[0], self.gt_nifti.affine)
#
#         # Find 3 biggest GT
#         if three_biggest:
#             tumors_with_diameter_gt_unique, tumors_with_diameter_gt, tumors_with_diameter = \
#                 self.find_3_biggest_tumors(tumors_with_diameter_gt_unique)
#             unique_gt = nib.Nifti1Image(tumors_with_diameter_gt_unique[0], self.gt_nifti.affine)
#             # nib.save(unique_gt, 'GT_unique.nii.gz')
#
#         # calculate diameter for each lesion in Predictions
#         tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
#             self.mask_by_diameter(self.unique_predictions, diameter, oper)
#
#         # Find 3 biggest GT
#         if three_biggest:
#             tumors_with_diameter_predictions_matrix_unique, tumors_with_diameter_predictions_matrix, tumors_with_diameter_predictions = \
#                 self.find_3_biggest_tumors(tumors_with_diameter_predictions_matrix_unique)
#             unique_gt = nib.Nifti1Image(tumors_with_diameter_predictions_matrix_unique[0], self.gt_nifti.affine)
#             # nib.save(unique_gt, 'Pred_unique.nii.gz')
#
#         # Find predicted tumor that touches 1 tumor of the predicition
#         # and calculating ASSDs ans Hausdorff metrices
#         """ Go over the gt tumors with the chosen diameter: if there is at least one predicted tumor with the correct diameter
#         that has a common voxel with the current gt tumor, defined the current gt tumor 'touched'. Go over the predicted tumors
#         and define 'touched' all those tumors that touched the current gt tumor.
#         """
#         ASSDs: List[float] = []
#         HDs: List[float] = []
#         for i in tumors_with_diameter:
#             current_1_tumor = (self.unique_gt[0] == i)
#             unique_predictions = list(np.unique((current_1_tumor * tumors_with_diameter_predictions_matrix_unique[0])))
#             unique_predictions.pop(0)
#             if len(unique_predictions) > 0:
#                 gt_lesions_touches[current_1_tumor] = 1
#             for j in unique_predictions:
#                 predict_lesions_touches[tumors_with_diameter_predictions_matrix_unique[0] == j] = 1
#                 assd, hd = 0,0#assd_and_hd(current_1_tumor, tumors_with_diameter_predictions_matrix_unique[0] == j,
#                            #            voxelspacing=self.pix_dims, connectivity=2)
#                 ASSDs.append(assd)
#                 HDs.append(hd)
#         mean_ASSDs = float(format(np.mean(ASSDs), '.3f')) if ASSDs else np.nan
#         mean_HDs = float(format(np.mean(HDs), '.3f')) if HDs else np.nan
#         max_HDs = float(format(np.max(HDs), '.3f')) if HDs else np.nan
#
#         # Segmentation statistics
#
#         seg_TP, seg_FP, seg_FN = \
#             self.Segmentation_statistics(tumors_with_diameter_gt, predict_lesions_touches, debug=False)
#
#         Total_tumor_GT = float(format((tumors_with_diameter_gt > 0).sum() * self.voxel_volume * 0.001, '.3f'))
#         Total_tumor_pred = float(
#             format((tumors_with_diameter_predictions_matrix > 0).sum() * self.voxel_volume * 0.001, '.3f'))
#         Total_tumor_GT_without_FN = float(format((gt_lesions_touches > 0).sum() * self.voxel_volume * 0.001, '.3f'))
#         Total_tumor_pred_without_FP = float(format((predict_lesions_touches > 0).sum() * self.voxel_volume * 0.001, '.3f'))
#         if not self.is_nnunet:
#             Liver_cc = self.roi.sum() * self.voxel_volume * 0.001
#         else:
#             Liver_cc = 1
#
#         if (Total_tumor_GT + Total_tumor_pred) == 0:
#             delta_percentage = 0
#         else:
#             delta_percentage = ((Total_tumor_GT - Total_tumor_pred) / (Total_tumor_GT + Total_tumor_pred)) * 100
#
#         if (Total_tumor_GT_without_FN + Total_tumor_pred_without_FP) == 0:
#             delta_percentage_TP_only = 0
#         else:
#             delta_percentage_TP_only = ((Total_tumor_GT_without_FN - Total_tumor_pred_without_FP) / (Total_tumor_GT_without_FN + Total_tumor_pred_without_FP)) * 100
#
#         # Detection statistics
#         try:
#             detection_TP, detection_FP, detection_FN, precision, recall = \
#                 self.Detection_statistics(predict_lesions_touches, tumors_with_diameter_gt_unique,
#                                           tumors_with_diameter_predictions_matrix_unique, three_biggest)
#         except:
#             print('im here')
#
#         if precision + recall > 0:
#             f1_score = float(format(2*precision*recall/(precision+recall), '.3f'))
#         else:
#             f1_score = 0
#
#         return {'Filename': self.case_name,
#                 'Num of lesion': len(tumors_with_diameter),
#                 'Dice': self.dice(gt_lesions_touches, predict_lesions_touches),
#                 # 'Dice with FP and FN': self.dice(tumors_with_diameter_gt, tumors_with_diameter_predictions_matrix),
#                 'Dice with FN': self.dice(tumors_with_diameter_gt, predict_lesions_touches),
#                 'Segmentation TP (cc)': float(format(seg_TP.sum() * self.voxel_volume * 0.001, '.3f')),
#                 'Segmentation FP (cc)': float(format(seg_FP.sum() * self.voxel_volume * 0.001, '.3f')),
#                 'Segmentation FN (cc)': float(format(seg_FN.sum() * self.voxel_volume * 0.001, '.3f')),
#                 'Total tumor volume_cal GT (cc)': Total_tumor_GT,
#                 'Total tumor volume_cal Predictions (cc)': Total_tumor_pred,
#                 'Delta between total tumor volumes (cc)': Total_tumor_GT - Total_tumor_pred,
#                 'Delta between total tumor volumes (%)': delta_percentage,
#                 'Delta between total tumor volumes (TP only) (cc)': Total_tumor_GT_without_FN - Total_tumor_pred_without_FP,
#                 'Delta between total tumor volumes (TP only) (%)': delta_percentage_TP_only,
#                 'Tumor Burden GT (%)': float(format(Total_tumor_GT / Liver_cc, '.3f')) * 100,
#                 'Tumor Burden Pred (%)': float(format(Total_tumor_pred / Liver_cc, '.3f')) * 100,
#                 'Tumor Burden Delta (%)': float(format((Total_tumor_GT - Total_tumor_pred) / Liver_cc, '.3f')) * 100,
#                 'Detection TP (per lesion)': detection_TP,
#                 'Detection FP (per lesion)': detection_FP,
#                 'Detection FN (per lesion)': detection_FN,
#                 'Precision': float(format(precision, '.3f')),
#                 'Recall': float(format(recall, '.3f')),
#                 'F1 Score': f1_score,
#                 'Mean ASSD (mm)': mean_ASSDs,
#                 'Mean Hausdorff (mm)': mean_HDs,
#                 'Max Hausdorff (mm)': max_HDs}
#         # 'diameter': diameter,
#         # 'oper': oper}
#
#     def Segmentation_statistics(self, tumors_with_diameter_gt, predict_lesions_touches, debug=False):
#         """
#         :param tumors_with_diameter_gt: binary mask of the scan, in which there are ones in the voxels belonging to the gt tumors
#         with the chosen diameter.
#         :param predict_lesions_touches: binary mask of the scan, in which there are ones in the voxels belonging to the predicted
#         tumors with the chosen diameter that have at least one voxel in common with tumors_with_diameter_gt
#         :returns: seg_TP: a binary mask of the voxels that are both in input masks.
#         seg_FP: mask of the voxels of the predicted tumors touching the gt tumors, that are not part of the gt tumors
#         seg_FN: mask of the voxels of the gt tumors touching the predicted tumors, that are not part of the predicted tumors
#         """
#         seg_TP = (tumors_with_diameter_gt * predict_lesions_touches)
#         seg_FP = (predict_lesions_touches - (tumors_with_diameter_gt * predict_lesions_touches))
#         seg_FN = (tumors_with_diameter_gt - (tumors_with_diameter_gt * predict_lesions_touches))
#         if debug:
#             unique_gt = nib.Nifti1Image(seg_FP, self.gt_nifti.affine)
#             nib.save(unique_gt, 'FP.nii.gz')
#             unique_gt = nib.Nifti1Image(seg_FN, self.gt_nifti.affine)
#             nib.save(unique_gt, 'FN.nii.gz')
#
#         return seg_TP, seg_FP, seg_FN
#
#     def mask_by_diameter(self, labeled_unique, diameter, oper):
#         """
#         :param labeled_unique: a tuple made of: a. scan with all tumors labeled
#         :param diameter: the diameter to which we will compare (through oper) the diameter of each tumor. We will take
#         the tumors for which the comparison is positive
#         :param oper: the operation to make the comparison
#         :return:
#         1. a tuple made by a. scan with tumors whose diameter is > than #diameter - the scan is labeled. b. the num of labels
#         2. a binary mask of the tumors whose diameter is > than #diameter
#         3. a list of integers: the numbers used as labels in #labeled_unique of those tumors with the required diameter
#         """
#         tumors_with_diameter_list = []
#         debug = []
#         tumors_with_diameter_mask = np.zeros(self.gt.shape)
#         for i in range(1, labeled_unique[1] + 1):
#             current_1_tumor = (labeled_unique[0] == i)
#             num_of_voxels = current_1_tumor.sum()
#             tumor_volume = num_of_voxels * self.voxel_volume
#             approx_diameter = self.approximate_diameter(tumor_volume)
#             if oper(approx_diameter, diameter):
#                 tumors_with_diameter_list.append(i)
#                 debug.append(num_of_voxels)
#                 tumors_with_diameter_mask[current_1_tumor] = 1
#         tumors_with_diameter_labeled = measure.label(tumors_with_diameter_mask, connectivity=1)
#         tumors_with_diameter_labeled = tuple((tumors_with_diameter_labeled, tumors_with_diameter_labeled.max()))
#         return tumors_with_diameter_labeled, tumors_with_diameter_mask, tumors_with_diameter_list
#
#     def find_3_biggest_tumors(self, tumors_with_diameter_labeling):
#         tumors_with_diameter_list = []
#         tumors_with_diameter_labeling_copy = deepcopy(tumors_with_diameter_labeling)
#         three_biggest = np.bincount(tumors_with_diameter_labeling_copy[0].flatten())
#         three_biggest[0] = 0
#         three_biggest = np.argsort(three_biggest)
#         three_biggest = three_biggest[1:]
#         if three_biggest.__len__() >= 3:
#             three_biggest = three_biggest[-3:]
#
#         tumors_with_diameter_mask = np.zeros(self.gt.shape)
#         for i in three_biggest:
#             tumors_with_diameter_list.append(i)
#             tumors_with_diameter_mask[tumors_with_diameter_labeling[0] == i] = 1
#         tumors_with_diameter_mask_labeled = measure.label(tumors_with_diameter_mask, connectivity=1)
#         tumors_with_diameter_mask_labeled = tuple(
#             (tumors_with_diameter_mask_labeled, tumors_with_diameter_mask_labeled.max()))
#         return tumors_with_diameter_mask_labeled, tumors_with_diameter_mask, tumors_with_diameter_list
#
#     @staticmethod
#     def Detection_statistics(predict_lesions_touches, tumors_with_diameter_gt_unique,
#                              tumors_with_diameter_predictions_matrix_unique, three_biggest, debug=False, ):
#         """
#         :param predict_lesions_touches: binary mask of the voxels of the predicted tumors that touch the gt tumors
#         :param tumors_with_diameter_gt_unique: tuple of a. mask of the labeled gt tumors, b. number of gt tumors
#         :param tumors_with_diameter_predictions_matrix_unique: tuple of a. mask of the labeled pred tumors, b. number of pred tumors
#         :return: detection_TP: how many gt tumors have at least one voxel segmented by the prediction (how many gt tumors are touched by a predicted tumor)
#                  detection_FP: how many predicted tumors minus how many predicted tumors touch the gt tumors (how many predicted tumors don't touch gt tumors)
#                  detection_FN: how many gt tumors minus detection_TP (how many gt tumors are not touched by a predicted tumor)
#                  precision: standard formula
#                  recall: standard formula
#         """
#         detection_TP = len(list(np.unique((predict_lesions_touches * tumors_with_diameter_gt_unique[0])))) - 1
#         if three_biggest:
#             detection_FP = int(tumors_with_diameter_predictions_matrix_unique[1] - detection_TP)
#         else:
#             detection_FP = int(tumors_with_diameter_predictions_matrix_unique[1] - \
#                            (len(list(np.unique(
#                                (predict_lesions_touches * tumors_with_diameter_predictions_matrix_unique[0])))) - 1))
#         detection_FN = int(tumors_with_diameter_gt_unique[1] - detection_TP)
#
#         try:
#             precision = detection_TP / (detection_TP + detection_FP)
#         except:
#             precision = 1
#         try:
#             recall = detection_TP / (detection_FN + detection_TP)
#         except:
#             recall = 1
#
#         return detection_TP, detection_FP, detection_FN, precision, recall
#
#     # @staticmethod
#     # def getLargestCC(segmentation):
#     #     labels = measure.label(segmentation, connectivity=1)
#     #     assert (labels.max() != 0)  # assume at least 1 CC
#     #     largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
#     #     return largestCC
#
#     @staticmethod
#     def CC(Map, min_size):
#         """
#         Remove Small connected component
#         :param Map:
#         :return:
#         """
#         label_img = measure.label(Map, connectivity=1)
#         cc_num = label_img.max()
#         cc_areas = ndimage.sum(Map, label_img, range(cc_num + 1))
#         area_mask = (cc_areas < min_size)
#         label_img[area_mask[label_img]] = 0
#         return_value = measure.label(label_img, connectivity=1)
#         return return_value, return_value.max()
#
#     @staticmethod
#     def dice(gt_seg, prediction_seg):
#         """
#         compute dice coefficient
#         :param gt_seg:
#         :param prediction_seg:
#         :return: dice coefficient between gt and predictions
#         """
#         seg1 = np.asarray(gt_seg).astype(bool)
#         seg2 = np.asarray(prediction_seg).astype(bool)
#
#         # Compute Dice coefficient
#         intersection = np.logical_and(seg1, seg2)
#         if seg1.sum() + seg2.sum() == 0:
#             return 1
#         return float(format(2. * intersection.sum() / (seg1.sum() + seg2.sum()), '.3f'))
#
#     @staticmethod
#     def approximate_diameter(tumor_volume):
#         r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
#         diameter = 2 * r
#         return diameter

class TumorStatistics():
    def __init__(self, gt, predictions, patient_name, date):
        # Loading 3 niftis files
        self.patient_name = patient_name
        self.date = date
        self.case_name = f'{self.patient_name}-{self.date}'

        gt_nifti = nib.as_closest_canonical(nib.load(gt))
        gt_ = gt_nifti.get_fdata()
        self.gt = gt_ > 0
        #self.gt = gt_ > 0
        #self.gt_labeled = get_labeled_segmentation(self.gt, size_filtering=20)
        self.gt_labeled = gt_

        pred_nifti = nib.as_closest_canonical(nib.load(predictions))
        predictions_ = pred_nifti.get_fdata()
        #self.predictions = predictions_ > 0
        #self.predictions_labeled = get_labeled_segmentation(self.predictions, size_filtering=20)
        self.predictions_labeled = predictions_

        # unique lesions for gt and predictions
        self.labels_ids_gt = self.get_label_list(self.gt_labeled)
        self.num_of_gt_lesions = len(self.labels_ids_gt)
        self.labels_ids_pred = self.get_label_list(self.predictions_labeled)
        self.num_of_pred_lesions = len(self.labels_ids_pred)

        self.pix_dims = gt_nifti.header.get_zooms()
        self.voxel_volume = self.pix_dims[0] * self.pix_dims[1] * self.pix_dims[2]

    def calculate_statistics_by_diameter(self, diameter, oper=operator.gt):

        # calculate diameter for each lesion in GT
        if diameter == 0:
            gt_labeled_with_diameter = self.gt_labeled
            predictions_labeled_with_diameter = self.predictions_labeled
        else:
            gt_labeled_with_diameter = self.mask_by_diameter(self.gt_labeled, self.labels_ids_gt, diameter, oper)
            predictions_labeled_with_diameter = self.mask_by_diameter(self.predictions_labeled, self.labels_ids_pred, diameter, oper)

        prediction_mask_tp = np.zeros(self.gt.shape, dtype=np.int8)
        gt_mask_tp = np.zeros(self.gt.shape, dtype=np.int8)

        lbs_with_diameters_gt = self.get_label_list(gt_labeled_with_diameter)

        for lb in lbs_with_diameters_gt:
            current_gt_tumor_maks = (self.gt_labeled == lb)
            touched_pred_lb = self.get_label_list(current_gt_tumor_maks * predictions_labeled_with_diameter)
            if len(touched_pred_lb) > 0:
                gt_mask_tp[current_gt_tumor_maks] = 1
            for pred_lb in touched_pred_lb:
                prediction_mask_tp[predictions_labeled_with_diameter == pred_lb] = 1

        # seg_TP, seg_FP, seg_FN = \
        #     self.segmentation_statistics(gt_mask_tp, prediction_mask_tp)

        detection_TP, detection_FP, detection_FN, precision, recall = \
            self.detection_statistics(gt_mask_tp, prediction_mask_tp, gt_labeled_with_diameter, predictions_labeled_with_diameter)

        if precision + recall > 0:
            f1_score = float(format(2 * precision * recall / (precision + recall), '.3f'))
        else:
            f1_score = 0

        return {'Filename': self.case_name,
                'Num of gt lesions': self.num_of_gt_lesions,
                'Num of pred lesions' : self.num_of_pred_lesions,
                'Dice': self.dice(gt_mask_tp, prediction_mask_tp),
                # 'Segmentation TP (cc)': float(format(seg_TP.sum() * self.voxel_volume * 0.001, '.3f')),
                # 'Segmentation FP (cc)': float(format(seg_FP.sum() * self.voxel_volume * 0.001, '.3f')),
                # 'Segmentation FN (cc)': float(format(seg_FN.sum() * self.voxel_volume * 0.001, '.3f')),
                'Detection TP': detection_TP,
                'Detection FP': detection_FP,
                'Detection FN': detection_FN,
                'Precision': float(format(precision, '.3f')),
                'Recall': float(format(recall, '.3f')),
                'F1 Score': f1_score}

    @staticmethod
    def segmentation_statistics(gt_mask_tp, prediction_mask_tp):
        """
        :param gt_mask_tp: binary mask of the scan, in which there are ones in the voxels belonging to the gt
        tumors with the chosen diameter that have at least one voxel in common with the predicted segmentation
        :param prediction_mask_tp: binary mask of the scan, in which there are ones in the voxels belonging to the predicted
        tumors with the chosen diameter that have at least one voxel in common with the gt segmentation
        :returns:
            seg_TP: a binary mask of the voxels that are in both input masks.
            seg_FP: mask of the voxels of the predicted tumors touching the gt tumors, that are not part of the gt tumors
            (voxels of detected tumors that exceed the gt borders)
            seg_FN: mask of the voxels of the gt tumors touching the predicted tumors, that are not part of the predicted tumors
            (voxels of detected tumors that should have been selected as part of the tumor)
        """
        seg_TP = (gt_mask_tp * prediction_mask_tp)
        seg_FP = (prediction_mask_tp - seg_TP)
        seg_FN = (gt_mask_tp - seg_TP)

        return seg_TP, seg_FP, seg_FN

    def detection_statistics(self, gt_mask_tp, prediction_mask_tp, gt_labeled_with_diameter,
                             predictions_labeled_with_diameter):
        """
        :param gt_mask_tp: binary mask of the voxels of the gt tumors that touch the gt pred tumors
        :param prediction_mask_tp: binary mask of the voxels of the predicted tumors that touch the gt tumors
        :param gt_labeled_with_diameter: labeled image of gt tumors (with the chosen diameter)
        :param predictions_labeled_with_diameter: labeled image of pred tumors (with the chosen diameter)
        :return: detection_TP: how many gt tumors have at least one voxel segmented by the prediction (how many gt tumors are touched by a predicted tumor)
                 detection_FP: how many predicted tumors minus how many predicted tumors touch the gt tumors (how many predicted tumors don't touch gt tumors)
                 detection_FN: how many gt tumors minus detection_TP (how many gt tumors are not touched by a predicted tumor)
                 precision: standard formula
                 recall: standard formula
        """
        num_gt_labels = len(self.get_label_list(gt_labeled_with_diameter))
        gt_labels_tp = self.get_label_list(gt_mask_tp * gt_labeled_with_diameter)

        num_pred_labels = len(self.get_label_list(predictions_labeled_with_diameter))
        pred_labels_tp = self.get_label_list(prediction_mask_tp * predictions_labeled_with_diameter)

        detection_TP = len(gt_labels_tp)
        detection_FP = int(num_pred_labels - len(pred_labels_tp))
        detection_FN = int(num_gt_labels - len(gt_labels_tp))

        try:
            precision = detection_TP / (detection_TP + detection_FP)
        except:
            precision = 1
        try:
            recall = detection_TP / (detection_TP + detection_FN)
        except:
            recall = 1

        return detection_TP, detection_FP, detection_FN, precision, recall

    def mask_by_diameter(self, labeled_image, label_list, diameter, oper):
        """
        :param labeled_image: a segmentation image, with the lesions labeled
        :param label_list: a list of the labels appearing in labeled_image
        :param diameter: the diameter used for the masking in mm
        :param oper: the operation that should be done to filter out improper lesions
        :return tumors_with_diameter_labeled: the labeled_image with lesion filtered out according to oper and diameter
        """
        tumors_with_diameter_list = []
        tumors_with_diameter_mask = np.zeros(self.gt.shape, dtype=np.uint8)
        tumors_with_diameter_labeled = np.zeros(self.gt.shape, dtype=np.uint8)

        for lb in label_list:
            current_tumor = (labeled_image == lb)
            num_of_voxels = np.sum(current_tumor)
            tumor_volume = num_of_voxels * self.voxel_volume
            approx_diameter = self.approximate_diameter(tumor_volume)

            if oper(approx_diameter, diameter):
                tumors_with_diameter_list.append(lb)
                tumors_with_diameter_mask[current_tumor] = 1

        tumors_with_diameter_labeled[tumors_with_diameter_mask.astype(bool)] = labeled_image[tumors_with_diameter_mask.astype(bool)]

        return tumors_with_diameter_labeled

    @staticmethod
    def get_label_list(labeled_segmentation: np.array):
        """Get an np.array: a labeled segmentation. Return a list of the labels (without background 0 label), present
        in the matrix"""
        labels = np.unique(labeled_segmentation)
        les_labels = labels[labels > 0]
        return list(les_labels)


    @staticmethod
    def dice(gt_seg, prediction_seg):
        """
        compute dice coefficient
        :param gt_seg:
        :param prediction_seg:
        :return: dice coefficient between gt and predictions
        """
        seg1 = np.asarray(gt_seg).astype(bool)
        seg2 = np.asarray(prediction_seg).astype(bool)

        # Compute Dice coefficient
        intersection = np.logical_and(seg1, seg2)
        if seg1.sum() + seg2.sum() == 0:
            return 1
        return float(format(2. * intersection.sum() / (seg1.sum() + seg2.sum()), '.3f'))

    @staticmethod
    def approximate_diameter(tumor_volume):
        r = ((3 * tumor_volume) / (4 * np.pi)) ** (1 / 3)
        diameter = 2 * r
        return diameter

def write_to_excel(sheet_name, df, writer, columns_order):
    df = df.set_index(columns_order[0])
    columns_order.pop(0)
    df = df.append(df.agg(['mean', 'std', 'min', 'max', 'sum']))
    workbook = writer.book
    # cell_format = workbook.add_format()
    cell_format = workbook.add_format({'num_format': '#,##0.00'})
    cell_format.set_font_size(16)

    df.to_excel(writer, sheet_name=sheet_name, columns=columns_order, startrow=1, startcol=1, header=False, index=False)
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'font_size': 16,
        'valign': 'top',
        'border': 1})

    max_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#E6FFCC'})
    min_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#FFB3B3'})
    last_format = workbook.add_format({
        'font_size': 16,
        'bg_color': '#C0C0C0',
        'border': 1,
        'num_format': '#,##0.00'})

    worksheet = writer.sheets[sheet_name]
    worksheet.freeze_panes(1, 1)
    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4), {'type': 'formula',
                                                                         'criteria': '=B2=B$' + str(len(df.axes[0])),
                                                                         'format': max_format})

    worksheet.conditional_format(f'$B$2:${xl_col_to_name(len(columns_order))}$' + str(len(df.axes[0]) - 4), {'type': 'formula',
                                                                         'criteria': '=B2=B$' + str(
                                                                             len(df.axes[0]) - 1),
                                                                         'format': min_format})

    n = df.shape[0] - 5
    for col in np.arange(len(columns_order)) + 1:
        for i, measure in enumerate(['AVERAGE', 'STDEV', 'MIN', 'MAX', 'SUM'], start=1):
            col_name = xl_col_to_name(col)
            worksheet.write(f'{col_name}{n + i + 1}', f'{{={measure}({col_name}2:{col_name}{n + 1})}}')

    for i in range(len(df.axes[0]) - 4, len(df.axes[0]) + 1):
        worksheet.set_row(i, None, last_format)

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


def calculate_runtime(t):
    t2 = gmtime(time() - t)
    return f'{t2.tm_hour:02.0f}:{t2.tm_min:02.0f}:{t2.tm_sec:02.0f}'


def replace_in_file_name(file_name, old_part, new_part):
    if old_part not in file_name:
        raise Exception(f'The following file doesn\'t contain the part "{old_part}": {file_name}')
    new_file_name = file_name.replace(old_part, new_part)
    if not os.path.isfile(new_file_name):
        raise Exception(f'It looks like the following file doesn\'t exist: {new_file_name}')
    return new_file_name


def calculate_stats_nnunet_(gt_segm_path: str, nnunet_results_dir: str, assd_and_hd=False):

    pat_name = gt_segm_path.split('/')[-2]
    date, _ = get_date_longitudinal(gt_segm_path.split('/')[-1])

    pred_path = f"{nnunet_results_dir}/{pat_name}{date}.nii.gz"

    one_case = 1 #tumors_statistics(roi=None, gt=gt_segm_path, predictions=pred_path, case_name=f"{pat_name}{date}",
                                 #nnunet=True)

    calculate_ASSD = assd_and_hd
    calculate_HD = assd_and_hd

    res = (
        one_case.calculate_statistics_by_diameter(0, three_biggest=False, calculate_ASSD=calculate_ASSD,
                                                  calculate_HD=calculate_HD),
        one_case.calculate_statistics_by_diameter(5, three_biggest=False, calculate_ASSD=calculate_ASSD,
                                                  calculate_HD=calculate_HD),
        one_case.calculate_statistics_by_diameter(10, three_biggest=False, calculate_ASSD=calculate_ASSD,
                                                  calculate_HD=calculate_HD)
    )
    return res

def calculate_stats_nnunet(arg):
    gt_segm_path = arg[0]
    nnunet_results_dir = arg[1]
    ind = arg[2]
    num_cases = arg[3]

    pat_name = gt_segm_path.split('/')[-2]
    date, _ = get_date_longitudinal(gt_segm_path.split('/')[-1])

    pred_path = f"{nnunet_results_dir}/{pat_name}/{name.segm_name_pred(date)}"

    one_case = TumorStatistics(gt=gt_segm_path, predictions=pred_path, patient_name=pat_name, date=date)

    res = (
        one_case.calculate_statistics_by_diameter(0),
        one_case.calculate_statistics_by_diameter(5),
        one_case.calculate_statistics_by_diameter(10)
    )
    print(f"{ind}/{num_cases}")
    return res

def scans_sort_key(file, full_path_given=True):
    if full_path_given:
        file = os.path.basename(os.path.dirname(file))
    file = file.replace('-', "_")
    file = file.replace("_nifti", "")
    split = file.split('_')
    return '_'.join(c for c in split if not c.isdigit()), int(split[-1]), int(split[-2]), int(split[-3])


def pairs_sort_key(file):
    file = os.path.basename(os.path.dirname(file))
    file = file.replace('BL_', '')
    file = file.replace("_nifti", "")
    bl_name, fu_name = file.split('_FU_')
    return (*scans_sort_key(bl_name, full_path_given=False), *scans_sort_key(fu_name, full_path_given=False))


def measure_test_nn_unet(task: str):

    if task == Task.liver:
        task_folder = 'Task500_Liver'
        main_folder = 'liver_pipeline'
    elif task == Task.lungs:
        task_folder = 'Task501_Lungs'
        main_folder = 'lungs_pipeline'
    elif task == Task.brain:
        task_folder = 'Task502_Brain'
        main_folder = 'brain_pipeline'
    else:
        raise ValueError("Bad task definition")

    #nnunet_results = f"/cs/casmip/bennydv/nn_unet/nnUNet_raw_data_base/nnUNet_raw_data/{task_folder}/labelsTs"
    nnunet_results = f"/cs/casmip/bennydv/{main_folder}/pred_data/size_filtered/labeled_no_reg"
    dataset_dir_name = f"/cs/casmip/bennydv/{main_folder}/gt_data/size_filtered/labeled_no_reg"
    print(f'-------------- start calculating measures for dataset: {task_folder} --------------')

    t = time()

    results_diameter_0 = pd.DataFrame()
    results_diameter_5 = pd.DataFrame()
    results_diameter_10 = pd.DataFrame()

    # sort according to patient names

    gt_tumors_paths = sorted(glob.glob(f'{dataset_dir_name}/*/lesions_gt*'))
    #gt_tumors_paths = glob.glob(f"{dataset_dir_name}/E_N_/lesions_gt*20_01*")
    #gt_tumors_paths = ["/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_S_S_/lesions_gt_19_10_2017.nii.gz"]
    map_args = [[gt_path, nnunet_results, ind, len(gt_tumors_paths)] for ind, gt_path in enumerate(gt_tumors_paths)]

    with Pool(10) as p:
        res = p.map(calculate_stats_nnunet, map_args)
    # res = calculate_stats_nnunet(map_args[0])

    res_sorted = sorted(res, key= lambda r:r[0]['Filename'])
    for r in res_sorted:
        results_diameter_0 = results_diameter_0.append(r[0], ignore_index=True)
        results_diameter_5 = results_diameter_5.append(r[1], ignore_index=True)
        results_diameter_10 = results_diameter_10.append(r[2], ignore_index=True)

    # i = 0
    # for gt_path in gt_tumors_paths:
    #     i += 1
    #     print(f"{i}/{len(gt_tumors_paths)}")
    #     res = calculate_stats_nnunet(gt_segm_path=gt_path,
    #                                   nnunet_results_dir=nnunet_results)
    #
    #     results_diameter_0 = results_diameter_0.append(res[0], ignore_index=True)
    #     results_diameter_5 = results_diameter_5.append(res[1], ignore_index=True)
    #     results_diameter_10 = results_diameter_10.append(res[2], ignore_index=True)


    writer = pd.ExcelWriter(
        f"/cs/casmip/bennydv/nn_unet/nnUNet_raw_data_base/nnUNet_raw_data/{task_folder}/nnunet_test_measures_4.xlsx",
        engine='xlsxwriter')

    cols = list(res_sorted[0][0].keys())
    print(cols)

    write_to_excel('diameter_0', results_diameter_0, writer, cols)
    write_to_excel('diameter_5', results_diameter_5, writer, cols)
    write_to_excel('diameter_10', results_diameter_10, writer, cols)
    writer.save()

    dice0 = float(format(np.mean(results_diameter_0['Dice']), '.2f'))
    dice5 = float(format(np.mean(results_diameter_5['Dice']), '.2f'))
    dice10 = float(format(np.mean(results_diameter_10['Dice']), '.2f'))
    prec0 = float(format(np.mean(results_diameter_0['Precision']), '.2f'))
    prec5 = float(format(np.mean(results_diameter_5['Precision']), '.2f'))
    prec10 = float(format(np.mean(results_diameter_10['Precision']), '.2f'))
    recall0 = float(format(np.mean(results_diameter_0['Recall']), '.2f'))
    recall5 = float(format(np.mean(results_diameter_5['Recall']), '.2f'))
    recall10 = float(format(np.mean(results_diameter_10['Recall']), '.2f'))

    print(f"        prec        recall      dice\n"
          f"diam10  {prec10}    {recall10}  {dice10}\n"
          f"diam5   {prec5}     {recall5}   {dice5}\n"
          f"diam0   {prec0}     {recall0}   {dice0}")

    print(f'Finished in {calculate_runtime(t)} hh:mm:ss')


def get_patient_from_file(filename: str):
    """
    A1_02-02-23 -> A1_
    A_Ab_17_01_22 -> A_Ab_
    """
    components = re.split('_|-', filename)
    res = ""
    for comp in components:
        if comp.isdigit():
            break
        res += comp + "_"
    return res

def get_date_from_file(filename: str):
    """
    :param : filename <patient_name>_<date>. E.g = A_A_01_12_17.nii.gz
    :return: date: E.g: 01_12_17
    """
    patient_name = get_patient_from_file(filename)
    scan_date = filename.replace(patient_name, '').replace('.nii.gz', '')
    scan_date = scan_date.replace('-', '_')
    return scan_date

def statistics_train_nnunet(task: str):

    scan_res = []
    les_per_scan = []

    train_labels = sorted(glob.glob(f"/cs/casmip/bennydv/nn_unet/nnUNet_raw_data_base/nnUNet_raw_data/*{task}/labelsTr/*nii.gz"))
    pat2dates = dict()
    for i, lb_path in enumerate(train_labels):
        print(f"{i+1}/{len(train_labels)}")
        f = os.path.basename(lb_path)
        pat_name = get_patient_from_file(f)
        date = get_date_from_file(f)
        if pat_name in pat2dates:
            pat2dates[pat_name].append(date)
        else:
            pat2dates.update({pat_name: [date]})

        scan_path = lb_path.replace('labelsTr', 'imagesTr').replace('.nii.gz', '_0000.nii.gz')

        _, nifti = load_nifti_data(scan_path)
        les, _ = load_nifti_data(lb_path)
        scan_res.append(nifti.header.get_zooms())
        n_les = len(np.unique(get_labeled_segmentation(les)))
        if n_les > 0:
            n_les = n_les - 1
        les_per_scan.append(n_les)

    dates_per_pat = [len(dates) for dates in pat2dates.values()]
    print("scans_per_pat")
    print(f"{np.mean(dates_per_pat)} // {np.std(dates_per_pat)}")
    print("tot # of scans")
    print(f"{np.sum(dates_per_pat)}")
    print("tot # of pat")
    print(f"{len(dates_per_pat)}")

    print("les_per_scan")
    print(f"{np.mean(les_per_scan)} // {np.std(les_per_scan)}")

    print("tot # of lesions")
    print(f"{np.sum(les_per_scan)}")

    print("scan res")
    scan_res_array = np.array(scan_res)
    print(f"minimum {np.min(scan_res_array, axis=0)}")
    print(f"max {np.max(scan_res_array, axis=0)}")


if __name__ == '__main__':
    from config import Name
    from general_utils import load_nifti_data, get_labeled_segmentation
    name = Name()
    #measure_test_nn_unet(Task.brain)
    statistics_train_nnunet(Task.lungs)
