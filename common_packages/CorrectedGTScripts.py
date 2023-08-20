import os
import shutil

import nibabel
import pandas as pd
from LongGraphPackage import *
import numpy as np
PIPELINE = 'brain_pipeline'
TEMP_FOLDER = '/cs/usr/bennydv/Desktop/pach'

def check_gt_corrections():
    """
    Print the nodes added and removed
    """
    for pat in get_patients_list():
        print("")
        #if pat != 'C_A_': continue
        for pat_date in get_patient_dates(pat):
            gt_p = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_no_reg/{pat}/{name.segm_name_gt(pat_date)}"
            gt_c_p = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_corrected/{pat}/{name.segm_name_gt_corr(pat_date)}"
            if os.path.islink(gt_c_p):
                continue
            gt, _ = load_nifti_data(gt_p)
            gt_c, _ = load_nifti_data(gt_c_p)

            gt_labels = np.unique(gt)[1:].astype(np.int8)
            if len(gt_labels) == 0:
                max_lb = 0
            else:
                max_lb = np.max(gt_labels)

            gt_mask = np.zeros_like(gt_c, dtype=bool)
            gt_mask[gt>0] = True
            gt_c_mask = np.zeros_like(gt_c, dtype=bool)
            gt_c_mask[gt_c > 0] = True
            # in both
            tp_mask = np.zeros_like(gt_c, dtype=bool)
            tp_mask[np.bitwise_and(gt_mask, gt_c_mask)] = True
            # only in gt_c
            fn_mask = np.zeros_like(gt_c, dtype=bool)
            fn_mask[np.bitwise_and(gt_c_mask, np.bitwise_not(gt_mask))] = True

            gt_c_new = np.zeros_like(gt_c, dtype=np.int8)
            gt_c_new[tp_mask] = gt[tp_mask]
            fn_lesions_labels = get_labeled_segmentation(fn_mask, size_filtering=0)
            fn_bad_labels = np.unique(fn_lesions_labels)[1:]
            fn_lesions_good_labels = np.zeros_like(fn_lesions_labels, dtype=np.int8)
            good_lb = max_lb + 1
            for fn_bad_l in fn_bad_labels:
                fn_lesions_good_labels[fn_lesions_labels == fn_bad_l] = good_lb
                good_lb += 1
            gt_c_new += fn_lesions_good_labels
            gt_c_labels = np.unique(gt_c_new)[1:]
            fn_labels = set(gt_c_labels) - set(gt_labels)
            fp_labels = set(gt_labels) - set(gt_c_labels)
            if len(fn_labels) > 0:
                print(f"{pat} {pat_date}: fn labels: {fn_labels}")
            if len(fp_labels) > 0:
                print(f"{pat} {pat_date}: fp labels: {fp_labels}")


def relabel_gt_corrections():
    """
    Label the newly added lesions with *new* labels.
    """
    for pat in get_patients_list():
        print(f"{pat}")
        if pat != 'TM0': continue
        for pat_date in get_patient_dates(pat):
            print(f"{pat_date}")
            gt_p = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_no_reg/{pat}/{name.segm_name_gt(pat_date)}"
            gt_c_p = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_corrected/{pat}/{name.segm_name_gt_corr(pat_date)}"
            if os.path.islink(gt_c_p):
                continue

            gt, nifti = load_nifti_data(gt_p)
            gt_c, _ = load_nifti_data(gt_c_p)

            gt_labels = np.unique(gt)[1:].astype(np.int8)
            if len(gt_labels) == 0:
                max_lb = 0
            else:
                max_lb = np.max(gt_labels)

            gt_mask = np.zeros_like(gt_c, dtype=bool)
            gt_mask[gt>0] = True
            gt_c_mask = np.zeros_like(gt_c, dtype=bool)
            gt_c_mask[gt_c > 0] = True
            # in both
            tp_mask = np.zeros_like(gt_c, dtype=bool)
            tp_mask[np.bitwise_and(gt_mask, gt_c_mask)] = True
            # only in gt_c
            fn_mask = np.zeros_like(gt_c, dtype=bool)
            fn_mask[np.bitwise_and(gt_c_mask, np.bitwise_not(gt_mask))] = True

            gt_c_new = np.zeros_like(gt_c, dtype=np.int8)
            gt_c_new[tp_mask] = gt[tp_mask]
            fn_lesions_labels = get_labeled_segmentation(fn_mask, size_filtering=0)
            fn_bad_labels = np.unique(fn_lesions_labels)[1:]
            fn_lesions_good_labels = np.zeros_like(fn_lesions_labels, dtype=np.int8)
            good_lb = max_lb + 1
            for fn_bad_l in fn_bad_labels:
                fn_lesions_good_labels[fn_lesions_labels == fn_bad_l] = good_lb
                good_lb += 1
            gt_c_new += fn_lesions_good_labels
            #gt_c_labels = np.unique(gt_c_new)[1:]
            shutil.move(src=gt_c_p, dst=f"{TEMP_FOLDER}/{pat}{name.segm_name_gt_corr(pat_date)}")
            nib.save(nib.Nifti1Image(gt_c_new.astype(np.int8), nifti.affine), gt_c_p)


def check_relabeled_gt_corrections():
    """
    same as check_gt corrections, but now fp and fn labels are much easier to find
    """
    for pat in get_patients_list():
        print(f"")
        # if pat != 'C_A_': continue
        for pat_date in get_patient_dates(pat):
            gt_p = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_no_reg/{pat}/{name.segm_name_gt(pat_date)}"
            gt_c_p = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_corrected/{pat}/{name.segm_name_gt_corr(pat_date)}"
            if os.path.islink(gt_c_p):
                continue
            gt, _ = load_nifti_data(gt_p)
            gt_c, _ = load_nifti_data(gt_c_p)

            gt_labels = set(np.unique(gt)[1:])
            gt_c_labels = set(np.unique(gt_c)[1:])

            fn_labels = gt_c_labels - gt_labels
            fp_labels = gt_labels - gt_c_labels
            if len(fn_labels) > 0:
                print(f"{pat} {pat_date}: fn labels: {fn_labels}")
            if len(fp_labels) > 0:
                print(f"{pat} {pat_date}: fp labels: {fp_labels}")


def report_corrections():
    class Cols:
        tp = 'tp'
        # errors found by skip edges (se):
        fp_e = 'fp_e'
        fn_e = 'fn_e'
        N_se = '#se'
        N_se_corr = '#se corr'
        # errors found by lone lesions (l)
        fp_l = 'fp_l'
        fn_l = 'fn_l'
        N_l = '#lone>5mm'
        N_l_corr = '#lone>5mm corr'
        list_all = [tp, fp_e, fn_e, N_se, N_se_corr, fp_l, fn_l, N_l, N_l_corr]

    patients = get_patients_list()
    cols = Cols()
    data_empty = np.zeros((len(patients), len(cols.list_all)), dtype=int)
    df = pd.DataFrame(data=data_empty, index=patients, columns=cols.list_all)
    for pat in patients:
        # if pat != 'F_Y_Ga_': continue
        print(pat)
        dates = get_patient_dates(pat)

        corr_loader = LoaderSimpleFromJson(f"{path.LESIONS_MATCH_GT_CORRECTED}/{pat}glong_gt.json")
        corr = LongitClassification(corr_loader, pat, dates)
        corr.classify_nodes()
        corr_nodes = corr.get_graph().nodes(data=True)
        corr_lone_lesions = [l for l, attr in corr_nodes if attr[NodeAttr.CHANGES] == NodesChanges.LONE]
        corr_skip_edges = [(v1, v2) for v1, v2, attr in corr.get_graph().edges(data=True) if attr[EdgeAttr.IS_SKIP]]

        gt_loader = LoaderSimpleFromJson(f"{path.LESIONS_MATCH_GT_ORIGINAL}/{pat}glong_gt.json")
        gt = LongitClassification(gt_loader, pat, dates)
        gt.classify_nodes()
        gt_nodes = gt.get_graph().nodes(data=True)
        gt_lone_lesions = [l for l, attr in gt_nodes if attr[NodeAttr.CHANGES] == NodesChanges.LONE]
        gt_skip_edges = [(v1, v2) for v1, v2, attr in gt.get_graph().edges(data=True) if attr[EdgeAttr.IS_SKIP]]

        for time_ind, date in enumerate(dates):
            corr_segm, nifti = load_nifti_data(
                f"{path.GT_FILTERED_LABELED_CORRECTED}/{pat}/{name.segm_name_gt_corr(date)}", type_integer=True)
            gt_segm, _ = load_nifti_data(
                f"{path.GT_FILTERED_LABELED_NO_REG}/{pat}/{name.segm_name_gt(date)}", type_integer=True)

            corr_lb = set(np.unique(corr_segm)[1:])
            gt_lb = set(np.unique(gt_segm)[1:])

            # tp
            df.loc[pat, cols.tp] += len(corr_lb & gt_lb)
            # fp
            all_fp = gt_lb - corr_lb
            for fp_label in all_fp:
                # check if the fp is lone in gt_graph.
                fp_les = f"{fp_label}_{time_ind}"
                if gt_nodes[fp_les][NodeAttr.CHANGES] == NodesChanges.LONE:
                    df.loc[pat, cols.fp_l] += 1
                else:
                    df.loc[pat, cols.fp_e] += 1
            # fn
            all_fn = corr_lb - gt_lb
            for fn_label in all_fn:
                fn_les = f"{fn_label}_{time_ind}"
                neighbors_are_lone = [neigh_les in gt_lone_lesions for neigh_les in corr.get_graph().neighbors(fn_les)]
                if any(neighbors_are_lone):
                    # a fn was found by lone lesion if some neighbour its a lone lesion
                    df.loc[pat, cols.fn_l] += 1
                else:
                    df.loc[pat, cols.fn_e] += 1

            if 0 < time_ind < len(dates) - 1:
                corr_date_lone_labels = np.array([int(l.split('_')[0]) for l in corr_lone_lesions
                                                if int(l.split('_')[1]) == time_ind])
                gt_date_lone_labels = np.array([int(l.split('_')[0]) for l in gt_lone_lesions
                                                if int(l.split('_')[1]) == time_ind])

                gt_labels_area = ndimage.sum(gt_segm>0, gt_segm, gt_date_lone_labels) * np.product(nifti.header.get_zooms())
                gt_bigger_than_5 = gt_date_lone_labels[np.round((6*gt_labels_area/np.pi)**(1/3),1) > 5]
                corr_bigger_than_5 = corr_date_lone_labels[np.isin(corr_date_lone_labels, gt_bigger_than_5)]
                df.loc[pat, cols.N_l] += len(gt_bigger_than_5)
                df.loc[pat, cols.N_l_corr] += len(corr_bigger_than_5)

        df.loc[pat, cols.N_se_corr] += len(corr_skip_edges)
        df.loc[pat, cols.N_se] += len(gt_skip_edges)

    os.makedirs(f"{path.LESIONS_MATCHING}/lesion_matching_database_correction", exist_ok=True)
    df.to_excel(f"{path.LESIONS_MATCHING}/lesion_matching_database_correction/correction_report.xlsx")


def report_predicted_lesion_vs_corrections_vs_gt():
    """
    Check the if some FP/FN predicted lesion with respect to unreviewed GT are instead present/absent in corrected GT
    """
    CORR = 1
    GT = 2
    BOTH = GT + CORR

    class Cols:
        pred_and_corr_and_gt = 'pred_and_corr_and_gt' # TP
        pred_and_corr_only = 'pred_and_corr_only' #previously: FP, reviewed: TP
        pred_and_gt_only = 'pred_and_gt_only' #previously: TP, reviewed:FP
        pred_only = 'pred_only' #FP
        corr_and_gt_only = 'corr_and_gt_only' #FN
        corr_only = 'corr_only' #previously: TN (uncounted), reviewed: FN
        gt_only = 'gt_only' #previously: FN, reviewed TP

        list_all = [pred_and_corr_and_gt, pred_and_corr_only, pred_and_gt_only, corr_and_gt_only, pred_only, gt_only,
                    corr_only]

    patients = get_patients_list()
    cols = Cols()
    data_empty = np.zeros((len(patients), len(cols.list_all)), dtype=int)
    df = pd.DataFrame(data=data_empty, index=patients, columns=cols.list_all)
    for pat in patients:
        if pat != 'E_N_': continue
        print(pat)
        dates = get_patient_dates(pat)
        for time_ind, date in enumerate(dates):
            corr_segm, _ = load_nifti_data(
                f"{path.GT_FILTERED_LABELED_CORRECTED}/{pat}/{name.segm_name_gt_corr(date)}", type_integer=True)
            gt_segm, _ = load_nifti_data(
                f"{path.GT_FILTERED_LABELED_NO_REG}/{pat}/{name.segm_name_gt(date)}", type_integer=True)
            pred_segm, _ = load_nifti_data(
                f"{path.PRED_FILTERED_LABELED_NO_REG}/{pat}/{name.segm_name_pred(date)}", type_integer=True)

            corr_lb = set(np.unique(corr_segm)[1:])
            gt_lb = set(np.unique(gt_segm)[1:])
            pred_lb = set(np.unique(pred_segm)[1:])

            #
            corr_mask = np.zeros_like(corr_segm, dtype=np.int8)
            corr_mask[corr_segm > 0] = 1
            gt_mask = np.zeros_like(gt_segm, dtype=np.int8)
            gt_mask[gt_segm > 0] = 1
            pred_mask = np.zeros_like(pred_segm, dtype=np.int8)
            pred_mask[pred_segm>0] = 1

            # pred_and_corr_and_gt_lb = set()
            # pred_and_corr_only_lb = set()
            # pred_and_gt_only_lb = set()
            # pred_only = set()

            matched_gt_lb = set(np.unique(pred_mask * gt_segm)[1:])
            matched_corr_lb = set(np.unique(pred_mask * corr_segm)[1:])
            matched_pred_lb = set(np.unique((gt_mask + corr_mask > 0) * pred_segm)[1:])

            # corr_not_pred_lb = corr_lb - pred_and_corr_and_gt_lb - pred_and_corr_only_lb
            # gt_not_pred_lb = gt_lb - pred_and_corr_and_gt_lb - pred_and_gt_only_lb

            corr_and_gt_only_lb = (corr_lb - matched_corr_lb) & (gt_lb - matched_gt_lb)
            gt_only = gt_lb - matched_gt_lb - corr_and_gt_only_lb
            curr_only = corr_lb - matched_corr_lb - corr_and_gt_only_lb

            # for pred_l in pred_lb:
            #     gt_match_pred = gt_segm[pred_segm == pred_l]
            #     corr_match_pred = corr_segm[pred_segm == pred_l]
            #     match_gt_lb = set(np.unique(gt_match_pred)[1:])
            #     match_corr_lb = set(np.unique(corr_match_pred)[1:])
            #     if len(match_gt_lb) == 0 and len(match_corr_lb) == 0:
            #         pred_only = pred_only | {pred_l}
            #     else:
            #         pred_and_corr_and_gt_lb = pred_and_corr_and_gt_lb | (match_gt_lb & match_corr_lb)
            #         pred_and_corr_only_lb = pred_and_corr_only_lb | (match_corr_lb - match_gt_lb)
            #         pred_and_gt_only_lb = pred_and_gt_only_lb | (match_gt_lb - match_corr_lb)

            df.loc[pat, cols.pred_and_corr_and_gt] += len(matched_gt_lb & matched_corr_lb)
            df.loc[pat, cols.pred_and_corr_only] += len(matched_corr_lb - matched_gt_lb)
            df.loc[pat, cols.pred_and_gt_only] += len(matched_gt_lb - matched_corr_lb)
            df.loc[pat, cols.pred_only] += len(pred_lb - matched_pred_lb)

            df.loc[pat, cols.corr_and_gt_only] += len(corr_and_gt_only_lb)
            df.loc[pat, cols.gt_only] += len(gt_only)
            df.loc[pat, cols.corr_only] += len(curr_only)

    os.makedirs(f"{path.LESIONS_MATCHING}/lesion_matching_database_correction", exist_ok=True)
    df.to_excel(f"{path.LESIONS_MATCHING}/lesion_matching_database_correction/correction_report_triple_comparison__.xlsx")

            #
            # all_fp_labels = gt_lb - corr_lb
            # num_gt_fp = len(all_fp_labels)
            # if num_gt_fp > 0:
            #     gt_fp_mask = np.zeros_like(gt_segm, dtype=np.int8)
            #     gt_fp_mask[np.isin(gt_segm, all_fp_labels)] = 1
            #
            #
            #
            #
            # all_fn_labels = corr_segm - gt_segm
            # if len(all_fn_labels) > 0:
            #     gt_fn_mask = np.zeros_like(gt_segm, dtype=np.int8)
            #     gt_fn_mask[np.isin(corr_segm, all_fn_labels)] = 1


def prepare_labeled_correct_folder():
    for pat in get_patients_list():
        uncorrected_paths = glob.glob(f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_no_reg/{pat}/*")
        os.makedirs(f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_corrected/{pat}", exist_ok=True)
        for p in uncorrected_paths:
            corr_path = p.replace('labeled_no_reg', 'labeled_corrected').replace('lesions_gt', 'lesions_gt_corr')
            if not os.path.exists(corr_path):
                os.symlink(src=p, dst=corr_path)






if __name__ == "__main__":
    from config import *
    from general_utils import *
    name = Name()
    #check_gt_corrections()

    #relabel_gt_corrections()
    #check_relabeled_gt_corrections()
    report_corrections()
    #prepare_labeled_correct_folder()
    # p = "/cs/usr/bennydv/Downloads/from_benny/AZ0_02-06-2020"
    # gt = f"{p}/lesions_gt_02_06_2020.nii.gz"
    # yg = f"{p}/Yigal.nii.gz"
    # gt_l, nifti = load_nifti_data(gt)
    # yg_l, _ = load_nifti_data(yg)
    # im = gt_l + yg_l
    # nib.save(nib.Nifti1Image(im.astype(np.int8), nifti.affine), f"{p}/composed.nii.gz")

    #report_predicted_lesion_vs_corrections_vs_gt()
    # p_bnny = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_corrected/E_N_/lesions_gt_corr_20_01_2020.nii.gz"
    # p_r = "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_corrected/E_N_/r1-lesions_gt_20_01_2020.nii.gz"
    # bnny_im, nifti = load_nifti_data(p_bnny)
    # rich_im, _ = load_nifti_data(p_r)
    # bnny_im[rich_im == 1] = 1
    # nib.save(nib.Nifti1Image(bnny_im.astype(np.int8), nifti.affine), "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_corrected/E_N_/lesions_gt_corrB_20_01_2020.nii.gz")
    #

            # labels_c = np.unique(label_gt_c)
            # for lb in labels_c:
            #     if lb == 0: continue
            #     gt_lb_list = np.unique(gt[gt_c == lb])
            #     if len(gt_lb_list) != 1:
            #         raise ValueError("Not 1:1!")
            #     gt_lb = gt_lb_list[0]
            #     if gt_lb != 0: # tp lesion:
            #         gt_c_new[gt_c == lb] = gt_lb



