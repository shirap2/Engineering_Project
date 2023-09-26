import numpy as np

from general_utils import load_nifti_data, get_labeled_segmentation

def get_voxels_count_per_label(scan):
    labels, counts = np.unique(scan, return_counts=True)
    combined_labels_and_counts = np.column_stack((labels, counts))
    counts_per_label = dict(np.core.records.fromarrays(combined_labels_and_counts.T, names='label, count', formats='i4, i4'))
    del counts_per_label[0]
    return counts_per_label

def from_voxels_count_to_volume_mm_3(counts_per_label :dict, nifti):
    one_voxel_volume = nifti.header.get_zooms()[0] * nifti.header.get_zooms()[1] * nifti.header.get_zooms()[2]
    volume_per_label = {(key, val*one_voxel_volume) for key, val in counts_per_label.items()}
    return volume_per_label

def from_mask_to_volume():
    scan, nifti = load_nifti_data(
        "/cs/casmip/bennydv/liver_pipeline/gt_data/size_filtered/labeled_no_reg/A_S_H_/lesions_gt_01_09_2019.nii.gz",
        type_integer=True)

    labeled_scan = get_labeled_segmentation(scan)
    counts_per_label = get_voxels_count_per_label(labeled_scan)
    volume_per_label = from_voxels_count_to_volume_mm_3(counts_per_label, nifti)
    print(volume_per_label)



from_mask_to_volume()