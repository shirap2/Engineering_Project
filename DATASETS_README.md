Scans and labeled lesions masks:
ORGAN = lungs, liver, brain
each patient has a folder named PATIENT_NAME
DATE is one of the dates in which the patient was scanned.

A) Scans:
cs/casmip/bennydv/{ORGAN}_pipeline/gt_data/size_filtered/labeled_no_reg/{PATIENT_NAME}/scan_{DATE}.nii.gz

B) Ground-truth lesion masks, labeled
cs/casmip/bennydv/{ORGAN}_pipeline/gt_data/size_filtered/labeled_no_reg/{PATIENT_NAME}/lesions_gt_{DATE}.nii.gz

[C) Ground-truth organ segmentation
cs/casmip/bennydv/{ORGAN}_pipeline/gt_data/size_filtered/labeled_no_reg/{PATIENT_NAME}/{ORGAN}_{DATE}.nii.gz
]

D) Ground-truth longitudinal matching graphs, as json files
cs/casmip/bennydv/{ORGAN}_pipeline/lesions_matching/longitudinal_gt/original/{PATIENT_NAME}glong_gt.json

E) Ground-truth longitudinal matching graphs corrected with the workflow, as json files
cs/casmip/bennydv/{ORGAN}_pipeline/lesions_matching/longitudinal_gt/original_corrected/{PATIENT_NAME}glong_gt.json

F) Computed longitudinal matching graphs on gt labeled lesions (B)
/cs/casmip/bennydv/lungs_pipeline/lesions_matching/results/gt_segmentation_gw13/{PATIENT_NAME}/gw13.json

