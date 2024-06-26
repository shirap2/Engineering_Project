import shutil


def copy_files(src_paths, dest_paths):
    if len(src_paths) != len(dest_paths):
        raise ValueError("The number of source paths and destination paths must be the same.")

    for src, dest in zip(src_paths, dest_paths):
        try:
            shutil.copy2(src, dest)
            print(f"Copied {src} to {dest}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except PermissionError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

patient_name = 'A_S_H_'
organ = 'liver'
dates = ['01_09_2019', '06_01_2020', '06_04_2020', '20_02_2020'
         ]
# hospital = 'Har_Hatzofim'
hospital = 'Ein_Kerem'


# patient_name = 'A_S_S_'
# organ = 'liver'
# dates = ['03_05_2017', '09_07_2017', '19_10_2017'
#          ]

# patient_name = 'C_A_'
# organ = 'liver'
# dates = ['05_11_2017', '07_04_2020', '10_01_2018', '13_12_2018',
#          '14_01_2020',
#          '22_04_2018', '28_04_2019', '31_03_2019', '31_05_2020'
#          ]


src_paths = [f'/cs/CASMIP/public/for_aviv/all_data/{hospital}/{patient_name}{date}/scan.nii.gz'
             for date in dates] + [f"/cs/casmip/archive/bennydv/{organ}_pipeline/lesions_matching/longitudinal_gt/original_corrected/{patient_name}glong_gt.json"] + [
    f'/cs/CASMIP/archive/bennydv/{organ}_pipeline/gt_data/size_filtered/labeled_no_reg/{patient_name}/lesions_gt_{date}.nii.gz' for date in dates]


dest_paths = [f'/cs/usr/talia.dym/Desktop/Engineering_Project/DATASET/{organ}_pipeline/gt_data_nifti/{patient_name}/scan_{date}.nii.gz'
                for date in dates] + [f'/cs/usr/talia.dym/Desktop/Engineering_Project/DATASET/{organ}_pipeline/lesions_matching_json/{patient_name}glong_gt.json'] + [
    f'/cs/usr/talia.dym/Desktop/Engineering_Project/DATASET/{organ}_pipeline/gt_data_nifti/{patient_name}/lesions_gt_{date}.nii.gz' for date in dates
]

copy_files(src_paths, dest_paths)
