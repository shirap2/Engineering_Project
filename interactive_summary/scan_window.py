import subprocess
import os



def scan_path(organ, name, date):
    return f'/cs/casmip/archive/bennydv/{organ}_pipeline/gt_data/size_filtered/labeled_no_reg/{name}/scan_{date}.nii.gz'


def gt_segmentation_path(organ, name, date):
    return f'/cs/casmip/archive/bennydv/{organ}_pipeline/gt_data/size_filtered/labeled_no_reg/{name}/lesions_gt_{date}.nii.gz'


def open_itksnap_on_slice(organ, name, date, slice_idx=0):
    """
    Opens a specified slice of a NIfTI scan and segmentation file in ITK-SNAP.

    Parameters:
    slice_idx (int): The index of the slice to view.
    scan_file (str): The path to the NIfTI scan file (.nii.gz).
    seg_file (str): The path to the NIfTI segmentation file (.nii.gz).
    """
    scan_file = scan_path(organ, name, date)
    seg_file = gt_segmentation_path(organ, name, date)
    # Ensure ITK-SNAP is installed
    if subprocess.run(["which", "itksnap"], capture_output=True).returncode != 0:
        raise EnvironmentError("ITK-SNAP is not installed or not found in the system PATH.")

    # Check if the files exist
    if not os.path.exists(scan_file):
        raise FileNotFoundError(f"Scan file not found: {scan_file}")
    if not os.path.exists(seg_file):
        raise FileNotFoundError(f"Segmentation file not found: {seg_file}")

    # Construct the ITK-SNAP command
    command = [
        "itksnap",
        "-g", scan_file,
        "-s", seg_file
        # , "--slice", "z", str(slice_idx)
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Opened ITK-SNAP on slice {slice_idx} for scan {scan_file} and segmentation {seg_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to open ITK-SNAP: {e}")

# organ = "liver"
# name = "C_A_"
# date = "14_01_2020"
# open_itksnap_on_slice(scan_path(organ, name, date), gt_segmentation_path(organ, name, date))
