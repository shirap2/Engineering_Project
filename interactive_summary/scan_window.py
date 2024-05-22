import subprocess
import os
import networkx as nx


def get_slice(longit):
    nodes2slice = nx.get_node_attributes(longit.get_graph(), NodeAttr.SLICE)


def open_itksnap_on_slice(slice_idx, scan_file, seg_file):
    """
    Opens a specified slice of a NIfTI scan and segmentation file in ITK-SNAP.

    Parameters:
    slice_idx (int): The index of the slice to view.
    scan_file (str): The path to the NIfTI scan file (.nii.gz).
    seg_file (str): The path to the NIfTI segmentation file (.nii.gz).
    """
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
        "-s", seg_file,
        "--slice", "z", str(slice_idx)
    ]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Opened ITK-SNAP on slice {slice_idx} for scan {scan_file} and segmentation {seg_file}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to open ITK-SNAP: {e}")


open_itksnap_on_slice(50, "path/to/scan.nii.gz", "path/to/segmentation.nii.gz")
