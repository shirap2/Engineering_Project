import numpy as np

from general_utils import *
import glob
from scipy import ndimage
from multiprocessing import Pool
import matplotlib.pyplot as plt

def all_sizes_les(organ):
    PIPELINE = f"{organ}_pipeline"
    gt_folder = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_no_reg/"
    all_les_paths = glob.glob(f"{gt_folder}/*/lesions_gt*")
    all_les_size = []
    for i,les_p in enumerate(all_les_paths):
        print(f"{i+1}/{len(all_les_paths)}")
        les, nifti = load_nifti_data(les_p, type_integer=True)
        labels = np.unique(les)[1:]
        areas = ndimage.sum(les>0, les, labels)
        all_les_size += list(areas*np.prod(nifti.header.get_zooms()))
    return all_les_size

def all_size_org(organ):
    PIPELINE = f"{organ}_pipeline"
    gt_folder = f"/cs/casmip/bennydv/{PIPELINE}/gt_data/size_filtered/labeled_no_reg/"
    all_les_paths = glob.glob(f"{gt_folder}/*/{organ}*")
    all_les_size = []
    for i, les_p in enumerate(all_les_paths):
        print(f"{i + 1}/{len(all_les_paths)}")
        les, nifti = load_nifti_data(les_p, type_integer=True)
        areas = np.sum(les)
        all_les_size += [areas * np.prod(nifti.header.get_zooms())]
    return all_les_size


organs =  ['lungs', 'liver', ]#'brain']
# with Pool(3) as p:
#     res = p.map(all_sizes_les, organs)
with Pool(2) as p:
    res = p.map(all_size_org, organs)


import json
with open("/cs/casmip/bennydv/org_dimension.json", 'w') as f:
    json.dump({organs[i]: res[i] for i in range(2)}, f)
with open("/cs/casmip/bennydv/org_dimension.json", 'r') as f:
    res = json.load(f)

for org, v in res.items():
    vols = np.array(v)
    print(f"{org}: vol cc: {np.mean(vols/1000)}+-{np.std(vols/1000)}, #org {len(vols)}")




