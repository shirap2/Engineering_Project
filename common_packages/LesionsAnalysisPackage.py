import pandas as pd
import numpy as np
from typing import List
import os
from skimage.measure import regionprops
from itertools import combinations
from functools import partial
from tqdm.contrib.concurrent import process_map
from time import time
from common_packages.MatchingGroupwisePackage import ImagesCreator
from common_packages.BaseClasses import *
from common_packages.ComputedGraphsMapping import MapComputedGraphs
from common_packages.LongGraphPackage import LoaderSimpleFromJson
from common_packages.LongGraphClassification import LongitClassification


class LesionAttr:
    REG_CENTROID = 'centroid (reg)'
    VOLUME = 'volume'
    DIAMETER = 'extr. diameter'
    MAX_LESION_Z = 'max lesion slice'
    CALLIPER_DIAMETER = 'calliper diameter'
    LESION_CHANGE = 'lesion_change'
    PATTERN_CLASSIFICATION = 'pattern_classification'
    DETECTION = 'detection'
    GT_MAPPING = 'gt_mapping'


class Lesion:
    def __init__(self, label=None, layer=None, lb_layer=None):
        if label is None or layer is None:
            label, layer = lb_layer.split('_')
        self._label = int(label)
        self._layer = int(layer)
        self._reg_centroid = None
        self._volume = None
        self._diameter = None

    def label(self):
        return self._label

    def layer(self):
        return self._layer

    def name(self):
        return f"{self._label}_{self._layer}"

    def set(self, attr, value):
        if attr == LesionAttr.REG_CENTROID:
            if value is None:
                return
            if self._reg_centroid is not None:
                raise ValueError(f"{attr} has already a value!")
            if len(value) > 1:
                reg_cent = tuple([round(v) + 1 for v in value])
                self._reg_centroid = reg_cent
            else:
                raise ValueError("Bad centroid")
        elif attr == LesionAttr.VOLUME:
            if self._volume is not None:
                raise ValueError(f"{attr} has already a value!")
            self._volume = value
        elif attr == LesionAttr.DIAMETER:
            if self._diameter is not None:
                raise ValueError(f"{attr} has already a value!")
            self._diameter = value

    def get(self, attr):
        if attr == LesionAttr.REG_CENTROID:
            return self._reg_centroid
        elif attr == LesionAttr.VOLUME:
            return self._volume
        elif attr == LesionAttr.DIAMETER:
            return self._diameter

    @staticmethod
    def extrapulate_diameter(volume):
        return (6*volume/np.pi)**(1/3)

    def __hash__(self):
        return self._label + self._layer*1000


class MatchAttr:
    CENTROID_DIST = 'centroid dist'
    VOL_DIFFERENCE = 'volume diff'
    NUM_OVERLAP_VOXELS = 'num overlap voxels'
    DICE = 'dice'


class Match:
    def __init__(self, lesion0: Lesion, lesion1: Lesion):
        layer0 = lesion0.layer()
        layer1 = lesion1.layer()
        if layer0 == layer1:
            raise ValueError("Matching between two lesions in the same layer is invalid")
        if layer0 < layer1:
            self._lesion0 = lesion0
            self._lesion1 = lesion1
        else:
            self._lesion0 = lesion1
            self._lesion1 = lesion0
        self._centroid_dist = None
        self._num_overlap_voxels = None
        self._vol_difference = None
        self._dice = None

    def get_lesions_names(self):
        return self._lesion0.name(), self._lesion1.name()

    def __eq__(self, other):
        other_l0, other_l1 = other.get_lesions_names()
        l0, l1 = self.get_lesions_names()
        return other_l0 == l0 and other_l1 == l1

    def __hash__(self):
        return self._lesion0.__hash__() * 1000 + self._lesion1.__hash__()

    def set(self, attr, value):
        if attr == MatchAttr.CENTROID_DIST:
            if self._centroid_dist is not None:
                raise ValueError(f"{attr} has already a value!")
            self._centroid_dist = round(value, 2)

        elif attr == MatchAttr.NUM_OVERLAP_VOXELS:
            if self._num_overlap_voxels is not None:
                raise ValueError(f"{attr} has already a value!")
            self._num_overlap_voxels = value

        elif attr == MatchAttr.VOL_DIFFERENCE:
            if self._vol_difference is not None:
                raise ValueError(f"{attr} has already a value!")
            self._vol_difference = value

        elif attr == MatchAttr.DICE:
            if self._dice is not None:
                raise ValueError(f"{attr} has already a value!")
            self._dice = value

    def get(self, attr):
        if attr == MatchAttr.CENTROID_DIST:
            return self._centroid_dist
        elif attr == MatchAttr.NUM_OVERLAP_VOXELS:
            return self._num_overlap_voxels
        elif attr == MatchAttr.VOL_DIFFERENCE:
            return self._vol_difference
        elif attr == MatchAttr.DICE:
            return self._dice


class TableManager:
    def __init__(self, tables_folder: str, patient_list: List[str]):
        assert os.path.exists(tables_folder)
        self._tables_folder = tables_folder
        self._patient_list = patient_list
        self._lesion_path = f"{self._tables_folder}/lesions_data.xlsx"
        self._match_path = f"{self._tables_folder}/match_data.xlsx"
        self._matching_graph_paths = None
        self._matching_graph_name = None
        self._mapping_path = None
        self._mapping_name = None
    
    def run(self):
        les_tb = TableLesionCreator(table_path=self._lesion_path,
                                    patient_list=self._patient_list,
                                    series_loader=self.local_load_series)
        les_tb.run()
        
        # match_tb = TableMatchesCreator(matches_table_path=self._match_path,
        #                                lesion_table_path=self._lesion_path,
        #                                patient_list=self._patient_list,
        #                                series_loader=self.local_load_reg_series)
        # match_tb.run()

    def lesions_update(self):
        les_tb = TableLesionCreator(table_path=self._lesion_path,
                                    patient_list=self._patient_list,
                                    series_loader=self.local_load_orig_series)
        les_tb.update()

    def lesion_labels_update(self):
        les_tb = TableLesionCreator(table_path=self._lesion_path,
                                    patient_list=self._patient_list,
                                    series_loader=None)
        les_tb.update_lesion_labels(matching_path=self._matching_graph_paths,
                                    matching_name=self._matching_graph_name,
                                    mapping_path=self._mapping_path,
                                    mapping_name=self._mapping_name)
    
        
class TableCreator:
    """Class to create a patient table. The class holds the patient list and the table path. Its non-abstract methods
    perform excel-pandas configurations"""

    def __init__(self, table_path: str, patient_list: List, local_load_series=None, load_load_reg_series=None):
        """
        table_path: the path of the table you want to create. 
        patient_list: list of patients names (str)
        local_load_series= function that loads reg series, non registered series and voxel volume
        local_load_reg_series = function that loads reg series
        """
        self._table_path = table_path
        self._patient_list = patient_list
        self._writer = pd.ExcelWriter(self._table_path, engine='xlsxwriter')
        self._local_load_series_fun = local_load_series
        self._load_load_reg_series_fun = load_load_reg_series

    def save(self):
        self._writer.save()

    def write_to_excel(self, sheet_name, df):
        """Function to write the excel with a nice layout"""
        if df.empty:
            df.to_excel(self._writer, sheet_name=sheet_name)
            return
        columns_order = list(df.columns)
        workbook = self._writer.book
        cell_format = workbook.add_format({'num_format': '#,##0.00'})
        cell_format.set_font_size(16)

        df.to_excel(self._writer, sheet_name=sheet_name, startrow=1, startcol=1, header=False, index=False)
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'font_size': 16,
            'valign': 'top',
            'border': 1})

        worksheet = self._writer.sheets[sheet_name]
        worksheet.freeze_panes(1, 1)

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

    def load_patient_series(self, patient_name):
        """Return reg_series, no_reg_series, voxel_dim_series"""
        if self._local_load_series_fun is None:
            raise ValueError("Abstract method! Must override")
        else:
            return self._local_load_series_fun(patient_name)

    def load_patient_registered_scans(self, patient_name):
        """Return reg_series"""
        if self._load_load_reg_series_fun is None:
            raise ValueError("Abstract method! Must override")
        else:
            return self._load_load_reg_series_fun(patient_name)

    def run(self):
        """Run: create a table for each patient"""
        raise ValueError("Abstract method! Must override")


class TableLesionCreator(TableCreator):
    def __init__(self, table_path: str, patient_list: List, series_loader):
        super().__init__(table_path, patient_list, local_load_series=series_loader)

    def run(self):
        for patient in self._patient_list:
            print(f"Working on patient {patient}")
            reg_series, no_reg_series, voxel_dim_series = self.load_patient_series(patient)
            patient_sheet = LesionPatientSheet(reg_series, no_reg_series, voxel_dim_series)
            df = patient_sheet.create()
            self.write_to_excel(sheet_name=patient, df=df)
        self.save()


    def update(self):

        for patient in self._patient_list:
            print(f"Working on patient {patient}")
            orig_series, voxel_dim_series = self.load_patient_series(patient)
            df = pd.read_excel(self._table_path, sheet_name=patient, col_index=0)
            for les in df.index:
                l = Lesion(lb_layer=les)
                scan = orig_series[l.layer()]
                voxel_dim = voxel_dim_series[l.layer()]
                if voxel_dim[0] != voxel_dim[1]:
                    print(f"layer: {l.layer()}, X and Y res are different! {voxel_dim[0]} and {voxel_dim[1]}")

                working_scan = np.zeros_like(scan).astype(int)
                working_scan[scan == l.label()] = 1
                slice_lesion_max_area = np.argmax(np.sum(working_scan, axis=(0,1)))
                slice_scan = working_scan[:,:,slice_lesion_max_area]
                rp = regionprops(slice_scan)
                calliper_diam = (rp[0].feret_diameter_max)*voxel_dim[0]
                df.loc[les, LesionAttr.MAX_LESION_Z] = slice_lesion_max_area + 1
                df.loc[les, LesionAttr.CALLIPER_DIAMETER] = calliper_diam

            self.write_to_excel(sheet_name=patient, df=df)
        self.save()

    def update_lesion_labels(self, matching_path, matching_name, mapping_path, mapping_name):
        """Add to the table the TP, FP labels and the CC and Change labels
        Inputs: {m*_path}/patient_name/{m*_name}"""
        if any([inp is None for inp in [matching_path, matching_name, mapping_path, mapping_name]]):
            raise ValueError("One or more update_lesion_labels are None!")
        match_folder = os.path.basename(matching_path)
        print(f"Matching folder is {match_folder}")
        for patient in self._patient_list:
            print(f"Working on patient {patient}")
            ld = LoaderSimpleFromJson(f"{matching_path}/{patient}/{matching_name}")
            if len(ld.get_nodes()) == 0:
                self.write_to_excel(sheet_name=patient, df=pd.DataFrame())
                continue
            l = LongitClassification(ld, patient)
            l.classify_nodes()
            l.classify_cc()
            l.classify_nodes_detection(mapping_path=f"{mapping_path}/{patient}/{mapping_name}")
            graph_nodes = l.get_graph().nodes(data=True)

            mapping = MapComputedGraphs()
            mapping.load_mapping(f"{mapping_path}/{patient}/{mapping_name}")

            df = pd.read_excel(self._table_path, sheet_name=patient, col_index=0)
            for les in df.index:
                if les not in graph_nodes:
                    # This is a lesion that was deleted by the registration
                    continue
                df.loc[les, LesionAttr.DETECTION] = graph_nodes[les][NodeAttr.DETECTION]
                df.loc[les, LesionAttr.LESION_CHANGE] = graph_nodes[les][NodeAttr.CHANGES]
                df.loc[les, LesionAttr.PATTERN_CLASSIFICATION] = graph_nodes[les][NodeAttr.CC_PATTERNS]
                df.loc[les, LesionAttr.GT_MAPPING] = str(mapping.get_matches_for_pred_node(les)).replace('[','').replace(']','')

            self.write_to_excel(sheet_name=patient, df=df)
        self.save()



class TableMatchesCreator(TableCreator):
    def __init__(self, matches_table_path: str, lesion_table_path: str, patient_list: List, series_loader):
        super().__init__(matches_table_path, patient_list, load_load_reg_series=series_loader)
        self._lesion_table_path = lesion_table_path

    def run(self):
        for patient in self._patient_list:
            print(f"Working on patient {patient}")
            reg_series = self.load_patient_registered_scans(patient)
            patient_lesion_df = pd.read_excel(self._lesion_table_path, sheet_name=patient, index_col=0)
            patient_sheet = MatchPatientSheet(reg_series, patient_lesion_df)
            patient_dfs_dict = patient_sheet.create()
            for stat_name, patient_df in patient_dfs_dict.items():
                self.write_to_excel(sheet_name=f"{patient}{stat_name}", df=patient_df)
        self.save()


class PatientSheetInterface:
    def __init__(self):
        pass

    def extract_lesions_list(self):
        """Get a list of Lesion for the current patient"""
        raise ValueError("Abstract Method!")

    def initialize_df(self):
        """Define rows and columns for the table. Open new dataframe"""
        raise ValueError("Abstract Method!")

    def create(self):
        """Return the dataframe filled with data"""
        raise ValueError("Abstract Method!")


class LesionPatientSheet(PatientSheetInterface):
    def __init__(self, reg_series: List[np.array], no_reg_series: List[np.array], voxel_dim_series: List[float]):
        super().__init__()
        assert len(reg_series) == len(no_reg_series)
        assert len(voxel_dim_series) == len(no_reg_series)
        self._reg_series = reg_series
        self._no_reg_series = no_reg_series
        self._voxel_dim_series = voxel_dim_series
        self._num_time_points = len(self._no_reg_series)
        self._lesions_list = self.extract_lesions_list()
        if len(self._lesions_list) == 0:
            return
        self._df = self.initialize_df()

    def extract_lesions_list(self):
        lesions_list = list()
        for tpoint in range(self._num_time_points):
            unique_labels = np.unique(self._no_reg_series[tpoint])
            unique_labels = unique_labels[unique_labels != 0]
            for lb in unique_labels:
                lesions_list.append(Lesion(lb, tpoint))
        return lesions_list

    def initialize_df(self):
        col_names = [LesionAttr.REG_CENTROID, LesionAttr.VOLUME, LesionAttr.DIAMETER]
        row_names = [l.name() for l in self._lesions_list]
        n_col = len(col_names)
        n_row = len(row_names)
        empty_data = [[None] * n_col] * n_row
        return pd.DataFrame(data=empty_data, columns=col_names, index=row_names)

    def calculate_volume_and_diameter(self):
        lb2volumes = list()
        for tpoint in range(self._num_time_points):
            tpoint_scan = self._no_reg_series[tpoint]
            tpoint_voxel_dim = self._voxel_dim_series[tpoint]
            rprop = regionprops(tpoint_scan, cache=False)
            lb2volumes.append({r.label: r.area*tpoint_voxel_dim for r in rprop})
            #lb2diam.append({lb: Lesion.extrapulate_diameter(vol) for lb, vol in lb2volumes[tpoint].items()})

        for lesion in self._lesions_list:
            les_volume = lb2volumes[lesion.layer()][lesion.label()]
            lesion.set(attr=LesionAttr.VOLUME, value=les_volume)
            lesion.set(attr=LesionAttr.DIAMETER, value=Lesion.extrapulate_diameter(les_volume))

        # for lesion in self._lesions_list:
        #     lb = lesion.label()
        #     tpoint = lesion.layer()
        #     tpoint_scan = self._no_reg_series[tpoint]
        #     tpoint_voxel_dim = self._voxel_dim_series[tpoint]
        #     les_volume = np.sum(tpoint_scan[tpoint_scan == lb]) * tpoint_voxel_dim
        #     les_diameter = Lesion.extrapulate_diameter(les_volume)
        #     lesion.set(attr=LesionAttr.VOLUME, value=les_volume)
        #     lesion.set(attr=LesionAttr.DIAMETER, value=les_diameter)

    def calculate_reg_centroid(self):
        lb2centroid = list()
        for tpoint in range(self._num_time_points):
            tpoint_scan = self._reg_series[tpoint]
            rprop = regionprops(tpoint_scan, cache=False)
            lb2centroid.append({r.label: r.centroid for r in rprop})

        for lesion in self._lesions_list:
            try:
                les_centroid = lb2centroid[lesion.layer()][lesion.label()]
            except KeyError:
                # the label has been deleted during registration
                les_centroid = None
            lesion.set(attr=LesionAttr.REG_CENTROID, value=les_centroid)

    def create(self):
        if len(self._lesions_list) == 0:
            return pd.DataFrame()
        self.calculate_volume_and_diameter()
        self.calculate_reg_centroid()
        for lesion in self._lesions_list:
            for attr in self._df.columns:
                self._df.loc[lesion.name(), attr] = lesion.get(attr)

        return self._df


class MatchPatientSheet(PatientSheetInterface):
    def __init__(self, reg_series, patient_lesion_df):
        super().__init__()
        self._reg_series = reg_series
        self._lesion_df = patient_lesion_df
        self._lesions_list = self.extract_lesions_list()
        self._matches_set = self.make_matches_set()
        self._sheet_names = \
            [MatchAttr.CENTROID_DIST, MatchAttr.NUM_OVERLAP_VOXELS, MatchAttr.VOL_DIFFERENCE] #, MatchAttr.DICE]
        self._dfs = self.initialize_df()

    def make_matches_set(self):
        matches_set = set()
        for l0 in self._lesions_list:
            for l1 in self._lesions_list:
                if l0.layer() < l1.layer():
                    matches_set.add(Match(l0, l1))
        return matches_set

    def extract_lesions_list(self):
        lesions_names = list(self._lesion_df.index)
        return [Lesion(lb_layer=les_name) for les_name in lesions_names]

    def initialize_df(self):
        """Return a dict {attr_name: dataframe}"""
        row_col_names = [l.name() for l in self._lesions_list]
        n_row_col = len(row_col_names)
        empty_data = [[None] * n_row_col] * n_row_col
        return {sheet_name: pd.DataFrame(data=empty_data, columns=row_col_names, index=row_col_names)
                for sheet_name in self._sheet_names}

    def create(self):
        self.calculate_from_lesion_df()
        self.calculate_from_scans()
        for match in self._matches_set:
            l0_name, l1_name = match.get_lesions_names()
            for attr in self._sheet_names:
                self._dfs[attr].loc[l0_name, l1_name] = match.get(attr)
        return self._dfs

    def calculate_from_lesion_df(self):
        for match in self._matches_set:
            l0_name, l1_name = match.get_lesions_names()
            try:
                l0_cntr = self.extract_np_array_from_cell(self._lesion_df.loc[l0_name, LesionAttr.REG_CENTROID])
                l1_cntr = self.extract_np_array_from_cell(self._lesion_df.loc[l1_name, LesionAttr.REG_CENTROID])
            except:
                a = 1
            l0_vol = self._lesion_df.loc[l0_name, LesionAttr.VOLUME]
            l1_vol = self._lesion_df.loc[l1_name, LesionAttr.VOLUME]
            match.set(MatchAttr.CENTROID_DIST, np.linalg.norm(l0_cntr - l1_cntr, ord=2))
            match.set(MatchAttr.VOL_DIFFERENCE, l1_vol - l0_vol)

    def calculate_from_scans(self):
        n_layers = len(self._reg_series)
        reg_series_tensor = np.stack(self._reg_series, axis=-1)
        matching_tensor = np.reshape(reg_series_tensor, (np.product(reg_series_tensor.shape[0:-1]), n_layers)) # dim (x*y*z, n_img)
        unique_matchings, matching_cnt = np.unique(matching_tensor, axis=0, return_counts=True)
        #matching_tensor_list = list(matching_tensor)
        #matching2count = {m: matching_tensor_list.count(m) for m in unique_matching}
        pairmatch2count = {m: 0 for m in self._matches_set}
        for matching_ind, matching in enumerate(unique_matchings):
            m_names = [Lesion(lb, layer) for layer, lb in enumerate(matching) if lb != 0]
            if len(m_names) < 2:
                continue
            les_combinations = combinations(m_names, 2)
            for pairmatch in les_combinations:
                # l0 = Lesion(lb_layer=pairmatch[0])
                # l1 = Lesion(lb_layer=pairmatch[1])
                pairmatch2count[Match(*pairmatch)] += matching_cnt[matching_ind]
        for match in self._matches_set:
            match.set(MatchAttr.NUM_OVERLAP_VOXELS, pairmatch2count[match])

    @staticmethod
    def extract_np_array_from_cell(cell_str: str):
        "cell_str is a string '({val1}, {val2}, ..)' "
        # remove parenthesis
        cell_str = cell_str[1:-1]
        numbers = [int(n) for n in cell_str.split(', ')]
        return np.array(numbers)

                
class TableManagerTester(TableManager):
    def __init__(self):
        table_path = "/cs/usr/bennydv/Desktop"
        print(f"Creating tables in {table_path}")
        patient_list = ['O_O_']
        super().__init__(table_path, patient_list)

    @staticmethod
    def local_load_series(patient_name):
        im = ImagesCreator(show_im=True)
        reg_series = list(im.run(get_image_list=True))
        return reg_series, reg_series, [1]*len(reg_series)

    @staticmethod
    def local_load_reg_series(patient_name):
        im = ImagesCreator(show_im=False)
        reg_series = list(im.run(get_image_list=True))
        return reg_series


if __name__ == "__main__":
    a = 1
    from general_utils import *
    df_dict = pd.read_excel("/cs/casmip/bennydv/lungs_pipeline/lesions_matching/lesion_matching_database_nnunet/lesions_data.xlsx", index_col=0, sheet_name=None)
    df_res = pd.read_excel(f"/cs/casmip/bennydv/nn_unet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Lungs/nnunet_test_measures_3.xlsx", index_col=0, sheet_name='diameter_0')
    pats = get_patients_list()
    pats = [p for p in pats if p!='B_B_S_']
    for pat_i, df in enumerate(df_dict.values()):
        pat_name = pats[pat_i]
        print(pat_name)
        dates = get_patient_dates(pat_name)
        all_ind = [ind.split('_')[1] for ind in df.index]
        df['ind'] = all_ind
        for ind in np.unique(all_ind):
            date = dates[int(ind)]
            fp_les = sum((df['ind'] == ind)&(df[LesionAttr.DETECTION]=='fp'))
            fp_nnunet = df_res.loc[f'{pat_name}-{date}', 'Detection FP']
            del_in_reg = sum((df['ind'] == ind)&(pd.isnull(df[LesionAttr.DETECTION])))
            if fp_nnunet == del_in_reg + fp_les:
                continue
    
            print(f"date={date}, t={int(ind)}: #FP_lesion_data: {sum((df['ind'] == ind)&(df[LesionAttr.DETECTION]=='fp'))},"
                  f" deleted in reg: {del_in_reg}, #FP_nnunet_measure: {df_res.loc[f'{pat_name}-{date}', 'Detection FP']}")
        print("")
    for pat_name in get_patients_list():
        dates = get_patient_dates(pat_name)
        print(pat_name)
        for date in dates:
            #img, _ = load_nifti_data(f"/cs/casmip/bennydv/nn_unet/nnUNet_raw_data_base/nnUNet_raw_data/Task501_Lungs/labelsTs/{pat_name}{date}.nii.gz")
            label_img, _ = load_nifti_data(f"/cs/casmip/bennydv/lungs_pipeline/pred_data/size_filtered/labeled_no_reg/{pat_name}/lesions_pred_{date}.nii.gz")
            #label_img = label(img, connectivity=1)
            label_img = label_img.astype(int)
            cc_num = label_img.max()
            cc_areas = ndimage.sum(label_img>0, label_img, range(cc_num + 1))
            area_mask = (cc_areas != 20)
            label_img[area_mask[label_img]] = 0
            n_20_voxel_pred = len(np.unique(label_img)) - 1
            if n_20_voxel_pred > 1:
                print(f"{date}: #{n_20_voxel_pred} lesions with 20 voxels")




    l0 = Lesion(1, 0)
    l1 = Lesion(1, 1)
    l2 = Lesion(2, 2)
    l3 = Lesion(2, 3)
    
    m1 = Match(l0, l1)
    m2 = Match(l1, l0)
    m3 = Match(l1, l2)
    
    mlist = [m1, m2]
    
    a = 0

    tb = TableManagerTester()
    tb.run()
