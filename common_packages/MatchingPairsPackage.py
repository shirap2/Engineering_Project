from common_packages.LongGraphPackage import *
from scipy.ndimage.morphology import distance_transform_edt

class PwMatching:
    def __init__(self, layer2scan_pairs, layer2voxel_spacing, dilate_param):
        """
        Apply match_2_cases_v5 on scan pairs defined by layer2scan_pairs
        :param layer2scan_pairs: a dictionary of {(layer_id_prev, layer_id_curr): (S_reg[layer_id - 1], S[layer_id])},
            where S is np arrays containing the registered or fix scan to match.
        :param layer2voxel_spacing: a dictionary {layer_id: [x, y, z voxel size]}.
            for a series of N scans the dict length will be N-1
        :param dilate_param: a param for the algorithm
        """
        self.layer2scan_pairs = layer2scan_pairs
        self.layer2voxel_spacing = layer2voxel_spacing
        self.dilate_param = dilate_param
        self.n_layers = len(layer2voxel_spacing) + 1
        self.longit_loader = None

    def run(self):
        nodes = []
        nodes_dict = {}
        edges = []
        for layer_ids, scan_pair in self.layer2scan_pairs.items():
            prev_scan = scan_pair[0]
            prev_layer = layer_ids[0]
            curr_scan = scan_pair[1]
            curr_layer = layer_ids[1]
            curr_scan_voxel_space = self.layer2voxel_spacing[curr_layer]
            if prev_layer not in nodes_dict.keys():
                prev_lesions = list(np.unique(prev_scan).astype(int))
                prev_labels = [f"{les}_{prev_layer}" for les in prev_lesions if les != 0]
                nodes_dict.update({prev_layer: prev_labels})
            if curr_layer == self.n_layers - 1 and curr_layer not in nodes_dict.keys():
                last_lesions = list(np.unique(curr_scan).astype(int))
                last_labels = [f"{les}_{self.n_layers - 1}" for les in last_lesions if les != 0]
                nodes_dict.update({self.n_layers - 1: last_labels})

            matches = PwMatching.match_2_cases_v5(baseline_moved_labeled=prev_scan,
                                       followup_labeled=curr_scan,
                                       voxelspacing=curr_scan_voxel_space,
                                       max_dilate_param=self.dilate_param)

            for m in matches:
                n0 = f"{int(m[0])}_{prev_layer}"
                n1 = f"{int(m[1])}_{curr_layer}"
                edges += [[n0, n1]]

        for nodes_list in nodes_dict.values():
            nodes += nodes_list
        self.longit_loader = LoaderSimple(labels_list=nodes, edges_list=edges)
        return self.longit_loader

    def get_longit(self):
        l = Longit(self.longit_loader)
        return l

    @staticmethod
    def expand_labels(label_image, distance=1, voxelspacing=None, distance_cache=None, return_distance_cache=False):
        """

        This function is based on the same named function in skimage.segmentation version 0.18.3

        expand_labels is derived from code that was
        originally part of CellProfiler, code licensed under BSD license.
        Website: http://www.cellprofiler.org

        Copyright (c) 2020 Broad Institute
        All rights reserved.

        Original authors: CellProfiler team


        Expand labels in label image by ``distance`` pixels without overlapping.
        Given a label image, ``expand_labels`` grows label regions (connected components)
        outwards by up to ``distance`` pixels without overflowing into neighboring regions.
        More specifically, each background pixel that is within Euclidean distance
        of <= ``distance`` pixels of a connected component is assigned the label of that
        connected component.
        Where multiple connected components are within ``distance`` pixels of a background
        pixel, the label value of the closest connected component will be assigned (see
        Notes for the case of multiple labels at equal distance).
        Parameters
        ----------
        label_image : ndarray of dtype int
            label image
        distance : float
            Euclidean distance in pixels by which to grow the labels. Default is one.
        voxelspacing : float or sequence of floats, optional
            The voxelspacing in a distance unit i.e. spacing of elements
            along each dimension. If a sequence, must be of length equal to
            the input rank; if a single number, this is used for all axes. If
            not specified, a grid spacing of unity is implied.
        distance_cache : a tuple with 2 ndarrays, optional
            This two ndarrays are distances calculated earlyer to use in the current calculation
            This is used, for example, if you want to run this function several times while changing only
            the ``distance`` parameter. The calculation will be more optimized.
        return_distance_cache : bool, optional
            If this is set to True, the distances cache will be returned too. By default it's False.
            See distance_cache decumentation.
        Returns
        -------
        enlarged_labels : ndarray of dtype int
            Labeled array, where all connected regions have been enlarged
        distance_cache : a tuple with 2 ndarrays
            This will be returned only if return_distance_cache is set to True.
            See distance_cache decumentation.
        Notes
        -----
        Where labels are spaced more than ``distance`` pixels are apart, this is
        equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
        However, in contrast to a morphological dilation, ``expand_labels`` will
        not expand a label region into a neighboring region.
        This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
        it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
        There is an important edge case when a pixel has the same distance to
        multiple regions, as it is not defined which region expands into that
        space. Here, the exact behavior depends on the upstream implementation
        of ``scipy.ndimage.distance_transform_edt``.
        See Also
        --------
        :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
        References
        ----------
        .. [1] https://cellprofiler.org
        .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
        Examples
        --------
        # >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
        # >>> expand_labels(labels, distance=1)
        array([1, 1, 1, 0, 0, 2, 2])
        Labels will not overwrite each other:
        # >>> expand_labels(labels, distance=3)
        array([1, 1, 1, 1, 2, 2, 2])
        In case of ties, behavior is undefined, but currently resolves to the
        label closest to ``(0,) * ndim`` in lexicographical order.
        # >>> labels_tied = np.array([0, 1, 0, 2, 0])
        # >>> expand_labels(labels_tied, 1)
        array([1, 1, 1, 2, 2])
        # >>> labels2d = np.array(
        # ...     [[0, 1, 0, 0],
        # ...      [2, 0, 0, 0],
        # ...      [0, 3, 0, 0]]
        # ... )
        # >>> expand_labels(labels2d, 1)
        array([[2, 1, 1, 0],
               [2, 2, 0, 0],
               [2, 3, 3, 0]])
        """
        if distance_cache is None:
            distances, nearest_label_coords = distance_transform_edt(
                label_image == 0, return_indices=True, sampling=voxelspacing
            )
        else:
            distances, nearest_label_coords = distance_cache
        labels_out = np.zeros_like(label_image)
        dilate_mask = distances <= distance
        # build the coordinates to find nearest labels,
        # in contrast to [1] this implementation supports label arrays
        # of any dimension
        masked_nearest_label_coords = [
            dimension_indices[dilate_mask]
            for dimension_indices in nearest_label_coords
        ]
        nearest_labels = label_image[tuple(masked_nearest_label_coords)]
        labels_out[dilate_mask] = nearest_labels
        if return_distance_cache:
            return labels_out, (distances, nearest_label_coords)
        return labels_out

    @staticmethod
    def match_2_cases_v5(baseline_moved_labeled, followup_labeled, voxelspacing=None, max_dilate_param=5,
                         return_iteration_indicator=False):
        """
        • This version removes the tumors only at the end of the iterations.
        • This version works with minimum python 'for' iterations and mostly with numpy optimizations.
        • This version's match criteria is only the ratio of intersection between tumors, and it doesn't take a maximum
            intersection as a match.
        • This version dilates the images once in the beginning.
        """

        if np.all(baseline_moved_labeled == 0) or np.all(followup_labeled == 0):
            return []

        distance_cache_bl, distance_cache_fu = None, None

        pairs = np.array([]).reshape([0, 3 if return_iteration_indicator else 2])
        # Hyper-parameter for sensitivity of the matching (5 means that 10 pixels between the scans will be same)
        for dilate in range(max_dilate_param):

            # dilation without overlap, and considering resolution
            working_baseline_moved_labeled, distance_cache_bl = PwMatching.expand_labels(baseline_moved_labeled,
                                                                                          distance=dilate + 1,
                                                                                          voxelspacing=voxelspacing,
                                                                                          distance_cache=distance_cache_bl,
                                                                                          return_distance_cache=True)
            working_followup_labeled, distance_cache_fu = PwMatching.expand_labels(followup_labeled, distance=dilate + 1,
                                                                                                    voxelspacing=voxelspacing,
                                                                                                    distance_cache=distance_cache_fu,
                                                                                                    return_distance_cache=True)

            if dilate > 0:
                # zero the BL tumor and the FU tumor in the matches
                working_baseline_moved_labeled[np.isin(working_baseline_moved_labeled, pairs[:, 0])] = 0
                working_followup_labeled[np.isin(working_followup_labeled, pairs[:, 1])] = 0

                if np.all(working_baseline_moved_labeled == 0) or np.all(working_followup_labeled == 0):
                    break

            # find pairs of intersection of tumors
            pairs_of_intersection = np.stack([working_baseline_moved_labeled, working_followup_labeled]).astype(
                np.int16)
            pairs_of_intersection, overlap_vol = np.unique(
                pairs_of_intersection[:, np.all(pairs_of_intersection != 0, axis=0)].T, axis=0, return_counts=True)

            if pairs_of_intersection.size > 0:

                relevant_bl_tumors, relevant_bl_tumors_vol = np.unique(working_baseline_moved_labeled[
                                                                           np.isin(working_baseline_moved_labeled,
                                                                                   pairs_of_intersection[:, 0])],
                                                                       return_counts=True)
                relevant_fu_tumors, relevant_fu_tumors_vol = np.unique(
                    working_followup_labeled[np.isin(working_followup_labeled, pairs_of_intersection[:, 1])],
                    return_counts=True)

                # intersection_matrix_overlap_vol[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j]
                # intersection_matrix_overlap_vol is a matrix in which each row is a bl tumor and each col a fu tumor.
                # mat[i,j]: how many voxel in common the bl lesion i and the fu lesion j have.
                intersection_matrix_overlap_vol = np.zeros((relevant_bl_tumors.size, relevant_fu_tumors.size))
                intersection_matrix_overlap_vol[np.searchsorted(relevant_bl_tumors, pairs_of_intersection[:, 0]),
                                                np.searchsorted(relevant_fu_tumors,
                                                                pairs_of_intersection[:, 1])] = overlap_vol

                # intersection_matrix_overlap_with_bl_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_bl_tumors[i]
                intersection_matrix_overlap_with_bl_ratio = intersection_matrix_overlap_vol / relevant_bl_tumors_vol.reshape(
                    [-1, 1])

                # intersection_matrix_overlap_with_fu_ratio[i,j] = #voxels_in_the_overlap_of_relevant_bl_tumors[i]_and_relevant_fu_tumors[j] / #voxels_in_relevant_fu_tumors[j]
                intersection_matrix_overlap_with_fu_ratio = intersection_matrix_overlap_vol / relevant_fu_tumors_vol.reshape(
                    [1, -1])

                current_pairs_inds = np.array([], dtype=np.int16).reshape([0, 2])
                # take as a match any intersection with a overlap ratio with either bl of fu above 10 percent

                valid_pairs_inds = np.concatenate(
                    [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_bl_ratio > 0.1)).T])
                # if valid_pairs_inds.shape[0]>0: #added
                current_pairs_inds = np.unique(valid_pairs_inds, axis=0)
                # else: #added
                #     current_pairs_inds = np.array([], dtype=np.int64).reshape([0, 2]) #added

                valid_pairs_inds = np.concatenate(
                    [current_pairs_inds, np.stack(np.where(intersection_matrix_overlap_with_fu_ratio > 0.1)).T])
                # if valid_pairs_inds.shape[0]>0: #added
                current_pairs_inds = np.unique(valid_pairs_inds, axis=0)
                # else: #added
                #     current_pairs_inds = np.array([], dtype=np.int64).reshape([0, 2]) #added

                if return_iteration_indicator:
                    current_pairs = np.stack(
                        [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]],
                         np.repeat(dilate + 1, current_pairs_inds.shape[0])]).T
                else:
                    current_pairs = np.stack(
                        [relevant_bl_tumors[current_pairs_inds[:, 0]], relevant_fu_tumors[current_pairs_inds[:, 1]]]).T

                pairs = np.concatenate([pairs, current_pairs])

        if return_iteration_indicator:
            return [(p[2], (p[0], p[1])) for p in pairs]
        return [tuple(p) for p in pairs]


# class PwMatchingAllCombinations(PwMatching):
#     """Apply pw matching on all combinations of pairs. Here layer2scan_pairs will be: {layer_id"""
#     def __init__(self, layer2scan_pairs, layer2voxel_spacing, dilate_param):
#         super().__init__(layer2scan_pairs, layer2voxel_spacing, dilate_param)
#
#     def run(self):
#         nodes = []
#         edges = []
#         for layer_id in range(1, self.n_layers):
#             scan_pair = self.layer2scan_pairs[layer_id]
#             prev_scan = scan_pair[0]
#             curr_scan = scan_pair[1]
#             curr_scan_voxel_space = self.layer2voxel_spacing[layer_id]
#             prev_lesions = list(np.unique(prev_scan).astype(int))
#             prev_labels = [f"{les}_{layer_id - 1}" for les in prev_lesions if les != 0]
#             nodes += prev_labels
#
#             matches = PwMatching.match_2_cases_v5(baseline_moved_labeled=prev_scan,
#                                        followup_labeled=curr_scan,
#                                        voxelspacing=curr_scan_voxel_space,
#                                        max_dilate_param=self.dilate_param)
#
#             for m in matches:
#                 n0 = f"{int(m[0])}_{layer_id - 1}"
#                 n1 = f"{int(m[1])}_{layer_id}"
#                 edges += [[n0, n1]]
#
#         last_scan = self.layer2scan_pairs[self.n_layers - 1][1]
#         last_scan_lesions = list(np.unique(last_scan).astype(int))
#         curr_labels = [f"{les}_{self.n_layers - 1}" for les in last_scan_lesions if les != 0]
#         nodes += curr_labels
#         self.longit_loader = LoaderSimple(labels_list=nodes, edges_list=edges)
#         return self.longit_loader