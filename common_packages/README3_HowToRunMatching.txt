#Introduction#
1. This project is made of a lot of classes and sub-classes. Some classes in these scripts are obsolete.
To know which classes are useful follow the matching algorithm steps.
2. This project has two configuration scripts: config.py and general_utils.py.
They are used to store all the variables, paths and general functions. They assume the file organization I used, that is,
basically: "folder/{PATIENT_NAME}/{TYPE_OF_IMAGE}_{DATE1}_{DATE2}.nii.gz". PATIENT_NAME is the patient identifier,
usually patients initials. TYPE_OF_IMAGE is 'scan' or 'lesions_gt' or 'liver' or 'lesion_pred'. {DATE1} is the image date.
If {DATE2} is present it means that the image was taken in {DATE1} and registered to {DATE2}.

#Matching algorithm#
1. The algo is found in the file 'MatchingAlgo.py' (the class AllPairsMatchingAlgo).
2. To run the algo, compute pairwise registration between all consecutive pairs of scans in the series. For ex: [S1, S2, S3],
You need to perform the following registrations: (A) S1->S2; (B) S2->S3; (C) S1->S3. (A) and (B) are consecutive; (C) is
non-consecutive. Then apply the registration to the lesion masks. We denote Li the Lesion mask of Si.
*** Note that the lesions masks must be labeled BEFORE the registration ***
3. Run MatchingAlgo.py

3. The algo inputs all the pairs of registered and fix lesion masks. For ex: [(L1->L2, L2), (L1->L3, L3), ...]; the algo
gets the parameters r, d, p, as in Shalom's Algo.
*** r is the number of iterations. In the code is called "max dilation" ***
The algo outputs is an object of type Longit (self.longit), that represents the lesion matching graph. Can be saved as json.


The algo is made of two parts: (a) Consecutive Matching ("find_consecutive_edges") - works on consecutive pairs.
(b) Non-consecutive Matching ("find_skip_edges").

ALGO PSEUDOCODE
(written as a table in Benny's thesis)

(a)
For all consecutive pairs i=1,…,N-1 do
Repeat ¬r times
	Dilate by d voxels the lesion segmentations L^i and L^(i+1)
	For all vertices (v_j^i,v_l^(i+1) ) do
	Compute the intersection percentage of their corresponding lesion segmentations (l_j^i,l_l^(i+1)) as (max(|l_j^i∩l_l^(i+1) |)⁄|l_j^i | , (|l_j^i∩l_l^(i+1) |  )⁄(|l_l^(i+1) |)).
	If the lesion segmentation intersection percentage is ≥p then
	Add the consecutive edge e_(j,l)^(i,i+1)= (v_j^i,v_l^(i+1) ) to E_c.
	Remove the lesion segmentations l_j^i,l_l^(i+1) from L^i,L^(i+1), respectively
	If L^i  or L^(i+1) are empty, stop
Return G_C=(V,E_c)

(b)
Initialization: E_NC=∅ be an empty set of non-consecutive edges.
Compute CC={cc_m }, the set of connected components of G_C, by Depth First Search (DFS).
For each cc_m=(V_m,E_m)∈CC compute first_m and last_m, the first and last time-indices of the scans in which cc_m’s lesions appear, first_m=argmin_i {v_j^i∈V_m}; last_m=argmax_i {v_j^i∈V_m}.
Create G_CC=├ (V_cc,E_cc ┤), the connected component directed acyclic graph, initially empty:
	For all cc_m∈CC, add a vertex to V_CC
	For all 〖cc〗_m∈CC,〖cc〗_n∈CC do
        	If  (last_m+1< first_n)  and  (t^(first_n )-t^(last_m )<1.5 years)
The lesions in the two connected cc_m and cc_n appear in non-consecutive scans and the time difference between them is less than 1.5 years
	Then add the edge (cc_m,cc_n ) to E_CC
Create a dictionary D that maps all the ordered pairs of non-consecutive time indices to the candidate lesions for matching.
The keys of D are the set of ordered index pairs {<i,k>: 1≤i+1<k≤N}.
The values are sets of vertices pairs V ̃_(i,k) computed as follows.
For all (cc_m,cc_n )∈E_CC do
	Add  to V ̃_(last_m,first_n )all vertices v_j^(last_m )∈V_m and v_l^(first_n )∈V_n
For all keys <i,k> of D whose value V ̃_(i,k)≠∅ do
	Define candidate lesion masks L ̃^i={l_j^i:v_j^i∈V ̃_(i,k)} and L ̃^k={l_l^k:v_l^k∈V ̃_(i,k)}
	Compute the deformable registration R of scan S^i onto S^k
	Apply R to L ̃^i
	Compute the set of edges〖 E〗_(i,k) between L ̃^i, L ̃^k (lesion pairings) using the consecutive edges method in Table 1.
	Add E_(i,k) to E_NC
Return G=(V,E_C∪E_NC)


