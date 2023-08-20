#Introduction#
1. This project is made of a lot of classes and sub-classes. Some classes in these scripts are obsolete.
2. This project has two configuration scripts: config.py and general_utils.py.
They are used to store all the variables, paths and general functions. They assume the file organization I used, that is,
basically: "folder/{PATIENT_NAME}/{TYPE_OF_IMAGE}_{DATE1}_{DATE2}.nii.gz". PATIENT_NAME is the patient identifier,
usually patients initials. TYPE_OF_IMAGE is 'scan' or 'lesions_gt' or 'liver' or 'lesion_pred'. {DATE1} is the image date.
If {DATE2} is present it means that the image was taken in {DATE1} and registered to {DATE2}.

- common_packages contains the shared class for lungs, liver and brain.
- lesion_matching instead is the lungs specific implementation (and calls) of these classes. For brain and liver is basically
the same. Sometimes in 'common_packages' you will find abstract functions that are implemented in lesions_matching.

In BaseClasses.py you will find many 'structs' to define parameters and three basic building blocks:
1. Loader: this class takes a list of labels and edges and build and prepare them to be loaded into a networkx Graph
2. Longit: this class wraps the networkx Graph containing the lesion matching graph.
    Longit needs to be initialized with a Loader
3. Drawer: draws a Longit.

You will find along the code many child classes of these three classes.

#Evaluation#
MatchingEvaluationPackage.py
class ReportEvaluation:
        Class with methods for creating an excel table of the evaluation metrics. It creates both patient metrics and
        summary metrics.
        :param evaluation_type: is the type of evaluation you want to calculate. Must be one of the fields of EvaluationTypes"""

        The main loop of this class must be implemented, and it is organ/file organization specific.
        An example can be found in 'lesion_matching'

        - when evaluating predicted lesion matching graph, a mapping should be also provided. The mapping maps predicted
        lesion label to gt lesion labels. The mapping should be computed with map_pred_to_gt.py script (that inherits from ComputedGraphsMapping.py).

LesionChangesAnalysis.py
- class ChangesReport:
    How many Individual Lesion changes of each class, how many Patterns of lesion changes
    of each class


