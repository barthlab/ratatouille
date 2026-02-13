from collections import namedtuple

from kitchen.structure.hierarchical_data_structure import DataSet


PlotManual = namedtuple(
    "PlotManual", 
    [
        "timeline",

        "locomotion",
        "position",

        "lick", 

        "pupil", 
        "saccade",

        "tongue", 
        "nose",
        "whisker", 

        "fluorescence",
        "deconv_fluorescence",

        "potential",
        "potential_conv",

        "baseline_subtraction",
        "amplitude_sorting",

    ], 
    defaults=[True, False, False, False, False, False, False, False, False, False, False, False, False,
              
              None, None]
)

PLOT_MANUAL_NAME_TO_CHECK = (
    "timeline",
    "locomotion",
    "position",
    "lick", 
    "pupil", 
    "tongue", 
    "nose",
    "whisker", 
    "fluorescence",
    "potential",
    "potential_conv",
)

def CHECK_PLOT_MANUAL(dataset: DataSet, plot_manual: PlotManual) -> bool:
    """Check if the plot manual is valid for the dataset"""
    for node in dataset:
        for data_name, data_flag in plot_manual._asdict().items():
            if data_name in PLOT_MANUAL_NAME_TO_CHECK and data_flag and getattr(node.data, data_name) is None:
                return False
    return True
