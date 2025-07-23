from collections import namedtuple

from kitchen.structure.hierarchical_data_structure import DataSet


PlotManual = namedtuple(
    "PlotManual", 
    [
        "timeline",
        "locomotion",
        "lick", 
        "pupil", 
        "tongue", 
        "whisker", 
        "fluorescence"
    ], 
    defaults=[True, False, False, False, False, False, False]
)


def CHECK_PLOT_MANUAL(dataset: DataSet, plot_manual: PlotManual) -> bool:
    """Check if the plot manual is valid for the dataset"""
    for node in dataset:
        for data_name, data_flag in plot_manual._asdict().items():
            if data_flag and getattr(node.data, data_name) is None:
                return False
    return True
