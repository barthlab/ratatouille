
import matplotlib.pyplot as plt

from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.plotter.ax_plotter.basic_nodes_plot import node_flat_view
from kitchen.plotter.ax_plotter.basic_datasets_plot import datasets_avg_view
from kitchen.plotter.ax_plotter.basic_nodes_plot import node_avg_view
from kitchen.plotter.ax_plotter.basic_nodes_plot import node_flat_view
from kitchen.plotter.ax_plotter.basic_datasets_plot import datasets_avg_view
from kitchen.utils.sequence_kit import find_only_one

def flat_view(
        ax: plt.Axes,
        datasets: DataSet,
        
        locomotion_flag: bool = True,
        lick_flag: bool = True,
        pupil_flag: bool = True,
        tongue_flag: bool = True,
        whisker_flag: bool = True,
        fluorescence_flag: bool = True,
):
    """Flat view of the only node in the dataset"""
    node = find_only_one(datasets.nodes)
    
    y_offset = 0
        