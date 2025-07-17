import os
from collections import namedtuple
from typing import Callable, List, Optional, Tuple
import matplotlib.pyplot as plt
from functools import wraps

from kitchen.configs import routing
from kitchen.structure.hierarchical_data_structure import DataSet, Node


def default_style_single_ax(
        tight_layout: bool = True,    
        invisible_spines: bool = True, 
        xticks_flag: bool = True,
        yticks_flag: bool = False,   
        node_num_flag: bool = True,
        title: str = "", 
        xlabel: str = "", 
        ylabel: str = ""):
    """
    A decorator to default style a Matplotlib plot with a single axis.

    Args:
        tight_layout (bool): Whether to apply tight layout.
        invisible_spines (bool): Whether to make top and right spines invisible.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    def decorator(plot_func: Callable[..., Optional[Tuple[float, float]]]):
        @wraps(plot_func)
        def wrapper(dataset: DataSet | Node, save_name: Optional[str] = None, *args, **kwargs):            
            # Set default font size
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams['font.size'] = 5
            plt.rcParams.update({
                'xtick.labelsize': 5,      # X-axis tick labels
                'ytick.labelsize': 5,      # Y-axis tick labels
                'axes.labelsize': 6,       # X and Y axis labels
                'legend.fontsize': 6,      # Legend font size
                'axes.titlesize': 5,       # Plot title
                'figure.titlesize': 5     # Figure title (suptitle)
            })

            # Create figure and axis
            fig, ax = plt.subplots(1, 1)

            # Set axis labels and title
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            if not xticks_flag:
                ax.set_xticks([])
            if not yticks_flag:
                ax.set_yticks([])

            # Call plotting function
            dataset = dataset if isinstance(dataset, DataSet) else DataSet([dataset])
            figsize = plot_func(ax, dataset, *args, **kwargs)
            if figsize is None:
                plt.close(fig)
                return
            fig.set_size_inches(*figsize)

            # Apply default styles
            if invisible_spines:
                ax.spines[['top', 'right']].set_visible(False)
            if tight_layout:
                fig.tight_layout()
            if node_num_flag:
                ax.text(1, 1, f"n = {len(dataset)}", transform=ax.transAxes, va='top', ha='right')

            # Save or show the figure
            if save_name:
                save_path = routing.smart_path_append(routing.default_fig_path(dataset), save_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=600)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            # Close the figure
            plt.close(fig)
        return wrapper
    return decorator


def default_style_multi_ax(
        tight_layout: bool = True,    
        invisible_spines: bool = True, 
        xticks_flag: bool = True,
        yticks_flag: bool = False,   
        node_num_flag: bool = True,
        xlabel: str = "", 
        ylabel: str = ""):
    """
    A decorator to default style a Matplotlib plot with multiple axes.

    Args:
        tight_layout (bool): Whether to apply tight layout.
        invisible_spines (bool): Whether to make top and right spines invisible.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
    """
    def decorator(plot_func: Callable[..., Optional[Tuple[float, float]]]):
        @wraps(plot_func)
        def wrapper(datasets: DataSet | List[DataSet], save_name: Optional[str] = None, *args, **kwargs):
            # Set default font size
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams['font.size'] = 5
            plt.rcParams.update({
                'xtick.labelsize': 5,      # X-axis tick labels
                'ytick.labelsize': 5,      # Y-axis tick labels
                'axes.labelsize': 6,       # X and Y axis labels
                'legend.fontsize': 6,      # Legend font size
                'axes.titlesize': 5,       # Plot title
                'figure.titlesize': 5      # Figure title (suptitle)
            })


            # Create figure and axis
            if isinstance(datasets, DataSet):
                datasets = [datasets,]
            num_datasets = len(datasets)   
            fig, axs = plt.subplots(1, num_datasets, sharey=True)

            # Set axis labels and title
            for ax in axs:
                ax.tick_params(axis='y', labelleft=True)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                if not xticks_flag:
                    ax.set_xticks([])
                if not yticks_flag:
                    ax.set_yticks([])
                if invisible_spines:
                    ax.spines[['top', 'right']].set_visible(False)

            # Call plotting function
            figsize = plot_func(axs, datasets, *args, **kwargs)
            if figsize is None:
                plt.close(fig)
                return
            fig.set_size_inches(*figsize)

            # Apply default styles
            for ax, dataset in zip(axs, datasets):
                if node_num_flag:
                    ax.text(1, 1, f"n = {len(dataset)}", transform=ax.transAxes, va='top', ha='right')
            if tight_layout:
                fig.tight_layout()

            # Save or show the figure
            if save_name:
                first_nonempty_dataset = next((dataset for dataset in datasets if len(dataset) > 0))
                save_path = routing.smart_path_append(routing.default_fig_path(first_nonempty_dataset), save_name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=200)
                print(f"Plot saved to {save_path}")
            else:
                plt.show()
            
            # Close the figure
            plt.close(fig)
        return wrapper
    return decorator


BBOX = namedtuple("BBOX", ["ymin", "ymax"])

class TraceStacker:
    """
    A decorator that provides vertical offset to plotting functions, 
    allowing them to stack traces vertically.
    """
    def __init__(self, padding_factor: float = 0.1, base_padding: float = 1):
        self.padding_factor = padding_factor
        self.base_padding = base_padding
        self.offset: float = 0.0

    def __call__(self, plot_func: Callable[..., BBOX]):
        """The wrapping logic"""        
        @wraps(plot_func)
        def wrapper(*args, **kwargs):       
            bbox = plot_func(*args, y_offset=self.offset, **kwargs)          
            bbox_height = (bbox.ymax - bbox.ymin) * (1 + self.padding_factor) + self.base_padding
            self.offset += bbox_height
        return wrapper
