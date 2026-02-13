import os
from typing import Generator, List, Mapping, Tuple, Callable, Any, Optional
from matplotlib import pyplot as plt
import logging

from kitchen.plotter.plotting_params import DPI, NORMALIZED_PROGRESS_YLIM
from kitchen.plotter.utils.setup_plot import apply_plot_settings
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import zip_dicts

logger = logging.getLogger(__name__)


def coroutine_cycle(plot_coroutines: dict[Generator[float, float, None], Any], 
                  default_padding: float = 0.1, overlap_ratio: float = 0.0):
    """ 
    Use coroutines to synchronize the progress from all plot_func
    """
    
    # Prime non-progressive plot functions
    active_coroutines = {}
    for plot_coro, name in plot_coroutines.items():
        try:
            initial_progress = next(plot_coro)
            active_coroutines[plot_coro] = initial_progress
        except StopIteration:
            pass
        except Exception as e:
            raise ValueError(f"Error in {name} at initial progress: {e}")
    
    # Progressive plot functions
    progress = 0
    while active_coroutines:
        plot_heights = list(active_coroutines.values())
        progress += (1-overlap_ratio) * max(plot_heights) + default_padding

        # update the alive coroutines
        next_step_active_coroutines = {}
        for plot_coro, name in active_coroutines.items():
            try:
                next_progress = plot_coro.send(progress)
                next_step_active_coroutines[plot_coro] = next_progress
            except StopIteration:
                pass
            except Exception as e:
                coroutine_name = plot_coroutines[plot_coro].replace("\n", " ")
                raise ValueError(f"Error in {name} at progress {progress} at {coroutine_name}: {e}")
        active_coroutines = next_step_active_coroutines    
    return progress


def default_style(
        mosaic_style,
        content_dict: Mapping[Any, Tuple[Callable[..., Generator[float, float, None]], DataSet | List[DataSet]]],  
        figsize: Tuple[float, float],
        save_path: Optional[str] = None, 

        # style
        sharex: bool = True,
        sharey: bool = True,

        sharex_ticks: bool = True,
        sharey_ticks: bool = True,

        constrained_layout: bool = True,
        invisible_spines: bool = True, 
        xticks_flag: bool = True,
        yticks_flag: bool = False,   
        node_num_flag: bool = True,
        uniform_xlabel: str = "", 
        uniform_ylabel: str = "",
        auto_title: bool = True,
        auto_yscale: bool = False,
        default_padding: float = 0.1,
        overlap_ratio: float = 0.0,

        # Overide style
        plot_settings: Optional[dict[str, dict]] = None,
):
    default_plt_param()

    # Create figure and axis
    fig, axs = plt.subplot_mosaic(
        mosaic_style,
        figsize=figsize, 
        sharex=sharex, sharey=sharey, 
        constrained_layout=constrained_layout)

    # Set axis labels and title
    for label, ax in axs.items():       
        if sharex_ticks:
            ax.tick_params(axis='x', labelbottom=True)
        if sharey_ticks:
            ax.tick_params(axis='y', labelleft=True) 
        ax.set_xlabel(uniform_xlabel)
        ax.set_ylabel(uniform_ylabel)
        if not xticks_flag:
            ax.set_xticks([])
        if not yticks_flag:
            ax.set_yticks([])
        if invisible_spines:
            ax.spines[['top', 'right']].set_visible(False)
        if auto_title:
            ax.set_title(str(label), fontsize=40)
    
    # Main plotting
    plot_coroutines = {plot_func(ax=axs[label], datasets=datasets): label 
                       for label, (plot_func, datasets) in content_dict.items()}
    progress = coroutine_cycle(plot_coroutines, default_padding=default_padding, overlap_ratio=overlap_ratio)

    # Draw node number
    if node_num_flag:
        for _, (ax, (plot_func, datasets)) in zip_dicts(axs, content_dict):
            if isinstance(datasets, DataSet):
                text = f"n = {len(datasets)}"
            else:
                text = f"n = {', '.join(str(len(dataset)) for dataset in datasets)}"
            ax.text(0.99, 0.99, text, transform=ax.transAxes, va='top', ha='right')

    # Adjust figsize
    if auto_yscale:
        fig.set_size_inches(figsize[0], figsize[1] * progress / NORMALIZED_PROGRESS_YLIM)

    # Apply plot settings
    if plot_settings is not None:
        for ax_name, ax_setup_dict in plot_settings.items():
            apply_plot_settings(axs[ax_name], ax_setup_dict)
            
    # Save or show the figure
    default_exit_save(fig, save_path)


def default_plt_param():
    # Set default font size
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 5
    # plt.rcParams.update({
    #     'xtick.labelsize': 5,      # X-axis tick labels
    #     'ytick.labelsize': 5,      # Y-axis tick labels
    #     'axes.labelsize': 6,       # X and Y axis labels
    #     'legend.fontsize': 3,      # Legend font size
    #     'axes.titlesize': 5,       # Plot title
    #     'figure.titlesize': 5,     # Figure title (suptitle)
    #     "lines.linewidth": 0.5,    # Line width
    #     "lines.markersize": 3,     # Marker size
    # })



def default_exit_save(
        fig: plt.Figure,
        save_path: Optional[str] = None, 
        _transparent: bool = False,
):
    """
    Save or show the figure.
    """
    if save_path:        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, transparent=_transparent)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    # Close the figure
    plt.close(fig)