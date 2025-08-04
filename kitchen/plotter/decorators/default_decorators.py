import os
from typing import Generator, List, Mapping, Tuple, Callable, Any, Optional
from matplotlib import pyplot as plt

from kitchen.plotter.plotting_params import DPI, NORMALIZED_PROGRESS_YLIM
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.utils.sequence_kit import zip_dicts


def default_style(
        mosaic_style,
        content_dict: Mapping[Any, Tuple[Callable[..., Generator[float, float, None]], DataSet | List[DataSet]]],  
        figsize: Tuple[float, float],
        save_path: Optional[str] = None, 

        # style
        sharex: bool = True,
        sharey: bool = True,
        constrained_layout: bool = True,
        invisible_spines: bool = True, 
        xticks_flag: bool = True,
        yticks_flag: bool = False,   
        node_num_flag: bool = True,
        uniform_xlabel: str = "", 
        uniform_ylabel: str = "",
        auto_title: bool = True,
        auto_yscale: bool = True,
        default_padding: float = 0.5,
):
    # Set default font size
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 3
    plt.rcParams.update({
        'xtick.labelsize': 5,      # X-axis tick labels
        'ytick.labelsize': 5,      # Y-axis tick labels
        'axes.labelsize': 6,       # X and Y axis labels
        'legend.fontsize': 3,      # Legend font size
        'axes.titlesize': 5,       # Plot title
        'figure.titlesize': 5      # Figure title (suptitle)
    })

    # Create figure and axis
    fig, axs = plt.subplot_mosaic(
        mosaic_style,
        figsize=figsize, 
        sharex=sharex, sharey=sharey, 
        constrained_layout=constrained_layout)

    # Set axis labels and title
    for label, ax in axs.items():       
        if sharex:
            ax.tick_params(axis='x', labelbottom=True)
        if sharey:
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
            ax.set_title(str(label))
    
    # Main plotting
    """ 
    Use coroutines to synchronize the progress from all plot_func
    """
    plot_coroutines = {plot_func(ax=axs[label], datasets=datasets): label 
                       for label, (plot_func, datasets) in content_dict.items()}
    
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
        progress += max(plot_heights) + default_padding

        # update the alive coroutines
        next_step_active_coroutines = {}
        for plot_coro, name in active_coroutines.items():
            try:
                next_progress = plot_coro.send(progress)
                next_step_active_coroutines[plot_coro] = next_progress
            except StopIteration:
                pass
            except Exception as e:
                raise ValueError(f"Error in {name} at progress {progress}: {e}")
        active_coroutines = next_step_active_coroutines    
    
    # Draw node number
    if node_num_flag:
        for _, (ax, (plot_func, datasets)) in zip_dicts(axs, content_dict):
            if isinstance(datasets, DataSet):
                text = f"n = {len(datasets)}"
            else:
                text = f"n = {', '.join(str(len(dataset)) for dataset in datasets)}"
            ax.text(1, 1, text, transform=ax.transAxes, va='top', ha='right')

    # Adjust figsize
    if auto_yscale:
        fig.set_size_inches(figsize[0], figsize[1] * progress / NORMALIZED_PROGRESS_YLIM)

    # Save or show the figure
    if save_path:        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    # Close the figure
    plt.close(fig)


