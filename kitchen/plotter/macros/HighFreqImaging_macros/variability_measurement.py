import numpy as np
import matplotlib.pyplot as plt

from kitchen.calculator.basic_metric import AVERAGE_VALUE, DIFFERENCE
from kitchen.configs import routing
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.ax_plotter.basic_plot import heatmap_view
from kitchen.plotter.decorators.default_decorators import coroutine_cycle, default_exit_save
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.operator.split import split_dataset_by_trial_type



def variability_visualization_of_fluo_and_deconv(
        dataset: DataSet,

        fluo_amp_range: tuple[float, float] = (0, 2),
        spike_cnt_range: tuple[float, float] = (0, 1),
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Trial",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)
    plot_manual_deconv = PlotManual(deconv_fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    puff_evoked_content = {}
    for mice_node in dataset.select("mice"):
        for cell_session_node in dataset.subtree(mice_node).select("cellsession"):
            cs_subtree = dataset.subtree(cell_session_node)
            type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                       plot_manual=plot_manual_fluo,
                                                       _element_trial_level =_element_trial_level,)
            if len(type2dataset) == 0:
                continue

            puff_cs_dataset = sync_nodes(type2dataset["PuffOnly"], alignment_events, plot_manual=plot_manual_fluo)
            puff_evoked_fluo_amplitude = [AVERAGE_VALUE(one_trial.fluorescence.df_f0, fluo_amp_range) for one_trial in puff_cs_dataset]
            puff_evoked_deconv_amplitude = [DIFFERENCE(one_trial.fluorescence.deconv_f, spike_cnt_range, (-1, 0)) for one_trial in puff_cs_dataset]
            puff_evoked_content[cell_session_node] = {
                "fluo": puff_evoked_fluo_amplitude,
                "deconv": puff_evoked_deconv_amplitude,
                "dataset": puff_cs_dataset,
            }


    # fig, axs = plt.subplots(2, len(puff_evoked_content), figsize=(20, 4))
    # coroutines = {}
    # for col_idx, (cell_session_node, cell_session_content_dict) in enumerate(puff_evoked_content.items()):
    #     puff_evoked_fluo_amplitude = cell_session_content_dict["fluo"]
    #     puff_evoked_deconv_amplitude = cell_session_content_dict["deconv"]
    #     puff_cs_dataset = cell_session_content_dict["dataset"]

    #     specific_ploter = heatmap_view(ax=axs[0, col_idx], datasets=puff_cs_dataset, sync_events=alignment_events, 
    #                                             plot_manual=plot_manual_fluo, modality_name="fluorescence")
    #     coroutines[specific_ploter] = f"fluo_{cell_session_node.session_id}"

    #     specific_ploter = heatmap_view(ax=axs[1, col_idx], datasets=puff_cs_dataset, sync_events=alignment_events, 
    #                                             plot_manual=plot_manual_deconv, modality_name="deconv_fluorescence")
    #     coroutines[specific_ploter] = f"deconv_{cell_session_node.session_id}"

    # coroutine_cycle(coroutines)
    # for ax in axs.flatten():
    #     ax.tick_params(axis='y', labelleft=False)
    #     ax.tick_params(axis='x', labeltop=False)
    #     ax.set_yticks([])
    #     ax.set_xlim(-1, 5)
    #     ax.set_xticks([])

    # fig.subplots_adjust(left=0.01, bottom=0.05, right=0.99, top=0.95, wspace=0, hspace=0)
        
    # save_path = routing.default_fig_path(dataset, "VariabilityMeasurementOverview.png")
    # default_exit_save(fig, save_path)

    
    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(4, 2))
    all_fluo = np.concatenate([cell_session_content_dict["fluo"] 
                                for cell_session_content_dict in puff_evoked_content.values()])
    x_range = (np.percentile(all_fluo, 1), np.percentile(all_fluo, 99))
    xs = np.linspace(*x_range, 500)
    axs[0].hist(all_fluo, bins=np.linspace(*x_range, 50), density=True, alpha=0.5,)
    kde = gaussian_kde(all_fluo)
    axs[0].plot(xs, kde(xs), linewidth=1)
    axs[0].set_xlim(*x_range)
    axs[0].axvline(np.mean(all_fluo), color="red", linestyle="--", linewidth=1)
    axs[0].axvline(0, color="black", linestyle="--", linewidth=1)

    all_deconv = np.concatenate([cell_session_content_dict["deconv"] 
                                for cell_session_content_dict in puff_evoked_content.values()])
    x_range = (np.percentile(all_deconv, 1), np.percentile(all_deconv, 99))
    xs = np.linspace(*x_range, 500)
    axs[1].hist(all_deconv, bins=np.linspace(*x_range, 50), density=True, alpha=0.5,)
    kde = gaussian_kde(all_deconv)
    axs[1].plot(xs, kde(xs), linewidth=1)
    axs[1].set_xlim(*x_range)
    axs[1].axvline(np.mean(all_deconv), color="red", linestyle="--", linewidth=1)
    axs[1].axvline(0, color="black", linestyle="--", linewidth=1)
    plt.show()
    # save_path = routing.default_fig_path(dataset, "VariabilityMeasurementDistribution.png")
    # default_exit_save(fig, save_path)

