
import os.path as path
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging


from kitchen.operator.grouping import AdvancedTimeSeries
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_ClusteringAnalysis import get_putative_labels
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS, COHORT_BRIGHTER_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.potential import GCaMP_kernel
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)


def Visualize_ConvolutedPSTH_InClsuters(weight_tuple, psth_tuple, feature_space_name: str):
    
    psth_matrix, psth_node_names, _ = psth_tuple

    weight_matrix, node_names, cohort_names = weight_tuple
    component_projection = weight_matrix[:, 1:]
    n_components = component_projection.shape[1]
    n_cell, _ = weight_matrix.shape    

    assert np.all(psth_node_names == node_names), f"Node names do not match between PSTH and weight matrix:\nPSTH: {psth_node_names}\nWeight matrix: {node_names}"


    cluster_names, (n_clusters, embedding, normalized_weight_matrix) = get_putative_labels((weight_matrix, node_names, cohort_names))
    for x, y,z in zip(cohort_names, cluster_names, node_names):
        print(x, y, z)
        
    BINSIZE = 5/1000
    FEATURE_RANGE = (-1, 1.5)
    bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    BIN_FS = 1 / BINSIZE

    gcamp_kernel, kernel_t = GCaMP_kernel(BIN_FS)
    putative_df_f0 = np.stack([np.convolve(single_psth, gcamp_kernel, mode='full')[:len(bin_centers)]
                      for single_psth in psth_matrix], axis=0) * BINSIZE  # scale with firing rate unit
    baseline_range = np.searchsorted(bin_centers, (-0.5, 0))
    putative_df_f0 -= putative_df_f0[:, baseline_range[0]:baseline_range[1]].mean(axis=1, keepdims=True)

    calcium_inorder = []
    
    """
    Figure :
    PSTH
    Heatmap
    For each single putative clusters
    """
    from matplotlib.colors import SymLogNorm
    plt.rcParams['font.size'] = 6
    plt.rcParams['font.family'] = 'Arial'
    for cluster_id in range(n_clusters):
        for cohort_group, cohort_list in zip(("SST", "PV", "PYR", "ALL"), 
                                             (["SST_JUX", "SST_WC"], ["PV_JUX",], ["PYR_JUX",], ["SST_JUX", "SST_WC", "PV_JUX", "PYR_JUX"])):
            idx_in_cluster = [i for i, name in enumerate(cluster_names) if name == f"cluster {cluster_id}"]
            idx_in_cohort = [i for i, name in enumerate(cohort_names) if name in cohort_list]
            idx_in_both = list(set(idx_in_cluster).intersection(idx_in_cohort))

            if len(idx_in_both) > 0:
                fig, axs = plt.subplots(3, 1, figsize=(1.2, 2), constrained_layout=True, height_ratios=[1, 1, 1.5])

                ax = axs[0]
                group_psth = AdvancedTimeSeries(
                    t=bin_centers, v=putative_df_f0[idx_in_both, :].mean(axis=0), 
                    variance=putative_df_f0[idx_in_both, :].std(axis=0), 
                    raw_array=putative_df_f0[idx_in_both, :])
                oreo_plot(ax, group_psth, 0, 1,  
                          {"color": CLUSTER_COLORS[f"cluster {cluster_id}"], "lw": 1, "alpha": 1},
                          style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, })
                ax.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
                ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
                ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
                ax.set_xlim(-0.5, 1.5)
                ax.set_ylim(-0.2, 1.8)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axs[1]
                oreo_plot(ax, group_psth, 0, 1,  
                          {"color": CLUSTER_COLORS[f"cluster {cluster_id}"], "lw": 1, "alpha": 1},
                          style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, })
                ax.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
                ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
                ax.axvline(20/1000, color='black', linestyle='--', lw=0.5, alpha=0.5, zorder=5)
                ax.set_xlim(-50/1000, 200/1000 )
                ax.set_ylim(-0.2, 1.8)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axs[2]
                ax.tick_params(
                    length=1,
                    pad=1,
                )
                subset_psth_matrix = putative_df_f0[idx_in_both, :]
                mask = (bin_centers > 0) & (bin_centers < 0.5)
                display_mask = (bin_centers >= -0.5) & (bin_centers <= 1.5)
                sorted_order = np.argsort(np.mean(subset_psth_matrix[:, mask], axis=1),)[::-1]
                sns.heatmap(subset_psth_matrix[sorted_order][:, display_mask],  
                            ax=ax, cbar=False, 
                            cmap='RdYlBu_r', vmin=-1, vmax=1,)

                if cohort_group == "SST":
                    calcium_inorder.append(subset_psth_matrix[sorted_order][:, display_mask])

                desired_ticks = [0, 0.5]
                tick_positions = [np.argmin(np.abs(bin_centers[display_mask] - tick)) for tick in desired_ticks]
                ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

                # if "SST_WC" in cohort_list:
                #     heatmap_cohort_names = [cohort_names[i] for i in np.array(idx_in_both)[sorted_order]]
                #     WC_indices = [i for i, name in enumerate(heatmap_cohort_names) if name == "SST_WC"]
                #     ax.set_yticks(0.5 + np.array(WC_indices), [str(wc_idx+1) for wc_idx in WC_indices], rotation=0)
                # else:
                ax.set_yticks([0.5, len(idx_in_both)-0.5], 
                            ["1", f"{len(idx_in_both)}"], rotation=0)
                
                ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)

                
                save_path = path.join(get_saving_path(), "ClusteringAnalysis_ConvolutedPSTH", 
                                      f"SubCluster_{feature_space_name}_{cohort_group}_cluster{cluster_id}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=500, transparent=True)
                plt.close(fig)
                logger.info("Plot saved to " + save_path)


    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    for ax, dataset_name in zip(axs, ( "SST_JUX", "PV_JUX", "PYR_JUX",) ):    
        ax.tick_params(
            pad=1       # smaller gap between ticks and tick labels
        )
            
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        cohort_indices = [i for i, name in enumerate(cohort_names) if name == dataset_name]
        if dataset_name == "SST_JUX":
            cohort_indices += [i for i, name in enumerate(cohort_names) if name == "SST_WC"]
        sub_psth = psth_matrix[cohort_indices, :]
        sub_matrix = putative_df_f0[cohort_indices, :]
        mask = (bin_centers > 0) & (bin_centers < 0.5)
        sorted_order = np.argsort(np.mean(sub_psth[:, mask], axis=1))
        display_mask = (bin_centers >= -0.5) & (bin_centers <= 1.5)
        sns.heatmap(sub_matrix[sorted_order][:, display_mask],
                    ax=ax, cbar=False,
                    cmap='RdYlBu_r', vmin=-1, vmax=1,)
        
        desired_ticks = [0, 0.5]
        tick_positions = [np.argmin(np.abs(bin_centers[display_mask] - tick)) for tick in desired_ticks]
        ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

        ax.invert_yaxis()
        ax.set_xlabel("Time [s]")
        ax.set_yticks([0.5, len(sub_matrix)-0.5], 
                    [f"{len(sub_matrix)}", "1"], rotation=0)
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)
    fig.set_size_inches(3.5, 1.5)

    save_path = path.join(get_saving_path(), "ClusteringAnalysis_ConvolutedPSTH",  f"PlainHeatmap_PSTH.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    return calcium_inorder