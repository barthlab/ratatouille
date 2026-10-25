from collections import defaultdict
import logging

from matplotlib.colors import Normalize
from pyparsing import Optional
from sklearn.decomposition import TruncatedSVD

from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import AdvancedTimeSeries, calculate_group_tuple, grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.macros.HighFreqImaging_macros.posthoc_staining_analysis import register_staining_info
from kitchen.plotter.macros.HighFreqImaging_macros.sorting_activity import get_celltype_deconv_fluo, get_saving_path2
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_ConvolutionAnalysis import Visualize_ConvolutedPSTH_InClsuters
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_weight_tuple_Decomposition_Weights, get_weight_tuple_PSTH
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.unit_plotter.unit_heatmap import default_ax_realign, label_heatmap_y_ticklabels
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Node
from kitchen.configs.routing import default_data_path, search_pattern_file
from kitchen.structure.neural_data_structure import TimeSeries, TimeSeries_concat
from kitchen.utils import numpy_kit
from kitchen.utils.sequence_kit import find_only_one, select_truthy_items
from kitchen.plotter.utils.tick_labels import add_textonly_legend
from kitchen.utils.ordering_kit import linkage_order


import os.path as path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distinctipy


logger = logging.getLogger(__name__)


plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 6



plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)




def deconv_clustering_analysis(
        _element_trial_level: str = "trial",
        _alignment_style: str = "Aligned2Trial",
        fluo_range: tuple = (0., 1.),
        deconv_range: tuple = (0.3, 0.5),
        _visualize: bool = False,
):

    alignment_events = ALL_ALIGNMENT_STYLE[_alignment_style]
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 6


    fluo_order = {}
    deconv_order = {}

    bin_centers, puff_fluo, blank_fluo, puff_deconv, blank_deconv, red_cohen_d, node_name, day_id = get_celltype_deconv_fluo()

    for single_puff_fluo, single_puff_deconv, single_node_name in zip(puff_fluo, puff_deconv, node_name):
        
        fluo_range_in_frame = np.searchsorted(bin_centers, fluo_range)
        fluo_order[single_node_name] = np.nanmean(single_puff_fluo[fluo_range_in_frame[0]:fluo_range_in_frame[1]])

        deconv_range_in_frame = np.searchsorted(bin_centers, deconv_range)
        deconv_order[single_node_name] = np.nanmean(single_puff_deconv[deconv_range_in_frame[0]:deconv_range_in_frame[1]])


    sorted_fluo_order = sorted(fluo_order.items(), key=lambda x: x[1], reverse=True)
    sorted_deconv_order = sorted(deconv_order.items(), key=lambda x: x[1], reverse=True)
    
    fluo_range_str = f"{fluo_range[0]}-{fluo_range[1]}s"
    deconv_range_str = f"{deconv_range[0]}-{deconv_range[1]}s"

    final_order = sorted_deconv_order

    final_order_index = [list(node_name).index(name) for name, _ in final_order]
    reordered_puff_fluo = puff_fluo[final_order_index, :]
    reordered_blank_fluo = blank_fluo[final_order_index, :]
    reordered_puff_deconv = puff_deconv[final_order_index, :]
    reordered_blank_deconv = blank_deconv[final_order_index, :]
    reordered_red_cohen_d = red_cohen_d[final_order_index]
    reordered_day_id = day_id[final_order_index]
    reordered_node_name = [name for name, _ in final_order]

    red_cohen_d_list_masked = np.ma.masked_invalid(np.array(reordered_red_cohen_d).reshape(-1, 1))
    day_id_list_masked = np.ma.masked_invalid(np.array(reordered_day_id).reshape(-1, 1))

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "black_magenta",
        ["#000000", "gray", "#ff00ff"]
    )
    cmap.set_bad("gray", alpha=0.5)
    norm = Normalize(vmin=0, vmax=2)

    day3_mask = np.where(reordered_day_id == 3)[0]
    day3_puff_fluo = reordered_puff_fluo[day3_mask, :]
    day3_puff_deconv = reordered_puff_deconv[day3_mask, :]
    red_cohen_d_day3 = reordered_red_cohen_d[day3_mask]
    node_name_day3 = [reordered_node_name[i] for i in day3_mask]
    
    cell_colors = cmap(norm(red_cohen_d_day3))


    n_cell, n_timepoints = day3_puff_deconv.shape
    print(f"Performing stability test for SVD with on "
          f"deconvolution data with {n_cell} cells and {n_timepoints} timepoints.")

    select_mask = (bin_centers >= -0.1) & (bin_centers <= .5)
    zscore_day3_puff_deconv = numpy_kit.zscore(day3_puff_deconv[:, select_mask], axis=1)
    focus_window = (bin_centers[select_mask] >= 0.3) & (bin_centers[select_mask] <= 0.5)
    pc1_projection = np.nanmean(zscore_day3_puff_deconv[:, focus_window], axis=1)

    from sklearn.mixture import GaussianMixture
    from scipy.stats import gaussian_kde, norm
    
    N_GMM_COMPONENTS = 3
    X = pc1_projection.reshape(-1, 1)
    gmm = GaussianMixture(
            n_components=N_GMM_COMPONENTS,
            random_state=42,
        )

    gmm.fit(X)
    gmm_labels = gmm.predict(X)    
    gmm_labels = (3 - gmm_labels) % N_GMM_COMPONENTS  # Reorder the labels to match the expected order

    if _visualize:
        grid = np.linspace(pc1_projection.min() - 0.5, pc1_projection.max() + 0.5, 1000)
        fig, ax = plt.subplots(1, 1, figsize=(1., 1.5), constrained_layout=True)
        ax.tick_params(length=1, pad=1)
        rng = np.random.default_rng(seed=42)
        # Plot each GMM component
        for label_id in range(N_GMM_COMPONENTS):
            component_id = (3-label_id) % N_GMM_COMPONENTS  # Reorder the component index to match the label

            cluster = pc1_projection[gmm_labels == label_id]
            mean = gmm.means_[component_id, 0]
            variance = gmm.covariances_[component_id, 0, 0]
            std = np.sqrt(variance)
            weight = gmm.weights_[component_id]

            fitted_gaussian = (weight * norm.pdf(grid, loc=mean, scale=std, ))

            ax.plot(
                fitted_gaussian,
                grid,
                linestyle="--",
                linewidth=1,
                color=CLUSTER_COLORS[label_id],
                label=(
                    f"GMM {label_id}: "
                    f"μ={mean:.2f}, σ={std:.2f}"
                ),
            )
            ax.scatter(
                rng.uniform(-0.05, 0.05, size=len(cluster)), 
                cluster,
                edgecolor=CLUSTER_COLORS[label_id],
                facecolor="white",
                s=6, alpha=0.9
            )
        ax.set_xticks([])
        ax.spines[['right', 'top']].set_visible(False)
        save_path = path.join(get_saving_path2(), f"FeatureSpace", f"deconv_clustering.png")
        default_exit_save(fig, save_path, _transparent=True)


    all_calcium_inorder = []
    for gmm_cluster in range(N_GMM_COMPONENTS):
        select_mask = (bin_centers >= -0.5) & (bin_centers <= 1.5)
        selected_bin_centers = bin_centers[select_mask]
        cluster_node_names = [node_name_day3[i] for i in range(len(node_name_day3)) if gmm_labels[i] == gmm_cluster]
        cluster_fluo = day3_puff_fluo[gmm_labels == gmm_cluster, :][:, select_mask]


        fig, axs = plt.subplots(3, 1, figsize=(1.2, 2), constrained_layout=True, height_ratios=[1, 1, 1.5])

        ax = axs[0]
        group_psth = AdvancedTimeSeries(
            t=selected_bin_centers, v=cluster_fluo.mean(axis=0),
            variance=cluster_fluo.std(axis=0), 
            raw_array=cluster_fluo)
        oreo_plot(ax, group_psth, 0, 1,  
                    {"color": CLUSTER_COLORS[f"cluster {gmm_cluster}"], "lw": 1, "alpha": 1},
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
                    {"color": CLUSTER_COLORS[f"cluster {gmm_cluster}"], "lw": 1, "alpha": 1},
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
        
        mask = (selected_bin_centers > 0) & (selected_bin_centers < 0.5)
        display_mask = (selected_bin_centers >= -0.5) & (selected_bin_centers <= 1.5)
        sorted_order = np.argsort(np.mean(cluster_fluo[:, mask], axis=1),)[::-1]
        sns.heatmap(cluster_fluo[sorted_order][:, display_mask],
                    ax=ax, cbar=False, 
                    cmap='RdYlBu_r', vmin=-1, vmax=1,)

        
        all_calcium_inorder.append(cluster_fluo[sorted_order][:, display_mask])

        desired_ticks = [0, 0.5]
        tick_positions = [np.argmin(np.abs(selected_bin_centers[display_mask] - tick)) for tick in desired_ticks]
        ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

        # if "SST_WC" in cohort_list:
        #     heatmap_cohort_names = [cohort_names[i] for i in np.array(idx_in_both)[sorted_order]]
        #     WC_indices = [i for i, name in enumerate(heatmap_cohort_names) if name == "SST_WC"]
        #     ax.set_yticks(0.5 + np.array(WC_indices), [str(wc_idx+1) for wc_idx in WC_indices], rotation=0)
        # else:
        ax.set_yticks([0.5, len(cluster_fluo)-0.5], 
                    ["1", f"{len(cluster_fluo)}"], rotation=0)
        
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)

        
        save_path = path.join(get_saving_path2(), "ClusteringAnalysis_GMM", 
                                f"SubCluster_cluster{gmm_cluster}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)


    #  Grand Correlation
    PSTH_tuple = get_weight_tuple_PSTH("raw", 5/1000)    
    svd_5_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", 5, True, _visualize=False)
    juxta_calcium_in_order = Visualize_ConvolutedPSTH_InClsuters(svd_5_tuple, PSTH_tuple, "SVD5")

    calcium_in_order_size = [len(cluster) for cluster in all_calcium_inorder]
    juxta_calcium_in_order_size = [len(cluster) for cluster in juxta_calcium_in_order]
    calcium_in_order_boundary = np.cumsum(calcium_in_order_size)[:-1]
    juxta_calcium_in_order_boundary = np.cumsum(juxta_calcium_in_order_size)[:-1]

    calcium_in_order_total_matrix = np.concatenate(all_calcium_inorder, axis=0)
    juxta_calcium_in_order_total_matrix = np.concatenate(juxta_calcium_in_order, axis=0)  # doube fs
    juxta_calcium_in_order_total_matrix = (juxta_calcium_in_order_total_matrix[:, 1::2] + \
                                           juxta_calcium_in_order_total_matrix[:, 0::2]) / 2  # downsample to match the fs of calcium_in_order_total_matrix

    print(f"Calcium in order total matrix shape: {calcium_in_order_total_matrix.shape}, boundaries: {calcium_in_order_boundary}")
    print(f"Juxta calcium in order total matrix shape: {juxta_calcium_in_order_total_matrix.shape}, boundaries: {juxta_calcium_in_order_boundary}")
    C = np.corrcoef(calcium_in_order_total_matrix, juxta_calcium_in_order_total_matrix)
    correlation_matrix = C[:juxta_calcium_in_order_total_matrix.shape[0], juxta_calcium_in_order_total_matrix.shape[0]:]


    x_starts = np.concatenate([[0], np.cumsum(calcium_in_order_size)[:-1]])
    x_centers = x_starts + np.asarray(calcium_in_order_size) / 2

    y_starts = np.concatenate([[0], np.cumsum(juxta_calcium_in_order_size)[:-1]])
    y_centers = y_starts + np.asarray(juxta_calcium_in_order_size) / 2

                                      
    fig, ax = plt.subplots(
        1, 1,
        figsize=(1.7, 1.5),
        constrained_layout=True,
    )

    im = ax.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        interpolation="none",
        # aspect="equal",
        rasterized=True,
    )

    for boundary in calcium_in_order_boundary:
        position = boundary - 0.5
        ax.axvline(
            position,
            color="white",
            linewidth=0.5,
            ls="--"
        )
    for boundary in juxta_calcium_in_order_boundary:
        position = boundary - 0.5
        ax.axhline(
            position,
            color="white",
            linewidth=0.5,
            ls="--"
        )

    # ------------------------------------------------------------
    # Cohort labels instead of individual cell labels
    # ------------------------------------------------------------
    ax.set_xticks(x_centers)
    ax.set_yticks(y_centers)

    # ax.set_xticklabels(
    #     ["SST", "PV", "PYR"],
    #     fontsize=7,
    # )
    # for label, color in zip(
    #     ax.get_xticklabels(),
    #     [COHORT_COLORS["SST_JUX"],
    #         COHORT_COLORS["PV_JUX"],
    #         COHORT_COLORS["PYR_JUX"],]
    # ):
    #     label.set_color(color)

    # ax.set_yticklabels(
    #     ["SST", "PV", "PYR"],
    #     fontsize=7,
    #     rotation=90,
    #     va="center", ha="right"
    # )
    # for label, color in zip(
    #     ax.get_yticklabels(),
    #     [COHORT_COLORS["SST_JUX"],
    #         COHORT_COLORS["PV_JUX"],
    #         COHORT_COLORS["PYR_JUX"],]
    # ):
    #     label.set_color(color)

    ax.tick_params(
        length=0,
        pad=1,
    )

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    save_path = path.join(get_saving_path2(), "ClusteringAnalysis_GMM", 
                            f"GrandCorrelation.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)