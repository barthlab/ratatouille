import logging
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os.path as path
import os
import pandas as pd
import _pickle as pkl
from scipy.stats import spearmanr

from kitchen.calculator import calculate_physiology
from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.grouping import AdvancedTimeSeries, grouping_timeseries
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SPVtypes import get_grand_cluster_result
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.utils.ordering_kit import linkage_order
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_BRIGHTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import get_100ms_puff_trials, get_500ms_puff_trials
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_PSTH_mean, get_all_cellsession_Waveform_mean, get_saving_path
from kitchen.settings.potential import NARROW_SPIKE_RANGE_FOR_VISUALIZATION, SPIKE_ANNOTATION_EARLY_WINDOW
from kitchen.structure.neural_data_structure import TimeSeries
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)



def corr_matrix_compare_modality(physiology_matrix, physiology_node_names, svd_matrix, svd_node_names):

    import matplotlib.pyplot as plt    
    import seaborn as sns

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6


    cohort_sizes = (27, 36, 11)
    boundaries = (27, 63)
    
    physiology_names = list(physiology_node_names)
    aligned_physiology_order = [
        physiology_names.index(name) for name in svd_node_names
    ]
    physiology_matrix_order_by_svd = physiology_matrix[
        np.ix_(aligned_physiology_order, aligned_physiology_order)
    ]
    mixed_matrix_order_by_svd = np.triu(svd_matrix, 1) + np.tril(physiology_matrix_order_by_svd, 0)


    svd_names = list(svd_node_names)
    aligned_svd_order = [
        svd_names.index(name) for name in physiology_names if name in svd_names
    ]
    svd_matrix_order_by_physiology = svd_matrix[
        np.ix_(aligned_svd_order, aligned_svd_order)
    ]
    physiology_valid_order = [i for i, name in enumerate(physiology_names) if name in svd_names]
    physiology_matrix_valid = physiology_matrix[np.ix_(physiology_valid_order, physiology_valid_order)]
    mixed_matrix_order_by_physiology = np.triu(svd_matrix_order_by_physiology, 1) + np.tril(physiology_matrix_valid, 0)


    starts = np.concatenate([[0], np.cumsum(cohort_sizes)[:-1]])
    centers = starts + (np.asarray(cohort_sizes) - 1) / 2

    for postfix, mixed_matrix in [("order_by_svd", mixed_matrix_order_by_svd), ("order_by_physiology", mixed_matrix_order_by_physiology)]:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(1.7, 1.5),
            constrained_layout=True,
        )

        im = ax.imshow(
            mixed_matrix,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            interpolation="none",
            aspect="equal",
            rasterized=True,
        )

        # Thin white lines separating cohorts
        # -0.5 aligns lines exactly between matrix pixels
        for boundary in boundaries:
            position = boundary - 0.5

            ax.axhline(
                position,
                color="white",
                linewidth=0.5,
                ls="--"
            )
            ax.axvline(
                position,
                color="white",
                linewidth=0.5,
                ls="--"
            )

        # ------------------------------------------------------------
        # Cohort labels instead of individual cell labels
        # ------------------------------------------------------------
        ax.set_xticks(centers)
        ax.set_yticks(centers)

        ax.set_xticklabels(
            ["PV", "SST", "PYR"],
            fontsize=7,
        )
        for label, color in zip(
            ax.get_xticklabels(),
            [COHORT_COLORS["PV_JUX"],
                COHORT_COLORS["SST_JUX"],
                COHORT_COLORS["PYR_JUX"],]
        ):
            label.set_color(color)

        ax.set_yticklabels(
            ["PV", "SST", "PYR"],
            fontsize=7,
            rotation=-90,
            va="center", ha="right"
        )
        for label, color in zip(
            ax.get_yticklabels(),
            [COHORT_COLORS["PV_JUX"],
                COHORT_COLORS["SST_JUX"],
                COHORT_COLORS["PYR_JUX"],]
        ):
            label.set_color(color)

        ax.tick_params(
            length=1,
            pad=1,
        )

        save_path = path.join(get_saving_path(), "CompareModality", f"MixedMatrix_{postfix}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)

    # Compare each unique cell pair using the same SVD ordering in both matrices.
    rows, cols = np.triu_indices(len(svd_matrix), k=1)
    cohorts = np.repeat(["PV", "SST", "PYR"], cohort_sizes)
    pair_groups = np.where(cohorts[rows] == cohorts[cols], cohorts[rows], "Cross-cohort")

    fig, ax = plt.subplots(figsize=(1.3, 1.3), constrained_layout=True)
    for group in ["Cross-cohort", "PV", "SST", "PYR"]:
        mask = pair_groups == group
        color = "gray" if group == "Cross-cohort" else COHORT_BRIGHTER_COLORS[group + "_JUX"]
        ax.scatter(physiology_matrix_order_by_svd[rows[mask], cols[mask]],
                   svd_matrix[rows[mask], cols[mask]],
                   facecolors=color,  
                   s=4, alpha=0.7, edgecolors="none", clip_on=False)
        spearman_corr = spearmanr(physiology_matrix_order_by_svd[rows[mask], cols[mask]],
                                  svd_matrix[rows[mask], cols[mask]])[0]
        print(f"{group} spearman correlation: {spearman_corr}")
    ax.set_xlabel("Physiology cosine similarity", labelpad=1)
    ax.set_ylabel("SVD cosine similarity", labelpad=1)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1], ["-1", "0", "1"], rotation=90)
    ax.spines[["top", "right",]].set_visible(False)
    ax.tick_params(length=1, pad=1)
    save_path = path.join(get_saving_path(), "CompareModality", "PairwiseSimilarity_scatter.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)


    fig, ax = plt.subplots(figsize=(1.2, 0.9), constrained_layout=True)
    for group_id, group in enumerate(["SST", "PV", "PYR", "Cross-cohort"]):
        mask = pair_groups == group
        color = "gray" if group == "Cross-cohort" else COHORT_COLORS[group + "_JUX"]
        avg_corr_phys = np.mean(physiology_matrix_order_by_svd[rows[mask], cols[mask]])
        avg_corr_svd = np.mean(svd_matrix[rows[mask], cols[mask]])
        ax.bar(group_id - 0.18, avg_corr_phys, width=0.32, facecolor=color,  edgecolor=color, linewidth=0.5,
               zorder=3)
        ax.bar(group_id + 0.18, avg_corr_svd, width=0.32, facecolor="none",
               edgecolor=color, linewidth=0.5, zorder=3,)
    ax.set_ylabel("Mean similarity", labelpad=1)
    ax.set_xlim(-0.55, 3.55)
    ax.axhline(0, color="black", ls=':', alpha=0.5, linewidth=0.5, zorder=-1)
    ax.set_axisbelow(True)
    ax.set_yticks([-0.3, 0, 0.3, 0.6])
    for y in (-0.3, 0, 0.3, 0.6):
        ax.axhline(y, color="gray", ls='--', alpha=0.5, linewidth=0.5, zorder=-1)
    ax.set_xticks([0, 1, 2, 3], ["SST", "PV", "PYR", "C.C."])
    for ticklabel, tickcolor in zip(ax.get_xticklabels(), 
                                    [COHORT_COLORS["SST_JUX"], COHORT_COLORS["PV_JUX"], COHORT_COLORS["PYR_JUX"], "gray"]):
        ticklabel.set_color(tickcolor)
        
    ax.spines[["top", "right",]].set_visible(False)
    ax.tick_params(length=1, pad=1)
    save_path = path.join(get_saving_path(), "CompareModality", "PairwiseSimilarity_bar_avg_corr.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    fig, axs = plt.subplots(1, 4, figsize=(3.2, 0.9), sharex='all', sharey='all', constrained_layout=True)
    for group_id, group in enumerate(["SST", "PV", "PYR", "Cross-cohort"]):
        mask = pair_groups == group
        color = "gray" if group == "Cross-cohort" else COHORT_COLORS[group + "_JUX"]
        values_phys = physiology_matrix_order_by_svd[rows[mask], cols[mask]].flatten()
        values_svd = svd_matrix[rows[mask], cols[mask]].flatten()
        axs[group_id].plot(np.sort(values_phys), np.arange(1, len(values_phys) + 1)/len(values_phys),
                 color =color, linewidth=1, zorder=3)
        axs[group_id].plot(np.sort(values_svd), np.arange(1, len(values_svd) + 1)/len(values_svd),
                 color =color, linewidth=1, zorder=3, ls='--')
        axs[group_id].set_xticks([-1, 0, 1])
    axs[0].set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])

    for ax in axs:
        ax.spines[["top", "right",]].set_visible(False)
        ax.tick_params(length=1, pad=1)
    
    save_path = path.join(get_saving_path(), "CompareModality", f"PairwiseSimilarity_CD.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    # plt.close(fig)
    logger.info("Plot saved to " + save_path)

    for group_id, group in enumerate(["SST", "PV", "PYR", "Cross-cohort"]):
        mask = pair_groups == group
        color = "gray" if group == "Cross-cohort" else COHORT_COLORS[group + "_JUX"]
        values_phys = physiology_matrix_order_by_svd[rows[mask], cols[mask]].flatten()
        values_svd = svd_matrix[rows[mask], cols[mask]].flatten()
        basic_ttest.stats_pairwise_ks(
            data={
                "Physiology": values_phys,
                "SVD": values_svd
            },
            ax=axs[group_id],
            bon_ferroni_adj=4
        )
    save_path = path.join(get_saving_path(), "CompareModality", f"PairwiseSimilarity_CD_annotated.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def corr_matrix_compare_modality_by_clusters(physiology_matrix, physiology_node_names, svd_matrix, svd_node_names):
    """Compare modalities for shared cells, grouped by clusters 1 through 6."""

    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6

    node2label, _ = get_grand_cluster_result()
    cluster_ids = list(range(1, 7))
    physiology_names = list(physiology_node_names)
    svd_names = list(svd_node_names)
    shared_names = set(physiology_names) & set(svd_names) & set(node2label)
    if not shared_names:
        raise ValueError("No cells with cluster labels are shared by both modalities.")

    # Stable sorting preserves each modality's cell order within each cluster.
    svd_names_by_cluster = sorted(
        [name for name in svd_names if name in shared_names],
        key=lambda name: node2label[name],
    )
    physiology_names_by_cluster = sorted(
        [name for name in physiology_names if name in shared_names],
        key=lambda name: node2label[name],
    )
    aligned_physiology_order = [physiology_names.index(name) for name in svd_names_by_cluster]
    aligned_svd_order = [svd_names.index(name) for name in svd_names_by_cluster]
    physiology_matrix_by_cluster = physiology_matrix[
        np.ix_(aligned_physiology_order, aligned_physiology_order)
    ]
    svd_matrix_by_cluster = svd_matrix[np.ix_(aligned_svd_order, aligned_svd_order)]
    mixed_matrix_order_by_svd = (
        np.triu(svd_matrix_by_cluster, 1) + np.tril(physiology_matrix_by_cluster, 0)
    )

    physiology_order = [svd_names_by_cluster.index(name) for name in physiology_names_by_cluster]
    mixed_matrix_order_by_physiology = (
        np.triu(svd_matrix_by_cluster[np.ix_(physiology_order, physiology_order)], 1)
        + np.tril(physiology_matrix_by_cluster[np.ix_(physiology_order, physiology_order)], 0)
    )

    clusters = np.array([node2label[name] for name in svd_names_by_cluster])
    boundaries = np.flatnonzero(clusters[1:] != clusters[:-1]) + 1
    present_clusters = np.unique(clusters)
    centers = [np.flatnonzero(clusters == cluster_id).mean() for cluster_id in present_clusters]
    save_dir = path.join(get_saving_path(), "CompareModality", "by_clusters")
    os.makedirs(save_dir, exist_ok=True)

    for postfix, mixed_matrix in [("order_by_svd", mixed_matrix_order_by_svd),
                                  ("order_by_physiology", mixed_matrix_order_by_physiology)]:
        fig, ax = plt.subplots(figsize=(1.7, 1.5), constrained_layout=True)
        ax.imshow(mixed_matrix, cmap="coolwarm", vmin=-1, vmax=1,
                  interpolation="none", aspect="equal", rasterized=True)

        for boundary in boundaries:
            ax.axhline(boundary - 0.5, color="white", linewidth=0.5, ls="--")
            ax.axvline(boundary - 0.5, color="white", linewidth=0.5, ls="--")

        ax.set_xticks(centers, [str(c) for c in present_clusters], fontsize=7)
        ax.set_yticks(centers, [str(c) for c in present_clusters], fontsize=7,
                      rotation=-90, va="center", ha="right")
        for labels in [ax.get_xticklabels(), ax.get_yticklabels()]:
            for label, cluster_id in zip(labels, present_clusters):
                label.set_color(CLUSTER_COLORS[cluster_id])
        ax.tick_params(length=1, pad=1)

        save_path = path.join(save_dir, f"MixedMatrix_{postfix}.png")
        fig.savefig(save_path, dpi=500, transparent=True)
        plt.close(fig)
        logger.info("Plot saved to " + save_path)

    # Compare unique cell pairs with the same cell ordering in both modalities.
    rows, cols = np.triu_indices(len(clusters), k=1)
    # Zero denotes pairs whose cells belong to different clusters.
    pair_groups = np.where(clusters[rows] == clusters[cols], clusters[rows], 0)
    groups = cluster_ids + [0]
    group_colors = [CLUSTER_COLORS[c] for c in cluster_ids] + ["gray"]
    group_labels = [str(c) for c in cluster_ids] + ["C.C."]

    fig, ax = plt.subplots(figsize=(1.3, 1.3), constrained_layout=True)
    for group in [0] + cluster_ids:
        mask = pair_groups == group
        color = "gray" if group == 0 else CLUSTER_COLORS[group]
        values_phys = physiology_matrix_by_cluster[rows[mask], cols[mask]]
        values_svd = svd_matrix_by_cluster[rows[mask], cols[mask]]
        ax.scatter(values_phys, values_svd, facecolors=color,
                   s=4, alpha=0.7, edgecolors="none", clip_on=False)
        spearman_corr = spearmanr(values_phys, values_svd)[0] if len(values_phys) > 1 else np.nan
        group_name = "Cross-cluster" if group == 0 else f"Cluster {group}"
        print(f"{group_name} spearman correlation: {spearman_corr}")
    ax.set_xlabel("Physiology cosine similarity", labelpad=1)
    ax.set_ylabel("SVD cosine similarity", labelpad=1)
    ax.set_xlim(-1, 1)
    ax.set_xticks([-1, 0, 1])
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1], ["-1", "0", "1"], rotation=90)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=1, pad=1)
    save_path = path.join(save_dir, "PairwiseSimilarity_scatter.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    fig, ax = plt.subplots(figsize=(2.1, 0.9), constrained_layout=True)
    for group_id, (group, color) in enumerate(zip(groups, group_colors)):
        mask = pair_groups == group
        if not np.any(mask):
            continue
        avg_corr_phys = np.mean(physiology_matrix_by_cluster[rows[mask], cols[mask]])
        avg_corr_svd = np.mean(svd_matrix_by_cluster[rows[mask], cols[mask]])
        ax.bar(group_id - 0.18, avg_corr_phys, width=0.32, facecolor=color,
               edgecolor=color, linewidth=0.5, zorder=3)
        ax.bar(group_id + 0.18, avg_corr_svd, width=0.32, facecolor="none",
               edgecolor=color, linewidth=0.5, zorder=3)
    ax.set_ylabel("Mean similarity", labelpad=1)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.axhline(0, color="black", ls=":", alpha=0.5, linewidth=0.5, zorder=-1)
    ax.set_axisbelow(True)
    ax.set_yticks([-0.3, 0, 0.3, 0.6])
    for y in (-0.3, 0, 0.3, 0.6):
        ax.axhline(y, color="gray", ls="--", alpha=0.5, linewidth=0.5, zorder=-1)
    ax.set_xticks(range(len(groups)), group_labels)
    for ticklabel, color in zip(ax.get_xticklabels(), group_colors):
        ticklabel.set_color(color)
        ticklabel.set_fontweight("bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=1, pad=1)
    save_path = path.join(save_dir, "PairwiseSimilarity_bar_avg_corr.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)

    fig, axs = plt.subplots(1, len(groups), figsize=(2.3, 0.35),
                            sharex="all", sharey="all", constrained_layout=True)
    for group_id, (group, color) in enumerate(zip(groups, group_colors)):
        mask = pair_groups == group
        values_phys = physiology_matrix_by_cluster[rows[mask], cols[mask]]
        values_svd = svd_matrix_by_cluster[rows[mask], cols[mask]]
        if len(values_phys):
            axs[group_id].plot(np.sort(values_phys), np.arange(1, len(values_phys) + 1) / len(values_phys),
                               color=color, linewidth=0.7, zorder=3)
            axs[group_id].plot(np.sort(values_svd), np.arange(1, len(values_svd) + 1) / len(values_svd),
                               color=color, linewidth=0.7, zorder=3, ls="--")
        axs[group_id].set_xticks([])
        # axs[group_id].set_xticks([-1, 0, 1])
        # axs[group_id].set_xlabel(group_labels[group_id], color=color, labelpad=1)
    # axs[0].set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    axs[0].set_yticks([])
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=1, pad=1)

    save_path = path.join(save_dir, "PairwiseSimilarity_CD.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    logger.info("Plot saved to " + save_path)

    for group_id, group in enumerate(groups):
        mask = pair_groups == group
        if not np.any(mask):
            continue
        basic_ttest.stats_pairwise_ks(
            data={
                "Physiology": physiology_matrix_by_cluster[rows[mask], cols[mask]],
                "SVD": svd_matrix_by_cluster[rows[mask], cols[mask]],
            },
            ax=axs[group_id],
            bon_ferroni_adj=len(groups),
        )
    fig.set_size_inches(5, 0.9)
    save_path = path.join(save_dir, "PairwiseSimilarity_CD_annotated.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)
