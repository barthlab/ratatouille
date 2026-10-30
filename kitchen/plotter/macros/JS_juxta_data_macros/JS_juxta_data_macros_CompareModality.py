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


    fig, ax = plt.subplots(figsize=(1.2, 0.7), constrained_layout=True)
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
    ax.set_yticks([0, 0.5])
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

    fig, axs = plt.subplots(1, 4, figsize=(2, 0.4), sharex='all', sharey='all', constrained_layout=True)
    for group_id, group in enumerate(["SST", "PV", "PYR", "Cross-cohort"]):
        mask = pair_groups == group
        color = "gray" if group == "Cross-cohort" else COHORT_COLORS[group + "_JUX"]
        values_phys = physiology_matrix_order_by_svd[rows[mask], cols[mask]].flatten()
        values_svd = svd_matrix[rows[mask], cols[mask]].flatten()
        axs[group_id].plot(np.sort(values_phys), np.arange(1, len(values_phys) + 1)/len(values_phys),
                 color =color, linewidth=0.5, zorder=3)
        axs[group_id].plot(np.sort(values_svd), np.arange(1, len(values_svd) + 1)/len(values_svd),
                 color =color, linewidth=0.5, zorder=3, ls='--')
    # axs[0].set_ylabel("Cum. dist.", labelpad=1)
    axs[0].set_yticks([])
    axs[0].set_xticks([])

    for ax in axs:
        ax.spines[["top", "right",]].set_visible(False)
        ax.tick_params(length=1, pad=1)
    
    save_path = path.join(get_saving_path(), "CompareModality", f"PairwiseSimilarity_CD.png")
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)