from collections import defaultdict

from matplotlib.colors import Normalize

from kitchen.calculator.basic_metric import AVERAGE_VALUE, PEARSON_CORRELATION
from kitchen.configs import routing
from kitchen.operator.grouping import grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.plotting_params import LOCOMOTION_BIN_SIZE
from kitchen.plotter.unit_plotter.unit_heatmap import default_ax_realign, label_heatmap_y_ticklabels
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.configs.routing import default_data_path, search_pattern_file
from kitchen.structure.neural_data_structure import TimeSeries_concat
from kitchen.utils.sequence_kit import find_only_one, select_truthy_items
from kitchen.plotter.utils.tick_labels import add_textonly_legend

import os.path as path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import distinctipy
from scipy.stats import linregress, mannwhitneyu, ttest_ind


def visualize_trial_level_correlation(trial_fluos, trial_locos, trial_whiskers, title, save_path):

    def regression_panel(ax, x, y, color, xlabel, ylabel, title):
        # Convert to arrays and remove NaN / inf pairs
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        avg_y = np.nanmean(np.abs(y))

        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        # Regression statistics
        res = linregress(x, y)

        # Scatter + regression line + 95% CI
        sns.regplot(
            x=x,
            y=y,
            ax=ax,
            ci=95,
            color=color,
            scatter_kws={
                "s": 15,
                "alpha": 0.65,
                "edgecolor": "white",
            },
            line_kws={
                "lw": 2,
            },
        )

        # Labels
        ax.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        # Regression result
        r2 = res.rvalue**2
        p_text = f"{res.pvalue:.2e}" if res.pvalue < 0.001 else f"{res.pvalue:.3f}"
        ax.text(
            0.05, 0.95,
            rf"$\hat{{\beta}}$ = {res.slope/avg_y:.2f}" + "\n" +
            rf"$R^2$ = {r2:.2f}" + "\n" +
            rf"$p$ = {p_text}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6,
        )

        # Minimal appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    # ----------------------------
    # Plot
    # ----------------------------

    sns.set_theme(
        style="ticks",
        context="paper",
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(5, 2.5),
        constrained_layout=True,
    )

    fig.suptitle(title, fontsize=9)

    regression_panel(
        axes[0],
        trial_whiskers, trial_fluos,
        color="green",       # green
        xlabel="trial whiskering [A.U.]",
        ylabel=f"trial evoked {DF_F0_SIGN}",
        title="Whisker vs. Fluorescence",
    )

    regression_panel(
        axes[1],
        trial_locos, trial_fluos,
        color="blue",       # blue
        xlabel="trial locomotion [A.U.]",
        ylabel=f"trial evoked {DF_F0_SIGN}",
        title="Locomotion vs. Fluorescence",
    )

    default_exit_save(fig, save_path)


def visualize_all_cell_state_results(dataset: DataSet, save_path):

    all_mice_names = [mice_node.mice_id for mice_node in dataset.select("mice")]
    mice_color = distinctipy.get_colors(len(all_mice_names))
    cs_node_color = [mice_color[all_mice_names.index(cs_node.mice_id)] for cs_node in dataset.select("cellsession")]

    cs_node_cohen_d_red = [cs_node.info.get("red_cohen_d", np.nan) for cs_node in dataset.select("cellsession")]
    red_cohen_d_list_masked = np.ma.masked_invalid(np.array(cs_node_cohen_d_red).reshape(-1, 1))
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("lightgray", alpha=0.5)
    norm = Normalize(vmin=0, vmax=2)
    celltype_colors = cmap(norm(red_cohen_d_list_masked))


    def scatter_panel(ax, p_results, slope_values, color, xlabel, ylabel, title):
        # Convert to arrays and remove NaN / inf pairs
        p_results = np.asarray(p_results, dtype=float)
        slope_values = np.asarray(slope_values, dtype=float)

        mask = np.isfinite(p_results) & np.isfinite(slope_values)
        p_results, slope_values = p_results[mask], slope_values[mask]

        # Scatter plot
        ax.scatter(
            slope_values,
            -np.log10(p_results),
            color=color,
            s=15,
            # alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )

        # Labels
        ax.set_xlabel(xlabel, fontsize="x-small")
        ax.set_ylabel(ylabel, fontsize="x-small")
        ax.set_title(title, fontsize="x-small")

        # Minimal appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.axhline(-np.log10(0.05), color='green', linestyle='--', linewidth=1, alpha=0.7)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        

        
    fig, axes = plt.subplots(
        2, 4, 
        figsize=(5, 5),
        width_ratios=[1, 0.2, 1, 0.2],
        # sharey="row",
        constrained_layout=True,
    )
    scatter_panel(
        axes[0, 0],
        [cs_node.info["trial_level_corr"]["whisker"].pvalue for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["whisker"].slope/cs_node.info["trial_level_corr"]["fluorescence_avg"]
          for cs_node in dataset.select("cellsession")],
        color=cs_node_color,
        xlabel="Whisker-Fluorescence\nregression slope (normalized)",
        ylabel="-log10(p-value)",
        title="Whisker vs. Fluorescence",
    )
    add_textonly_legend(axes[0, 0], {f"{mice_id}": {"color": mice_color[idx]} for idx, mice_id in enumerate(all_mice_names)}, 
                        loc="best", fontsize='xx-small')
    scatter_panel(
        axes[0, 2],
        [cs_node.info["trial_level_corr"]["locomotion"].pvalue for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["locomotion"].slope/cs_node.info["trial_level_corr"]["fluorescence_avg"]
         for cs_node in dataset.select("cellsession")],
        color=cs_node_color,
        xlabel="Locomotion-Fluorescence\nregression slope (normalized)",
        ylabel="-log10(p-value)",
        title="Locomotion vs. Fluorescence",
    )
    add_textonly_legend(axes[0, 2], {f"{mice_id}": {"color": mice_color[idx]} for idx, mice_id in enumerate(all_mice_names)}, 
                        loc="best", fontsize='xx-small')

    scatter_panel(
        axes[1, 0],
        [cs_node.info["trial_level_corr"]["whisker"].pvalue for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["whisker"].slope/cs_node.info["trial_level_corr"]["fluorescence_avg"]
         for cs_node in dataset.select("cellsession")],
        color=celltype_colors,
        xlabel="Whisker-Fluorescence\nregression slope (normalized)",
        ylabel="-log10(p-value)",
        title="Whisker vs. Fluorescence",
    )
    scatter_panel(
        axes[1, 2],
        [cs_node.info["trial_level_corr"]["locomotion"].pvalue for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["locomotion"].slope/cs_node.info["trial_level_corr"]["fluorescence_avg"]
         for cs_node in dataset.select("cellsession")],
        color=celltype_colors,
        xlabel="Locomotion-Fluorescence\nregression slope (normalized)",
        ylabel="-log10(p-value)",
        title="Locomotion vs. Fluorescence",
    )
    for ax_id in [0, 1]:
        axes[ax_id, 1].scatter(red_cohen_d_list_masked, 
                               -np.log10([cs_node.info["trial_level_corr"]["whisker"].pvalue for cs_node in dataset.select("cellsession")]),
                        color=celltype_colors, s=10, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[ax_id, 1].set_xticks([0, 2])
        axes[ax_id, 1].axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axes[ax_id, 1].axhline(-np.log10(0.05), color='green', linestyle='--', linewidth=1, alpha=0.7)
        axes[ax_id, 1].set_xlabel("Red Cohen's d", fontsize="x-small")
        axes[ax_id, 1].spines[["top", "right", "left"]].set_visible(False)
        axes[ax_id, 1].set_yticks([])
        axes[ax_id, 1].sharey(axes[ax_id, 0])
        

        axes[ax_id, 3].scatter(red_cohen_d_list_masked, 
                               -np.log10([cs_node.info["trial_level_corr"]["locomotion"].pvalue for cs_node in dataset.select("cellsession")]),
                        color=celltype_colors, s=10, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[ax_id, 3].set_xticks([0, 2])
        axes[ax_id, 3].axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axes[ax_id, 3].axhline(-np.log10(0.05), color='green', linestyle='--', linewidth=1, alpha=0.7)
        axes[ax_id, 3].set_xlabel("Red Cohen's d", fontsize="x-small")
        axes[ax_id, 3].spines[["top", "right", "left"]].set_visible(False)
        axes[ax_id, 3].set_yticks([])
        axes[ax_id, 3].sharey(axes[ax_id, 2])

    default_exit_save(fig, save_path)


def visualize_all_cell_variance_explained(dataset: DataSet, save_path):

    all_mice_names = [mice_node.mice_id for mice_node in dataset.select("mice")]
    mice_color = distinctipy.get_colors(len(all_mice_names))
    cs_node_color = [mice_color[all_mice_names.index(cs_node.mice_id)] for cs_node in dataset.select("cellsession")]

    cs_node_cohen_d_red = [cs_node.info.get("red_cohen_d", np.nan) for cs_node in dataset.select("cellsession")]
    red_cohen_d_list_masked = np.ma.masked_invalid(np.array(cs_node_cohen_d_red).reshape(-1, 1))
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("lightgray", alpha=0.5)
    norm = Normalize(vmin=0, vmax=2)
    celltype_colors = cmap(norm(red_cohen_d_list_masked))

    def scatter_panel(ax, var_values, r2_values, color, xlabel, ylabel, title):
        # Convert to arrays and remove NaN / inf pairs
        var_values = np.asarray(var_values, dtype=float)
        r2_values = np.asarray(r2_values, dtype=float)

        mask = np.isfinite(var_values) & np.isfinite(r2_values)
        var_values, r2_values = var_values[mask], r2_values[mask]

        # Scatter plot
        ax.scatter(
            var_values,
            r2_values,
            color=color,
            s=20,
            # alpha=0.7,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xscale('log')

        # Labels
        ax.set(
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
        )

        # Minimal appearance
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, axes = plt.subplots(
        2, 4,
        figsize=(7, 7),
        width_ratios=[1, 0.2, 1, 0.2],
        constrained_layout=True,
    )
    scatter_panel(
        axes[0, 0],
        [cs_node.info["trial_level_corr"]["fluorescence_var"] for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["whisker"].rvalue**2 for cs_node in dataset.select("cellsession")],
        color=cs_node_color,
        xlabel="Fluorescence CV",
        ylabel="variance explained by whisker (R^2)",
        title="Whisker vs Fluorescence",
    )
    add_textonly_legend(axes[0, 0], {f"{mice_id}": {"color": mice_color[idx]} for idx, mice_id in enumerate(all_mice_names)},
                        loc="best", fontsize='x-small') 
    
    scatter_panel(
        axes[0, 2],
        [cs_node.info["trial_level_corr"]["fluorescence_var"] for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["locomotion"].rvalue**2 for cs_node in dataset.select("cellsession")],
        color=cs_node_color,
        xlabel="Fluorescence CV",
        ylabel="variance explained by locomotion (R^2)",
        title="Locomotion vs Fluorescence",
    )
    add_textonly_legend(axes[0, 2], {f"{mice_id}": {"color": mice_color[idx]} for idx, mice_id in enumerate(all_mice_names)},
                        loc="best", fontsize='x-small') 

    scatter_panel(
        axes[1, 0],
        [cs_node.info["trial_level_corr"]["fluorescence_var"] for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["whisker"].rvalue**2 for cs_node in dataset.select("cellsession")],
        color=celltype_colors,
        xlabel="Fluorescence CV",
        ylabel="variance explained by whisker (R^2)",
        title="Whisker vs Fluorescence",
    )

    scatter_panel(
        axes[1, 2],
        [cs_node.info["trial_level_corr"]["fluorescence_var"] for cs_node in dataset.select("cellsession")],
        [cs_node.info["trial_level_corr"]["locomotion"].rvalue**2 for cs_node in dataset.select("cellsession")],
        color=celltype_colors,
        xlabel="Fluorescence CV",
        ylabel="variance explained by locomotion (R^2)",
        title="Locomotion vs Fluorescence",
    )

    for ax_id in [0, 1]:
        axes[ax_id, 1].scatter(red_cohen_d_list_masked,
                               [cs_node.info["trial_level_corr"]["whisker"].rvalue**2 for cs_node in dataset.select("cellsession")],
                        color=celltype_colors, s=10, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[ax_id, 1].set_xticks([0, 2])
        axes[ax_id, 1].axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axes[ax_id, 1].set_xlabel("Red Cohen's d")
        axes[ax_id, 1].spines[["top", "right", "left"]].set_visible(False)
        axes[ax_id, 1].set_yticks([])

        axes[ax_id, 3].scatter(red_cohen_d_list_masked,
                               [cs_node.info["trial_level_corr"]["locomotion"].rvalue**2 for cs_node in dataset.select("cellsession")],
                        color=celltype_colors, s=10, alpha=0.8, edgecolor='white', linewidth=0.5)
        axes[ax_id, 3].set_xticks([0, 2])
        axes[ax_id, 3].axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        axes[ax_id, 3].set_xlabel("Red Cohen's d")
        axes[ax_id, 3].spines[["top", "right", "left"]].set_visible(False)
        axes[ax_id, 3].set_yticks([])
    default_exit_save(fig, save_path)

def visualize_all_cell_variance_explained_rank(dataset: DataSet, save_path):

    all_mice_names = [mice_node.mice_id for mice_node in dataset.select("mice")]
    mice_color = distinctipy.get_colors(len(all_mice_names))
    cs_node_color = [mice_color[all_mice_names.index(cs_node.mice_id)] for cs_node in dataset.select("cellsession")]

    cs_node_cohen_d_red = [cs_node.info.get("red_cohen_d", np.nan) for cs_node in dataset.select("cellsession")]
    red_cohen_d_list_masked = np.ma.masked_invalid(np.array(cs_node_cohen_d_red).reshape(-1, 1))
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("gray")
    norm = Normalize(vmin=0, vmax=2)
    celltype_colors = cmap(norm(red_cohen_d_list_masked))


    fig, axs = plt.subplots(2, 1, figsize=(8, 5), constrained_layout=True)

    cell_type_order = np.argsort(red_cohen_d_list_masked, axis=0).squeeze()
    x_pos = np.arange(len(cell_type_order))
    sorted_celltype_colors = celltype_colors[cell_type_order]

    explained_var = np.array([cs_node.info["trial_level_corr"]["fluorescence_var"] * cs_node.info["trial_level_corr"]["whisker"].rvalue**2
                      for cs_node in dataset.select("cellsession")])[cell_type_order]
    unexplained_var = np.array([cs_node.info["trial_level_corr"]["fluorescence_var"] * (1 - cs_node.info["trial_level_corr"]["whisker"].rvalue**2)
                      for cs_node in dataset.select("cellsession")])[cell_type_order]
    total_var = explained_var + unexplained_var
    
    bars = axs[0].bar(x_pos, total_var, facecolor="lightgray", edgecolor=sorted_celltype_colors, ls='--', lw=0.5, alpha=0.8, label="Unexplained Variance")
    axs[0].bar(x_pos, explained_var, color=sorted_celltype_colors, label="Explained Variance")
    axs[0].bar_label(bars, [f"{(explained_var[i] / (explained_var[i] + unexplained_var[i])) * 100:.1f}%" for i in range(len(cell_type_order))],
                      label_type='edge', color='black', fontsize=3, padding=2, rotation=30)
    axs[0].set_title("Whisker vs Fluorescence")
    axs[0].set_xlabel("Cells (sorted by Red Cohen's d)")
    axs[0].set_ylabel("Fluorescence CV")
    # axs[0].legend(fontsize='x-small', loc='best')

    explained_var = np.array([cs_node.info["trial_level_corr"]["fluorescence_var"] * cs_node.info["trial_level_corr"]["locomotion"].rvalue**2
                      for cs_node in dataset.select("cellsession")])[cell_type_order]
    unexplained_var = np.array([cs_node.info["trial_level_corr"]["fluorescence_var"] * (1 - cs_node.info["trial_level_corr"]["locomotion"].rvalue**2)
                      for cs_node in dataset.select("cellsession")])[cell_type_order]
    total_var = explained_var + unexplained_var

    bars = axs[1].bar(x_pos, total_var, facecolor="lightgray", edgecolor=sorted_celltype_colors, ls='--', lw=0.5, alpha=0.8, label="Unexplained Variance")
    axs[1].bar(x_pos, explained_var, color=sorted_celltype_colors, label="Explained Variance")
    axs[1].bar_label(bars, [f"{(explained_var[i] / (explained_var[i] + unexplained_var[i])) * 100:.1f}%" for i in range(len(cell_type_order))],
                      label_type='edge', color='black', fontsize=3, padding=2, rotation=30)
    axs[1].set_title("Locomotion vs Fluorescence")
    axs[1].set_xlabel("Cells (sorted by Red Cohen's d)")
    axs[1].set_ylabel("Fluorescence CV")
    # axs[1].legend(fontsize='x-small', loc='best')

    default_exit_save(fig, save_path)



def visualize_simple_metric_bar_graph(dataset: DataSet, save_path):
    cs_node_cohen_d_red = [cs_node.info.get("red_cohen_d", np.nan) for cs_node in dataset.select("cellsession")]
    red_cohen_d_list_masked = np.ma.masked_invalid(np.array(cs_node_cohen_d_red).reshape(-1, 1))
    CR_pos = np.where(red_cohen_d_list_masked > 1)[0]
    CR_neg = np.where(red_cohen_d_list_masked < 1)[0]

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad("gray")
    norm = Normalize(vmin=0, vmax=2)
    celltype_colors = cmap(norm(red_cohen_d_list_masked))

    CR_pos_colors = celltype_colors[CR_pos]
    CR_neg_colors = celltype_colors[CR_neg]
    
    sns.set_theme(
        style="ticks",
        context="paper",
    )

    def simple_two_bars(ax, data1, data2, label1, label2, color1, color2, ylabel):
        df = pd.DataFrame({
            "Group": [label1] * len(data1) + [label2] * len(data2),
            "Value": data1 + data2,
        })

        df["dot_id"] = np.arange(len(df))
        palette = {
            i: tuple(color[0])
            for i, color in enumerate(color1 + color2)
        }

        # Welch's t-test
        # t, p = ttest_ind(data1, data2)

        # Mann-Whitney U test
        u_stat, p = mannwhitneyu(data1, data2)

        sns.swarmplot(
            data=df,
            x="Group",
            y="Value",
            hue="dot_id",
            palette=palette,
            legend=False,
            dodge=False,
            size=5,
            ax=ax
        )

        # Mean
        means = df.groupby("Group")["Value"].mean()

        ax.scatter(
            [0, 1],
            means,
            marker="_",
            s=300,
            linewidth=2,
            color="black",
            zorder=10
        )

        # Statistical comparison
        y = df["Value"].max() + 0.2

        ax.plot([0, 0, 1, 1],
                [y, y+0.05, y+0.05, y],
                color="black")

        
        p_text = f"{p:.2e}" if p < 0.001 else f"{p:.3f}"
        ax.text(
            0.5,
            y + 0.07,
            f"p = {p_text}",
            ha="center"
        )

        # Clean formatting
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)


    
    fig, axs = plt.subplots(1, 2 + 2*2, figsize=(12, 4), constrained_layout=True)

    # axs 0: fluo avg
    fluo_avg_cr_pos = np.array([cs_node.info["trial_level_corr"]["fluorescence_avg"] 
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_pos])
    fluo_avg_cr_neg = np.array([cs_node.info["trial_level_corr"]["fluorescence_avg"]
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_neg])
    simple_two_bars(axs[0], fluo_avg_cr_pos.tolist(), fluo_avg_cr_neg.tolist(),
                    "CR+", "CR-", CR_pos_colors.tolist(), CR_neg_colors.tolist(), f"Average of Response {DF_F0_SIGN}")

    # axs 1: fluo var
    fluo_var_cr_pos = np.array([cs_node.info["trial_level_corr"]["fluorescence_var"]
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_pos])
    fluo_var_cr_neg = np.array([cs_node.info["trial_level_corr"]["fluorescence_var"]
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_neg])
    simple_two_bars(axs[1], fluo_var_cr_pos.tolist(), fluo_var_cr_neg.tolist(),
                    "CR+", "CR-", CR_pos_colors.tolist(), CR_neg_colors.tolist(), f"Variance of Response {DF_F0_SIGN}")


    # axs 2: whisker -log10(p)
    whisker_p_cr_pos = np.array([cs_node.info["trial_level_corr"]["whisker"].pvalue
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_pos])
    whisker_p_cr_neg = np.array([cs_node.info["trial_level_corr"]["whisker"].pvalue
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_neg])
    simple_two_bars(axs[2], list(-np.log10(whisker_p_cr_pos)), list(-np.log10(whisker_p_cr_neg)),
                    "CR+", "CR-", CR_pos_colors.tolist(), CR_neg_colors.tolist(), "-log10(p) of regression\nWhisker vs Fluorescence")

    # axs 3: locomotion -log10(p)
    locomotion_p_cr_pos = np.array([cs_node.info["trial_level_corr"]["locomotion"].pvalue
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_pos])
    locomotion_p_cr_neg = np.array([cs_node.info["trial_level_corr"]["locomotion"].pvalue
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_neg])
    simple_two_bars(axs[3], list(-np.log10(locomotion_p_cr_pos)), list(-np.log10(locomotion_p_cr_neg)),
                    "CR+", "CR-", CR_pos_colors.tolist(), CR_neg_colors.tolist(), "-log10(p) of regression\nLocomotion vs Fluorescence")

    # axs 4: whisker R^2
    whisker_r2_cr_pos = np.array([cs_node.info["trial_level_corr"]["whisker"].rvalue**2
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_pos])
    whisker_r2_cr_neg = np.array([cs_node.info["trial_level_corr"]["whisker"].rvalue**2
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_neg])
    simple_two_bars(axs[4], whisker_r2_cr_pos.tolist(), whisker_r2_cr_neg.tolist(),
                    "CR+", "CR-", CR_pos_colors.tolist(), CR_neg_colors.tolist(), "R^2\nWhisker vs Fluorescence")

    # axs 5: locomotion R^2
    locomotion_r2_cr_pos = np.array([cs_node.info["trial_level_corr"]["locomotion"].rvalue**2
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_pos])
    locomotion_r2_cr_neg = np.array([cs_node.info["trial_level_corr"]["locomotion"].rvalue**2
                                for idx, cs_node in enumerate(dataset.select("cellsession")) if idx in CR_neg])
    simple_two_bars(axs[5], locomotion_r2_cr_pos.tolist(), locomotion_r2_cr_neg.tolist(),
                    "CR+", "CR-", CR_pos_colors.tolist(), CR_neg_colors.tolist(), "R^2\nLocomotion vs Fluorescence")

    default_exit_save(fig, save_path)


def visualize_single_cell_state_dependent(dataset: DataSet,

        _element_trial_level: str = "trial",
        _alignment_style: str = "Aligned2Trial",
        plot_small_pic: bool = False,):
    
    alignment_events = ALL_ALIGNMENT_STYLE[_alignment_style]
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    for behav_range in [(0, 1), (0, 2), (-1, 1), (-2, 2)]:
        for cs_node, cs_subtree in dataset.select_subtree("cellsession"):
            if cs_node.fluorescence is None:
                continue

            print(f"Processing {cs_node.session_id}")

            # fluo_detrend_zscore = cs_node.fluorescence.detrend_z_score
            # whisker = cs_node.whisker
            # locomotion = cs_node.locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE)
            # trial_onset_t = cs_node.data.timeline.filter("TrialOn").t
            # concat_fluo = TimeSeries_concat(fluo_detrend_zscore.batch_segment(ts=trial_onset_t, segment_range=(-5, 5), _auto_align=False))
            # concat_whisker = TimeSeries_concat(whisker.batch_segment(ts=trial_onset_t, segment_range=(-5, 5), _auto_align=False))
            # concat_locomotion = TimeSeries_concat(locomotion.batch_segment(ts=trial_onset_t, segment_range=(-5, 5), _auto_align=False))

            # concat_corr_fluo_whisker = PEARSON_CORRELATION(concat_fluo, concat_whisker)
            # concat_corr_fluo_locomotion = PEARSON_CORRELATION(concat_fluo, concat_locomotion)

            all_puff_trials = sync_nodes(cs_subtree.select("trial", _self=lambda x: x.info.get("trial_type") == "PuffOnly"),
                                        alignment_events, plot_manual=plot_manual_fluo)
            all_trial_fluo = [AVERAGE_VALUE(node.data.fluorescence.df_f0.squeeze(0), (0, 1)) for node in all_puff_trials]
            all_trial_loco = [AVERAGE_VALUE(node.data.locomotion, behav_range) for node in all_puff_trials]
            all_trial_whisker = [AVERAGE_VALUE(node.data.whisker, behav_range) for node in all_puff_trials]

            if plot_small_pic:
                save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
                                    f"state_dependent_correlation_{cs_node.session_id}_cell_{cs_node.cell_id}.png",)
                visualize_trial_level_correlation(all_trial_fluo, all_trial_loco, all_trial_whisker, save_path=save_path, 
                                                title=f"Session {cs_node.session_id} - Cell {cs_node.cell_id}\n")

            cs_node.info["trial_level_corr"] = {
                "whisker": linregress(all_trial_whisker, all_trial_fluo),
                "locomotion": linregress(all_trial_loco, all_trial_fluo),
                "fluorescence_avg": np.nanmean(np.abs(all_trial_fluo)),
                "fluorescence_var": np.nanvar(all_trial_fluo)/np.nanmean(np.abs(all_trial_fluo)),
            }

        behav_range_str = f"{behav_range[0]}_{behav_range[1]}"
        save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
                            f"state_dependent_correlation_all_cells_{behav_range_str}.png",)
        visualize_all_cell_state_results(dataset, save_path=save_path)
        # save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
        #                     f"state_dependent_variance_explained_all_cells_{behav_range_str}.png",)
        # visualize_all_cell_variance_explained(dataset, save_path=save_path)
        save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
                            f"state_dependent_variance_explained_rank_all_cells_{behav_range_str}.png",)
        visualize_all_cell_variance_explained_rank(dataset, save_path=save_path)

        save_path = path.join(routing.default_fig_path(dataset, fov_skip=True), "trial_level_corr", 
                            f"state_dependent_simple_metric_bar_graph_{behav_range_str}.png",)
        visualize_simple_metric_bar_graph(dataset, save_path=save_path)





