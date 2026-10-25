from collections import defaultdict

from matplotlib.colors import Normalize

from kitchen.configs import routing
from kitchen.operator.grouping import grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.unit_plotter.unit_heatmap import default_ax_realign, label_heatmap_y_ticklabels
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, Node
from kitchen.configs.routing import default_data_path, search_pattern_file
from kitchen.structure.neural_data_structure import TimeSeries, TimeSeries_concat
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

plt.rcParams["font.family"] = "Arial"


def cohens_d(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    nx, ny = len(x), len(y)
    pooled_sd = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) +
        (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )

    return (np.mean(x) - np.mean(y)) / pooled_sd

def welch_ttest(x, y):
    from scipy import stats
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    _, p_value = stats.ttest_ind(x, y, equal_var=False)
    return p_value


def visualize_staining_small(soma_red, soma_green, neuropil_red, neuropil_green, title, save_path):
    
    fig, ax = plt.subplots(2, 2, figsize=(3.2, 3.2))
    fig.suptitle(title, fontsize='small')
    rng = np.random.default_rng(1)  
    sns.kdeplot(x=soma_red, ax=ax[0, 0],
                color="red", linewidth=1, label="Soma")

    sns.kdeplot(x=neuropil_red, ax=ax[0, 0],
                color="red", linewidth=1, linestyle="--", label="Neuropil")


    sns.stripplot(
        data=pd.DataFrame({
            "Intensity": np.concatenate([soma_red, neuropil_red]),
            "Region": ["Soma"] * len(soma_red) +
                    ["Neuropil"] * len(neuropil_red),}),
        x="Region",
        y="Intensity",
        color="red",
        size=3,
        alpha=0.6,
        ax=ax[1, 0]
    )

    d1 = cohens_d(soma_red, neuropil_red)
    p1 = welch_ttest(soma_red, neuropil_red)
    ax[0, 0].text(
        0.97, 0.93, f"Cohen's d\n{d1:.2f}",
        transform=ax[0, 0].transAxes,
        ha="right", va="top", fontsize='small'
    )
    ax[1, 0].text(
        0.97, 0.93, f"Welch's t-test\np = {p1:.3f}",
        transform=ax[1, 0].transAxes,
        ha="right", va="top", fontsize='small'
    )

    ax[0, 0].set_title("red channel")
    ax[0, 0].set_xlabel("raw pixel intensity")
    # ax[0, 0].legend(loc="best", fontsize='xx-small', frameon=False)


    sns.kdeplot(x=soma_green, ax=ax[0, 1],
                color="green", linewidth=1, label="Soma")

    sns.kdeplot(x=neuropil_green, ax=ax[0, 1],
                color="green", linewidth=1, linestyle="--", label="Neuropil")

    sns.stripplot(
        data=pd.DataFrame({
            "Intensity": np.concatenate([soma_green, neuropil_green]),
            "Region": ["Soma"] * len(soma_green) +
                    ["Neuropil"] * len(neuropil_green),}),
        x="Region",
        y="Intensity",
        color="green",
        size=3,
        alpha=0.6,
        ax=ax[1, 1]
    )
                    
    d2 = cohens_d(soma_green, neuropil_green)
    p2 = welch_ttest(soma_green, neuropil_green)
    ax[0, 1].text(
        0.97, 0.93, f"Cohen's d\n{d2:.2f}",
        transform=ax[0, 1].transAxes, fontsize='small',
        ha="right", va="top"
    )
    ax[1, 1].text(
        0.97, 0.93, f"Welch's t-test\np = {p2:.3f}",
        transform=ax[1, 1].transAxes, fontsize='small',
        ha="right", va="top"
    )

    ax[0, 1].set_title("green channel")
    ax[0, 1].set_xlabel("raw pixel intensity")
    # ax[0, 1].legend(loc="best", fontsize='xx-small', frameon=False)
    ax[1, 0].set_ylabel("raw pixel intensity")

    ax[1, 1].set_xlabel("")
    ax[1, 0].set_xlabel("")
    
    ax[0, 1].set_yticks([])
    ax[0, 0].set_yticks([])
    ax[1, 1].set_yticks([])
    ax[1, 0].set_yticks([])
    ax[0, 0].spines[["left", "top", "right"]].set_visible(False)
    ax[0, 1].spines[["left", "top", "right"]].set_visible(False)

    default_exit_save(fig, save_path)


def visualize_all_cell_staining(dataset: DataSet, save_dir: str):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
    "black_magenta",
    ["#000000", "gray", "#ff00ff"]
    )
    cmap.set_bad("gray", alpha=0.01)
    norm = Normalize(vmin=0, vmax=2)

    fig, axs = plt.subplots(2, 1, figsize=(1.5, 2), gridspec_kw={'height_ratios': [1, 3]}, constrained_layout=True)

    ax = axs[1]
    ax.tick_params(length=1, pad=1)
    all_mice_idx = sorted(list(set([node.mice_id for node in dataset.select("cell") 
                             if "red_cohen_d" in node.info and "green_cohen_d" in node.info])))
    # mice_marker = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "H", "X"]
    all_cohen_red = []
    for cell_node in dataset.select("cell"):
        if "red_cohen_d" not in cell_node.info or "green_cohen_d" not in cell_node.info:
            continue

        mice_idx = all_mice_idx.index(cell_node.mice_id)

        ax.scatter(cell_node.info["red_cohen_d"], cell_node.info["green_cohen_d"],
                   facecolor=cmap(norm(cell_node.info["red_cohen_d"])), edgecolor='white', lw=0.5, alpha=0.8, s=8, 
                   marker='o')
        all_cohen_red.append(cell_node.info["red_cohen_d"])

    # WC recording from J.S.
    ax.scatter([1.925, 2.606, 1.045], [2.511, 2.040, 2.919], facecolor=cmap(norm([1.925, 2.606, 1.045])), 
               edgecolor='black', lw=0.5, alpha=1, s=20, marker='*', zorder=100)
    
    ax.set_xlabel("Red Channel\nEffect Size (Cohen's d)", color="red", labelpad=1)
    ax.spines["bottom"].set_color("red")
    ax.set_ylabel("Green Channel\nEffect Size (Cohen's d)", color="green", labelpad=1)
    ax.spines["left"].set_color("green")
    # add_textonly_legend(ax, {f"{mice_id}": {"color": mice_colors[idx]} for idx, mice_id in enumerate(all_mice_idx)}, 
    #                     loc="best", fontsize='x-small')

    ax.set_xlim(-2, 4)
    ax.set_ylim(0, 5)
    ax.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax = axs[0]
    ax.tick_params(length=1, pad=1)
    sns.kdeplot(x=all_cohen_red, ax=ax,
                color="gray", linewidth=0, fill=True)
    ax.set_xlim(-2, 4)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines[["left", "top", "right", "bottom"]].set_visible(False)
    ax.axvline(1, color='red', linestyle='--', linewidth=1, alpha=0.7)

    default_exit_save(fig, save_dir, _transparent=True)

def pack_images(a_path, b_path, c_path, output_path):
    from PIL import Image
    A = Image.open(a_path).convert("RGBA")
    B = Image.open(b_path).convert("RGBA")
    C = Image.open(c_path).convert("RGBA")

    # Resize A to the same width as B, preserving aspect ratio
    new_a_width = B.width
    new_a_height = round(A.height * new_a_width / A.width)

    A = A.resize(
        (new_a_width, new_a_height),
        Image.Resampling.LANCZOS
    )

    # Size of the A+B column
    ab_width = B.width
    ab_height = A.height + B.height

    # Final canvas size
    canvas_width = ab_width + C.width
    canvas_height = max(ab_height, C.height)

    # Transparent output canvas
    canvas = Image.new(
        "RGBA",
        (canvas_width, canvas_height),
        (0, 0, 0, 0)
    )

    # Vertically center AB and C relative to each other
    ab_y = (canvas_height - ab_height) // 2
    c_y = (canvas_height - C.height) // 2

    # Paste A
    canvas.paste(A, (0, ab_y), A)

    # Paste B directly below A
    canvas.paste(B, (0, ab_y + A.height), B)

    # Paste C to the right
    canvas.paste(C, (ab_width, c_y), C)

    canvas.save(output_path)

def register_staining_info(dataset: DataSet, create_small_pic: bool = False, create_big_pic: bool = False):

    for mice_node, mice_subtree in dataset.select_subtree("mice"):
        mouse_data_path = default_data_path(mice_node)
        staining_data_path = mouse_data_path.replace(mice_node.cohort_id, mice_node.cohort_id+"_info")
        
        all_cell_nodes = mice_subtree.select("cell")
        all_date_list = sorted(list(set([node.fov_id.split("_")[0] for node in mice_subtree.select("session")])))

        for cellsession_node in mice_subtree.select("cellsession"):
            date_str, roi_str = cellsession_node.fov_id.split("_")
            cellsession_node.info["day_id"] = 3 + all_date_list.index(date_str)

        if not path.exists(staining_data_path):
            print(f"Staining data for {mice_node.mice_id} doesn't exist, skipping...")
            continue

        short_mice_id = "_".join(mice_node.mice_id.split("_")[1:])
        pixel_values_file = path.join(staining_data_path, f"{short_mice_id}.csv")
        assert path.exists(pixel_values_file), f"Pixel values file for {mice_node} doesn't exist"
        pixel_values_df = pd.read_csv(pixel_values_file)
        
        example_preview_pics = search_pattern_file(f"*_crop.png", staining_data_path)


        n_cell_found = 0
        for session_node, session_subtree in mice_subtree.select_subtree("session"):
            date_str, roi_str = session_node.fov_id.split("_")
            mutliple_cell_session_flag = len(session_subtree.select("cellsession")) > 1
            for cell_idx, cellsession_node in enumerate(session_subtree.select("cellsession")):
                cell_node = find_only_one(all_cell_nodes,  _self=lambda node: node.contains(cellsession_node))
                if mutliple_cell_session_flag:
                    target_id = f"day{cellsession_node.info['day_id']}_{roi_str}_cell{cell_idx}"
                else:
                    target_id = f"day{cellsession_node.info['day_id']}_{roi_str}"

                try:
                    corresponding_preview_pic = find_only_one(example_preview_pics, _self=lambda pic: target_id in pic)
                    print(f"Found corresponding preview pic for {target_id}")
                    n_cell_found += 1
                except Exception as e:
                    print(f"Cannot find corresponding preview pic for {target_id}, skipping...")
                    continue

                cell_df = pixel_values_df[pixel_values_df["cell"] == target_id]

                soma_red = cell_df.loc[cell_df["region"] == "soma", "CH1"].to_numpy()
                soma_green = cell_df.loc[cell_df["region"] == "soma", "CH2"].to_numpy()

                neuropil_red = cell_df.loc[cell_df["region"] == "neuropil", "CH1"].to_numpy()
                neuropil_green = cell_df.loc[cell_df["region"] == "neuropil", "CH2"].to_numpy()

                if create_small_pic:
                    save_path = path.join(path.dirname(corresponding_preview_pic), f"{target_id}_pixels.png")
                    visualize_staining_small(soma_red, soma_green, neuropil_red, neuropil_green, 
                                             title=f"{short_mice_id}_{target_id}", save_path=save_path)
                if create_big_pic:
                    big_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\cuisine\PassivePuff_HighFreqImaging\HighFreqImaging_Combine_ROIs\CelluarOverview"
                    image_A_dir = corresponding_preview_pic
                    image_B_dir = path.join(path.dirname(corresponding_preview_pic), f"{target_id}_pixels.png")                    
                    image_C_dir = find_only_one(search_pattern_file(f"*{cellsession_node.session_id}*Cell{cellsession_node.cell_id}.png",
                                                      path.join(big_path, mice_node.mice_id)))
                    save_path = path.join(path.dirname(corresponding_preview_pic), f"Pack_{target_id}.png")
                    pack_images(image_A_dir, image_B_dir, image_C_dir, save_path)                    

                cell_node.info["red_cohen_d"] = cohens_d(soma_red, neuropil_red)
                cell_node.info["green_cohen_d"] = cohens_d(soma_green, neuropil_green)
                cellsession_node.info["red_cohen_d"] = cohens_d(soma_red, neuropil_red)
                cellsession_node.info["green_cohen_d"] = cohens_d(soma_green, neuropil_green)

        print(f"Found {n_cell_found}/{len(all_cell_nodes)} cells with staining info for {mice_node.mice_id}\n")

    visualize_all_cell_staining(dataset, save_dir=path.join(staining_data_path, "..",  "staining_summary.png"))



