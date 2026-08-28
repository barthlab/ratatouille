from collections import defaultdict

from matplotlib.colors import Normalize

from kitchen.configs import routing
from kitchen.operator.grouping import grouping_timeseries
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter import style_dicts
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
    fig, axs = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [3, 1]}, constrained_layout=True)

    ax = axs[0]

    all_mice_idx = sorted(list(set([node.mice_id for node in dataset.select("cell") 
                             if "red_cohen_d" in node.info and "green_cohen_d" in node.info])))
    mice_colors = [f"C{idx}" for idx in range(len(all_mice_idx))]
    all_cohen_red = []
    for cell_node in dataset.select("cell"):
        if "red_cohen_d" not in cell_node.info or "green_cohen_d" not in cell_node.info:
            continue

        mice_idx = all_mice_idx.index(cell_node.mice_id)

        ax.scatter(cell_node.info["red_cohen_d"], cell_node.info["green_cohen_d"],
                   color=mice_colors[mice_idx], alpha=0.8, s=10)
        all_cohen_red.append(cell_node.info["red_cohen_d"])

    # ax.set_xlabel("Cohen's d (red channel)", color="red")
    ax.set_ylabel("Cohen's d (green channel)", color="green")
    add_textonly_legend(ax, {f"{mice_id}": {"color": mice_colors[idx]} for idx, mice_id in enumerate(all_mice_idx)}, 
                        loc="best", fontsize='x-small')

    ax.set_box_aspect(5/6)
    ax.set_xlim(-2, 4)
    ax.set_ylim(0, 5)
    ax.axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax = axs[1]
    sns.kdeplot(x=all_cohen_red, ax=ax,
                color="red", linewidth=1,)
    ax.set_xlim(-2, 4)
    ax.set_yticks([])
    ax.set_xlabel("Cohen's d (red channel)", color="red")
    ax.set_ylabel("Density")
    ax.axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    default_exit_save(fig, save_dir)

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




def sort_all_cell_activity(
        dataset: DataSet,

        _element_trial_level: str = "trial",
        _alignment_style: str = "Aligned2Trial",
        fluo_range: tuple = (0., 1.),
        deconv_range: tuple = (0.3, 0.5),
):

    alignment_events = ALL_ALIGNMENT_STYLE[_alignment_style]
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9
    
    fovs = list(set([cs_node.fov_id for cs_node in dataset.select("cellsession")]))
    mice = list(set([cs_node.mice_id for cs_node in dataset.select("cellsession")]))
    color_by_mice = distinctipy.get_colors(len(mice))

    fluo_order = {}
    deconv_order = {}
    # zscore_order = {}

    fluo_linkage_order = {}
    deconv_linkage_order = {}
    # zscore_linkage_order = {}

    fluo_diff_order = {}
    deconv_diff_order = {}

    def get_linkage_order_from_group_timeseries(raw_order_dict: dict[Node, TimeSeries]):
        group_series = grouping_timeseries([ts for ts in raw_order_dict.values()], baseline_subtraction=None)
        node_list = list(raw_order_dict.keys())
        for node_idx in linkage_order(group_series.raw_array):
            node = node_list[node_idx]
            yield node, raw_order_dict[node]


    for cs_node, cs_subtree in dataset.select_subtree("cellsession"):
        
        type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                    plot_manual=plot_manual_fluo,
                                                    _element_trial_level =_element_trial_level,)
        if len(type2dataset) == 0:
            continue
        
        # get amp diff
        group_fluorescence = grouping_timeseries([node.data.fluorescence.df_f0.squeeze(0) 
                                                  for node in sync_nodes(cs_subtree.select(
                                                       "trial", _self=lambda x: x.info.get("trial_type") == "PuffOnly"),
                                                         alignment_events, plot_manual=plot_manual_fluo)], 
                                            baseline_subtraction=None)
        amplitude_range_in_frame1 = np.searchsorted(group_fluorescence.t, fluo_range)
        fluo_avg_amp = np.nanmean(group_fluorescence.raw_array[:, amplitude_range_in_frame1[0]:amplitude_range_in_frame1[1]], axis=0)
        fluo_order[cs_node] = np.nanmean(fluo_avg_amp)
        fluo_linkage_order[cs_node] = group_fluorescence.mean_ts

        group_fluorescence_blank = grouping_timeseries([node.data.fluorescence.df_f0.squeeze(0)
                                                    for node in sync_nodes(cs_subtree.select(
                                                        "trial", _self=lambda x: x.info.get("trial_type") == "BlankOnly"),
                                                        alignment_events, plot_manual=plot_manual_fluo)],
                                            baseline_subtraction=None, _predefined_t=group_fluorescence.t)
        amplitude_range_in_frame1_blank = np.searchsorted(group_fluorescence_blank.t, fluo_range)
        fluo_avg_amp_blank = np.nanmean(group_fluorescence_blank.raw_array[:, amplitude_range_in_frame1_blank[0]:amplitude_range_in_frame1_blank[1]], axis=0)
        fluo_diff_order[cs_node] = np.nanmean(fluo_avg_amp - fluo_avg_amp_blank)
        
        # get amp deconv 
        group_deconv_fluo = grouping_timeseries([node.data.fluorescence.delta_deconv_f.squeeze(0)
                                                 for node in sync_nodes(cs_subtree.select(
                                                     "trial", _self=lambda x: x.info.get("trial_type") == "PuffOnly"),
                                                     alignment_events, plot_manual=plot_manual_fluo)], 
                                            baseline_subtraction=None)
        amplitude_range_deconv1 = np.searchsorted(group_deconv_fluo.t, deconv_range)
        deconv_avg_amp = np.nanmean(group_deconv_fluo.raw_array[:, amplitude_range_deconv1[0]:amplitude_range_deconv1[1]], axis=0)
        deconv_order[cs_node] = np.nanmean(deconv_avg_amp)
        deconv_linkage_order[cs_node] = group_deconv_fluo.mean_ts

        group_deconv_fluo_blank = grouping_timeseries([node.data.fluorescence.delta_deconv_f.squeeze(0)
                                                 for node in sync_nodes(cs_subtree.select(
                                                     "trial", _self=lambda x: x.info.get("trial_type") == "BlankOnly"),
                                                     alignment_events, plot_manual=plot_manual_fluo)], 
                                            baseline_subtraction=None, _predefined_t=group_deconv_fluo.t)
        amplitude_range_deconv1_blank = np.searchsorted(group_deconv_fluo_blank.t, deconv_range)
        deconv_avg_amp_blank = np.nanmean(group_deconv_fluo_blank.raw_array[:, amplitude_range_deconv1_blank[0]:amplitude_range_deconv1_blank[1]], axis=0)
        deconv_diff_order[cs_node] = np.nanmean(deconv_avg_amp - deconv_avg_amp_blank)


    sorted_fluo_order = sorted(fluo_order.items(), key=lambda x: x[1], reverse=True)
    sorted_fluo_diff_order = sorted(fluo_diff_order.items(), key=lambda x: x[1], reverse=True)

    # sorted_zscore_order = sorted(zscore_order.items(), key=lambda x: x[1], reverse=True)

    sorted_deconv_order = sorted(deconv_order.items(), key=lambda x: x[1], reverse=True)
    sorted_deconv_diff_order = sorted(deconv_diff_order.items(), key=lambda x: x[1], reverse=True)

    sorted_red_cohen_d_order = sorted([(cs_node, cs_node.info.get("red_cohen_d", -100)) 
                                       for cs_node in fluo_order.keys()], key=lambda x: x[1], reverse=True)
    
    fluo_range_str = f"{fluo_range[0]}-{fluo_range[1]}s"
    deconv_range_str = f"{deconv_range[0]}-{deconv_range[1]}s"

    for order_name, final_order in [
        ("sortby_fluo", sorted_fluo_order), 
        ("sortby_deconv", sorted_deconv_order), 
        # ("sortby_fluo_diff", sorted_fluo_diff_order),
        # ("sortby_deconv_diff", sorted_deconv_diff_order),
        # ("sortby_zscorefluo", sorted_zscore_order),
        # ("sortby_fluo_linkage", get_linkage_order_from_group_timeseries(fluo_linkage_order)),
        # ("sortby_deconv_linkage", get_linkage_order_from_group_timeseries(deconv_linkage_order)),
        # ("sortby_zscorefluo_linkage", get_linkage_order_from_group_timeseries(zscore_linkage_order)),
        # ("sortby_red", sorted_red_cohen_d_order)
                                    ]:
        avg_fluo = defaultdict(list)
        avg_PSTH = defaultdict(list)
        avg_zscore_fluo = defaultdict(list)
        type_specific_timeline = defaultdict(list)
        red_cohen_d_list = []
        day_id_list = []
        range_str = fluo_range_str if "fluo" in order_name else deconv_range_str if "deconv" in order_name else "NA"

        for cs_node, order_score in final_order:
            cs_subtree = dataset.subtree(cs_node)
            type2dataset = split_dataset_by_trial_type(cs_subtree, 
                                                        plot_manual=plot_manual_fluo,
                                                        _element_trial_level =_element_trial_level,)
            if len(type2dataset) == 0:
                continue
            
            for trial_type, raw_type_dataset in type2dataset.items():
                type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)

                type_specific_timeline[trial_type].append(type_dataset.nodes[0].data.timeline)

                group_fluorescence = grouping_timeseries([single_fluorescence.df_f0.squeeze(0) 
                                                            for single_fluorescence in select_truthy_items(
                                                                [node.data.fluorescence for node in type_dataset])], 
                                                baseline_subtraction=None)
                avg_fluo[trial_type].append(group_fluorescence.mean_ts)

                group_zscore_fluorescence = grouping_timeseries([single_fluorescence.z_score.squeeze(0)
                                                            for single_fluorescence in select_truthy_items(
                                                                [node.data.fluorescence for node in type_dataset])],
                                                baseline_subtraction=None)
                avg_zscore_fluo[trial_type].append(group_zscore_fluorescence.mean_ts)
            
                type_dataset = sync_nodes(raw_type_dataset, alignment_events, plot_manual=plot_manual_fluo)
                group_delta_deconv_fluorescence = grouping_timeseries([single_fluorescence.delta_deconv_f.squeeze(0) 
                                                                    for single_fluorescence in select_truthy_items(
                                                                        [node.data.fluorescence for node in type_dataset])], 
                                                                    baseline_subtraction=None)
                avg_PSTH[trial_type].append(group_delta_deconv_fluorescence.mean_ts)

            red_cohen_d_list.append(cs_node.info.get("red_cohen_d", np.nan))
            day_id_list.append(cs_node.info.get("day_id", np.nan))
        
        avg_PSTH = {k: grouping_timeseries(v) for k, v in avg_PSTH.items()} 
        avg_fluo = {k: grouping_timeseries(v) for k, v in avg_fluo.items()} 
        avg_zscore_fluo = {k: grouping_timeseries(v) for k, v in avg_zscore_fluo.items()}

        red_cohen_d_list_masked = np.ma.masked_invalid(np.array(red_cohen_d_list).reshape(-1, 1))
        day_id_list_masked = np.ma.masked_invalid(np.array(day_id_list).reshape(-1, 1))

        CR_pos = np.where(red_cohen_d_list_masked > 1)[0]
        CR_neg = np.where(red_cohen_d_list_masked < 1)[0]

        cmap = plt.cm.coolwarm.copy()
        cmap.set_bad("gray", alpha=0.01)
        norm = Normalize(vmin=0, vmax=2)
        cell_colors = cmap(norm(red_cohen_d_list_masked))

        def unit_plot(plotting_dict, plotting_red_cohen_list, plotting_cell_colors, v_range, xlim, ylim, save_path, theme_color="black"):
            fig, ax = plt.subplots(2, len(plotting_dict) + 1, figsize=(4, 3), 
                                width_ratios=[0.2] + [1]*len(plotting_dict),
                                height_ratios=[1, 0.5],
                                constrained_layout=True)

            row_height = 9/len(plotting_red_cohen_list)
            ax[0, 0].scatter(plotting_red_cohen_list, np.linspace(1 + row_height/2, 10 - row_height/2, len(plotting_red_cohen_list)), 
                             color=plotting_cell_colors, s=6, alpha=0.8)
            ax[0, 0].set_xticks([0, 2])
            ax[0, 0].axvline(1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            default_ax_realign(ax[0, 0])
            label_heatmap_y_ticklabels(ax[0, 0], len(plotting_red_cohen_list), (1, 10))
            ax[0, 0].set_title("Red Cohen's d")

            for idx, (trial_type, group_fluorescence) in enumerate(plotting_dict.items()):
                heatmap_extent = (group_fluorescence.t[0], group_fluorescence.t[-1], 10, 1)
                ax[0, idx + 1].imshow(group_fluorescence.raw_array, extent=heatmap_extent, cmap="RdYlBu_r", vmin=-v_range, vmax=v_range,
                            **style_dicts.HEATMAP_STYLE)
                ax[0, idx + 1].set_title(trial_type)
                default_ax_realign(ax[0, idx + 1])
                label_heatmap_y_ticklabels(ax[0, idx + 1], len(group_fluorescence.raw_array), (1, 10))
                # unit_plot_timeline(timeline=type_specific_timeline[trial_type], ax=ax[idx], y_offset=0, ratio=1.0)
                ax[0, idx + 1].set_xlim(*xlim)
                ax[0, idx + 1].set_xticks([ 0, 0.5])

                oreo_plot(ax[1, idx + 1], group_fluorescence, 
                          y_offset=0, ratio=1.0, 
                          trace_style={"color": theme_color, "linewidth": 1}, 
                          fill_between_style={"lw": 0, "alpha": 0.2})
                ax[1, idx + 1].set_xlim(*xlim)
                ax[1, idx + 1].set_ylim(*ylim)
                ax[1, idx + 1].set_xticks([ 0, 0.5])
                if idx == 0:
                    ax[1, idx + 1].sharey(ax[1, 1])
            default_exit_save(fig, save_path)


        PSTH_v_range = 1.0
        fluo_v_range = 1.0
        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}.png")
        unit_plot(avg_PSTH, red_cohen_d_list_masked, cell_colors, v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.5, 2.5), save_path=save_path)

        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}_long.png")
        unit_plot(avg_PSTH, red_cohen_d_list_masked, cell_colors, v_range=PSTH_v_range, xlim=(-2, 2.5), ylim=(-0.5, 2.5), save_path=save_path)

        CR_pos_avg_PSTH = {k: v.subset(CR_pos) for k, v in avg_PSTH.items()}
        CR_neg_avg_PSTH = {k: v.subset(CR_neg) for k, v in avg_PSTH.items()}
        red_cohen_d_list_masked_CR_pos = red_cohen_d_list_masked[CR_pos]
        red_cohen_d_list_masked_CR_neg = red_cohen_d_list_masked[CR_neg]
        cell_colors_CR_pos = cell_colors[CR_pos]
        cell_colors_CR_neg = cell_colors[CR_neg]

        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}_CRpos.png")
        unit_plot(CR_pos_avg_PSTH, red_cohen_d_list_masked_CR_pos, cell_colors_CR_pos, v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.5, 2.5), save_path=save_path)
        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}_CRneg.png")
        unit_plot(CR_neg_avg_PSTH, red_cohen_d_list_masked_CR_neg, cell_colors_CR_neg, v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.5, 2.5), save_path=save_path)
        

        save_path = routing.default_fig_path(dataset, f"AverageFluorescence_{order_name}_{range_str}.png")
        unit_plot(avg_fluo, red_cohen_d_list_masked, cell_colors, v_range=fluo_v_range, xlim=(-2, 2.5), ylim=(-0.1, 0.5), save_path=save_path)

        CR_pos_avg_fluo = {k: v.subset(CR_pos) for k, v in avg_fluo.items()}
        CR_neg_avg_fluo = {k: v.subset(CR_neg) for k, v in avg_fluo.items()}

        save_path = routing.default_fig_path(dataset, f"AverageFluorescence_{order_name}_{range_str}_CRpos.png")
        unit_plot(CR_pos_avg_fluo, red_cohen_d_list_masked_CR_pos, cell_colors_CR_pos, v_range=fluo_v_range, xlim=(-2, 2.5), ylim=(-0.1, 0.5), save_path=save_path)
        save_path = routing.default_fig_path(dataset, f"AverageFluorescence_{order_name}_{range_str}_CRneg.png")
        unit_plot(CR_neg_avg_fluo, red_cohen_d_list_masked_CR_neg, cell_colors_CR_neg, v_range=fluo_v_range, xlim=(-2, 2.5), ylim=(-0.1, 0.5), save_path=save_path)


        day3_cs = np.where(day_id_list_masked == 3)[0]
        day4_cs = np.where(day_id_list_masked == 4)[0]

        day3_avg_PSTH = {k: v.subset(day3_cs) for k, v in avg_PSTH.items()}
        day4_avg_PSTH = {k: v.subset(day4_cs) for k, v in avg_PSTH.items()}
        day3_avg_fluo = {k: v.subset(day3_cs) for k, v in avg_fluo.items()}
        day4_avg_fluo = {k: v.subset(day4_cs) for k, v in avg_fluo.items()}

        red_cohen_d_list_masked_day3 = red_cohen_d_list_masked[day3_cs]
        red_cohen_d_list_masked_day4 = red_cohen_d_list_masked[day4_cs]
        cell_colors_day3 = cell_colors[day3_cs] 
        cell_colors_day4 = cell_colors[day4_cs]

        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}_day3.png")
        unit_plot(day3_avg_PSTH, red_cohen_d_list_masked_day3, cell_colors_day3, v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.5, 2.5), save_path=save_path)
        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}_day4.png")
        unit_plot(day4_avg_PSTH, red_cohen_d_list_masked_day4, cell_colors_day4, v_range=PSTH_v_range, xlim=(-0.5, 1), ylim=(-0.5, 2.5), save_path=save_path)
        
        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}_day3_long.png")
        unit_plot(day3_avg_PSTH, red_cohen_d_list_masked_day3, cell_colors_day3, v_range=PSTH_v_range, xlim=(-2, 2.5), ylim=(-0.5, 2.5), save_path=save_path)
        save_path = routing.default_fig_path(dataset, f"AveragePSTH_{order_name}_{range_str}_day4_long.png")
        unit_plot(day4_avg_PSTH, red_cohen_d_list_masked_day4, cell_colors_day4, v_range=PSTH_v_range, xlim=(-2, 2.5), ylim=(-0.5, 2.5), save_path=save_path)

        save_path = routing.default_fig_path(dataset, f"AverageFluorescence_{order_name}_{range_str}_day3.png")
        unit_plot(day3_avg_fluo, red_cohen_d_list_masked_day3, cell_colors_day3, v_range=fluo_v_range, xlim=(-2, 2.5), ylim=(-0.1, 0.5), save_path=save_path)
        save_path = routing.default_fig_path(dataset, f"AverageFluorescence_{order_name}_{range_str}_day4.png")
        unit_plot(day4_avg_fluo, red_cohen_d_list_masked_day4, cell_colors_day4, v_range=fluo_v_range, xlim=(-2, 2.5), ylim=(-0.1, 0.5), save_path=save_path)
        
