
from collections import defaultdict
import json
import os
from typing import Optional
import distinctipy
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import _pickle as pkl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import SpectralClustering

from kitchen.configs import routing
from kitchen.operator.grouping import grouping_timeseries
from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.color_scheme import CLUSTER_COLORS, num_to_hex_color
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.utils.tick_labels import add_textonly_legend
from kitchen.settings.fluorescence import DF_F0_SIGN
from kitchen.settings.timeline import ALL_ALIGNMENT_STYLE
from kitchen.structure.hierarchical_data_structure import DataSet, DataSet, FovTrial, Node, Trial


def get_session_index(cs_node: Node) -> int:
    session_index_str = cs_node.session_id.split("_")[3]
    if session_index_str == "P1":
        return 0
    elif session_index_str == "P2":
        return 6
    else:
        assert session_index_str.startswith("S"), f"Unexpected session id format: {cs_node.session_id}"
        return int(session_index_str[1:])

def get_session_index2(cs_node: Node) -> int:
    session_index_str = cs_node.session_id.split("_")[3]
    if session_index_str == "P1":
        return 0
    elif session_index_str == "P2":
        if ("A50" in cs_node.session_id) or ("A100" in cs_node.session_id):
            return 1
        else:
            return 6
    else:
        assert session_index_str.startswith("S"), f"Unexpected session id format: {cs_node.session_id}"
        return int(session_index_str[1:])


def get_day_index(cs_node: Node) -> Optional[int]:
    day_id_int = int(cs_node.day_id)
    if cs_node.mice_id == "SUS6F":
        if day_id_int == 9:
            return None
        elif day_id_int > 9:    
            return day_id_int - 1
    return day_id_int
    

def is_passive1(any_node):
    return get_session_index(any_node) == 0
def is_passive2(any_node):
    return get_session_index(any_node) == 6
def is_training(any_node):
    session_index = get_session_index(any_node)
    return session_index != 0 and session_index != 6


def has_passive_puff(any_node):
    return any_node.info.get("trial_type") == "PuffOnly"
def has_passive_blank(any_node):
    return any_node.info.get("trial_type") == "BlankOnly"
def has_passive_water(any_node):
    return any_node.info.get("trial_type") in ["CueWater", ]
def has_passive_nowater(any_node):
    return any_node.info.get("trial_type") in ["CueNoWater", ]

def has_training_puff(any_node):
    return any_node.info.get("trial_type") in ["CuePuffWater", "CuePuffNoWater"]
def has_training_blank(any_node):
    return any_node.info.get("trial_type") in ["CueBlankWater", "CueBlankNoWater"]
def has_training_water(any_node):
    return any_node.info.get("trial_type") in ["CuePuffWater", "CueBlankWater"]
def has_training_nowater(any_node):
    return any_node.info.get("trial_type") in ["CuePuffNoWater", "CueBlankNoWater"]


DEFAULT_FS = 5.11
PREDEFINED_T = np.linspace(-5, 5, int(10 * DEFAULT_FS * 2))

def fluorescence_snapshot(trial_nodes: DataSet, alignment_events: tuple[str], plot_manual: PlotManual) -> np.ndarray:
    sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual)
    group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                            for single_trial in sync_trial_nodes], 
                                        baseline_subtraction=None,
                                        _predefined_t=PREDEFINED_T)
    return np.nanmean(group_fluo.raw_array, axis=0)

def store_dataset_snapshot(dataset: DataSet, save_path: str, cell_level: str):
    assert cell_level in ("cellday", "cellsession"), f"Unsupported cell_level: {cell_level}"

    tosave_nodes = []
    for celllevel_node in dataset.select(cell_level):
        if get_day_index(celllevel_node) not in (4, 5, 6, 7, 8, 9, 10, 11):
            continue
        print(f"Processing {celllevel_node.coordinate}...")
        celllevel_subtree = dataset.subtree(celllevel_node)
        celllevel_node.__dict__["snapshot"] = {}
        for trial_type_name, trial_type_func in zip([
            "passive_puff", "passive_blank", "acc_water", "acc_nowater",
            "training_puff", "training_blank", 
        ], [
            has_passive_puff, has_passive_blank, has_passive_water, has_passive_nowater,
            has_training_puff, has_training_blank,
        ]):
            type_nodes = celllevel_subtree.select("trial", _self=lambda node: trial_type_func(node), _empty_warning=False)
            if len(type_nodes) == 0:
                continue
            fluo_array = fluorescence_snapshot(type_nodes, alignment_events=ALL_ALIGNMENT_STYLE["Aligned2Adaptive"], 
                                               plot_manual=PlotManual(fluorescence=True))
            assert fluo_array.shape == (102,), f"Unexpected fluorescence array shape: {fluo_array.shape} at {celllevel_node.coordinate} {trial_type_name}"
            celllevel_node.__dict__["snapshot"][trial_type_name] = fluo_array
        tosave_nodes.append(celllevel_node)
    tosave_dataset = dataset.subset(_self=lambda node: (not isinstance(node, Trial)) and (not isinstance(node, FovTrial)) and any(node.coordinate.contains(single_node.coordinate) for single_node in tosave_nodes))
    with open(save_path, "wb") as f:
        pkl.dump(tosave_dataset, f)


def get_dataset_snapshot(save_path: str) -> DataSet:
    assert os.path.exists(save_path), f"Dataset snapshot not found at {save_path}. Please run store_dataset_snapshot."
    with open(save_path, "rb") as f:
        dataset = pkl.load(f)
    print(dataset)
    for celllevel_node in dataset.select("cellsession"):
        print(celllevel_node.coordinate, f"\nSnapshots: {'; '.join(['{}: {}'.format(k, v.shape) for k, v in celllevel_node.snapshot.items()])}")
        for trial_type_name, fluo_array in celllevel_node.snapshot.items():
            if np.isnan(fluo_array).any():
                print(f"NaN values found in snapshot at {celllevel_node.coordinate} {trial_type_name}: {fluo_array}")
                celllevel_node.snapshot[trial_type_name] = np.nan_to_num(fluo_array, nan=0.0)

    print(f"Loaded dataset snapshot from {save_path}")
    return dataset
    

def peak_reliability_clustering(
        dataset: DataSet,

        color: str,
        line_color: str,
        activity_range: tuple[float, float],

        responsive_percentile: int, 
        axis_func_name: str,
        _element_trial_level: str = "trial",
        _aligment_style: str = "Aligned2Adaptive",
):

    alignment_events = ALL_ALIGNMENT_STYLE[_aligment_style]
    print(f"Alignment events: {alignment_events}")
    plot_manual_fluo = PlotManual(fluorescence=True, baseline_subtraction=None)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 9

    
    def calculate_trial_auc(trial_nodes: DataSet, period_range: tuple[float, float]) -> np.ndarray:
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        max_fs = max([single_trial.data.fluorescence.df_f0.fs for single_trial in sync_trial_nodes])
        predefined_t = np.linspace(period_range[0], period_range[1], 
                                    int((period_range[1] - period_range[0]) * max_fs * 2))
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None,
                                            _predefined_t=predefined_t)
        early_mask = (group_fluo.t >= period_range[0]) & (group_fluo.t <= period_range[1])
        return np.trapezoid(group_fluo.raw_array[:, early_mask], group_fluo.t[early_mask])

    def calculate_trial_peak(trial_nodes: DataSet, period_range: tuple[float, float]) -> np.ndarray:
        sync_trial_nodes = sync_nodes(trial_nodes, alignment_events, plot_manual=plot_manual_fluo)
        max_fs = max([single_trial.data.fluorescence.df_f0.fs for single_trial in sync_trial_nodes])
        predefined_t = np.linspace(period_range[0], period_range[1], 
                                    int((period_range[1] - period_range[0]) * max_fs * 2))
        group_fluo = grouping_timeseries([single_trial.data.fluorescence.df_f0.squeeze(0)
                                                for single_trial in sync_trial_nodes], 
                                            baseline_subtraction=None,
                                            _predefined_t=predefined_t)
        early_mask = (group_fluo.t >= period_range[0]) & (group_fluo.t <= period_range[1])
        return np.max(group_fluo.raw_array[:, early_mask], axis=1)

    assert axis_func_name in ("auc", "peak"), f"Unsupported axis_func_name: {axis_func_name}"
    axis_func = calculate_trial_auc if axis_func_name == "auc" else calculate_trial_peak

    range_str = f"{activity_range[0]:.1f}s_{activity_range[1]:.1f}s"
    dataset_name = "PSE" if "PSE" in dataset.name else "SAT" 
    baseline_acc_days = (4, 5, 6)   
    target_training_days = (7, 8, 9, 10, 11)
    n_training_days = len(target_training_days)

    y_axis_reliability_dict, x_axis_puff_dict = {}, {}
    metric_function_dict = defaultdict(dict)

    # calculate responsive_threshold
    all_acc456_trial_nodes = dataset.select(
        _element_trial_level, _self=lambda node: (get_day_index(node) in baseline_acc_days) and has_passive_puff(node)
    )
    all_x_values = axis_func(all_acc456_trial_nodes, period_range=activity_range)
    responsive_threshold = np.nanpercentile(all_x_values, responsive_percentile)

    for cell_node in dataset.select("cell"):
        cell_subtree = dataset.subtree(cell_node)
        
        acc456_trial_nodes = cell_subtree.select(
            _element_trial_level, _self=lambda node: (get_day_index(node) in baseline_acc_days) and has_passive_puff(node))
        
        x_values = axis_func(acc456_trial_nodes, period_range=activity_range)
        x_axis_puff_dict[cell_node] = np.nanmean(x_values)
        y_axis_reliability_dict[cell_node] = np.sum(x_values > responsive_threshold) / len(x_values)

        for metric_name, metric_function in [("peak", calculate_trial_peak), ("auc", calculate_trial_auc)]:
            print(f"Processing cell {cell_node.coordinate} for metric {metric_name}...")
            for trial_type_name in ("puff", "blank"):
                if trial_type_name == "puff":
                    baseline_trial_nodes = cell_subtree.select(
                        _element_trial_level, _self=lambda node: (get_day_index(node) in baseline_acc_days) and has_passive_puff(node))
                    target_trial_nodes_group = cell_subtree.select(
                        _element_trial_level, _self=lambda node: (get_day_index(node) in target_training_days) and has_puff(node))
                elif trial_type_name == "blank":
                    baseline_trial_nodes = cell_subtree.select(
                        _element_trial_level, _self=lambda node: (get_day_index(node) in baseline_acc_days) and has_passive_blank(node))
                    target_trial_nodes_group = cell_subtree.select( 
                        _element_trial_level, _self=lambda node: (get_day_index(node) in target_training_days) and has_blank(node))
                else:
                    raise ValueError(f"Unsupported trial type: {trial_type_name}")
                
                baseline_value = np.nanmean(metric_function(baseline_trial_nodes, period_range=activity_range))

                for target_trial_group in ("passive", "training"):
                    if target_trial_group == "passive":
                        target_trial_nodes = target_trial_nodes_group.select(
                            _element_trial_level, _self=lambda node: is_passive1(node) or is_passive2(node))
                    elif target_trial_group == "training":
                        target_trial_nodes = target_trial_nodes_group.select(
                            _element_trial_level, _self=lambda node: is_training(node)) 
                    else:
                        raise ValueError(f"Unsupported target trial group: {target_trial_group}")
                    
                    for target_day in target_training_days:
                        target_trail_nodes_at_day = target_trial_nodes.select(
                            _element_trial_level, _self=lambda node: get_day_index(node) == target_day)
                        target_value = np.nanmean(metric_function(target_trail_nodes_at_day, period_range=activity_range))

                        for metric_type in ("raw", "foldchange"):
                            if metric_type == "raw":
                                final_value = target_value
                            elif metric_type == "foldchange":
                                final_value = target_value / (abs(baseline_value) + 1e-6)
                            else:
                                raise ValueError(f"Unsupported metric type: {metric_type}")
                            
                            final_metric_name = f"{trial_type_name} {target_trial_group}\n{metric_name} {range_str} {metric_type}\n{dataset_name}{target_day-6}"
                            metric_function_dict[cell_node][final_metric_name] = final_value

    n_metrics = 4

    fig, axs = plt.subplots(n_metrics+1, n_training_days, 
                            figsize=(n_training_days*2.2, (n_metrics+1)*2.0), constrained_layout=True)
    
    mice_names = ["SUS6F", "SUT2M", "SUT4M", "RZJ4M", "RZJ5M", "SMW7F"]
    mice_colors = distinctipy.get_colors(len(mice_names))

    def plot_scatter_at_ax(ax, metric_name):
        x_values = [x_axis_puff_dict[cell_node] for cell_node in dataset.select("cell")]
        y_values = [y_axis_reliability_dict[cell_node] for cell_node in dataset.select("cell")]
        if metric_name == "mice_id":
            cell_colors = [mice_colors[mice_names.index(cell_node.mice_id)] for cell_node in dataset.select("cell")]
            ax.scatter(x_values, y_values, c=cell_colors, alpha=0.8, s=8)
            add_textonly_legend(ax, {mice_name: {"color": mice_colors[i], "alpha": 0.8} for i, mice_name in enumerate(mice_names)}, title="Mice ID", ncol=2)
        elif metric_name is None:
            ax.scatter(x_values, y_values, color=line_color, alpha=0.8, s=8)
        else:
            metric_values = [metric_function_dict[cell_node][metric_name] for cell_node in dataset.select("cell")]
            if "foldchange" in metric_name:
                norm = LogNorm(vmin=0.5, vmax=2)
                sc = ax.scatter(x_values, y_values, c=metric_values, cmap="RdYlBu_r", norm=norm, alpha=0.8, s=8)
                colorbar = ax.figure.colorbar(sc, ax=ax)
                colorbar.set_ticks([0.5, 1, 2])
                colorbar.set_ticklabels(["0.5", "1.0", "2.0"])
                colorbar.minorticks_off()
            else:
                sc = ax.scatter(x_values, y_values, c=metric_values, cmap="viridis", alpha=0.8, s=8)
                ax.figure.colorbar(sc, ax=ax)
        ax.set_xlabel(f"Puff {axis_func_name} ({DF_F0_SIGN})")
        ax.set_ylabel(f"Puff {axis_func_name} reliability (%)")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        # ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, None)
        ax.spines[['right', 'top']].set_visible(False)
    
    plot_scatter_at_ax(axs[0, 0], metric_name=None)
    plot_scatter_at_ax(axs[0, 1], metric_name="mice_id")

    row_offset = 1
    for metric_name in ("peak", "auc"):
        for trial_type_name in ("puff", ):
            for target_trial_group in ("training", "passive"):
                for metric_type in ( "foldchange",):
                    for col_id, target_day in enumerate(target_training_days):
                        full_metric_name = f"{trial_type_name} {target_trial_group}\n{metric_name} {range_str} {metric_type}\n{dataset_name}{target_day-6}"
                        plot_scatter_at_ax(axs[row_offset, col_id], full_metric_name)
                        axs[row_offset, col_id].set_title(f"{target_trial_group} {trial_type_name}\n{metric_name} {metric_type}\n{dataset_name}{target_day-6}", fontsize=8)
                    row_offset += 1
    
    task_str = f"{axis_func_name}_{range_str}_thres{responsive_threshold}"
    save_path = routing.default_fig_path(dataset, f"Peak_Reliability_Clustering\\Task_{task_str}_" + f"_{{}}_{_aligment_style}.png", fov_skip=True)
    default_exit_save(fig, save_path, _transparent=True)



def trial_response_shape_clustering(sat_dataset: DataSet, pse_dataset: DataSet, preprocess: str, 
                                    cell_level: str,
                                    _include_blank: bool, _normalize: bool = True):
    key_stages = [
        (-5, -1),
        (-1, 0),
        (0, 0.25),
        (0.25, 0.5),
        (0.5, 1),
        (1, 1.5),
        (1.5, 2),
        (2, 2.25),
        (2.25, 2.5),
        (2.5, 3),
        (3, 3.5),
        (3.5, 5)
    ]
    key_stage_midpoints = [(start + end) / 2 for start, end in key_stages]

    # n_b_choices = (15, )
    # metric_choices = ("euclidean",)
    # n_b_choices = (5, 10, 15, 20, 25, 30, 40, 50)
    # metric_choices = ("euclidean", "correlation", "cosine", "chebyshev", "manhattan", "minkowski")

    sat_celllevel_nodes = sat_dataset.select(cell_level)
    pse_celllevel_nodes = pse_dataset.select(cell_level)

    combine_mice = [mouse_node.mice_id for mouse_node in sat_dataset.select("mice")] + [mouse_node.mice_id for mouse_node in pse_dataset.select("mice")]
    n_mice = len(combine_mice)

    combined_celllevel_nodes = sat_celllevel_nodes + pse_celllevel_nodes
    raw_celllevel_data = [
        (celllevel_node, trial_type_name, fluo_array)
        for celllevel_node in combined_celllevel_nodes
        if get_day_index(celllevel_node) in ( 7, )
        for trial_type_name, fluo_array in celllevel_node.snapshot.items()
        # if trial_type_name in ("training_puff",  "training_blank", "passive_puff", "passive_blank")
        if trial_type_name in (("training_puff",  "training_blank") if _include_blank else ("training_puff", ))
    ]

    def downsample(fluo_array: np.ndarray) -> np.ndarray:
        stage_masks = [(start <= PREDEFINED_T) & (PREDEFINED_T <= end) for start, end in key_stages]
        # print(stage_masks)
        downsampled_array = np.array([np.nanmean(fluo_array[mask]) for mask in stage_masks])
        return downsampled_array
        
    def norm(arr: np.ndarray) -> np.ndarray:
        if _normalize:
            return arr/np.nanstd(arr)
        else:
            return arr
        

    if preprocess == "raw":
        process_data = np.vstack([norm(fluo_array) for _, _, fluo_array in raw_celllevel_data])
    elif preprocess == "downsample":
        process_data = np.vstack([downsample(norm(fluo_array)) for _, _, fluo_array in raw_celllevel_data])
    else:
        raise ValueError(f"Unsupported preprocess type: {preprocess}")
    
    setting_str = f"{preprocess}_{'with_blank' if _include_blank else 'no_blank'}_{'normalized' if _normalize else 'unnormalized'}_{cell_level}"
    print(f"{setting_str} processed data shape: {process_data.shape}")

    # embeddings_2d = {}
    # embeddings_2d["PCA"] = PCA(n_components=2).fit_transform(process_data)
    # for n_b in n_b_choices:
    #     for metr in metric_choices:
    #         print(f"Calculating 2D UMAP embedding with n_b={n_b}, metric={metr}...")
    #         embedding_name = f"UMAP_nb{n_b}_{metr}"
    #         embeddings_2d[embedding_name] = UMAP(n_components=2, n_neighbors=n_b, metric=metr).fit_transform(process_data)
            
    # embeddings_3d = {}
    # embeddings_3d["PCA"] = PCA(n_components=3).fit_transform(process_data)
    # for n_b in n_b_choices:
    #     for metr in metric_choices:
    #         print(f"Calculating 3D UMAP embedding with n_b={n_b}, metric={metr}...")
    #         embedding_name = f"UMAP_nb{n_b}_{metr}"
    #         embeddings_3d[embedding_name] = UMAP(n_components=3, n_neighbors=n_b, metric=metr).fit_transform(process_data)

    embedding_2d = UMAP(n_components=2, n_neighbors=15, metric="euclidean", random_state=42).fit_transform(process_data)
    embedding_3d = UMAP(n_components=3, n_neighbors=15, metric="euclidean", random_state=42).fit_transform(process_data)

    from mpl_toolkits.mplot3d import Axes3D

    def part1():
        def regular_plotting():
            n_b_choices = (15, 30,)
            metric_choices = ("euclidean", "correlation")

            fig = plt.figure(figsize=(4.5 * 2, 4 * len(n_b_choices) * len(metric_choices)), constrained_layout=True)
            for i, n_b in enumerate(n_b_choices):
                for j, metr in enumerate(metric_choices):
                    tmp_setting_str = f"UMAP_nb{n_b}_{metr}"
                    tmp_embedding_2d = UMAP(n_components=2, n_neighbors=n_b, metric=metr, random_state=42).fit_transform(process_data)
                    tmp_embedding_3d = UMAP(n_components=3, n_neighbors=n_b, metric=metr, random_state=42).fit_transform(process_data)
                    ax = fig.add_subplot(len(n_b_choices)*len(metric_choices), 2, (i*len(metric_choices)+j)*2+1)
                    ax.scatter(tmp_embedding_2d[:, 0], tmp_embedding_2d[:, 1], alpha=0.8, s=6, color="k", edgecolors="w", lw=0.1)
                    ax.set_title(f"2D UMAP Embedding ({tmp_setting_str})")
                    ax.set_xlabel("UMAP Dimension 1")
                    ax.set_ylabel("UMAP Dimension 2")
                    ax.spines[['right', 'top']].set_visible(False)

                    ax = fig.add_subplot(len(n_b_choices)*len(metric_choices), 2, (i*len(metric_choices)+j)*2+2, projection='3d')
                    ax.scatter(tmp_embedding_3d[:, 0], tmp_embedding_3d[:, 1], tmp_embedding_3d[:, 2], alpha=0.8, s=6, color="k", edgecolors="w", lw=0.2)
                    ax.view_init(elev=20, azim=-150)
                    ax.set_title(f"3D UMAP Embedding ({tmp_setting_str})")
                    ax.set_xlabel("UMAP Dimension 1")
                    ax.set_ylabel("UMAP Dimension 2")
                    ax.set_zlabel("UMAP Dimension 3")
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False)
        
        def rotating_3d_video():
            from matplotlib.animation import FuncAnimation
            fig = plt.figure(figsize=(4, 4), constrained_layout=True)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], alpha=0.8, s=6, color="k", edgecolors="w", lw=0.2)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")

            def update(angle):
                ax.view_init(elev=20, azim=angle)
                return ax,

            ani = FuncAnimation(
                fig,
                update,
                frames=np.linspace(-150, -150+360, 720),
                interval=1000 / 60,
                blit=False,
            )
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_rotating.gif", fov_skip=True)
            ani.save(save_path, writer='ffmpeg', fps=60, dpi=200)

        def examples_plotting(n_example=50):
            random_indices = np.random.choice(embedding_2d.shape[0], size=n_example, replace=False)
            fig = plt.figure(figsize=(3 * 3, n_example*3.5), constrained_layout=True)
            for i, random_index in enumerate(random_indices):
                random_node = raw_celllevel_data[random_index][0]

                ax = fig.add_subplot(n_example, 3, i*3+1)
                ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.2, s=6, color="k", edgecolors="w", lw=0.1)
                ax.scatter(embedding_2d[random_index, 0], embedding_2d[random_index, 1], alpha=1.0, s=50, color="r", edgecolors="w", lw=0.5)
                ax.spines[['right', 'top']].set_visible(False)

                ax = fig.add_subplot(n_example, 3, i*3+2, projection='3d')
                ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], alpha=0.2, s=6, color="k", edgecolors="w", lw=0.2)
                ax.scatter(embedding_3d[random_index, 0], embedding_3d[random_index, 1], embedding_3d[random_index, 2], 
                        alpha=1.0, s=50, color="r", edgecolors="w", lw=0.5, zorder=10)
                ax.view_init(elev=20, azim=-150)
                ax.set_title(f"Fluorescence Trace ({setting_str})\n{random_node.object_uid}\nDay {get_day_index(random_node)} {raw_celllevel_data[random_index][1]}")

                raw_fluo = raw_celllevel_data[random_index][2]
                downsampled_fluo = downsample(raw_fluo)
                ax = fig.add_subplot(n_example, 3, i*3+3)
                ax.plot(PREDEFINED_T, raw_fluo, label="Raw", color="gray", alpha=0.7)
                ax.plot(key_stage_midpoints, downsampled_fluo, label="Downsampled", color="black", alpha=0.7)
                
                ax.spines[['right', 'top']].set_visible(False)
                ax.set_ylim(-0.15, 0.5)
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_examples.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False)

        def interactive_plotting():
            from mpl_toolkits.mplot3d import proj3d

            fig = plt.figure(figsize=(12, 4), constrained_layout=True)
            ax2d = fig.add_subplot(131)
            ax3d = fig.add_subplot(132, projection='3d')
            axtrace = fig.add_subplot(133)

            selected_index = 0

            # ---------- background scatters ----------
            ax2d.scatter(
                embedding_2d[:, 0], embedding_2d[:, 1],
                alpha=0.2, s=6, color="k", edgecolors="w", lw=0.1
            )
            selected_2d = ax2d.scatter(
                embedding_2d[selected_index, 0], embedding_2d[selected_index, 1],
                alpha=1.0, s=50, color="r", edgecolors="w", lw=0.5, zorder=10
            )
            ax2d.spines[['right', 'top']].set_visible(False)
            ax2d.set_title("2D embedding")

            ax3d.scatter(
                embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
                alpha=0.2, s=6, color="k", edgecolors="w", lw=0.2
            )
            selected_3d = ax3d.scatter(
                embedding_3d[selected_index, 0],
                embedding_3d[selected_index, 1],
                embedding_3d[selected_index, 2],
                alpha=1.0, s=50, color="r", edgecolors="w", lw=0.5, zorder=10
            )
            ax3d.view_init(elev=20, azim=-150)

            # ---------- trace panel ----------
            raw_fluo = raw_celllevel_data[selected_index][2]
            downsampled_fluo = downsample(raw_fluo)

            trace_raw_line, = axtrace.plot(
                PREDEFINED_T, raw_fluo, label="Raw", color="gray", alpha=0.7
            )
            trace_ds_line, = axtrace.plot(
                key_stage_midpoints, downsampled_fluo, label="Downsampled", color="black", alpha=0.7
            )

            axtrace.spines[['right', 'top']].set_visible(False)
            axtrace.set_ylim(-0.15, 0.5)
            axtrace.legend(frameon=False)

            def update_selected(idx):
                nonlocal selected_index

                selected_index = idx
                random_node = raw_celllevel_data[idx][0]

                # update 2D selected point
                selected_2d.set_offsets(embedding_2d[[idx]])

                # update 3D selected point
                selected_3d._offsets3d = (
                    [embedding_3d[idx, 0]],
                    [embedding_3d[idx, 1]],
                    [embedding_3d[idx, 2]],
                )

                # update 3D title
                ax3d.set_title(
                    f"Fluorescence Trace ({setting_str})\n"
                    f"{random_node.object_uid}\n"
                    f"Day {get_day_index(random_node)} S{get_session_index(random_node)} {raw_celllevel_data[idx][1]}"
                )

                # update trace
                raw_fluo = raw_celllevel_data[idx][2]
                downsampled_fluo = downsample(raw_fluo)
                trace_raw_line.set_ydata(raw_fluo)
                trace_ds_line.set_ydata(downsampled_fluo)

                axtrace.relim()
                axtrace.autoscale_view(scalex=False, scaley=True)
                # axtrace.set_ylim(-0.15, 0.5)

                fig.canvas.draw_idle()

            def nearest_point_2d(event):
                xy_pixels = ax2d.transData.transform(embedding_2d)
                mouse_xy = np.array([event.x, event.y])
                d2 = np.sum((xy_pixels - mouse_xy) ** 2, axis=1)
                idx = np.argmin(d2)
                return idx, d2[idx]

            def nearest_point_3d(event):
                x3, y3, z3 = embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2]
                x2, y2, _ = proj3d.proj_transform(x3, y3, z3, ax3d.get_proj())
                xy_proj = np.column_stack([x2, y2])
                xy_pixels = ax3d.transData.transform(xy_proj)

                mouse_xy = np.array([event.x, event.y])
                d2 = np.sum((xy_pixels - mouse_xy) ** 2, axis=1)
                idx = np.argmin(d2)
                return idx, d2[idx]

            def on_click(event):
                if event.inaxes is None:
                    return
                if event.x is None or event.y is None:
                    return

                if event.inaxes == ax2d:
                    idx, dist2 = nearest_point_2d(event)
                    if dist2 < 100:   # ~10 pixel threshold
                        update_selected(idx)

                elif event.inaxes == ax3d:
                    idx, dist2 = nearest_point_3d(event)
                    if dist2 < 200:   # slightly looser threshold for 3D
                        update_selected(idx)

            fig.canvas.mpl_connect("button_press_event", on_click)

            # initialize title
            random_node = raw_celllevel_data[selected_index][0]
            ax3d.set_title(
                f"Fluorescence Trace ({setting_str})\n"
                f"{random_node.object_uid}\n"
                f"Day {get_day_index(random_node)} {raw_celllevel_data[selected_index][1]}"
            )

            plt.show()


        regular_plotting()
        rotating_3d_video()
        examples_plotting()
        interactive_plotting()

    def part2():
        def regular_plot_with_color(colors: list[str], title_suffix: str, legend_dict: dict[str, dict], **kwargs):
            fig = plt.figure(figsize=(9, 4), constrained_layout=True)
            ax = fig.add_subplot(121)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.8, s=6, color=colors, edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            add_textonly_legend(ax, legend_dict, loc="lower left", **kwargs)

            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], alpha=0.8, s=6, color=colors, edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_{title_suffix}.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)
        
        # color by dataset
        dataset_colors = [
            "deepskyblue" if "SAT" in cellday_node.cohort_id else "orangered"
            for cellday_node, trial_type_name, _ in raw_celllevel_data
        ]
        regular_plot_with_color(dataset_colors, title_suffix="by_dataset", legend_dict={"SAT": {"color": "deepskyblue"}, "PSE": {"color": "orangered"}})

        dataset_trial_colors = [
            "deepskyblue" if ("SAT" in cellday_node.cohort_id and "puff" in trial_type_name) else
            "#D4E6F6" if ("SAT" in cellday_node.cohort_id and "blank" in trial_type_name) else
            "orangered" if ("PSE" in cellday_node.cohort_id and "puff" in trial_type_name) else
            "#FFD5D5"
            for cellday_node, trial_type_name, _ in raw_celllevel_data
        ]
        regular_plot_with_color(dataset_trial_colors, title_suffix="by_dataset_trialtype", legend_dict={
            "SAT Puff": {"color": "deepskyblue"},
            "SAT Blank": {"color": "#D4E6F6"},
            "PSE Puff": {"color": "orangered"},
            "PSE Blank": {"color": "#FFD5D5"}
        })

        from distinctipy import distinctipy
        colors = distinctipy.get_colors(n_mice)
        # colors = ["#1D4ED8", "#2563EB", "#38BDF8", "#147891",
        #           "#E63946", "#C1121F", "#FF6B6B", "#A4161A"]
        mice_colors = [
            colors[combine_mice.index(cellday_node.mice_id)]
            for cellday_node, trial_type_name, _ in raw_celllevel_data
        ]
        regular_plot_with_color(mice_colors, title_suffix="by_mouse", 
                                legend_dict={mice_id: {"color": colors[i]} for i, mice_id in enumerate(combine_mice)},
                                ncol=2)
        
        
        def values_to_cmap(values, cmap_name="viridis", vmin=None, vmax=None):
            """
            Map a list/array of floats to RGBA colors from a matplotlib colormap.
            """
            values = np.asarray(values)

            if vmin is None:
                vmin = np.nanmin(values)
            if vmax is None:
                vmax = np.nanmax(values)

            norm = LogNorm(vmin=vmin, vmax=vmax)
            cmap = plt.get_cmap(cmap_name)

            return cmap(norm(values))
        peak_colors = values_to_cmap([np.nanmax(fluo_array) for _, _, fluo_array in raw_celllevel_data], cmap_name="bwr")
        regular_plot_with_color(peak_colors, title_suffix="by_peak", legend_dict={"Higher Peak": {"color": "red"}, "Lower Peak": {"color": "blue"}})


    def get_item_index(celllevel_node):
        if cell_level == "cellday":
            return get_day_index(celllevel_node) - 7
        elif cell_level == "cellsession":
            return get_session_index(celllevel_node) - 1
        else:
            raise ValueError(f"Unsupported cell level: {cell_level}")
        

    def part3():
        sat_day_colors = ["#9EC5FE", "#5B9BFF", "#2563EB", "#1D3FA6", "#0A2A66"]
        pse_day_colors = ["#F2A3AA", "#E57373", "#D94A4A", "#B71C1C", "#6E0F14"]
        from matplotlib.colors import to_rgba

        def hex_alpha_to_rgba(hex_colors, alphas):
            """
            Convert lists of hex colors and alpha values to RGBA tuples.

            Parameters
            ----------
            hex_colors : list[str]
                Hex colors, e.g. ["#ff0000", "#00ff00", "#0000ff"].
            alphas : list[float]
                Alpha values between 0 and 1, same length as hex_colors.

            Returns
            -------
            list[tuple[float, float, float, float]]
                RGBA colors usable in matplotlib.
            """
            if len(hex_colors) != len(alphas):
                raise ValueError("hex_colors and alphas must have the same length")

            return [to_rgba(color, alpha=alpha) for color, alpha in zip(hex_colors, alphas)]


        def plot_daywise_plasticity():

            fig = plt.figure(figsize=(9, 8), constrained_layout=True)

            sat_dot_colors = [
                sat_day_colors[get_item_index(celllevel_node)] if "SAT" in celllevel_node.cohort_id else
                "gray"
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            sat_dot_alphas = [
                0.8 if "SAT" in celllevel_node.cohort_id else 0.1
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            legend_dict = {f"SAT{day+1}": {"color": sat_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(221)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=sat_dot_alphas, s=6, color=sat_dot_colors, edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(222, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), 
                       edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            pse_dot_colors = [
                pse_day_colors[get_item_index(celllevel_node)] if "PSE" in celllevel_node.cohort_id else
                "gray"
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            pse_dot_alphas = [
                0.8 if "PSE" in celllevel_node.cohort_id else 0.1
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            legend_dict = {f"PSE{day+1}": {"color": pse_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(223)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=pse_dot_alphas, s=6, color=pse_dot_colors, edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(224, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_daywise_plasticity.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)

        def plot_daywise_plasticity_PB():
            fig = plt.figure(figsize=(9, 8), constrained_layout=True)

            sat_dot_colors = [
                sat_day_colors[get_item_index(celllevel_node)] if "SAT" in celllevel_node.cohort_id and trial_type_name == "training_puff" else
                "gray"
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            sat_dot_alphas = [
                0.8 if "SAT" in celllevel_node.cohort_id and trial_type_name == "training_puff" else 0.1
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            legend_dict = {f"SAT{day+1} Puff": {"color": sat_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(221)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(222, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            pse_dot_colors = [
                pse_day_colors[get_item_index(celllevel_node)] if "PSE" in celllevel_node.cohort_id and trial_type_name == "training_puff" else
                "gray"
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            pse_dot_alphas = [
                0.8 if "PSE" in celllevel_node.cohort_id and trial_type_name == "training_puff" else 0.1
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            legend_dict = {f"PSE{day+1} Puff": {"color": pse_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(223)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(224, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_daywise_plasticityPuff.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)

            fig = plt.figure(figsize=(9, 8), constrained_layout=True)

            sat_dot_colors = [
                sat_day_colors[get_item_index(celllevel_node)] if "SAT" in celllevel_node.cohort_id and trial_type_name == "training_blank" else
                "gray"
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            sat_dot_alphas = [
                0.8 if "SAT" in celllevel_node.cohort_id and trial_type_name == "training_blank" else 0.1
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            legend_dict = {f"SAT{day+1} Blank": {"color": sat_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(221)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(222, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            pse_dot_colors = [
                pse_day_colors[get_item_index(celllevel_node)] if "PSE" in celllevel_node.cohort_id and trial_type_name == "training_blank" else
                "gray"
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            pse_dot_alphas = [
                0.8 if "PSE" in celllevel_node.cohort_id and trial_type_name == "training_blank" else 0.1
                for celllevel_node, trial_type_name, _ in raw_celllevel_data
            ]
            legend_dict = {f"PSE{day+1} Blank": {"color": pse_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(223)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(224, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_daywise_plasticityBlank.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)

        def plot_daywise_plasticity_PB_single_cell():
            all_cells = sat_dataset.select("cell") + pse_dataset.select("cell")
            cell_colors = distinctipy.get_colors(len(all_cells))
            fig = plt.figure(figsize=(9, 8), constrained_layout=True)

            def get_cell_index(cellday_node):
                for i, cell_node in enumerate(all_cells):
                    if cell_node.contains(cellday_node):
                        return i
                raise ValueError(f"Cell ID {cellday_node.cell_id} not found in all_cells")
            sat_dot_colors = [
                cell_colors[get_cell_index(cellday_node)] if "SAT" in cellday_node.cohort_id and trial_type_name == "training_puff" else
                "gray"
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            sat_dot_alphas = [
                0.8 if "SAT" in cellday_node.cohort_id and trial_type_name == "training_puff" else 0.1
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            legend_dict = {f"SAT{day+1} Puff": {"color": sat_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(221)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            # add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(222, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            pse_dot_colors = [
                cell_colors[get_cell_index(cellday_node)] if "PSE" in cellday_node.cohort_id and trial_type_name == "training_puff" else
                "gray"
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            pse_dot_alphas = [
                0.8 if "PSE" in cellday_node.cohort_id and trial_type_name == "training_puff" else 0.1
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            legend_dict = {f"PSE{day+1} Puff": {"color": pse_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(223)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            # add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(224, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_daywise_plasticityPuff_SingleCell.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)

            fig = plt.figure(figsize=(9, 8), constrained_layout=True)

            sat_dot_colors = [
                cell_colors[get_cell_index(cellday_node)] if "SAT" in cellday_node.cohort_id and trial_type_name == "training_blank" else
                "gray"
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            sat_dot_alphas = [
                0.8 if "SAT" in cellday_node.cohort_id and trial_type_name == "training_blank" else 0.1
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            legend_dict = {f"SAT{day+1} Blank": {"color": sat_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(221)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            # add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(222, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            pse_dot_colors = [
                cell_colors[get_cell_index(cellday_node)] if "PSE" in cellday_node.cohort_id and trial_type_name == "training_blank" else
                "gray"
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            pse_dot_alphas = [
                0.8 if "PSE" in cellday_node.cohort_id and trial_type_name == "training_blank" else 0.1
                for cellday_node, trial_type_name, _ in raw_cellday_data
            ]
            legend_dict = {f"PSE{day+1} Blank": {"color": pse_day_colors[day], "alpha": 0.8} for day in range(5)}
            ax = fig.add_subplot(223)
            ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.1)
            ax.set_title(f"2D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.spines[['right', 'top']].set_visible(False)
            # add_textonly_legend(ax, legend_dict, loc="lower left")

            ax = fig.add_subplot(224, projection='3d')
            ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.2)
            ax.view_init(elev=20, azim=-150)
            ax.set_title(f"3D UMAP Embedding ({setting_str})")
            ax.set_xlabel("UMAP Dimension 1")
            ax.set_ylabel("UMAP Dimension 2")
            ax.set_zlabel("UMAP Dimension 3")
            
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_daywise_plasticityBlank_SingleCell.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)


        def plot_daywise_single_plasticity():

            for day_id in (7, 8, 9, 10, 11):
                fig = plt.figure(figsize=(9, 8), constrained_layout=True)

                sat_dot_colors = [
                    sat_day_colors[day_id-7] if "SAT" in cellday_node.cohort_id and get_day_index(cellday_node) == day_id and trial_type_name == "training_puff" else
                    "gray"
                    for cellday_node, trial_type_name, _ in raw_cellday_data
                ]
                sat_dot_alphas = [
                    0.8 if "SAT" in cellday_node.cohort_id and get_day_index(cellday_node) == day_id and trial_type_name == "training_puff" else 0.1
                    for cellday_node, trial_type_name, _ in raw_cellday_data
                ]
                legend_dict = {f"SAT{day+1} Puff": {"color": sat_day_colors[day], "alpha": 0.8} for day in range(5)}
                ax = fig.add_subplot(221)
                ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.1)
                ax.set_title(f"2D UMAP Embedding ({setting_str})")
                ax.set_xlabel("UMAP Dimension 1")
                ax.set_ylabel("UMAP Dimension 2")
                ax.spines[['right', 'top']].set_visible(False)
                add_textonly_legend(ax, legend_dict, loc="lower left")

                ax = fig.add_subplot(222, projection='3d')
                ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(sat_dot_colors, sat_dot_alphas), edgecolors="w", lw=0.2)
                ax.view_init(elev=20, azim=-150)
                ax.set_title(f"3D UMAP Embedding ({setting_str})")
                ax.set_xlabel("UMAP Dimension 1")
                ax.set_ylabel("UMAP Dimension 2")
                ax.set_zlabel("UMAP Dimension 3")
                
                pse_dot_colors = [
                    pse_day_colors[day_id-7] if "PSE" in cellday_node.cohort_id and get_day_index(cellday_node) == day_id and trial_type_name == "training_puff" else
                    "gray"
                    for cellday_node, trial_type_name, _ in raw_cellday_data
                ]
                pse_dot_alphas = [
                    0.8 if "PSE" in cellday_node.cohort_id and get_day_index(cellday_node) == day_id and trial_type_name == "training_puff" else 0.1
                    for cellday_node, trial_type_name, _ in raw_cellday_data
                ]
                legend_dict = {f"PSE{day+1} Puff": {"color": pse_day_colors[day], "alpha": 0.8} for day in range(5)}
                ax = fig.add_subplot(223)
                ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.1)
                ax.set_title(f"2D UMAP Embedding ({setting_str})")
                ax.set_xlabel("UMAP Dimension 1")
                ax.set_ylabel("UMAP Dimension 2")
                ax.spines[['right', 'top']].set_visible(False)
                add_textonly_legend(ax, legend_dict, loc="lower left")

                ax = fig.add_subplot(224, projection='3d')
                ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color=hex_alpha_to_rgba(pse_dot_colors, pse_dot_alphas), edgecolors="w", lw=0.2)
                ax.view_init(elev=20, azim=-150)
                ax.set_title(f"3D UMAP Embedding ({setting_str})")
                ax.set_xlabel("UMAP Dimension 1")
                ax.set_ylabel("UMAP Dimension 2")
                ax.set_zlabel("UMAP Dimension 3")
                
                save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_daywise_single_plasticity_day{day_id}.png", fov_skip=True)
                default_exit_save(fig, save_path, _transparent=False,)
    
        plot_daywise_plasticity()
        plot_daywise_plasticity_PB()
        # plot_daywise_plasticity_PB_single_cell()
        # plot_daywise_single_plasticity()

    def part4():
        sat_day_colors = ["#9EC5FE", "#5B9BFF", "#2563EB", "#1D3FA6", "#0A2A66"]
        pse_day_colors = ["#F2A3AA", "#E57373", "#D94A4A", "#B71C1C", "#6E0F14"]
        def plot_each_single_cell():
            all_cells = sat_dataset.select("cell") + pse_dataset.select("cell")
            row_num = min(len(sat_dataset.select("cell")), len(pse_dataset.select("cell")), 20)  # Limit to 20 rows for visibility
            fig = plt.figure(figsize=(2 * 4, row_num * 3), constrained_layout=True)

            for row_id, (sat_cell, pse_cell) in enumerate(zip(sat_dataset.select("cell"), pse_dataset.select("cell"))):
                if row_id >= 20:  # Limit to 20 rows for visibility
                    break
                for cell_node in (sat_cell, pse_cell):
                    cell_colors = sat_day_colors if "SAT" in cell_node.cohort_id else pse_day_colors
                    cell_puff_index = np.array( [
                        i 
                        for item_index in range(5)
                        for i, (celllevel_node, trial_type_name, _) in enumerate(raw_celllevel_data)
                        if cell_node.contains(celllevel_node) and trial_type_name == "training_puff" and get_item_index(celllevel_node) == item_index
                    ])
                    cell_blank_index = np.array( [
                        i 
                        for item_index in range(5)
                        for i, (celllevel_node, trial_type_name, _) in enumerate(raw_celllevel_data)
                        if cell_node.contains(celllevel_node) and trial_type_name == "training_blank" and get_item_index(celllevel_node) == item_index
                    ])
                    
                    ax = fig.add_subplot(row_num, 2, row_id*2+1)
                    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.1, s=6, color="gray", edgecolors="w", lw=0.1)
                    ax.set_title(f"{cell_node.object_uid}")
                    ax.set_xlabel("UMAP Dimension 1")
                    ax.set_ylabel("UMAP Dimension 2")
                    ax.spines[['right', 'top']].set_visible(False)
                    
                    for day_id in range(4):
                        ax.plot(
                            [embedding_2d[cell_puff_index[day_id], 0], embedding_2d[cell_puff_index[day_id + 1], 0]],
                            [embedding_2d[cell_puff_index[day_id], 1], embedding_2d[cell_puff_index[day_id + 1], 1]],
                            color=cell_colors[day_id + 1], alpha=0.8, lw=1.5, ls="-"
                        )
                        if _include_blank:
                            ax.plot(
                                [embedding_2d[cell_blank_index[day_id], 0], embedding_2d[cell_blank_index[day_id + 1], 0]],
                                [embedding_2d[cell_blank_index[day_id], 1], embedding_2d[cell_blank_index[day_id + 1], 1]],
                                color=cell_colors[day_id + 1], alpha=0.8, lw=1.5, ls="--"
                            )
                    

                    ax = fig.add_subplot(row_num, 2, row_id*2+2, projection='3d')
                    ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], s=6, color="gray", alpha=0.1, edgecolors="w", lw=0.2)
                    ax.view_init(elev=20, azim=-150)
                    ax.set_xlabel("UMAP Dimension 1")
                    ax.set_ylabel("UMAP Dimension 2")
                    ax.set_zlabel("UMAP Dimension 3")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])

                    
                    for day_id in range(4):
                        ax.plot(
                            [embedding_3d[cell_puff_index[day_id], 0], embedding_3d[cell_puff_index[day_id + 1], 0]],
                            [embedding_3d[cell_puff_index[day_id], 1], embedding_3d[cell_puff_index[day_id + 1], 1]],
                            [embedding_3d[cell_puff_index[day_id], 2], embedding_3d[cell_puff_index[day_id + 1], 2]],
                            color=cell_colors[day_id + 1], alpha=0.8, lw=1.5, ls="-"
                        )
                        if _include_blank:
                            ax.plot(
                                [embedding_3d[cell_blank_index[day_id], 0], embedding_3d[cell_blank_index[day_id + 1], 0]],
                                [embedding_3d[cell_blank_index[day_id], 1], embedding_3d[cell_blank_index[day_id + 1], 1]],
                                [embedding_3d[cell_blank_index[day_id], 2], embedding_3d[cell_blank_index[day_id + 1], 2]],
                                color=cell_colors[day_id + 1], alpha=0.8, lw=1.5, ls="--"
                            )
                
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\UMAP_{setting_str}_each_single_cell.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=True,)
        
        all_cells = sat_dataset.select("cell") + pse_dataset.select("cell")
        from distinctipy import distinctipy
        colors = distinctipy.get_colors(n_mice)

        all_cell_puff_embedding_2d = []
        all_cell_puff_embedding_3d = []
        cell_colors = []
        cell_mice_colors = []
        for row_id, cell_node in enumerate(all_cells):
            cell_puff_index = np.array( [
                i 
                for item_index in range(5)
                for i, (celllevel_node, trial_type_name, _) in enumerate(raw_celllevel_data)
                if cell_node.contains(celllevel_node) and trial_type_name == "training_puff" and get_item_index(celllevel_node) == item_index
            ])
            cell_puff_embedding_2d = embedding_2d[cell_puff_index].flatten()
            cell_puff_embedding_3d = embedding_3d[cell_puff_index].flatten()
            if _include_blank:
                cell_blank_index = np.array( [
                    i
                    for item_index in range(5)
                    for i, (celllevel_node, trial_type_name, _) in enumerate(raw_celllevel_data)
                    if cell_node.contains(celllevel_node) and trial_type_name == "training_blank" and get_item_index(celllevel_node) == item_index
                ])
                cell_blank_embedding_2d = embedding_2d[cell_blank_index].flatten()
                cell_blank_embedding_3d = embedding_3d[cell_blank_index].flatten()
                cell_history_embedding_2d = np.concatenate([cell_puff_embedding_2d, cell_blank_embedding_2d])
                cell_history_embedding_3d = np.concatenate([cell_puff_embedding_3d, cell_blank_embedding_3d])
                
                all_cell_puff_embedding_2d.append(cell_history_embedding_2d)
                all_cell_puff_embedding_3d.append(cell_history_embedding_3d)
            else:
                all_cell_puff_embedding_2d.append(cell_puff_embedding_2d)
                all_cell_puff_embedding_3d.append(cell_puff_embedding_3d)
            cell_colors.append("deepskyblue" if "SAT" in cell_node.cohort_id else "orangered")
            cell_mice_colors.append(colors[combine_mice.index(cell_node.mice_id)])

        all_cell_puff_embedding_2d = np.vstack(all_cell_puff_embedding_2d)
        all_cell_puff_embedding_3d = np.vstack(all_cell_puff_embedding_3d)
        print(all_cell_puff_embedding_2d.shape, all_cell_puff_embedding_3d.shape, len(cell_colors))


        def group_cells_by_puff_response_history():
            pca_2d = PCA(n_components=2)
            umap_2d = UMAP(n_components=2, n_neighbors=10, random_state=42)
            fig, axes = plt.subplots(2, 4 + 4*4, figsize=(3*(4 + 4*4), 6), constrained_layout=True)

            def add_cluster_ellipses(ax, embedding, labels, cluster_num, n_std=2):
                plotted_embedding = embedding[:, :2]
                for cluster_id in range(cluster_num):
                    cluster_points = plotted_embedding[labels == cluster_id]
                    if len(cluster_points) < 3:
                        continue

                    cov = np.cov(cluster_points, rowvar=False)
                    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
                        continue

                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    eigenvalues = np.maximum(eigenvalues, 0)
                    if np.allclose(eigenvalues, 0):
                        continue

                    order = np.argsort(eigenvalues)[::-1]
                    eigenvalues = eigenvalues[order]
                    eigenvectors = eigenvectors[:, order]
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    width, height = 2 * n_std * np.sqrt(eigenvalues)
                    ellipse = Ellipse(
                        xy=np.mean(cluster_points, axis=0),
                        width=width,
                        height=height,
                        angle=angle,
                        facecolor="none",
                        edgecolor=CLUSTER_COLORS[cluster_id],
                        alpha=0.8,
                        ls='--',
                        lw=1.5,
                        zorder=-1,
                    )
                    ax.add_patch(ellipse)

            for row_id, dim_red_func in enumerate([pca_2d, umap_2d]):
                cell_puff_embedding_2d_red = dim_red_func.fit_transform(all_cell_puff_embedding_2d)
                cell_puff_embedding_3d_red = dim_red_func.fit_transform(all_cell_puff_embedding_3d)

            
                ax = axes[row_id, 0]
                ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], s=6, color=cell_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 2D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.spines[['right', 'top']].set_visible(False)

                ax = axes[row_id, 1]
                ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], s=6, color=cell_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 3D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.spines[['right', 'top']].set_visible(False)

                ax = axes[row_id, 2]
                ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], s=6, color=cell_mice_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 2D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.spines[['right', 'top']].set_visible(False)
                add_textonly_legend(ax, {combine_mice[i]: {"color": colors[i], "alpha": 0.8} for i in range(n_mice)}, ncols=2)

                ax = axes[row_id, 3]
                ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], s=6, color=cell_mice_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 3D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.spines[['right', 'top']].set_visible(False)
                add_textonly_legend(ax, {combine_mice[i]: {"color": colors[i], "alpha": 0.8} for i in range(n_mice)}, ncols=2)
                from sklearn.cluster import SpectralClustering
                for col_id, cluster_num in enumerate([2,3,4,5]):

                    """
                    cluster on reduced coordinate
                    KMeans 4 Clusters (3D, UMAP):
                    SAT in each cluster: 12/61 19.67% 0/61 0.00% 27/61 44.26% 22/61 36.07%
                    PSE in each cluster: 13/56 23.21% 33/56 58.93% 8/56 14.29% 2/56 3.57%
                    
                    
                    Spectral Clustering 4 Clusters (3D, UMAP):
                    SAT in each cluster: 0/61 0.00% 29/61 47.54% 11/61 18.03% 21/61 34.43%
                    PSE in each cluster: 32/56 57.14% 10/56 17.86% 13/56 23.21% 1/56 1.79%
                    """
                    labels_2d = SpectralClustering(n_clusters=cluster_num, random_state=42).fit_predict(cell_puff_embedding_2d_red)
                    n_sat_in_each_cluster = [np.sum([cell_colors[i] == "deepskyblue" for i in range(len(cell_colors)) if labels_2d[i] == cluster_id]) 
                                             for cluster_id in range(cluster_num)]
                    n_pse_in_each_cluster = [np.sum([cell_colors[i] == "orangered" for i in range(len(cell_colors)) if labels_2d[i] == cluster_id])
                                             for cluster_id in range(cluster_num)]
                    print(f"Spectral Clustering {cluster_num} Clusters (2D, {dim_red_func.__class__.__name__}):")
                    sat_str = ' '.join([f'{n_sat_single_cluster}/{np.sum(n_sat_in_each_cluster)} {100 * n_sat_single_cluster / np.sum(n_sat_in_each_cluster):.2f}%' 
                                        for n_sat_single_cluster in n_sat_in_each_cluster])
                    print(f"  SAT in each cluster: {sat_str}")
                    pse_str = ' '.join([f'{n_pse_single_cluster}/{np.sum(n_pse_in_each_cluster)} {100 * n_pse_single_cluster / np.sum(n_pse_in_each_cluster):.2f}%' 
                                        for n_pse_single_cluster in n_pse_in_each_cluster])
                    print(f"  PSE in each cluster: {pse_str}")
                    

                    labels_3d = SpectralClustering(n_clusters=cluster_num, random_state=42).fit_predict(cell_puff_embedding_3d_red)
                    n_sat_in_each_cluster = [np.sum([cell_colors[i] == "deepskyblue" for i in range(len(cell_colors)) if labels_3d[i] == cluster_id]) 
                                             for cluster_id in range(cluster_num)]
                    n_pse_in_each_cluster = [np.sum([cell_colors[i] == "orangered" for i in range(len(cell_colors)) if labels_3d[i] == cluster_id])
                                             for cluster_id in range(cluster_num)]
                    print(f"Spectral Clustering {cluster_num} Clusters (3D, {dim_red_func.__class__.__name__}):")
                    sat_str = ' '.join([f'{n_sat_single_cluster}/{np.sum(n_sat_in_each_cluster)} {100 * n_sat_single_cluster / np.sum(n_sat_in_each_cluster):.2f}%' 
                                        for n_sat_single_cluster in n_sat_in_each_cluster])
                    print(f"  SAT in each cluster: {sat_str}")    
                    pse_str = ' '.join([f'{n_pse_single_cluster}/{np.sum(n_pse_in_each_cluster)} {100 * n_pse_single_cluster / np.sum(n_pse_in_each_cluster):.2f}%' 
                                        for n_pse_single_cluster in n_pse_in_each_cluster])
                    print(f"  PSE in each cluster: {pse_str}")

                    ax = axes[row_id, 4 + col_id*4]
                    ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], s=10, color=cell_colors, edgecolors="w", lw=0.1, zorder=2)
                    ax.set_title(f"Cell Puff Response 2D History ({dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.spines[['right', 'top']].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # add_cluster_ellipses(ax, cell_puff_embedding_2d_red, labels_2d, cluster_num)

                            
                    ax = axes[row_id, 4 + col_id*4 + 2]
                    ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], s=10, color=cell_colors, edgecolors="w", lw=0.1, zorder=2)
                    ax.set_title(f"Cell Puff Response 3D History ({dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.spines[['right', 'top']].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # add_cluster_ellipses(ax, cell_puff_embedding_3d_red, labels_3d, cluster_num)


                    ax = axes[row_id, 4 + col_id*4 + 1]
                    ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], s=10, color=[CLUSTER_COLORS[label] for label in labels_2d], edgecolors="w", lw=0.1)
                    ax.set_title(f"KMeans {cluster_num} Clusters (2D, {dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.spines[['right', 'top']].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    ax = axes[row_id, 4 + col_id*4 + 3]
                    ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], s=10, color=[CLUSTER_COLORS[label] for label in labels_3d], edgecolors="w", lw=0.1)
                    ax.set_title(f"KMeans {cluster_num} Clusters (3D, {dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.spines[['right', 'top']].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\Cell_Puff_Response_History2D.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)


            pca_3d = PCA(n_components=3)
            umap_3d = UMAP(n_components=3, n_neighbors=10, random_state=42)
            fig, axes = plt.subplots(2, 4 + 4*4, figsize=(3*(4 + 4*4), 6), constrained_layout=True, subplot_kw={'projection': '3d'})

            for row_id, dim_red_func in enumerate([pca_3d, umap_3d]):
                cell_puff_embedding_2d_red = dim_red_func.fit_transform(all_cell_puff_embedding_2d)
                cell_puff_embedding_3d_red = dim_red_func.fit_transform(all_cell_puff_embedding_3d)

            
                ax = axes[row_id, 0]
                ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], cell_puff_embedding_2d_red[:, 2],
                            s=6, color=cell_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 2D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_zlabel("Dimension 3")
                ax.spines[['right', 'top']].set_visible(False)

                ax = axes[row_id, 1]
                ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], cell_puff_embedding_3d_red[:, 2],
                            s=6, color=cell_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 3D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_zlabel("Dimension 3")
                ax.spines[['right', 'top']].set_visible(False)

                ax = axes[row_id, 2]
                ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], cell_puff_embedding_2d_red[:, 2], s=6, color=cell_mice_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 2D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_zlabel("Dimension 3")
                ax.spines[['right', 'top']].set_visible(False)
                add_textonly_legend(ax, {combine_mice[i]: {"color": colors[i], "alpha": 0.8} for i in range(n_mice)}, ncols=2)

                ax = axes[row_id, 3]
                ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], cell_puff_embedding_3d_red[:, 2], s=6, color=cell_mice_colors, edgecolors="w", lw=0.1)
                ax.set_title(f"Cell Puff Response 3D History ({dim_red_func.__class__.__name__})")
                ax.set_xlabel("Dimension 1")
                ax.set_ylabel("Dimension 2")
                ax.set_zlabel("Dimension 3")
                ax.spines[['right', 'top']].set_visible(False)
                add_textonly_legend(ax, {combine_mice[i]: {"color": colors[i], "alpha": 0.8} for i in range(n_mice)}, ncols=2)
                for col_id, cluster_num in enumerate([2,3,4,5]):

                    """
                    cluster on reduced coordinate
                    KMeans 4 Clusters (3D, UMAP):
                    SAT in each cluster: 12/61 19.67% 0/61 0.00% 27/61 44.26% 22/61 36.07%
                    PSE in each cluster: 13/56 23.21% 33/56 58.93% 8/56 14.29% 2/56 3.57%
                    
                    
                    Spectral Clustering 4 Clusters (3D, UMAP):
                    SAT in each cluster: 0/61 0.00% 29/61 47.54% 11/61 18.03% 21/61 34.43%
                    PSE in each cluster: 32/56 57.14% 10/56 17.86% 13/56 23.21% 1/56 1.79%
                    """
                    labels_2d = SpectralClustering(n_clusters=cluster_num, random_state=42).fit_predict(cell_puff_embedding_2d_red)
                    n_sat_in_each_cluster = [np.sum([cell_colors[i] == "deepskyblue" for i in range(len(cell_colors)) if labels_2d[i] == cluster_id]) 
                                             for cluster_id in range(cluster_num)]
                    n_pse_in_each_cluster = [np.sum([cell_colors[i] == "orangered" for i in range(len(cell_colors)) if labels_2d[i] == cluster_id])
                                             for cluster_id in range(cluster_num)]
                    print(f"Spectral Clustering {cluster_num} Clusters (2D, {dim_red_func.__class__.__name__}):")
                    sat_str = ' '.join([f'{n_sat_single_cluster}/{np.sum(n_sat_in_each_cluster)} {100 * n_sat_single_cluster / np.sum(n_sat_in_each_cluster):.2f}%' 
                                        for n_sat_single_cluster in n_sat_in_each_cluster])
                    print(f"  SAT in each cluster: {sat_str}")
                    pse_str = ' '.join([f'{n_pse_single_cluster}/{np.sum(n_pse_in_each_cluster)} {100 * n_pse_single_cluster / np.sum(n_pse_in_each_cluster):.2f}%' 
                                        for n_pse_single_cluster in n_pse_in_each_cluster])
                    print(f"  PSE in each cluster: {pse_str}")
                    

                    labels_3d = SpectralClustering(n_clusters=cluster_num, random_state=42).fit_predict(cell_puff_embedding_3d_red)
                    n_sat_in_each_cluster = [np.sum([cell_colors[i] == "deepskyblue" for i in range(len(cell_colors)) if labels_3d[i] == cluster_id]) 
                                             for cluster_id in range(cluster_num)]
                    n_pse_in_each_cluster = [np.sum([cell_colors[i] == "orangered" for i in range(len(cell_colors)) if labels_3d[i] == cluster_id])
                                             for cluster_id in range(cluster_num)]
                    print(f"Spectral Clustering {cluster_num} Clusters (3D, {dim_red_func.__class__.__name__}):")
                    sat_str = ' '.join([f'{n_sat_single_cluster}/{np.sum(n_sat_in_each_cluster)} {100 * n_sat_single_cluster / np.sum(n_sat_in_each_cluster):.2f}%' 
                                        for n_sat_single_cluster in n_sat_in_each_cluster])
                    print(f"  SAT in each cluster: {sat_str}")    
                    pse_str = ' '.join([f'{n_pse_single_cluster}/{np.sum(n_pse_in_each_cluster)} {100 * n_pse_single_cluster / np.sum(n_pse_in_each_cluster):.2f}%' 
                                        for n_pse_single_cluster in n_pse_in_each_cluster])
                    print(f"  PSE in each cluster: {pse_str}")


                    ax = axes[row_id, 4 + col_id*4]
                    ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], cell_puff_embedding_2d_red[:, 2],
                                s=6, color=cell_colors, edgecolors="w", lw=0.1)
                    ax.set_title(f"Cell Puff Response 2D History ({dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.set_zlabel("Dimension 3")
                    ax.spines[['right', 'top']].set_visible(False)

                    ax = axes[row_id, 4 + col_id*4 + 2]
                    ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], cell_puff_embedding_3d_red[:, 2], s=6, color=cell_colors, edgecolors="w", lw=0.1)
                    ax.set_title(f"Cell Puff Response 3D History ({dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.set_zlabel("Dimension 3")
                    ax.spines[['right', 'top']].set_visible(False)
                    
                    ax = axes[row_id, 4 + col_id*4 + 1]
                    ax.scatter(cell_puff_embedding_2d_red[:, 0], cell_puff_embedding_2d_red[:, 1], cell_puff_embedding_2d_red[:, 2], s=6, color=[CLUSTER_COLORS[label] for label in labels_2d], edgecolors="w", lw=0.1)
                    ax.set_title(f"KMeans {cluster_num} Clusters (2D, {dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.set_zlabel("Dimension 3")
                    ax.spines[['right', 'top']].set_visible(False)

                    ax = axes[row_id, 4 + col_id*4 + 3]
                    ax.scatter(cell_puff_embedding_3d_red[:, 0], cell_puff_embedding_3d_red[:, 1], cell_puff_embedding_3d_red[:, 2], s=6, color=[CLUSTER_COLORS[label] for label in labels_3d], edgecolors="w", lw=0.1)
                    ax.set_title(f"KMeans {cluster_num} Clusters (3D, {dim_red_func.__class__.__name__})")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
                    ax.set_zlabel("Dimension 3")
                    ax.spines[['right', 'top']].set_visible(False)
            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\Cell_Puff_Response_History3D.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)


        def plot_daywise_plasticity_change_by_cluster():
            cluster_num = 3
            umap_2d = UMAP(n_components=2, n_neighbors=10, random_state=42)
            cell_puff_embedding_3d_red = umap_2d.fit_transform(all_cell_puff_embedding_3d)
            labels_2d = SpectralClustering(n_clusters=cluster_num, random_state=42).fit_predict(cell_puff_embedding_3d_red)
            labels = labels_2d

            n_sat_in_each_cluster = [np.sum([cell_colors[i] == "deepskyblue" for i in range(len(cell_colors)) if labels[i] == cluster_id]) 
                                        for cluster_id in range(cluster_num)]
            n_pse_in_each_cluster = [np.sum([cell_colors[i] == "orangered" for i in range(len(cell_colors)) if labels[i] == cluster_id])
                                        for cluster_id in range(cluster_num)]
            print(f"Spectral Clustering {cluster_num} Clusters (2D, UMAP):")
            sat_str = ' '.join([f'{n_sat_single_cluster}/{np.sum(n_sat_in_each_cluster)} {100 * n_sat_single_cluster / np.sum(n_sat_in_each_cluster):.2f}%' 
                                for n_sat_single_cluster in n_sat_in_each_cluster])
            print(f"  SAT in each cluster: {sat_str}")
            pse_str = ' '.join([f'{n_pse_single_cluster}/{np.sum(n_pse_in_each_cluster)} {100 * n_pse_single_cluster / np.sum(n_pse_in_each_cluster):.2f}%' 
                                for n_pse_single_cluster in n_pse_in_each_cluster])
            print(f"  PSE in each cluster: {pse_str}")
            
            fig, axs = plt.subplots(1, 6, figsize=(30, 5), constrained_layout=True)

            for ax in axs.flatten():
                ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.1, s=6, color="gray", edgecolors="w", lw=0.1)
                ax.set_xlabel("UMAP Dimension 1")
                ax.set_ylabel("UMAP Dimension 2")
                ax.spines[['right', 'top']].set_visible(False)


            add_textonly_legend(axs[0], {f"SAT{day+1}": {"color": sat_day_colors[day], "alpha": 0.8} for day in range(5)}, loc="lower left")
            add_textonly_legend(axs[1], {f"PSE{day+1}": {"color": pse_day_colors[day], "alpha": 0.8} for day in range(5)}, loc="lower left")
            add_textonly_legend(axs[2], {f"Cluster {cluster_id} Puff": {"color": CLUSTER_COLORS[cluster_id], "alpha": 0.8} for cluster_id in range(cluster_num)}, loc="lower left")
            add_textonly_legend(axs[3], {f"Cluster {cluster_id} Blank": {"color": CLUSTER_COLORS[cluster_id], "alpha": 0.8} for cluster_id in range(cluster_num)}, loc="lower left")
            
            for cell_node, cluster_id in zip(all_cells, labels):
                cohort_cell_colors = sat_day_colors if "SAT" in cell_node.cohort_id else pse_day_colors
                cell_puff_index = np.array( [
                    i 
                    for item_index in range(5)
                    for i, (celllevel_node, trial_type_name, _) in enumerate(raw_celllevel_data)
                    if cell_node.contains(celllevel_node) and trial_type_name == "training_puff" and get_item_index(celllevel_node) == item_index
                ])
                cell_blank_index = np.array( [
                    i 
                    for item_index in range(5)
                    for i, (celllevel_node, trial_type_name, _) in enumerate(raw_celllevel_data)
                    if cell_node.contains(celllevel_node) and trial_type_name == "training_blank" and get_item_index(celllevel_node) == item_index
                ])
                              
                
                for day_id in range(4):
                    axs[0 if "SAT" in cell_node.cohort_id else 1].scatter(
                        embedding_2d[cell_puff_index[day_id], 0], embedding_2d[cell_puff_index[day_id], 1],
                        color=cohort_cell_colors[day_id], s=8, edgecolors="w", lw=0.1
                    )
                    axs[2].scatter(
                        embedding_2d[cell_puff_index[day_id], 0], embedding_2d[cell_puff_index[day_id], 1],
                        color=CLUSTER_COLORS[cluster_id], s=8, edgecolors="w", lw=0.1, alpha=0.5 + 0.1 * day_id
                    )

                    axs[4].plot(
                        [embedding_2d[cell_puff_index[day_id], 0], embedding_2d[cell_puff_index[day_id + 1], 0]],
                        [embedding_2d[cell_puff_index[day_id], 1], embedding_2d[cell_puff_index[day_id + 1], 1]],
                        color=CLUSTER_COLORS[cluster_id], alpha=0.5 + 0.1 * day_id, lw=1.5, ls="-"
                    )
                    if _include_blank:
                        axs[3].scatter(
                            embedding_2d[cell_blank_index[day_id], 0], embedding_2d[cell_blank_index[day_id], 1],
                            color=CLUSTER_COLORS[cluster_id], s=8, edgecolors="w", lw=0.1, alpha=0.5 + 0.1 * day_id
                        )
                        axs[5].plot(
                            [embedding_2d[cell_blank_index[day_id], 0], embedding_2d[cell_blank_index[day_id + 1], 0]],
                            [embedding_2d[cell_blank_index[day_id], 1], embedding_2d[cell_blank_index[day_id + 1], 1]],
                            color=CLUSTER_COLORS[cluster_id], alpha=0.5 + 0.1 * day_id, lw=1.5, ls="--"
                        )

            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\Daywise_Plasticity_Change_by_Cluster.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=False,)

            
            def make_labels(counts, total):
                return [f"{c/total*100:.1f}%\n({c})" for c in counts]

            sat_labels = make_labels(n_sat_in_each_cluster, np.sum(n_sat_in_each_cluster))
            pse_labels = make_labels(n_pse_in_each_cluster, np.sum(n_pse_in_each_cluster))
            sat_counts = n_sat_in_each_cluster
            pse_counts = n_pse_in_each_cluster
            # -----------------------------
            # Plot
            # -----------------------------
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            

            for ax in axes:
                ax.axis("equal")

            # SAT donut chart
            axes[0].pie(
                sat_counts,
                labels=sat_labels,
                colors=[CLUSTER_COLORS[cluster_id] for cluster_id in range(cluster_num)],
                startangle=90,
                counterclock=False,
                labeldistance=0.76,
                wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2),
                textprops=dict(color="black", fontsize=13, ha="center")
            )

            axes[0].text(
                0, 0.08, "SAT",
                ha="center", va="center",
                fontsize=34,
                color="deepskyblue"
            )

            axes[0].text(
                0, -0.22, f"n={np.sum(n_sat_in_each_cluster)}",
                ha="center", va="center",
                fontsize=20,
                color="deepskyblue"
            )

            # PSE donut chart
            axes[1].pie(
                pse_counts,
                labels=pse_labels,
                colors=[CLUSTER_COLORS[cluster_id] for cluster_id in range(cluster_num)],
                startangle=90,
                counterclock=False,
                labeldistance=0.76,
                wedgeprops=dict(width=0.45, edgecolor="black", linewidth=1.2),
                textprops=dict(color="black", fontsize=13, ha="center")
            )

            axes[1].text(
                0, 0.08, "PSE",
                ha="center", va="center",
                fontsize=34,
                color="orangered"
            )

            axes[1].text(
                0, -0.22, f"n={np.sum(n_pse_in_each_cluster)}",
                ha="center", va="center",
                fontsize=20,
                color="orangered"
            )

            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\Cluster_Composition_Donut_Chart.png", fov_skip=True)
            default_exit_save(fig, save_path, _transparent=True,)




        plot_each_single_cell()
        group_cells_by_puff_response_history()
        plot_daywise_plasticity_change_by_cluster()

        

        cluster_num = 3
        umap_2d = UMAP(n_components=2, n_neighbors=10, random_state=42)
        cell_puff_embedding_3d_red = umap_2d.fit_transform(all_cell_puff_embedding_3d)
        labels_2d = SpectralClustering(n_clusters=cluster_num, random_state=42).fit_predict(cell_puff_embedding_3d_red)
        labels = labels_2d
        clustering_result_dict = {cell_node: int(labels[i]) for i, cell_node in enumerate(all_cells)}


        def save_clustering_result():
            import pandas as pd
            from kitchen.writer.excel_writer import write_boolean_dataframe

            save_path = routing.default_fig_path(sat_dataset, f"..\\Trial_Response_Shape_Clustering\\Cell_Cluster_Labels.xlsx", fov_skip=True)
            
            clustering_result_list = []
            for i_cell_node, (cell_node, cluster_id) in enumerate(clustering_result_dict.items()):
                clustering_result_list.append({
                    "Cohort": "SAT" if "SAT" in cell_node.cohort_id else "PSE",
                    "Mice": cell_node.mice_id,
                    "Cell": cell_node.cell_id,
                    "Cluster": cluster_id
                } | {
                    f"Embedding_Dim_{i}": embedding for i, embedding in enumerate(cell_puff_embedding_3d_red[i_cell_node])
                })
            df = pd.DataFrame(clustering_result_list)
            
            write_boolean_dataframe(df, f"Clustering Result", save_path)

        save_clustering_result()
        return clustering_result_dict
    

    part1()
    part2()
    part3()
    return part4()
