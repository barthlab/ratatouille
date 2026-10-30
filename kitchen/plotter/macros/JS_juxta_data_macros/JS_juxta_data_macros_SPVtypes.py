
from collections import Counter, defaultdict
from dataclasses import dataclass
import os.path as path
import os
from typing import Optional, Self, Tuple
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import pandas as pd
import logging
import seaborn as sns

from matplotlib.colors import SymLogNorm

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, optimal_leaf_ordering
from scipy.spatial.distance import squareform
from sklearn.metrics import davies_bouldin_score, jaccard_score, silhouette_score, calinski_harabasz_score

from kitchen.operator.grouping import AdvancedTimeSeries
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.decorators.default_decorators import default_exit_save
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_weight_tuple_Decomposition_Weights, get_weight_tuple_Physiology_Fingerprint, get_weight_tuple_Waveform
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS, COHORT_BRIGHTER_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path
from kitchen.plotter.stats_tests import basic_ttest
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.utils import numpy_kit, ordering_kit, sequence_kit

logger = logging.getLogger(__name__)

from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    SpectralClustering,
    HDBSCAN,
)
import umap
from sklearn.mixture import GaussianMixture


from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6


@dataclass
class CellFeature:
    features: np.ndarray
    node_name: str
    cohort_name: str

    @property
    def celltype(self) -> str:
        return self.cohort_name.split("_")[0]

    def drop_feature(self, feature_id: int) -> "CellFeature":
        new_feature = np.delete(self.features, feature_id)
        return CellFeature(
            features=new_feature,
            node_name=self.node_name,
            cohort_name=self.cohort_name
        )



def single_spv_clustering(
    full_feature_list: list[CellFeature],
    weighting: np.ndarray,
    clustering_method: str,
    num_of_cluster: int,
    clustering_random_seed: int,
    save_point_identifier: str,
    clustering_params: dict = {},
):
    n_cell = len(full_feature_list)
    n_feature = len(full_feature_list[0].features)
    print(f"{clustering_method} Clustering on {n_cell} cells with {n_feature} features.")

    weighting = np.asarray(weighting, dtype=float)
    assert weighting.shape == (n_feature,), (
        f"weighting must have shape ({n_feature},), "
        f"but got {weighting.shape}"
    )
    square_root_weighting = np.sqrt(weighting)

    feature_matrix = np.stack([cf.features for cf in full_feature_list], axis=0).astype(float)
    assert np.all(np.isfinite(feature_matrix)), "Feature matrix contains NaN or infinite values."
    normalized_feature_matrix = numpy_kit.zscore(feature_matrix, axis=0)
    
    weighted_feature_matrix = normalized_feature_matrix * square_root_weighting[None, :]
    
    if clustering_method == "Agg":
        assert clustering_random_seed is None, "Agg clustering does not use a random seed."
        model = AgglomerativeClustering(
            n_clusters=num_of_cluster,
            metric="euclidean",
            **clustering_params,
        )
        labels = model.fit_predict(weighted_feature_matrix)

    elif clustering_method == "KMeans":
        model = KMeans(
            n_clusters=num_of_cluster,
            random_state=clustering_random_seed,
            **clustering_params,
        )
        labels = model.fit_predict(weighted_feature_matrix)

    elif clustering_method == "Spectral":
        model = SpectralClustering(
            n_clusters=num_of_cluster,
            random_state=clustering_random_seed,
            **clustering_params,
        )
        labels = model.fit_predict(weighted_feature_matrix)

    elif clustering_method == "HDBSCAN":
        assert clustering_random_seed is None, "HDBSCAN does not use a random seed."
        assert num_of_cluster is None, "HDBSCAN does not use a fixed number of clusters."
        model = HDBSCAN(
            **clustering_params,
        )
        labels = model.fit_predict(weighted_feature_matrix)

    elif clustering_method == "Gaussian":
        model = GaussianMixture(
            n_components=num_of_cluster,
            random_state=clustering_random_seed,
            **clustering_params,
        )
        labels = model.fit_predict(weighted_feature_matrix)

    else:
        raise NotImplementedError(
            f"Unknown clustering method: {clustering_method}"
        )

    coclustering_matrix = numpy_kit.Labels2CoClusterMatrix(labels)
    print(f"Coclustering matrix computed for save point: {save_point_identifier}")
    return coclustering_matrix



def visualize_coclustering_matrix(feature_list: list[CellFeature],
                                  coclustering_matrix: np.ndarray, 
                                  save_path: str, _resample_indices = None):
    n_cell = len(feature_list)
    assert coclustering_matrix.shape == (n_cell, n_cell), \
        "Coclustering matrix shape does not match the number of cells. Expected " \
        f"({n_cell}, {n_cell}), got {coclustering_matrix.shape}."
    
    dissimilarity = np.nan_to_num(1.0 - coclustering_matrix, nan=0.)
    condensed_dissimilarity = squareform(dissimilarity, checks=True)

    Z = linkage(
        condensed_dissimilarity,
        method="average",
    )  
    
    Z_opt = optimal_leaf_ordering(Z, condensed_dissimilarity)


    embedding = umap.UMAP(
        metric="precomputed",
        random_state=42
    ).fit_transform(dissimilarity if _resample_indices is None else 
                    dissimilarity[np.ix_(_resample_indices, _resample_indices)])
    embed_color = [COHORT_BRIGHTER_COLORS[cf.cohort_name.split("_")[0] + "_JUX"] 
                   for cf in feature_list] if _resample_indices is None else \
                   [COHORT_BRIGHTER_COLORS[cf.cohort_name.split("_")[0] + "_JUX"] 
                    for cf in np.array(feature_list)[_resample_indices]]

    fig, axs = plt.subplots(1, 4, figsize=(4.5, 1.5), constrained_layout=True, 
                            width_ratios=[0.3, 1, 0.2, 1])
    
    dendro = dendrogram(Z_opt, ax=axs[0], orientation="left", no_labels=True)
    leaves_order = dendro['leaves']
    fcluster_labels = fcluster(Z_opt, t=0.5, criterion="distance")[leaves_order]
    boundary_indices = [i for i in range(1, len(fcluster_labels)) 
                        if fcluster_labels[i] != fcluster_labels[i - 1]]
    for boundary_index in boundary_indices:        
        axs[0].axhline(y=boundary_index * 10, color="black", linestyle="--", lw=1)
        for ax in axs[1:-1]:
            ax.axhline(y=boundary_index - 0.5, color="black", linestyle="--", lw=1)

    axs[1].imshow(coclustering_matrix[leaves_order, :][:, leaves_order], 
                  cmap="viridis", aspect="equal", vmin=0, vmax=1)

    COHORT_vis = {"SST" : (0, COHORT_BRIGHTER_COLORS["SST_JUX"]),
                  "PV": (1, COHORT_BRIGHTER_COLORS["PV_JUX"]),
                  "PYR": (2, COHORT_BRIGHTER_COLORS["PYR_JUX"])}
    for cf_id, cf_index in enumerate(leaves_order):
        cf = feature_list[cf_index]
        x_cor, cf_color = COHORT_vis[cf.cohort_name.split("_")[0]]
        axs[2].scatter(x_cor, cf_id, facecolors = cf_color, 
                       edgecolors='black', lw=0.4, s=10, alpha=0.9, clip_on=False)
    axs[2].set_ylim(-0.5, len(feature_list) - 0.5)

    axs[1].invert_yaxis()
    axs[2].set_xlim(-0.5, 2.5)
    for ax in axs:
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    
    axs[3].scatter(embedding[:, 0], embedding[:, 1], facecolors=embed_color, 
                   edgecolors='black', lw=0.4, s=10, alpha=0.9)
    axs[3].spines[["left", "bottom"]].set_visible(True)
    # axs[3].set_aspect('equal')

    # plt.show()
    default_exit_save(fig, save_path=save_path, _transparent=True)



def random_weight_vec(length: int, alpha: float, seed: int):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(alpha * np.ones(length))


def spv_clustering_search(cf_list: list[CellFeature], 
                          clustering_method: str, params_grid: dict, consensus_name: str,
                          task_id: str,
                          _overwrite: bool = False):

    archive_path = path.join(get_saving_path(), "spv_clustering", task_id,
                             consensus_name, clustering_method + "_coclustering_matrices.npy")
    if not _overwrite and path.exists(archive_path):
        coclustering_matrices = np.load(archive_path)
        print(f"Loaded coclustering matrices from {archive_path}")
        return coclustering_matrices
    
    coclustering_matrices = [
        single_spv_clustering(
            full_feature_list=cf_list,
            clustering_method=clustering_method,
            save_point_identifier=\
                path.join(
                    consensus_name, 
                    f"{clustering_method}_{idx:05}_{params.get('num_of_cluster', 'NA')}_{params.get('clustering_random_seed', 'NA')}"),
            **params
        )
        for idx, params in enumerate(sequence_kit.grid_on(params_grid))
    ]

    os.makedirs(path.dirname(archive_path), exist_ok=True)
    np.save(archive_path, coclustering_matrices)
    print(f"Saved coclustering matrices to {archive_path}")
    return coclustering_matrices


def consensus_grid_search(cf_list: list[CellFeature], 
                          num_of_cluster: int,
                          n_features: int, weight_alpha: float,
                          n_weight: int = 1000, n_rs: int = 10,
                          _feature_drop_id: Optional[int] = None,
                          _resample_tuple: Optional[Tuple[float, int]] = None,
                          
                          task_id: str = None,
                          _visualize: bool = False):
    n_cf_full = len(cf_list)
    full_cf_list = cf_list.copy()

    if _visualize:
        assert task_id is not None, "Task ID must be provided for visualization"

    # Feature dropping
    if _feature_drop_id is not None:
        cf_list = [cf.drop_feature(_feature_drop_id) for cf in cf_list]
        n_features -= 1

    # CF list resampling
    resample_indices = None
    if _resample_tuple is not None:
        fraction, seed = _resample_tuple
        rng = np.random.default_rng(seed)
        resample_indices = rng.choice(len(cf_list), size=int(len(cf_list) * fraction), replace=False)
        cf_list = [cf_list[i] for i in resample_indices]

    assert all(len(cf.features) == n_features for cf in cf_list), "Not all CellFeature instances have the expected number of features"

    consensus_group_name = f"consensus_feature{n_features}_alpha{weight_alpha}_cluster{num_of_cluster}"
    consensus_group_name += f"_FeatureDrop{_feature_drop_id}" if _feature_drop_id is not None else ""
    consensus_group_name += f"_Resample{_resample_tuple[0]}_{_resample_tuple[1]}" if _resample_tuple is not None else ""

    shared_search_space = {
        "weighting": [random_weight_vec(n_features, alpha=weight_alpha, seed=i) for i in range(n_weight)],
        "num_of_cluster": [num_of_cluster, ],
        "clustering_random_seed": [None],
    }

    agg_search_space = {
        "clustering_params": [{"linkage": method} 
                              for method in ("ward", "complete", "average")],
    }
    kmeans_search_space = {"clustering_random_seed": (i for i in range(n_rs))}
    spectral_search_space = {"clustering_random_seed": (i for i in range(n_rs))}
    hdbscan_search_space = {"num_of_cluster": [None]}
    gaussian_search_space = {"clustering_random_seed": (i for i in range(n_rs))}

    all_coclustering_results = {}
    for cluster_method, clustering_search_space in zip(
        ["KMeans", "Spectral", "Gaussian", "Agg",],
        [kmeans_search_space, spectral_search_space, gaussian_search_space, agg_search_space,]
    ):
        print(f"Running consensus grid search for {cluster_method}")
        c_result = spv_clustering_search(
            cf_list, cluster_method, shared_search_space | clustering_search_space, 
            consensus_group_name, task_id=task_id)
        
        if _resample_tuple is not None:
            c_result = numpy_kit.matrices_fillback(c_result, resample_indices, full_size=n_cf_full)

        if _visualize:
            save_path = path.join(get_saving_path(), "spv_clustering", task_id, consensus_group_name,
                                f"consensus_{cluster_method.lower()}.png")
            visualize_coclustering_matrix(
                full_cf_list, np.nanmean(c_result, axis=0), save_path=save_path,
                _resample_indices=resample_indices,
            )

        
        all_coclustering_results[cluster_method] = np.stack(c_result, axis=0)

    if _visualize:
        save_path = path.join(get_saving_path(), "spv_clustering", task_id, consensus_group_name,
                            f"consensus_all.png")
        # Give each method equal weight, regardless of its number of runs.
        method_means = [np.nanmean(result, axis=0)
                        for result in all_coclustering_results.values()]
        visualize_coclustering_matrix(
            full_cf_list, np.nanmean(method_means, axis=0), save_path=save_path,
            _resample_indices=resample_indices
        )

    return all_coclustering_results




def _core_tree_layout(clusterings, gap=0.015):
    """Place dominant-parent subtrees side by side, allowing for box widths.

    Inspired by clustree's core-edge layout; this is a width-aware tree layout,
    not an implementation of igraph's Reingold-Tilford algorithm. Equal parent
    overlaps choose the first cluster deterministically. All edges are drawn.
    The first row must contain one cluster: the whole population.
    """
    sizes = [np.bincount(labels) for labels in clusterings]
    flows = [np.bincount(a * len(sb) + b, minlength=len(sa) * len(sb))
             .reshape(len(sa), len(sb))
             for a, b, sa, sb in zip(clusterings, clusterings[1:], sizes, sizes[1:])]
    # flows[layer][parent, child] = number_of_shared_samples
    
    # Maximizing overlap also maximizes overlap / destination size.
    parents = [flow.argmax(axis=0) for flow in flows]
    children = [[np.flatnonzero(parent == c) for c in range(len(counts))]
                for parent, counts in zip(parents, sizes)]
    widths = [counts / len(clusterings[0]) for counts in sizes]
    spans = [width.copy() for width in widths]

    # Reserve enough horizontal space for each node and all its descendants.
    for layer in reversed(range(len(children))):
        for cluster, child_ids in enumerate(children[layer]):
            if child_ids.size:
                child_span = spans[layer + 1][child_ids].sum() + gap * (len(child_ids) - 1)
                spans[layer][cluster] = max(spans[layer][cluster], child_span)

    centers = [np.zeros(len(counts)) for counts in sizes]
    total_width = spans[0][0]
    centers[0][0] = total_width / 2
    for layer, groups in enumerate(children):
        for cluster, child_ids in enumerate(groups):
            if not child_ids.size:
                continue
            child_spans = spans[layer + 1][child_ids]
            left = centers[layer][cluster] - (child_spans.sum() + gap * (len(child_ids) - 1)) / 2
            centers[layer + 1][child_ids] = (
                left + np.r_[0, np.cumsum(child_spans[:-1] + gap)] + child_spans / 2
            )

    # Relabel left to right so ribbon endpoints are stacked in spatial order.
    layouts = []
    cell_width = 1 / (len(clusterings[0]) * total_width)
    for labels, center, width in zip(clusterings, centers, widths):
        order = np.argsort(center)
        starts = ((center - width / 2) / total_width)[order]
        layouts.append((np.argsort(order)[labels], starts, cell_width))
    return layouts


def _add_composition_label(ax, xy, counts, colors, fontsize=4):
    """Center colored counts separated by black colons, three points above xy."""
    parts = []
    for count, color in zip(counts, colors):
        if count == 0:
            continue
        if parts:
            parts.append(TextArea(":", textprops={"color": "black", "fontsize": fontsize}))
        parts.append(TextArea(str(count), textprops={"color": color, "fontsize": fontsize}))
    label = AnnotationBbox(
        HPacker(children=parts, align="baseline", pad=0, sep=0),
        xy, xycoords="data", xybox=(0, 0), boxcoords="offset points",
        box_alignment=(0.5, 0), frameon=False, pad=0, zorder=5,
    )
    ax.add_artist(label)
    return label


def visualize_cluster_sankey(
    similarity_matrices,
    cohort_names,
    cluster_nums,
    *,
    cohort_order = None,
    cohort_colors = None,
    text_colors = None,
    save_path = None,
    cocluster_ratio: float = 0.5,
    figsize=(2, 2.2),
):
    """Cluster each similarity matrix and plot how samples move between clusters.

    Pass a list of (n, n) similarity matrices, n cohort names in the same sample
    order for every matrix, and one cluster limit per matrix. Average linkage
    uses distance = 1 - similarity; tied distances may yield fewer clusters.
    Cells are colored by cohort, and gray ribbons show shared sample counts.
    A whole-population row precedes the matrices. A clustree-inspired layout
    groups each cluster beneath its strongest parent, spacing entire subtrees
    to fit the rectangles. Ties choose the first parent. Every ribbon remains
    visible; opacity increases with its fraction of the destination cluster.
    cohort_order lists every cohort once (default: sorted); it controls cells,
    composition labels, and the legend. cohort_colors is an optional mapping
    from cohort name to Matplotlib color; omitted colors use tab10.
    Returns (fig, ax); call plt.show() to display the plot.
    """
    cohort_names = np.asarray(cohort_names)
    n, layers = len(cohort_names), len(similarity_matrices) + 1
    if not n or not similarity_matrices or len(similarity_matrices) != len(cluster_nums):
        raise ValueError("Provide samples, similarity matrices, and one cluster limit per matrix.")

    cohorts = list(np.unique(cohort_names)) if cohort_order is None else list(cohort_order)
    if len(cohorts) != len(set(cohorts)) or set(cohorts) != set(cohort_names):
        raise ValueError("cohort_order must contain every cohort exactly once.")
    
    colors = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(cohorts)}
    if cohort_colors is not None:
        colors.update(cohort_colors)

    cohort_rank = {c: i for i, c in enumerate(cohorts)}
    sample_order = np.array([cohort_rank[c] for c in cohort_names])

    clusterings = [np.zeros(n, dtype=int)]  # Layer zero: the whole population.
    for similarity, k in zip(similarity_matrices, cluster_nums):
        distance = 1 - np.asarray(similarity)
        np.fill_diagonal(distance, 0)
        tree = linkage(squareform(distance, checks=False), method="average")
        # labels = fcluster(tree, k, criterion="maxclust")
        labels = fcluster(tree, cocluster_ratio, criterion="distance")
        clusterings.append(np.unique(labels, return_inverse=True)[1])
    layouts = _core_tree_layout(clusterings)
    box_height = 0.32
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for layer, (labels, starts, cell_width) in enumerate(layouts):
        sizes = np.bincount(labels)
        y = layers - 1 - layer

        for cluster, (x, size) in enumerate(zip(starts, sizes)):
            members = np.flatnonzero(labels == cluster)
            members = members[np.argsort(sample_order[members], kind="stable")]
            width = size * cell_width
            for j, sample in enumerate(members):
                ax.add_patch(Rectangle(
                    (x + j * cell_width, y - box_height / 2), cell_width, box_height,
                    facecolor=colors[cohort_names[sample]], edgecolor="white",
                    linewidth=0.1, zorder=3,
                ))
            ax.add_patch(Rectangle(
                (x, y - box_height / 2), width, box_height, fill=False,
                edgecolor="black", linewidth=0.3, zorder=4,
            ))

            if text_colors is not None:
                counts = np.bincount(sample_order[members], minlength=len(cohorts))
                _add_composition_label(
                    ax, (x + width / 2, y + box_height / 2),
                    counts, [text_colors[c] for c in cohorts],
                )

        title = "1" if layer == 0 else f"{cluster_nums[layer - 1]}"
        ax.text(-0.08, y, title, ha="center", va="center")

    # Count only cluster pairs that share samples, and connect them with ribbons.
    for layer, (upper, lower) in enumerate(zip(layouts, layouts[1:])):
        upper_labels, upper_starts, upper_width = upper
        lower_labels, lower_starts, lower_width = lower
        lower_sizes = np.bincount(lower_labels)
        upper_offset, lower_offset = upper_starts.copy(), lower_starts.copy()
        pairs, counts = np.unique(
            np.column_stack((upper_labels, lower_labels)), axis=0, return_counts=True,
        )
        y1 = layers - 1 - layer - box_height / 2
        y2 = layers - 2 - layer + box_height / 2
        mid_y = (y1 + y2) / 2
        for (source, target), count in zip(pairs, counts):
            in_prop = count / lower_sizes[target]
            x1, x2 = upper_offset[source], lower_offset[target]
            w1, w2 = count * upper_width, count * lower_width
            # Vertical tangents let both curved edges flow smoothly out of the boxes.
            path = Path(
                [(x1, y1), (x1 + w1, y1),
                 (x1 + w1, mid_y), (x2 + w2, mid_y), (x2 + w2, y2),
                 (x2, y2), (x2, mid_y), (x1, mid_y), (x1, y1), (x1, y1)],
                [Path.MOVETO, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4,
                 Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY],
            )
            ax.add_patch(PathPatch(
                path, facecolor="0.75", edgecolor="none",
                alpha=0.1 + 0.6 * in_prop, zorder=1 + in_prop,
            ))
            upper_offset[source] += w1
            lower_offset[target] += w2

    ax.set_xlim(-0.12, 1)
    ax.set_ylim(-0.6, layers - 0.4)
    ax.axis("off")

    if save_path is not None:
        default_exit_save(fig, save_path=save_path, _transparent=True)

    return fig, ax



def clustering_lineage(cf_list: list[CellFeature], task_id: str, top_k: int = 10, figsize = None, **kw):

    clustering_results = {
        k: consensus_grid_search(cf_list, num_of_cluster=k, task_id=task_id, **kw)
        for k in range(2, top_k + 1)  # Example range for number of clusters
    }

    # Average runs within each method before averaging methods equally.
    avg_coclustering_matrices = {
        k: np.nanmean([np.nanmean(result, axis=0)
                    for result in coclustering_result_k.values()], axis=0)
        for k, coclustering_result_k in clustering_results.items()
    }

    for k, result_k in avg_coclustering_matrices.items():
        print(f"Number of clusters: {k} {result_k.shape}")

    for cocluster_ratio in (0.5, 0.3, 0.1):
        save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                f"clustering_lineage_{kw.get('weight_alpha', 'default')}_cocluster_{cocluster_ratio}.png")
    
        visualize_cluster_sankey(
            save_path=save_path,
            similarity_matrices=list(avg_coclustering_matrices.values()),
            cohort_names=[cf.celltype for cf in cf_list],
            cluster_nums=list(avg_coclustering_matrices.keys()),
            cohort_order=["SST", "PV", "PYR"],
            cohort_colors={"SST": COHORT_BRIGHTER_COLORS["SST_JUX"],
                        "PV": COHORT_BRIGHTER_COLORS["PV_JUX"],
                        "PYR": COHORT_BRIGHTER_COLORS["PYR_JUX"]},
            text_colors={"SST": COHORT_COLORS["SST_JUX"],
                        "PV": COHORT_COLORS["PV_JUX"],
                        "PYR": COHORT_COLORS["PYR_JUX"]},
            cocluster_ratio=cocluster_ratio,
            figsize=figsize
        )



from sklearn.neighbors import kneighbors_graph
from scipy.sparse import eye, diags
from scipy.sparse.linalg import eigsh
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import laplacian

from sklearn.metrics import adjusted_rand_score


def eigen_gap_quality_test(cf_list: list[CellFeature], task_id: str):
    raw_feature = np.stack([cf.features for cf in cf_list], axis=0)
    normalized_feature_matrix = numpy_kit.zscore(raw_feature, axis=0)


    X = normalized_feature_matrix
    d = X.shape[1]

    all_gaps_result = []
    for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, ]:
        gamma = scale / d
        W = rbf_kernel(X, gamma=gamma)
        np.fill_diagonal(W, 0)

        L = laplacian(W, normed=True)
        eigvals = np.linalg.eigvalsh(L)
        eigvals.sort()

        gaps = np.diff(eigvals)
        all_gaps_result.append(gaps[1:10])
    all_gaps_result = np.array(all_gaps_result)

    fig, ax = plt.subplots(1, 1, figsize=(1.1, 1.), constrained_layout=True)

    ax.tick_params(length=1, pad=1)
    ax.errorbar(np.arange(2, 11), all_gaps_result.mean(axis=0), 
                yerr=all_gaps_result.std(axis=0),
                fmt="o", color='black',
                markersize=3, lw=1, alpha=0.9, markeredgewidth=0.5,
                ecolor='black', elinewidth=0.5,
                capsize=1.5, capthick=0.5,)

    ax.set_xlabel("Eigen index (k)", labelpad=1)
    ax.set_ylabel("Eigengap " + r"$\lambda_{k+1}-\lambda_k$", labelpad=1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10])

    save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                f"eigen_gap.png")
    default_exit_save(fig, save_path=save_path, _transparent=True)


    
def clustering_quality_test(cf_list: list[CellFeature], task_id: str, **kw):

    raw_feature = np.stack([cf.features for cf in cf_list], axis=0)
    normalized_feature_matrix = numpy_kit.zscore(raw_feature, axis=0)
    
    clustering_results = {
        k: consensus_grid_search(cf_list, num_of_cluster=k, task_id=task_id, **kw)
        for k in range(2, 11)  # Example range for number of clusters
    }

    ct_list = ["SST", "PV", "PYR"]
    all_celltype_str = np.array([cf.celltype for cf in cf_list])
    celltype_indices = {ct: np.where(all_celltype_str == ct)[0] for ct in ct_list}
    celltype_indices["ALL"] = np.arange(len(all_celltype_str))

    def SilhouetteScore(cocluster_result_dict: dict, specific_celltype) -> dict:
        print("Calculating Silhouette Scores")
        silhouette_scores = defaultdict(lambda: defaultdict(list))
        specific_indices = celltype_indices[specific_celltype]
        for k, clustering_result_at_k in cocluster_result_dict.items():
            for cluster_method, coclustering_result in clustering_result_at_k.items():
                rng = np.random.default_rng(42)
                resample_indices = rng.choice(len(coclustering_result), size=100, replace=False)
                labels_list = [numpy_kit.CoClusterMatrix2Labels(coclustering_result[i]) for i in resample_indices]

                for labels in labels_list:
                    elements, counts = np.unique(labels[specific_indices], return_counts=True)
                    specific_k = np.sum(counts >= 1)
                    if specific_k == 1:
                        continue
                    silhouette_scores[k][cluster_method].append(
                        silhouette_score(normalized_feature_matrix[specific_indices], 
                                        labels[specific_indices]))
        return silhouette_scores

    def CHIndex(cocluster_result_dict: dict, specific_celltype) -> dict:
        print("Calculating Calinski-Harabasz Indices")
        ch_indices = defaultdict(lambda: defaultdict(list))
        specific_indices = celltype_indices[specific_celltype]
        for k, clustering_result_at_k in cocluster_result_dict.items():
            for cluster_method, coclustering_result in clustering_result_at_k.items():
                rng = np.random.default_rng(42)
                resample_indices = rng.choice(len(coclustering_result), size=100, replace=False)
                labels_list = [numpy_kit.CoClusterMatrix2Labels(coclustering_result[i]) for i in resample_indices]

                for labels in labels_list:
                    elements, counts = np.unique(labels[specific_indices], return_counts=True)
                    specific_k = np.sum(counts >= 1)
                    if specific_k == 1:
                        continue
                    ch_indices[k][cluster_method].append(
                        calinski_harabasz_score(normalized_feature_matrix[specific_indices], 
                                                labels[specific_indices]))
        return ch_indices

    def DBIndex(cocluster_result_dict: dict, specific_celltype) -> dict:
        print("Calculating Davies-Bouldin Indices")
        db_indices = defaultdict(lambda: defaultdict(list))
        specific_indices = celltype_indices[specific_celltype]
        for k, clustering_result_at_k in cocluster_result_dict.items():
            for cluster_method, coclustering_result in clustering_result_at_k.items():
                rng = np.random.default_rng(42)
                resample_indices = rng.choice(len(coclustering_result), size=100, replace=False)
                labels_list = [numpy_kit.CoClusterMatrix2Labels(coclustering_result[i]) for i in resample_indices]

                for labels in labels_list:
                    elements, counts = np.unique(labels[specific_indices], return_counts=True)
                    specific_k = np.sum(counts >= 1)
                    if specific_k == 1:
                        continue
                    db_indices[k][cluster_method].append(
                        davies_bouldin_score(normalized_feature_matrix[specific_indices], 
                                             labels[specific_indices]))
        return db_indices

    def PAC(cocluster_result_dict: dict, within: bool, cross: bool, low=0.1, high=0.9) -> dict:
        print("Calculating PAC Scores")
        pac_scores = {}
        for cluster_method, coclustering_result in cocluster_result_dict.items():
            consensus_matrix = np.mean(coclustering_result, axis=0)
            if within:
                within_values = []

                for idx in celltype_indices.values():
                    sub = consensus_matrix[np.ix_(idx, idx)]
                    vals = sub[np.triu_indices_from(sub, k=1)]
                    within_values.append(vals)
                pac_values = np.sort(np.concatenate(within_values))
                pac_scores[cluster_method] = np.mean((pac_values >= low) & (pac_values <= high))
            elif cross:
                cross_values = []

                for i in range(len(ct_list)):
                    for j in range(i + 1, len(ct_list)):
                        vals = consensus_matrix[
                            np.ix_(celltype_indices[ct_list[i]], celltype_indices[ct_list[j]])
                        ].ravel()
                        cross_values.append(vals)
                cross_values = np.sort(np.concatenate(cross_values))
                pac_scores[cluster_method] = np.mean((cross_values >= low) & (cross_values <= high))

            else:
                values = consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)]
                values = np.sort(values)
                pac_scores[cluster_method] = np.mean((values >= low) & (values <= high))
        return pac_scores

    def PAC_st(cocluster_result_dict: dict, specific_celltype, low=0.1, high=0.9) -> dict:
        print("Calculating PAC Scores")
        specific_indices = celltype_indices[specific_celltype]
        pac_scores = {}
        for cluster_method, coclustering_result in cocluster_result_dict.items():
            consensus_matrix = np.mean(coclustering_result, axis=0)
            
            values = consensus_matrix[np.ix_(specific_indices, specific_indices)]
            values = values[np.triu_indices_from(values, k=1)]
            values = np.sort(values)
            pac_scores[cluster_method] = np.mean((values >= low) & (values <= high))
        return pac_scores

    


    sst_pac = {k: PAC_st(v, specific_celltype="SST") for k, v in clustering_results.items()}
    pv_pac = {k: PAC_st(v, specific_celltype="PV") for k, v in clustering_results.items()}
    pyr_pac = {k: PAC_st(v, specific_celltype="PYR") for k, v in clustering_results.items()}
    all_pac = {k: PAC_st(v, specific_celltype="ALL") for k, v in clustering_results.items()}

    sst_silhouette = SilhouetteScore(clustering_results, "SST")
    pv_silhouette = SilhouetteScore(clustering_results, "PV")
    pyr_silhouette = SilhouetteScore(clustering_results, "PYR")
    all_silhouette = SilhouetteScore(clustering_results, "ALL")
    
    method_list = ["KMeans", "Spectral", "Gaussian", "Agg", "ALL"]
    for metric_name, metric_dict in zip(
        ["PAC (%)", "Silhouette (AU)"],
        [
            {"SST": sst_pac, "PV": pv_pac, "PYR": pyr_pac, "ALL": all_pac},
            {"SST": sst_silhouette, "PV": pv_silhouette, "PYR": pyr_silhouette, "ALL": all_silhouette}
        ]
    ):
        def mean_metric(scores):
            return np.nanmean(scores) if np.size(scores) else np.nan

        for method_name in method_list:
            methods = method_list[:-1] if method_name == "ALL" else [method_name]
            celltype_method_metrics = [
                [mean_metric([mean_metric(metric_dict[ct][k][method]) for method in methods])
                 for k in np.arange(2, 11)]
                for ct in ["SST", "PV", "PYR", "ALL"]
            ]

            fig, ax = plt.subplots(1, 1, figsize=(1.1, 1.), constrained_layout=True)
            
            ax.tick_params(length=1, pad=1)


            for ct_offset, ct_str, ct_color, ct_method_metric in zip(
                [-0.2, 0, 0.2, 0],
                ["SST", "PV", "PYR", "ALL"], 
                [COHORT_BRIGHTER_COLORS["SST_JUX"], COHORT_BRIGHTER_COLORS["PV_JUX"], COHORT_BRIGHTER_COLORS["PYR_JUX"], "black"],
                celltype_method_metrics):
                if ct_str == "ALL":
                    ax.scatter(np.arange(2, 11) + ct_offset, ct_method_metric,
                            facecolor="white", edgecolor='black',
                            s=5, lw=0.5, alpha=0.9, zorder=10)
                else:
                    ax.bar(np.arange(2, 11) + ct_offset, ct_method_metric,
                        color=ct_color, edgecolor='black', lw=0.3,
                        width=0.2, alpha=0.9)
                if ct_str == "ALL":
                    ax.plot(np.arange(2, 11), ct_method_metric, color='black', lw=1, alpha=0.9, zorder=9)

            ax.set_xlabel("Cluster number (k)", labelpad=1)
            ax.set_ylabel(metric_name, labelpad=1)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10])
            if "PAC" in metric_name:
                ax.set_yticks([0, 0.2, 0.4, 0.6], ["0", "20", "40", "60"])

            save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                        f"{method_name}_{metric_name.split(' ')[0]}_alpha{kw.get('weight_alpha')}.png")
            default_exit_save(fig, save_path=save_path, _transparent=True)


    fig, axs = plt.subplots(4, 5, figsize=(6.5, 5.2), constrained_layout=True, sharey="col")

    def plot_metric(ax, metric_dict):
        grand_avg = []
        for method in method_list[:-1]:
            avg_values = [np.nanmean(metric_dict[k][method]) for k in np.arange(2, 11)]
            grand_avg.append(avg_values)
            ax.plot(np.arange(2, 11), avg_values, label=method, lw=0.5)
        ax.plot(np.arange(2, 11), np.mean(grand_avg, axis=0), label="Grand Avg", color='black', lw=1, ls='--')
        ax.legend(frameon=False, loc='best', fontsize=5, handlelength=1, labelspacing=0.2, handletextpad=0.4)
        ax.tick_params(length=1, pad=1)
        ax.set_xlabel("Num of cluster (K)", labelpad=1)
        ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10])

    axs[0, 0].set_title("Silhouette Score")
    plot_metric(axs[0, 0], SilhouetteScore(clustering_results, "SST"))
    axs[0, 0].set_ylabel("SST", labelpad=1)
    plot_metric(axs[1, 0], SilhouetteScore(clustering_results, "PV"))
    axs[1, 0].set_ylabel("PV", labelpad=1)
    plot_metric(axs[2, 0], SilhouetteScore(clustering_results, "PYR"))
    axs[2, 0].set_ylabel("PYR", labelpad=1)
    plot_metric(axs[3, 0], SilhouetteScore(clustering_results, "ALL"))
    axs[3, 0].set_ylabel("ALL cell types", labelpad=1)

    axs[0, 1].set_title("Calinski-Harabasz Index")
    plot_metric(axs[0, 1], CHIndex(clustering_results, "SST"))
    axs[0, 1].set_ylabel("SST", labelpad=1)
    plot_metric(axs[1, 1], CHIndex(clustering_results, "PV"))
    axs[1, 1].set_ylabel("PV", labelpad=1)
    plot_metric(axs[2, 1], CHIndex(clustering_results, "PYR"))
    axs[2, 1].set_ylabel("PYR", labelpad=1)
    plot_metric(axs[3, 1], CHIndex(clustering_results, "ALL"))
    axs[3, 1].set_ylabel("ALL cell types", labelpad=1)

    axs[0, 2].set_title("Davies-Bouldin Index")
    plot_metric(axs[0, 2], DBIndex(clustering_results, "SST"))
    axs[0, 2].set_ylabel("SST", labelpad=1)
    plot_metric(axs[1, 2], DBIndex(clustering_results, "PV"))
    axs[1, 2].set_ylabel("PV", labelpad=1)
    plot_metric(axs[2, 2], DBIndex(clustering_results, "PYR"))   
    axs[2, 2].set_ylabel("PYR", labelpad=1)
    plot_metric(axs[3, 2], DBIndex(clustering_results, "ALL"))
    axs[3, 2].set_ylabel("ALL cell types", labelpad=1)

    axs[0, 3].set_title("Percentage of Ambiguous\nClustering (PAC)")
    plot_metric(axs[0, 3], {k: PAC(v, within=False, cross=False) for k, v in clustering_results.items()})
    axs[0, 3].set_ylabel("Grand consensus", labelpad=1)
    plot_metric(axs[1, 3], {k: PAC(v, within=True, cross=False) for k, v in clustering_results.items()})
    axs[1, 3].set_ylabel("Within-cell type consensus", labelpad=1)
    plot_metric(axs[2, 3], {k: PAC(v, within=False, cross=True) for k, v in clustering_results.items()})
    axs[2, 3].set_ylabel("Cross-cell type consensus", labelpad=1)


    axs[0, 4].set_title("PAC for Specific\nCell Types")
    plot_metric(axs[0, 4], {k: PAC_st(v, specific_celltype="SST") for k, v in clustering_results.items()})
    axs[0, 4].set_ylabel("SST", labelpad=1)
    plot_metric(axs[1, 4], {k: PAC_st(v, specific_celltype="PV") for k, v in clustering_results.items()})
    axs[1, 4].set_ylabel("PV", labelpad=1)
    plot_metric(axs[2, 4], {k: PAC_st(v, specific_celltype="PYR") for k, v in clustering_results.items()})
    axs[2, 4].set_ylabel("PYR", labelpad=1)
    plot_metric(axs[3, 4], {k: PAC_st(v, specific_celltype="ALL") for k, v in clustering_results.items()})
    axs[3, 4].set_ylabel("ALL cell types", labelpad=1)

    save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                f"All_metrics_alpha{kw.get('weight_alpha')}.png")
    default_exit_save(fig, save_path=save_path, _transparent=True)



from scipy.stats import pearsonr, spearmanr

def clustering_stability_test(cf_list: list[CellFeature], weight_alpha, task_id, n_feature, num_fcluster):
    
    def grand_coclustering_matrix(grid_search_result):
        return np.nanmean([np.nanmean(grid_search_result[clustering_method], axis=0) 
                           for clustering_method in grid_search_result.keys()], axis=0)
    clustering_results = {
        k: grand_coclustering_matrix(
            consensus_grid_search(cf_list, num_of_cluster=k, weight_alpha=weight_alpha, n_features=n_feature, task_id=task_id)
        )
        for k in range(2, 11)  
    }

    resample_clustering_result_90 = {
        k: np.stack([
            grand_coclustering_matrix(
                consensus_grid_search(cf_list, num_of_cluster=k, 
                                      n_weight=10, weight_alpha=weight_alpha, 
                                      _resample_tuple=(0.9, resample_seed),
                                      n_features=n_feature, task_id=task_id)
            )
            for resample_seed in range(100)  
        ], axis=0)
        for k in range(2, 11)  
    }

    resample_clustering_result_75 = {
        k: np.stack([
            grand_coclustering_matrix(
                consensus_grid_search(cf_list, num_of_cluster=k, 
                                      n_weight=10, weight_alpha=weight_alpha, 
                                      _resample_tuple=(0.75, resample_seed),
                                      n_features=n_feature, task_id=task_id)
            )
            for resample_seed in range(100)  
        ], axis=0)
        for k in range(2, 11)  
    }


    resample_clustering_result_50 = {
        k: np.stack([
            grand_coclustering_matrix(
                consensus_grid_search(cf_list, num_of_cluster=k, 
                                      n_weight=10, weight_alpha=weight_alpha, 
                                      _resample_tuple=(0.5, resample_seed),
                                      n_features=n_feature, task_id=task_id)
            )
            for resample_seed in range(100)  
        ], axis=0)
        for k in range(2, 11)  
    }

    def compare_consensus(C, resample_C):
        n_samples = C.shape[0]
        assert C.shape == resample_C.shape == (n_samples, n_samples)
        iu = np.triu_indices(n_samples, k=1)
        x, y = C[iu], resample_C[iu]
        valid_indices = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid_indices], y[valid_indices]
        return {
            "pearson": pearsonr(x, y).statistic,
            "spearman": spearmanr(x, y).statistic,
            "mae": np.mean(np.abs(x - y)),
            "rmse": np.sqrt(np.mean((x - y) ** 2))
        }

    resample_comparison_90 = {
        k: [compare_consensus(clustering_results[k], resample_single_consensus)
            for resample_single_consensus in resample_clustering_result_90[k]]
        for k in range(2, 11)
    }

    resample_comparison_75 = {
        k: [compare_consensus(clustering_results[k], resample_single_consensus)
            for resample_single_consensus in resample_clustering_result_75[k]]
        for k in range(2, 11)
    }

    resample_comparison_50 = {
        k: [compare_consensus(clustering_results[k], resample_single_consensus)
            for resample_single_consensus in resample_clustering_result_50[k]]
        for k in range(2, 11)
    }

    fig, ax = plt.subplots(1, 1, figsize=(1.4, 1.1), constrained_layout=True)
    metric_name = "pearson"
    metric_description = "Similarity to consensus\n of full dataset"
    ax.tick_params(length=1, pad=1)
    for resample_comparison, resample_color, resample_str, resample_offset in zip(
        [resample_comparison_90, resample_comparison_75, resample_comparison_50],
        ["C0", "C1", "C2"],
        ["90%", "75%", "50%"],
        [-0.25, 0, 0.25]
    ):
        avg_metric = [np.mean([single_compare[metric_name] 
                        for single_compare in resample_comparison[k]] )
                        for k in range(2, 11)]
        std_metric = [np.std([single_compare[metric_name] 
                        for single_compare in resample_comparison[k]] )
                        for k in range(2, 11)]
        ax.errorbar(np.arange(2, 11) + resample_offset, avg_metric, 
                        yerr=std_metric,
                        fmt="o", color=resample_color,
                        markersize=2, lw=1, alpha=0.9, markeredgewidth=0.5,
                        ecolor=resample_color, elinewidth=0.5,
                        capsize=1., capthick=0.5,)
    ax.set_ylabel(metric_description, labelpad=1)
    ax.set_xlabel("num of cluster (K)", labelpad=1)
    ax.set_xticks([2, 3, 4, 5, 6, 7, 8, 9, 10])
    ax.set_yticks([0, 0.5, 1])
    ax.spines[["top", "right"]].set_visible(False)

    save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                f"resample_stability_alpha{weight_alpha}.png")
    default_exit_save(fig, save_path=save_path, _transparent=True)

    selected_ks = np.arange(2, 11)
    selected_postfix = "_full"

    grand_consensus = np.nanmean([clustering_results[k] for k in selected_ks], axis=0)
    grand_consensus_90 = np.nanmean([resample_clustering_result_90[k] for k in selected_ks], axis=0)
    grand_consensus_75 = np.nanmean([resample_clustering_result_75[k] for k in selected_ks], axis=0)
    grand_consensus_50 = np.nanmean([resample_clustering_result_50[k] for k in selected_ks], axis=0)

    def get_fcluster_labels(C):
        valid_indices = ~np.isnan(np.diagonal(C))
        dissimilarity = 1 - C[np.ix_(valid_indices, valid_indices)]
        condensed_dissimilarity = squareform(dissimilarity, checks=True)

        Z = linkage(
            condensed_dissimilarity,
            method="average",
        )  
        
        fcluster_labels = fcluster(Z, t=num_fcluster, criterion="maxclust")
        full_labels = np.full(C.shape[0], -1, dtype=int)
        full_labels[valid_indices] = fcluster_labels
        return full_labels

    def outlier_free_ari(y_true, y_pred):
        mask = (y_true != -1) & (y_pred != -1)

        ari = adjusted_rand_score(
            y_true[mask],
            y_pred[mask]
        )
        return ari

    
    grand_fcluster_labels = get_fcluster_labels(grand_consensus)
    n_cluster = len(np.unique(grand_fcluster_labels))
    grand_fcluster_labels_90 = np.stack([get_fcluster_labels(single_consensus) 
                                         for single_consensus in grand_consensus_90], axis=0)
    grand_fcluster_labels_75 = np.stack([get_fcluster_labels(single_consensus) 
                                         for single_consensus in grand_consensus_75], axis=0)
    grand_fcluster_labels_50 = np.stack([get_fcluster_labels(single_consensus) 
                                         for single_consensus in grand_consensus_50], axis=0)

    ari_score_90 = [outlier_free_ari(grand_fcluster_labels, single_labels) 
                    for single_labels in grand_fcluster_labels_90]
    ari_score_75 = [outlier_free_ari(grand_fcluster_labels, single_labels) 
                    for single_labels in grand_fcluster_labels_75]
    ari_score_50 = [outlier_free_ari(grand_fcluster_labels, single_labels) 
                    for single_labels in grand_fcluster_labels_50]

    fig, ax = plt.subplots(1, 1, figsize=(0.7, 0.7), constrained_layout=True)
    ax.tick_params(length=1, pad=1)

    for resample_ari_score, resample_color, resample_str, resample_offset in zip(
        [ari_score_90, ari_score_75, ari_score_50],
        ["C0", "C1", "C2"],
        ["90%", "75%", "50%"],
        [-0.7, 0, 0.7]
    ):
        avg_metric = np.mean(resample_ari_score)
        std_metric = np.std(resample_ari_score)
        ax.errorbar(resample_offset, avg_metric, 
                        yerr=std_metric,
                        fmt="o", color=resample_color,
                        markersize=2, lw=1, alpha=0.9, markeredgewidth=0.5,
                        ecolor=resample_color, elinewidth=0.5,
                        capsize=1., capthick=0.5,)
    ax.set_ylabel("Adjusted RI", labelpad=1)
    ax.set_xticks([-0.7, 0, 0.7], ["90%", "75%", "50%"], fontsize=5)
    for tick_id, tick in enumerate(ax.get_xticklabels()):
        tick.set_rotation(0)
        tick.set_color(f"C{tick_id}")
    ax.set_yticks([-1, 0, 1])
    ax.set_xlim(-0.9, 0.9)
    ax.spines[["top", "right"]].set_visible(False)
    save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                f"resample_stability_alpha{weight_alpha}_ari_{selected_postfix}.png")
    default_exit_save(fig, save_path=save_path, _transparent=True)




    def cluster_best_match_jaccard(grand_labels, resample_labels):
        best_jaccard_scores = defaultdict(list)
        for ref_cluster in np.unique(grand_labels):

            ref_mask = (grand_labels == ref_cluster)
            for single_resample_label in resample_labels:
                valid_res_mask = single_resample_label != -1
                if np.sum(ref_mask[valid_res_mask]) == 0:
                    continue

                best_jaccard = -1
                for res_cluster in np.unique(single_resample_label):

                    res_mask = (single_resample_label == res_cluster)
                    score = jaccard_score(
                        ref_mask[valid_res_mask],
                        res_mask[valid_res_mask],
                        average="binary",
                        zero_division=0,
                    )

                    if score > best_jaccard:
                        best_jaccard = score

                best_jaccard_scores[ref_cluster].append(best_jaccard)
        return best_jaccard_scores

    jaccard_scores_90 = cluster_best_match_jaccard(grand_fcluster_labels, grand_fcluster_labels_90)
    jaccard_scores_75 = cluster_best_match_jaccard(grand_fcluster_labels, grand_fcluster_labels_75)
    jaccard_scores_50 = cluster_best_match_jaccard(grand_fcluster_labels, grand_fcluster_labels_50)

    fig, ax = plt.subplots(1, 1, figsize=(1.2, 0.7), constrained_layout=True)
    ax.tick_params(length=1, pad=1)

    for resample_jaccard_score, resample_color, resample_str, resample_offset in zip(
            [jaccard_scores_90, jaccard_scores_75, jaccard_scores_50],
            ["C0", "C1", "C2"],
            ["90%", "75%", "50%"],
            [-0.25, 0, 0.25]
        ):
        avg_metric = [np.mean(resample_jaccard_score[ref_cluster]) for ref_cluster in range(1, n_cluster+1)]
        std_metric = [np.std(resample_jaccard_score[ref_cluster]) for ref_cluster in range(1, n_cluster+1)]
        ax.errorbar(np.arange(1, n_cluster+1) + resample_offset, avg_metric, 
                        yerr=std_metric,
                        fmt="o", color=resample_color,
                        markersize=2, lw=1, alpha=0.9, markeredgewidth=0.5,
                        ecolor=resample_color, elinewidth=0.5,
                        capsize=1., capthick=0.5,)
    ax.set_ylabel("Jaccard Index", labelpad=1)
    ax.set_xticks(list(range(1, n_cluster+1)))
    
    for tick in ax.get_xticklabels():
        tick.set_rotation(0)
        tick.set_color(CLUSTER_COLORS[int(tick.get_text())])
    ax.set_yticks([0, 0.5, 1],)
    ax.spines[["top", "right"]].set_visible(False)

    save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                f"resample_stability_alpha{weight_alpha}_jaccard_{selected_postfix}.png")
    default_exit_save(fig, save_path=save_path, _transparent=True)
    


def visualize_final_grand(feature_list: list[CellFeature],
                          coclustering_matrix: np.ndarray, 
                          feature_names,
                          save_path: str, _visualize: bool=True):
    n_cell = len(feature_list)
    assert coclustering_matrix.shape == (n_cell, n_cell), \
        "Coclustering matrix shape does not match the number of cells. Expected " \
        f"({n_cell}, {n_cell}), got {coclustering_matrix.shape}."
    
    dissimilarity = np.nan_to_num(1.0 - coclustering_matrix, nan=0.)
    condensed_dissimilarity = squareform(dissimilarity, checks=True)

    Z = linkage(
        condensed_dissimilarity,
        method="average",
    )  
    
    Z_opt = optimal_leaf_ordering(Z, condensed_dissimilarity)
    fcluster_labels = fcluster(Z_opt, t=6, criterion="maxclust")

    if not _visualize:
        return fcluster_labels, dendrogram(Z_opt)['leaves']

    cluster_leaf_map = {i: {i} for i in range(n_cell)}
    for i, row in enumerate(Z_opt):
        cluster_id = n_cell + i
        child1, child2 = int(row[0]), int(row[1])
        cluster_leaf_map[cluster_id] = cluster_leaf_map[child1].union(cluster_leaf_map[child2])

    def link_cluster_func(k):
        # cluster_id = k + n_samples
        leaf_indices = cluster_leaf_map[k]
        leaf_clusters = {fcluster_labels[i] for i in leaf_indices}
        
        if len(leaf_clusters) == 1:
            cluster_name = leaf_clusters.pop()
            return CLUSTER_COLORS[cluster_name]
        return 'gray'
    

    fig, axs = plt.subplots(2, 2, figsize=(1.5, 1.5), constrained_layout=True, 
                            width_ratios=[1, 0.2], height_ratios=[0.2, 1], sharey='row')

    
    with plt.rc_context({'lines.linewidth': 0.5}):
        dendro = dendrogram(Z_opt, ax=axs[0, 0], orientation="top", no_labels=True,
                            link_color_func=link_cluster_func)
    merge_distance = Z[:, 2]
    axs[0, 1].hist(merge_distance, bins=20, orientation='horizontal', color='grey')
        
    leaves_order = dendro['leaves']
    sorted_fcluster_labels = fcluster_labels[leaves_order]
    boundary_indices = [i for i in range(1, len(sorted_fcluster_labels)) 
                        if sorted_fcluster_labels[i] != sorted_fcluster_labels[i - 1]]
    for i in range(n_cell):
        axs[1, 1].axhline(y=i+0.5, color='white', linestyle="-", lw=0.1)
    for i in range(3):
        axs[1, 1].axvline(x=i+0.5, color='black', linestyle="-", lw=0.5)
    for boundary_index in boundary_indices:        
        axs[0, 0].axvline(x=boundary_index * 10, color="black", linestyle="--", lw=0.5)
        axs[1, 0].axvline(x=boundary_index - 0.5, color="white", linestyle="--", lw=0.5)
        axs[1, 0].axhline(y=boundary_index - 0.5, color="white", linestyle="--", lw=0.5)
        axs[1, 1].axhline(y=boundary_index - 0.5, color="black", linestyle="--", lw=0.5)

    axs[1, 0].imshow(coclustering_matrix[leaves_order, :][:, leaves_order], 
                     cmap="viridis", vmin=0, vmax=1)

    
    
    COHORT_vis = {"SST" : (0, COHORT_BRIGHTER_COLORS["SST_JUX"]),
                "PV": (1, COHORT_BRIGHTER_COLORS["PV_JUX"]),
                "PYR": (2, COHORT_BRIGHTER_COLORS["PYR_JUX"])}

    COHORT_DARKER_vis = {"SST" : (0, COHORT_COLORS["SST_JUX"]),
                "PV": (1, COHORT_COLORS["PV_JUX"]),
                "PYR": (2, COHORT_COLORS["PYR_JUX"])}

    
    ct_matrix = np.full((n_cell, 3, 4), np.nan)
    for cf_id, cf_index in enumerate(leaves_order):
        cf = feature_list[cf_index]
        x_cor, cf_color = COHORT_vis[cf.cohort_name.split("_")[0]]
        ct_matrix[cf_id, x_cor] = to_rgba(cf_color)
    
    axs[1, 1].imshow(ct_matrix[:, :], aspect="auto")
    
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axs[0, :]:        
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    for ax in axs[1, :]:        
        ax.spines[["top", "right", "left", "bottom"]].set_linewidth(0.5)
    
    default_exit_save(fig, save_path=save_path + ".png", _transparent=True, _dpi=1500)



    # def draw_ellipse(points, ax, **kwargs):
    #     from matplotlib.patches import Ellipse
    #     position = points.mean(axis=0)
    #     covariance = np.cov(points, rowvar=False)

    #     if covariance.shape == (2, 2):
    #         U, s, Vt = np.linalg.svd(covariance)
    #         angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
    #         width, height = 2 * np.sqrt(s)
    #     else:
    #         angle = 0
    #         width, height = 2 * np.sqrt(covariance)

    #     ell = Ellipse(xy=position, width=3 * width, height=3 * height, angle=angle, **kwargs)
        
    #     ax.add_patch(ell)
    #     return ell
    
    # for i in (7, 8, 9, 10, 11, 12, 13, 14, 15):
    #     for type_embed in ("feature", "similarity"):
    #         if type_embed == "feature":
    #             embedding = umap.UMAP(
    #                 # metric="precomputed",
    #                 n_neighbors=i,
    #                 random_state=42
    #             # ).fit_transform(dissimilarity)
    #             ).fit_transform(normalized_feature_matrix)
    #         else:                
    #             embedding = umap.UMAP(
    #                 metric="precomputed",
    #                 n_neighbors=i,
    #                 random_state=42
    #             ).fit_transform(dissimilarity)

    #         embed_color = [COHORT_BRIGHTER_COLORS[cf.cohort_name.split("_")[0] + "_JUX"] 
    #                     for cf in feature_list]
            
    #         fig, ax = plt.subplots(1, 1, figsize=(1.2, 1.2), constrained_layout=True)
    #         ax.scatter(embedding[:, 0], embedding[:, 1],
    #                 facecolors = embed_color, edgecolors="black", lw=0.4, s=10, alpha=0.9)
    #         n_cluster = len(np.unique(fcluster_labels))
    #         for cluster_id in range(1, n_cluster+1):
    #             cluster_points = embedding[fcluster_labels == cluster_id]
    #             draw_ellipse(cluster_points, ax, edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"], 
    #                         facecolor="none", lw=1, ls='--', alpha=0.8, zorder=-10)

    #         ax.spines[['right', 'top']].set_visible(False)
    #         ax.set_aspect('equal', adjustable='box') 
    #         ax.set_xlabel("UMAP 1")
    #         ax.set_ylabel("UMAP 2")
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         default_exit_save(fig, save_path=save_path + f"_umap_{i}_{type_embed}.png", _transparent=True)
    
    n_cluster = len(np.unique(fcluster_labels))
    cluster_per_cohort = pd.DataFrame({
        "Cohort": [cf.celltype for cf in feature_list],
        "Cluster": [str(int(single_cluster)) for single_cluster in fcluster_labels]
    })
    
    cluster_per_cohort = cluster_per_cohort.groupby(['Cohort', 'Cluster']).size().unstack(fill_value=0)
    cluster_per_cohort = cluster_per_cohort.reindex(index=["SST", "PV", "PYR"], fill_value=0)
    cluster_per_cohort = cluster_per_cohort.reindex(columns=[f"{i}" for i in range(1, n_cluster+1)], fill_value=0)
    cluster_per_cohort_pct = cluster_per_cohort.div(cluster_per_cohort.sum(axis=1), axis=0) * 100
    cluster_per_cohort_pct = cluster_per_cohort_pct.fillna(0) # handle cohorts with 0 cells
    
    fig, ax = plt.subplots(1, 1, figsize=(1.7, 1.), constrained_layout=True)

    ax.tick_params(
        length=1,
        pad=1,
    )
    annot_labels = cluster_per_cohort_pct.applymap(lambda x: f'{x:.0f}%\n' if x > 0 else '') +\
          cluster_per_cohort.copy().applymap(lambda x: f'({x})' if x > 0 else '')
    sns.heatmap(cluster_per_cohort, 
                annot=annot_labels, 
                fmt='s',           
                cmap='GnBu',      
                vmin=0, vmax=15,
                linewidths=0.3,
                linecolor='black',
                ax=ax,
                square=False,
                cbar=False,
                annot_kws={"size": 6})

    # Apply custom tick formatting
    # X-axis (Clusters)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(7)
        tick.set_rotation(0)
        tick.set_color(CLUSTER_COLORS[int(tick.get_text())])

    # Y-axis (Cohorts)
    for tick in ax.get_yticklabels():
        tick.set_fontsize(7)
        tick.set_rotation(0)
        tick.set_color(COHORT_DARKER_vis[tick.get_text()][1])

    ax.add_patch(
        Rectangle(
            (0, 0),
            cluster_per_cohort.shape[1],
            cluster_per_cohort.shape[0],
            fill=False,
            edgecolor="black",
            linewidth=0.3,
            clip_on=False,
            zorder=10,
        )
    )
    
    ax.set_ylabel("" )
    ax.set_xlabel("") 
    default_exit_save(fig, save_path=save_path + "_confusion.png", _transparent=True)


    raw_feature = np.stack([cf.features for cf in feature_list], axis=0)
    normalized_feature_matrix = numpy_kit.zscore(raw_feature, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.2), constrained_layout=True)
    ax.tick_params(axis='x', length=1, pad=1)
    ax.tick_params(axis='y', length=0, pad=1)
    for i in range(n_cell):
        ax.axhline(y=i+0.5, color='white', linestyle="-", lw=0.1)
    for i in np.cumsum([len(feature_subgroup) for feature_subgroup in feature_names]):
        ax.axvline(x=i-0.5, color='black', linestyle="-", lw=0.5)
    for boundary_index in boundary_indices:        
        ax.axhline(y=boundary_index - 0.5, color="black", linestyle="-", lw=0.5)

    ax.imshow(normalized_feature_matrix[leaves_order], 
              cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    concat_feature_names = sum(feature_names, [])
    ax.set_xticks(np.arange(len(concat_feature_names)))
    ax.set_xticklabels(concat_feature_names, rotation=30, ha='right')

    full_boundary_indices = [0] + boundary_indices + [n_cell]
    cluster_center = [
        (left_boundary + right_boundary-1) / 2
        for left_boundary, right_boundary in zip(full_boundary_indices[:-1], full_boundary_indices[1:])
    ]
    ax.set_yticks(cluster_center)
    ax.set_yticklabels([f"{sorted_fcluster_labels[i]}" for i in full_boundary_indices[:-1]])
    for tick in ax.get_yticklabels():
        tick.set_rotation(0)
        tick.set_color(CLUSTER_COLORS[int(tick.get_text())])

    ax.spines[["top", "right", "left", "bottom"]].set_linewidth(0.5)
    default_exit_save(fig, save_path=save_path + "_feature_matrix.png", _transparent=True)




    
    
    wave_weight_matrix, waveform_node_names, cohort_name = get_weight_tuple_Waveform()
    wave_node_to_row = {name: row for row, name in enumerate(waveform_node_names)}
    wave_t = np.linspace(-2/1000, 2/1000, wave_weight_matrix.shape[1])
    ephys_tuple = get_weight_tuple_Physiology_Fingerprint(with_spont_FR=True, 
                                                          _drop_ACG_rise_time=False, 
                                                          _log_ACG_rise_time=False)
    node_to_ephys = {name: {
        "Spont. FR": row[0],
        "Width": row[1],
        "Asym.": row[2],
        "CV2": row[3],
        "ACG " + r"$\tau$": row[4],
    } for name, row in zip(ephys_tuple[1], ephys_tuple[0])}
    ephys_feature_names = ("Spont. FR", "Width", "Asym.", "CV2", "ACG " + r"$\tau$")

    svd_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", n_components=5, 
                                                             append_baseline=False, 
                                                             BINSIZE=10/1000,
                                                             _visualize=False)
    node_to_svd = {name: {
        "PC1": row[0],
        "PC2": row[1],
        "PC3": row[2],
        "PC4": row[3],
        "PC5": row[4],
    } for name, row in zip(svd_tuple[1], svd_tuple[0])}
    svd_feature_names = ("PC1", "PC2", "PC3", "PC4", "PC5")
    
    
    from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_all_cellsession_PSTH_mean

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6
    fig, axs = plt.subplots(1, 3, constrained_layout=True)
    all_node_names = [cf.node_name for cf in feature_list]
    for ax, dataset_name_group in zip(axs, ( ("SST_JUX", "SST_WC"), ("PV_JUX",), ("PYR_JUX", "PYR_WC"),) ):    
        ax.tick_params(
            pad=1,       # smaller gap between ticks and tick labels
            length=1
        )
            
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name_group[0]]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name_group[0]]
        all_spikes_histogram, node_to_row, prev_group_cnt = [], {}, 0
        for dataset_name in dataset_name_group:
            bin_centers, single_group_all_spikes_histogram, _, node_names = get_all_cellsession_PSTH_mean(dataset_name, 5/1000, (-0.5, 1.0))
            for row, name in enumerate(node_names):
                node_to_row[name] = row + prev_group_cnt
            all_spikes_histogram.append(single_group_all_spikes_histogram)
            prev_group_cnt += len(node_names)
        all_spikes_histogram = np.concatenate(all_spikes_histogram, axis=0)

        fcluster_specific_order = [
            node_to_row[all_node_names[i]]
            for i in leaves_order
            if all_node_names[i] in node_to_row
        ]
        corresponding_labels = [fcluster_labels[i] for i in leaves_order if all_node_names[i] in node_to_row]
        boundary_indices = []
        for i, (prelabel, postlabel) in enumerate(zip(corresponding_labels[:-1], corresponding_labels[1:])):
            if prelabel != postlabel:
                ax.axhline(i + 1, color="black", linewidth=0.5)
                boundary_indices.append(i + 1)
        boundary_indices = [0] + boundary_indices + [len(corresponding_labels)]
        group_centers = [(boundary_indices[i] + boundary_indices[i+1]) / 2 for i in range(len(boundary_indices)-1)]
        group_cluster = [corresponding_labels[boundary_indices[i]] for i in range(len(boundary_indices)-1)]

        sns.heatmap(all_spikes_histogram[fcluster_specific_order],  
                    ax=ax, cbar=False,
                    cmap='YlOrBr', 
                    vmin=0, vmax=100,)
                    # norm=SymLogNorm(linthresh=10., vmin=0, vmax=150),)
        
        desired_ticks = [0, 0.5]
        tick_positions = [np.argmin(np.abs(bin_centers - tick)) + 0.5 for tick in desired_ticks]
        ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

        # ax.invert_yaxis()
        ax.set_xlabel("Time [s]")

        ax.set_yticks(group_centers)
        ax.set_yticklabels([f"{i}" for i in group_cluster])    
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
            tick.set_color(CLUSTER_COLORS[int(tick.get_text())])
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)

        corresponding_cfs = [feature_list[i] for i in leaves_order if all_node_names[i] in node_to_row]
        # wc_ytick = []
        # for i, cf in enumerate(corresponding_cfs):
        #     print(cf.cohort_name)
        #     if "WC" in cf.cohort_name:
        #         wc_ytick.append(i + 0.5)

        # twinax = ax.twinx()
        # twinax.set_yticks(wc_ytick, ["WC" for _ in wc_ytick])
        # twinax.set_ylim(ax.get_ylim())

        for cluster_id in group_cluster:
            cluster_corresponding_idx = np.where(np.array(corresponding_labels) == cluster_id)[0]
            cluster_psth = all_spikes_histogram[fcluster_specific_order][cluster_corresponding_idx]
            stim_window = np.searchsorted(bin_centers, (0, 0.5))
            avg_sorting_order = np.argsort(np.mean(cluster_psth[:, stim_window[0]:stim_window[1]], axis=1))[::-1]
            sorted_cluster_psth = AdvancedTimeSeries(
                t=bin_centers, v=cluster_psth[avg_sorting_order].mean(axis=0),
                variance=cluster_psth[avg_sorting_order].std(axis=0),
                raw_array=cluster_psth[avg_sorting_order]
            )
            corresponding_cfs_in_cluster = [corresponding_cfs[i] for i in cluster_corresponding_idx[avg_sorting_order]]

            figAB, axAB = plt.subplots(2, 1, figsize=(1, 1,), constrained_layout=True)
            axA = axAB[0]
            oreo_plot(axA, sorted_cluster_psth, 0, 1,  
                        {"color": 'black', "lw": 1, "alpha": 1},
                        # {"color": CLUSTER_COLORS[f"cluster {cluster_id}"], "lw": 1, "alpha": 1},
                        style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, })
            axA.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
            axA.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
            axA.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
            axA.set_xlim(-0.5, 1)
            axA.set_ylim(-30, 250)
            axA.set_xticks([])
            axA.set_yticks([])

            axB = axAB[1]
            oreo_plot(axB, sorted_cluster_psth, 0, 1,  
                        {"color": 'black', "lw": 1, "alpha": 1},
                        # {"color": CLUSTER_COLORS[f"cluster {cluster_id}"], "lw": 1, "alpha": 1},
                        style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, })
            axB.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
            axB.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
            axB.axvline(20/1000, color='black', linestyle='--', lw=0.5, alpha=0.5, zorder=5)
            axB.set_xlim(-50/1000, 100/1000 )
            axB.set_ylim(-30, 250)
            axB.set_xticks([])
            axB.set_yticks([])

            default_exit_save(figAB, save_path=save_path + f"_heatmap_{dataset_name_group}_{cluster_id}_AB.png", 
                              _transparent=True)

            figC, axC = plt.subplots(1, 1, figsize=(1.1, 1,), constrained_layout=True)
            axC.tick_params(
                length=1,
                pad=1,
            )            
            sns.heatmap(sorted_cluster_psth.raw_array,
                        ax=axC, cbar=False, 
                        cmap='YlOrBr', 
                        vmin=0, vmax=100,)
                        # norm=SymLogNorm(linthresh=10., vmin=0, vmax=150),)            
            desired_ticks = [0, 0.5]
            tick_positions = [np.argmin(np.abs(bin_centers - tick)) + 0.5 for tick in desired_ticks]
            axC.set_xticks(tick_positions, desired_ticks, rotation=0)  

            # if "SST_WC" in dataset_name_group:
            #     heatmap_cohort_names = [single_cf.cohort_name for single_cf in corresponding_cfs_in_cluster]
            #     WC_indices = [i for i, name in enumerate(heatmap_cohort_names) if name == "SST_WC"]
            #     axC.set_yticks(0.5 + np.array(WC_indices), [str(wc_idx+1) for wc_idx in WC_indices], rotation=0)
            # else:
            axC.set_yticks([0.5, len(sorted_cluster_psth)-0.5], 
                        ["1", f"{len(sorted_cluster_psth)}"], rotation=0)
            axC.set_xlabel("Time (s)", labelpad=1)
            
            axC.spines[['right', 'top', 'left', 'bottom']].set_visible(True)

            default_exit_save(figC, save_path=save_path + f"_heatmap_{dataset_name_group}_{cluster_id}_C.png", _transparent=True)


            figD, axD = plt.subplots(1, 1, figsize=(0.4, 0.5,), constrained_layout=True)
            wave_indices_in_cluster = [wave_node_to_row[single_cf.node_name] 
                                       for single_cf in corresponding_cfs_in_cluster
                                       if single_cf.node_name in wave_node_to_row]
            cluster_wave_weight_matrix = wave_weight_matrix[wave_indices_in_cluster]
            avg_waveform_in_cluster = np.mean(cluster_wave_weight_matrix, axis=0)
            axD.plot(wave_t, avg_waveform_in_cluster, lw=0.6, color='black', alpha=0.8)
            
            for single_waveform in cluster_wave_weight_matrix:
                axD.plot(wave_t, single_waveform, lw=0.3, color='gray', alpha=0.3, zorder=-1)
    
            axD.set_yticks([])
            axD.set_xticks([-2/1000, 0, 2/1000], ["-2", "0", "2"])
            # axD.set_xlabel("Time (ms)", labelpad=1)
            # axD.set_xlabel(f"{cluster_id}", color=CLUSTER_COLORS[f"cluster {cluster_id}"])
            axD.spines[['right', 'top', 'left',]].set_visible(False)
            axD.tick_params(length=1, pad=1)
            
            default_exit_save(figD, save_path=save_path + f"_heatmap_{dataset_name_group}_{cluster_id}_D.png", _transparent=True)


            figE, axEs = plt.subplots(1, len(ephys_feature_names), 
                                      figsize=(2.1, 0.6), constrained_layout=True,
                                      gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
            for f_id, single_ephys_feature_name in enumerate(ephys_feature_names):
                axE = axEs[f_id]
                ephys_feature_in_cluster = np.array([
                    node_to_ephys[single_cf.node_name][single_ephys_feature_name]
                    for single_cf in corresponding_cfs_in_cluster
                    if single_cf.node_name in node_to_ephys
                    ])

                
                rest_ephys_feature_in_cluster = np.array([
                    node_to_ephys[single_cf.node_name][single_ephys_feature_name]
                    for single_cf in feature_list
                    if single_cf.node_name in node_to_ephys 
                ])
                all_cf_feature_loc_in_rest = np.array([
                    single_cf.node_name
                    for single_cf in feature_list
                    if single_cf.node_name in node_to_ephys 
                ])
                rng = np.random.default_rng(seed=f_id)
                jitter_rest = rng.uniform(-0.12, 0.12, size=len(rest_ephys_feature_in_cluster))


                jitter = np.array([jitter_rest[all_cf_feature_loc_in_rest.tolist().index(single_cf.node_name)] 
                                   for single_cf in corresponding_cfs_in_cluster
                                   if single_cf.node_name in node_to_ephys])
                axE.scatter(
                    jitter, ephys_feature_in_cluster,
                    edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                    facecolor="white", lw=0.5, s=4, alpha=1, zorder=2,
                )
                axE.scatter(
                    jitter_rest, rest_ephys_feature_in_cluster,
                    edgecolor="none",
                    facecolor="gray", lw=0., s=3, alpha=0.3, zorder=-10,
                )
                axE.errorbar(0., np.nanmean(ephys_feature_in_cluster), 
                            yerr=np.nanstd(ephys_feature_in_cluster) if "ACG" not in single_ephys_feature_name else None,
                            fmt="o",
                            markeredgecolor="black",
                            markerfacecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                            markeredgewidth=0.5,
                            markersize=3,
                            ecolor='black',
                            elinewidth=0.5,
                            capsize=2,
                            capthick=0.5,
                            alpha=0.9,)
                axE.set_xlabel(single_ephys_feature_name, labelpad=1)
                axE.tick_params(length=1, pad=1, which='both')
                axE.spines[['right', 'top', ]].set_visible(False)
                axE.set_xticks([0,], [""])
                axE.set_xlim(-0.3, 0.3)
                if "ACG" in single_ephys_feature_name:
                    axE.set_yscale("log")
                    axE.set_yticks([0.1, 1, 10], [".1", "1", "10"])
                elif "Width" in single_ephys_feature_name:
                    axE.set_yticks([0, 1, 2])
                elif "Asym." in single_ephys_feature_name:
                    axE.set_yticks([-0.5, 0., 0.5], ["-.5", "0", ".5"])
                elif "CV2" in single_ephys_feature_name:
                    axE.set_yticks([1, 2], ["1", "2"])
            
            default_exit_save(figE, save_path=save_path + f"_heatmap_{dataset_name_group}_{cluster_id}_E.png",
                              _transparent=True)

            

            figF, axFs = plt.subplots(1, len(svd_feature_names), 
                                      figsize=(2.1, 0.6), constrained_layout=True,
                                      gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
            for f_id, single_svd_name in enumerate(svd_feature_names):
                axF = axFs[f_id]
                svd_feature_in_cluster = np.array([
                    node_to_svd[single_cf.node_name][single_svd_name]
                    for single_cf in corresponding_cfs_in_cluster
                    if single_cf.node_name in node_to_svd
                    ])

                
                rest_svd_feature_in_cluster = np.array([
                    node_to_svd[single_cf.node_name][single_svd_name]
                    for single_cf in feature_list
                    if single_cf.node_name in node_to_svd 
                ])
                all_cf_feature_loc_in_rest = np.array([
                    single_cf.node_name
                    for single_cf in feature_list
                    if single_cf.node_name in node_to_svd 
                ])
                rng = np.random.default_rng(seed=f_id)
                jitter_rest = rng.uniform(-0.12, 0.12, size=len(rest_svd_feature_in_cluster))


                jitter = np.array([jitter_rest[all_cf_feature_loc_in_rest.tolist().index(single_cf.node_name)] 
                                   for single_cf in corresponding_cfs_in_cluster
                                   if single_cf.node_name in node_to_svd])
                axF.scatter(
                    jitter, svd_feature_in_cluster,
                    edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                    facecolor="white", lw=0.5, s=4, alpha=1, zorder=2,
                )
                axF.scatter(
                    jitter_rest, rest_svd_feature_in_cluster,
                    edgecolor="none",
                    facecolor="gray", lw=0., s=3, alpha=0.3, zorder=-10,
                )
                axF.errorbar(0., np.nanmean(svd_feature_in_cluster), 
                            yerr=np.nanstd(svd_feature_in_cluster) if "ACG" not in single_svd_name else None,
                            fmt="o",
                            markeredgecolor="black",
                            markerfacecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                            markeredgewidth=0.5,
                            markersize=3,
                            ecolor='black',
                            elinewidth=0.5,
                            capsize=2,
                            capthick=0.5,
                            alpha=0.9,)
                axF.set_xlabel(single_svd_name, labelpad=1)
                axF.tick_params(length=1, pad=1)
                axF.spines[['right', 'top', ]].set_visible(False)
                axF.set_xticks([0,], [""])
                axF.set_xlim(-0.3, 0.3)
                axF.axhline(0., color='gray', lw=0.5, ls='--', zorder=-20)
                max_val = np.max(np.abs(axF.get_ylim()))
                axF.set_ylim(-max_val, max_val)
            
            default_exit_save(figF, save_path=save_path + f"_heatmap_{dataset_name_group}_{cluster_id}_F.png",
                              _transparent=True)

    fig.set_size_inches(6, 1.5)

    default_exit_save(fig, save_path=save_path + "_heatmap.png", _transparent=True)

    
    figE, axEs = plt.subplots(1, len(ephys_feature_names), 
                                figsize=(4., 1), constrained_layout=True,
                                gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    
    figF, axFs = plt.subplots(1, len(svd_feature_names), 
                                figsize=(4., 1), constrained_layout=True,
                                gridspec_kw={'hspace': 0.0, 'wspace': 0.0})
    annotation_E, annotation_F = {f_id: {} for f_id in range(len(ephys_feature_names))}, {f_id: {} for f_id in range(len(svd_feature_names))}
    corresponding_cfs = [feature_list[i] for i in leaves_order]
    corresponding_labels = [fcluster_labels[i] for i in leaves_order]
    for cluster_id in np.unique(corresponding_labels):
        cluster_corresponding_idx = np.where(np.array(corresponding_labels) == cluster_id)[0]
        corresponding_cfs_in_cluster = [corresponding_cfs[i] for i in cluster_corresponding_idx]

        for f_id, single_ephys_feature_name in enumerate(ephys_feature_names):
            axE = axEs[f_id]
            ephys_feature_in_cluster = np.array([
                node_to_ephys[single_cf.node_name][single_ephys_feature_name]
                for single_cf in corresponding_cfs_in_cluster
                if single_cf.node_name in node_to_ephys
                ])
            annotation_E[f_id][cluster_id] = ephys_feature_in_cluster
            
            rest_ephys_feature_in_cluster = np.array([
                node_to_ephys[single_cf.node_name][single_ephys_feature_name]
                for single_cf in feature_list
                if single_cf.node_name in node_to_ephys 
            ])
            all_cf_feature_loc_in_rest = np.array([
                single_cf.node_name
                for single_cf in feature_list
                if single_cf.node_name in node_to_ephys 
            ])
            rng = np.random.default_rng(seed=f_id)
            jitter_rest = rng.uniform(-0.2, 0.2, size=len(rest_ephys_feature_in_cluster))


            jitter = np.array([jitter_rest[all_cf_feature_loc_in_rest.tolist().index(single_cf.node_name)] 
                                for single_cf in corresponding_cfs_in_cluster
                                if single_cf.node_name in node_to_ephys])
            axE.scatter(
                jitter + cluster_id, ephys_feature_in_cluster,
                edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                facecolor="white", lw=0.5, s=4, alpha=1, zorder=2,
            )
            # axE.scatter(
            #     jitter_rest, rest_ephys_feature_in_cluster,
            #     edgecolor="none",
            #     facecolor="gray", lw=0., s=3, alpha=0.3, zorder=-10,
            # )
            axE.errorbar(cluster_id, np.nanmean(ephys_feature_in_cluster), 
                        yerr=np.nanstd(ephys_feature_in_cluster) if "ACG" not in single_ephys_feature_name else None,
                        fmt="o",
                        markeredgecolor="black",
                        markerfacecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                        markeredgewidth=0.5,
                        markersize=3,
                        ecolor='black',
                        elinewidth=0.5,
                        capsize=2,
                        capthick=0.5,
                        alpha=0.9,)
            axE.set_xlabel(single_ephys_feature_name, labelpad=1)
            axE.tick_params(length=1, pad=1, which='both')
            axE.spines[['right', 'top', ]].set_visible(False)
            axE.set_xticks([1, 2, 3, 4, 5, 6])
            for tick in axE.get_xticklabels():
                text = tick.get_text()
                tick.set_fontsize(7)
                tick.set_color(CLUSTER_COLORS[int(text)])
                tick.set_fontweight("bold")
            axE.set_xlim(1-0.5, 6+0.5)
            if "ACG" in single_ephys_feature_name:
                axE.set_yscale("log")
                axE.set_yticks([0.1, 1, 10], [".1", "1", "10"])
            elif "Width" in single_ephys_feature_name:
                axE.set_yticks([0, 1, 2])
            elif "Asym." in single_ephys_feature_name:
                axE.set_yticks([-0.5, 0., 0.5], ["-.5", "0", ".5"])
            elif "CV2" in single_ephys_feature_name:
                axE.set_yticks([1, 2], ["1", "2"])
        

        

        for f_id, single_svd_name in enumerate(svd_feature_names):
            axF = axFs[f_id]
            svd_feature_in_cluster = np.array([
                node_to_svd[single_cf.node_name][single_svd_name]
                for single_cf in corresponding_cfs_in_cluster
                if single_cf.node_name in node_to_svd
                ])
            annotation_F[f_id][cluster_id] = svd_feature_in_cluster
            
            rest_svd_feature_in_cluster = np.array([
                node_to_svd[single_cf.node_name][single_svd_name]
                for single_cf in feature_list
                if single_cf.node_name in node_to_svd 
            ])
            all_cf_feature_loc_in_rest = np.array([
                single_cf.node_name
                for single_cf in feature_list
                if single_cf.node_name in node_to_svd 
            ])
            rng = np.random.default_rng(seed=f_id)
            jitter_rest = rng.uniform(-0.2, 0.2, size=len(rest_svd_feature_in_cluster))


            jitter = np.array([jitter_rest[all_cf_feature_loc_in_rest.tolist().index(single_cf.node_name)] 
                                for single_cf in corresponding_cfs_in_cluster
                                if single_cf.node_name in node_to_svd])
            axF.scatter(
                jitter + cluster_id, svd_feature_in_cluster,
                edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                facecolor="white", lw=0.5, s=4, alpha=1, zorder=2,
            )
            # axF.scatter(
            #     jitter_rest, rest_svd_feature_in_cluster,
            #     edgecolor="none",
            #     facecolor="gray", lw=0., s=3, alpha=0.3, zorder=-10,
            # )
            axF.errorbar(cluster_id, np.nanmean(svd_feature_in_cluster), 
                        yerr=np.nanstd(svd_feature_in_cluster) if "ACG" not in single_svd_name else None,
                        fmt="o",
                        markeredgecolor="black",
                        markerfacecolor=CLUSTER_COLORS[f"cluster {cluster_id}"],
                        markeredgewidth=0.5,
                        markersize=3,
                        ecolor='black',
                        elinewidth=0.5,
                        capsize=2,
                        capthick=0.5,
                        alpha=0.9,)
            axF.set_xlabel(single_svd_name, labelpad=1)
            axF.tick_params(length=1, pad=1)
            axF.spines[['right', 'top', ]].set_visible(False)
            axF.set_xticks([1, 2, 3, 4, 5, 6])
            for tick in axF.get_xticklabels():
                text = tick.get_text()
                tick.set_fontsize(7)
                tick.set_color(CLUSTER_COLORS[int(text)])
                tick.set_fontweight("bold")
            axF.set_xlim(1-0.5, 6+0.5)
            axF.axhline(0., color='gray', lw=0.5, ls='--', zorder=-20)
            max_val = max(np.max(np.abs(axF.get_ylim())), 1.1 * np.nanmax(np.abs(svd_feature_in_cluster)))
            axF.set_ylim(-max_val, max_val)
    
    default_exit_save(figE, save_path=save_path + f"_heatmap_all_E.png",
                        _transparent=True, _no_close=True)
    default_exit_save(figF, save_path=save_path + f"_heatmap_all_F.png",
                        _transparent=True, _no_close=True)
    for f_id in range(len(ephys_feature_names)):
        basic_ttest.stats_anova(annotation_E[f_id], axEs[f_id], bon_ferroni_adj=len(ephys_feature_names)+len(svd_feature_names))
    for f_id in range(len(svd_feature_names)):
        basic_ttest.stats_anova(annotation_F[f_id], axFs[f_id], bon_ferroni_adj=len(ephys_feature_names)+len(svd_feature_names))
        axFs[f_id].autoscale(enable=True, axis='y')
    figE.set_size_inches(4, 4)
    default_exit_save(figE, save_path=save_path + f"_heatmap_all_E_annotate.png",
                        _transparent=True)
    figF.set_size_inches(4, 4)
    default_exit_save(figF, save_path=save_path + f"_heatmap_all_F_annotate.png",
                        _transparent=True)    


    figCDF, axCDF = plt.subplots(1, 2, figsize=(2.1, 1.15), sharey=True, sharex=True, constrained_layout=True)
    clusterss = np.unique(corresponding_labels)
    from scipy.spatial.distance import pdist
    
    for ax in axCDF:
        ax.tick_params( 
            length=1,
            pad=1     
        )
        ax.set_xlabel("Pairwise corr. (cosine)", labelpad=1)
        ax.spines[['right', 'top',]].set_visible(False)
        ax.set_yticks([0, 0.5, 1], ["0", "50", "100"], rotation=90, ha='right', va='center')
        ax.set_xticks([-1, 0, 1])
    
    axCDF[0].set_ylabel("Cumulative frac. (%)", labelpad=1)

    total_ephys_matrix, total_svd_matrix = [], []
    cdf_cluster_names = []
    for cluster_id in clusterss:
        ephys_matrix = np.stack([annotation_E[f_id][cluster_id] 
                                 for f_id in range(len(ephys_feature_names))], axis=1)
        svd_matrix = np.stack([annotation_F[f_id][cluster_id] 
                               for f_id in range(len(svd_feature_names))], axis=1)
        assert ephys_matrix.shape[0] == svd_matrix.shape[0]
        cdf_cluster_names.extend([cluster_id] * ephys_matrix.shape[0])
        total_ephys_matrix.append(ephys_matrix)
        total_svd_matrix.append(svd_matrix)

    cdf_cluster_names = np.array(cdf_cluster_names)
    total_ephys_matrix = np.concatenate(total_ephys_matrix, axis=0)
    total_ephys_matrix = numpy_kit.zscore(total_ephys_matrix, axis=0)
    total_svd_matrix = np.concatenate(total_svd_matrix, axis=0)
    total_svd_matrix = numpy_kit.zscore(total_svd_matrix, axis=0)
    for cluster_id in clusterss:
        ephys_matrix = total_ephys_matrix[cdf_cluster_names == cluster_id]
        svd_matrix = total_svd_matrix[cdf_cluster_names == cluster_id]
        pairwise_distance = pdist(ephys_matrix, metric="cosine")
        axCDF[0].plot(np.sort(1-pairwise_distance), np.arange(1, 1+len(pairwise_distance))/len(pairwise_distance), 
                      color=CLUSTER_COLORS[f"cluster {cluster_id}"], alpha=0.9, lw=1)
        pairwise_distance = pdist(svd_matrix, metric="cosine")
        axCDF[1].plot(np.sort(1-pairwise_distance), np.arange(1, 1+len(pairwise_distance))/len(pairwise_distance), 
                      color=CLUSTER_COLORS[f"cluster {cluster_id}"], alpha=0.9, lw=1)

    i, j = np.triu_indices(len(cdf_cluster_names), k=1)
    within_mask = cdf_cluster_names[i] == cdf_cluster_names[j]
    cross_mask = ~within_mask    

    cross_corr = 1 - pdist(total_ephys_matrix, metric="cosine")[cross_mask]
    axCDF[0].plot(np.sort(cross_corr), np.arange(1, 1+len(cross_corr))/len(cross_corr), color="black", alpha=0.9, lw=1, ls="--")
    cross_corr = 1 - pdist(total_svd_matrix, metric="cosine")[cross_mask]
    axCDF[1].plot(np.sort(cross_corr), np.arange(1, 1+len(cross_corr))/len(cross_corr), color="black", alpha=0.9, lw=1, ls="--")

    
    default_exit_save(figCDF, save_path=save_path + f"_CDF.png",
                        _transparent=True, _no_close=True)
    for ax, data_matrix in zip(axCDF, [total_ephys_matrix, total_svd_matrix]):
        results = basic_ttest.stats_pairwise_permutation(
            weight_matrix=data_matrix,
            cohort_name=cdf_cluster_names,
            ax=ax,
        )
    figCDF.set_size_inches(4, 2)
    default_exit_save(figCDF, save_path=save_path + f"_CDF_annotate.png",
                        _transparent=True, )
    

    
    fcluster_specific_order = [
        wave_node_to_row[all_node_names[i]]
        for i in leaves_order
        if all_node_names[i] in wave_node_to_row
    ]
    corresponding_labels = np.array([fcluster_labels[i] for i in leaves_order if all_node_names[i] in wave_node_to_row])
    sorted_weight_matrix = wave_weight_matrix[fcluster_specific_order]

    fig, ax = plt.subplots(1, n_cluster, figsize=(3, 0.8), constrained_layout=True)
    for cluster_id in range(1, n_cluster + 1):
        cluster_color = CLUSTER_COLORS[cluster_id]
        cluster_indices = np.where(corresponding_labels == cluster_id)[0]
        cluster_weight_matrix = sorted_weight_matrix[cluster_indices]
        avg_waveform = np.mean(cluster_weight_matrix, axis=0)
        ax[cluster_id - 1].plot(wave_t, avg_waveform, lw=1, color=cluster_color, alpha=0.8)

        for single_waveform in cluster_weight_matrix:
            ax[cluster_id - 1].plot(wave_t, single_waveform, lw=0.5, color='gray', alpha=0.3, zorder=-1)

        ax[cluster_id - 1].set_yticks([])
        ax[cluster_id - 1].set_xticks([-2/1000, 0, 2/1000], ["-2", "0", "2"])
        ax[cluster_id - 1].set_xlabel(f"{cluster_id}", color=cluster_color)
        ax[cluster_id - 1].spines[['right', 'top', 'left',]].set_visible(False)
        ax[cluster_id - 1].tick_params(length=1, pad=1)
    default_exit_save(fig, save_path=save_path + "_waveform.png", _transparent=True)





def grand_co_clustering_analysis(cf_list: list[CellFeature], weight_alpha, task_id, feature_names, **kw):

    def grand_coclustering_matrix(grid_search_result):
        return np.nanmean([np.nanmean(method_grid_result, axis=0) 
                           for method_grid_result in grid_search_result.values()], axis=0)
    clustering_results = {
        k: grand_coclustering_matrix(
            consensus_grid_search(cf_list, num_of_cluster=k, weight_alpha=weight_alpha, task_id=task_id, **kw)
        )
        for k in range(2, 11)  
    }

    full_avg_coclustering_matrix = np.nanmean([clustering_results[k] for k in range(2, 11)], axis=0)
    # stable_avg_coclustering_matrix = np.nanmean([clustering_results[k] for k in (2, 4, 6, 7)], axis=0)

    save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                f"full_avg_consensus_alpha{weight_alpha}")
    visualize_final_grand(cf_list, full_avg_coclustering_matrix, save_path=save_path, feature_names=feature_names)



def get_grand_cluster_result():
    
    def tuple2featurelist(feature_tuple) -> list[CellFeature]:
        cf_list = []
        for features, node_name, cohort_name in zip(feature_tuple[0], feature_tuple[1], feature_tuple[2]):
            cf_list.append(CellFeature(features=features, node_name=node_name, cohort_name=cohort_name))
        print(f"Total cells: {len(cf_list)}")
        print(Counter([cf.cohort_name for cf in cf_list]))
        return cf_list

    
    def combine_feature_lists(*feature_lists):
        shared_node_names = set.intersection(*(set(cf.node_name for cf in fl) for fl in feature_lists))
        combined_fl = []
        for single_shared_node_name in shared_node_names:
            combined_feature_vec, combined_cohort_name = [], []
            for fl in feature_lists:
                corresponding_cf = sequence_kit.find_only_one(fl, node_name=single_shared_node_name)
                combined_feature_vec.append(corresponding_cf.features)
                combined_cohort_name.append(corresponding_cf.cohort_name)
            assert len(set(combined_cohort_name)) == 1, f"Inconsistent cohort names for node {single_shared_node_name}"
            combined_fl.append(CellFeature(features=np.concatenate(combined_feature_vec),
                                        node_name=single_shared_node_name, cohort_name=combined_cohort_name[0]))
            assert np.all(np.isfinite(combined_fl[-1].features)), f"Non-finite values found in features for node {single_shared_node_name}"

        combined_fl = sorted(combined_fl, key=lambda cf: cf.node_name)
        return combined_fl
        
    svd_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", n_components=5, 
                                                            append_baseline=True, 
                                                            BINSIZE=10/1000,
                                                            _visualize=False)
    print(f"SVD 5 Feature List")
    svd_cf_list = tuple2featurelist(svd_tuple)
    combined_cf_list = svd_cf_list
    
    print(f"After combined, total cells: {len(combined_cf_list)}")
    print(Counter([cf.cohort_name for cf in combined_cf_list]))

    n_feature = len(combined_cf_list[0].features)
    
    feature_names = [
        ["Spont. FR",],
        [f"PC{i+1}" for i in range(5)],
    ]

    def grand_coclustering_matrix(grid_search_result):
        return np.nanmean([np.nanmean(method_grid_result, axis=0) 
                           for method_grid_result in grid_search_result.values()], axis=0)
    
    task_name = f"zscore_SVD{5}_bin{10}ms_{'exclude'}SpkFeature_{'exclude'}ACG"

    clustering_results = {
        k: grand_coclustering_matrix(
            consensus_grid_search(combined_cf_list, num_of_cluster=k, weight_alpha=50, task_id=task_name,
                                     n_features=n_feature)
        )
        for k in range(2, 11)  
    }

    full_avg_coclustering_matrix = np.nanmean([clustering_results[k] for k in range(2, 11)], axis=0)

    labels, leaves_order = visualize_final_grand(combined_cf_list, full_avg_coclustering_matrix, save_path='', 
                                 feature_names=feature_names, _visualize=False)
    return {node_name: single_labels for node_name, single_labels in zip([cf.node_name for cf in combined_cf_list], labels)}, leaves_order



def feature_contribution_analysis(cf_list: list[CellFeature], task_id, n_feature, weight_alpha, feature_names):
   
    clustering_methods = ["KMeans", "Spectral", "Gaussian", "Agg"]
    def grand_coclustering_matrix(grid_search_result):
        return np.nanmean([np.nanmean(grid_search_result[clustering_method], axis=0) 
                           for clustering_method in clustering_methods], axis=0)

    selected_ks = np.arange(2, 11)
    selected_postfix = "_full"


    grand_consensus_matrix = np.nanmean([
        grand_coclustering_matrix(
            consensus_grid_search(cf_list, num_of_cluster=k, weight_alpha=weight_alpha, task_id=task_id, n_features=n_feature), )
        for k in selected_ks
    ], axis=0)
    
    
    feature_drop_grand_consensus_matrix = {
        feature_id: np.nanmean([
            grand_coclustering_matrix(
                consensus_grid_search(cf_list, num_of_cluster=k, n_weight=100,
                                      weight_alpha=weight_alpha,
                                       _feature_drop_id=feature_id, task_id=task_id, n_features=n_feature),
            ) for k in selected_ks
        ], axis=0)
        for feature_id in range(n_feature)
    }

    def compare_consensus(C, resample_C):
        n_samples = C.shape[0]
        assert C.shape == resample_C.shape == (n_samples, n_samples)
        iu = np.triu_indices(n_samples, k=1)
        x, y = C[iu], resample_C[iu]
        valid_indices = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid_indices], y[valid_indices]
        return {
            "pearson": pearsonr(x, y).statistic,
            "spearman": spearmanr(x, y).statistic,
            "mae": np.mean(np.abs(x - y)),
            "rmse": np.sqrt(np.mean((x - y) ** 2))
        }
    
    fig, ax = plt.subplots(1, 1, figsize=(2.5, 1.2), constrained_layout=True)
    ax.tick_params(length=1, pad=1)
    x_offset = 0
    group_offset = 0.5
    xticks, xticklabels = [], []
    flatten_feature_names = sum(feature_names, [])
    for feature_group_id, feature_group_names in enumerate(feature_names):
        for feature_name in feature_group_names:
            dissimilarity = 1 - compare_consensus(
                grand_consensus_matrix,
                feature_drop_grand_consensus_matrix[flatten_feature_names.index(feature_name)]
            )["pearson"]
            ax.bar(x_offset, dissimilarity, color='black', lw=0.5, width=0.7, alpha=0.9)
            xticks.append(x_offset)
            xticklabels.append(feature_name)
            x_offset += 1
        if feature_group_id < len(feature_names) - 1:
            ax.axvline(x=x_offset+(group_offset - 1)/2, 
                    color='gray', lw=0.5, linestyle='--')
        x_offset += group_offset
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=30, ha='right')
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks([0, 0.1])
    ax.set_ylabel("Dissimilarity", labelpad=1)

    save_path = path.join(get_saving_path(), "spv_clustering", task_id, "overall",
                                    f"feature_contribution_alpha{weight_alpha}_{selected_postfix}.png")
    default_exit_save(fig, save_path=save_path, _transparent=True)
    
    
   


if __name__ == "__main__":
    for test_alpha in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 100.0]:
        test_vec = np.concatenate([random_weight_vec(length=13, alpha=test_alpha, seed=test_seed) 
                                   for test_seed in range(5)])
        print(f"Alpha: {test_alpha}, Vector: {np.sort(test_vec)}"\
                f": 90th: {np.percentile(test_vec, 90):.3f}, 10th: {np.percentile(test_vec, 10):.3f}, Median: {np.median(test_vec):.3f}, SD: {np.std(test_vec):.3f}")
