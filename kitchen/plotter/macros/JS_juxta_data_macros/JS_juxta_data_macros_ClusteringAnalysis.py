
import os.path as path
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import logging

from kitchen.operator.grouping import AdvancedTimeSeries
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)
   
DEFAULT_UMAP_KWS = {'n_components': 2, 'random_state': 42, 'n_neighbors': 9}

def get_putative_labels(weight_tuple):
    

    from umap import UMAP
    from sklearn.cluster import SpectralClustering

    weight_matrix, node_names, cohort_names = weight_tuple
    component_projection = weight_matrix[:, 1:]
    n_components = component_projection.shape[1]

    if n_components == 2:
        n_clusters = 4
    elif n_components == 5:
        n_clusters = 5
    elif n_components == 4:
        n_clusters = 3
    else:
        raise ValueError(f"n_components {n_components} not supported")
    

    normalized_weight_matrix = numpy_kit.zscore(weight_matrix, axis=0)
    embedding = UMAP( **DEFAULT_UMAP_KWS).fit_transform(normalized_weight_matrix)
    target_cluster_id = SpectralClustering(n_clusters=n_clusters, random_state=42,
                                           n_neighbors=DEFAULT_UMAP_KWS['n_neighbors']).fit_predict(embedding)

    cluster_names = [f"cluster {i}" for i in target_cluster_id]   
    for cluster_id in range(n_clusters):
        idx = [i for i, name in enumerate(cluster_names) if name == f"cluster {cluster_id}"]
        print(f"Cluster {cluster_id} has {len(idx)} cells")
        for x in node_names[idx]:
            print(x)
        print("------------------")
    return np.array(cluster_names), (n_clusters, embedding, normalized_weight_matrix)


def Visualize_ClusteringAnalysis(weight_tuple, psth_tuple, feature_space_name: str):
    from scipy.cluster.hierarchy import linkage, dendrogram
    import pandas as pd


    psth_matrix, psth_node_names, _ = psth_tuple

    weight_matrix, node_names, cohort_names = weight_tuple
    component_projection = weight_matrix[:, 1:]
    n_components = component_projection.shape[1]

    shared_node_names = np.intersect1d(psth_node_names, node_names)

    idx_in_both = numpy_kit.reorder_indices(psth_node_names, shared_node_names, _allow_leftover=True)
    psth_matrix = psth_matrix[idx_in_both, :]
    psth_node_names = psth_node_names[idx_in_both]

    idx_in_both = numpy_kit.reorder_indices(node_names, shared_node_names, _allow_leftover=True)
    weight_matrix = weight_matrix[idx_in_both, :]
    cohort_names = cohort_names[idx_in_both]
    node_names = node_names[idx_in_both]

    n_cell, _ = weight_matrix.shape    


    cluster_names, (n_clusters, embedding, normalized_weight_matrix) = get_putative_labels((weight_matrix, node_names, cohort_names))


    BINSIZE = 10/1000
    FEATURE_RANGE = (-1, 1.5)
    bins = np.arange(FEATURE_RANGE[0], FEATURE_RANGE[1] + BINSIZE, BINSIZE)
    bin_centers = (bins[:-1] + bins[1:]) / 2


    def draw_ellipse(points, ax, **kwargs):
        from matplotlib.patches import Ellipse
        position = points.mean(axis=0)
        covariance = np.cov(points, rowvar=False)

        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        ell = Ellipse(xy=position, width=3 * width, height=3 * height, angle=angle, **kwargs)
        
        ax.add_patch(ell)
        return ell
    

    
    """
    Overview figure of clustering composition.
    
    
    """
    from matplotlib.ticker import MaxNLocator
    import matplotlib.patheffects as pe

    plt.rcParams["font.family"] = "Arial"
    fig, axs = plt.subplots(1, 4, figsize=(12, 2.75), constrained_layout=True, width_ratios=[1, 1, 1.5, 1.5])

    for cohort in ("PV_JUX", "SST_JUX", "PYR_JUX", "SST_WC",):
        idx = [i for i, name in enumerate(cohort_names) if name == cohort]
        axs[0].scatter(embedding[idx, 0], embedding[idx, 1],
                       facecolors=COHORT_COLORS[cohort], edgecolors="black", lw=0.5, s=20, alpha=0.9)
        axs[1].scatter(embedding[idx, 0], embedding[idx, 1],
                       facecolors=COHORT_COLORS[cohort], edgecolors="black", lw=0.5, s=20, alpha=0.9)
        
    for cluster_id in range(n_clusters):
        idx = [i for i, name in enumerate(cluster_names) if name == f"cluster {cluster_id}"]
        draw_ellipse(embedding[idx, :], axs[1], edgecolor=CLUSTER_COLORS[f"cluster {cluster_id}"], 
                     facecolor=CLUSTER_COLORS[f"cluster {cluster_id}"], lw=1, ls='--', alpha=0.8, zorder=-10)
        
    
    for ax in axs[:2]:
        ax.spines[['right', 'top']].set_visible(False)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_xticks([])
        ax.set_yticks([])

    def tick_formatter_cell_type(tick_text):
        a, b = tick_text.split("_")
        return fr'$\mathrm{{{a}}}_\mathrm{{{b}}}$'
    def tick_formatter_cluster_type(tick_text):
        return tick_text.split("cluster ")[1]
    
    df = pd.DataFrame({
        "Cohort": cohort_names,
        "Cluster": cluster_names,
    })
    cohort_per_cluster = df.groupby(['Cluster', 'Cohort']).size().unstack(fill_value=0)
    cohort_per_cluster.plot(kind='bar', stacked=True, ax=axs[2], color=COHORT_COLORS, 
                            edgecolor='black', lw=0.5, legend=False)
    axs[2].set_title("Cohort Composition per Cluster")
    axs[2].set_xticklabels(cohort_per_cluster.index, rotation=0)
    axs[2].set_xlabel("Cluster")
    axs[2].set_ylabel("Number of Cells")
    axs[2].spines[['right', 'top']].set_visible(False)
    new_xticks = [tick_formatter_cluster_type(tick_text.get_text()) for tick_text in axs[2].get_xticklabels()]
    axs[2].set_xticklabels(new_xticks)     
    for tick_label in axs[2].get_xticklabels():
        tick_label.set_color(CLUSTER_COLORS[int(tick_label.get_text())])
        tick_label.set_fontweight('bold')
        tick_label.set_path_effects([pe.withStroke(linewidth=0.3, foreground='black')])

    axs[2].yaxis.set_major_locator(MaxNLocator(integer=True))

    cluster_per_cohort = df.groupby(['Cohort', 'Cluster']).size().unstack(fill_value=0)
    cluster_per_cohort.plot(kind='bar', stacked=True, ax=axs[3], color=CLUSTER_COLORS, legend=False,
                            edgecolor='black', lw=0.5)
    axs[3].set_title("Cluster Composition per Cohort")
    axs[3].set_xticklabels(cluster_per_cohort.index, rotation=0)
    axs[3].set_xlabel("Cohort")
    axs[3].set_ylabel("Number of Cells")
    axs[3].spines[['right', 'top']].set_visible(False)
    tick_colors = [COHORT_COLORS[tick_text.get_text()] for tick_text in axs[3].get_xticklabels()]
    new_xticks = [tick_formatter_cell_type(tick_text.get_text()) for tick_text in axs[3].get_xticklabels()]
    axs[3].set_xticklabels(new_xticks) 
    for tick_label, tick_color in zip(axs[3].get_xticklabels(), tick_colors):
        tick_label.set_color(tick_color)
        tick_label.set_fontweight('bold')
        tick_label.set_path_effects([pe.withStroke(linewidth=0.3, foreground='black')])

    axs[3].yaxis.set_major_locator(MaxNLocator(integer=True))

    save_path = path.join(get_saving_path(), "ClusteringAnalysis", f"ClusteringAnalysis_{feature_space_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)




    
    """
    Summary Heatmap with hierarchical dendrogram
    
    Weight shown aside
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from matplotlib.colors import SymLogNorm
    import seaborn as sns
    from scipy.cluster import hierarchy

    fig, axs = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True, width_ratios=[0.3, 1, 0.2])

    linked_weight = linkage(normalized_weight_matrix, method='ward', metric='euclidean')

    cluster_leaf_map = {i: {i} for i in range(n_cell)}
    for i, row in enumerate(linked_weight):
        cluster_id = n_cell + i
        child1, child2 = int(row[0]), int(row[1])
        cluster_leaf_map[cluster_id] = cluster_leaf_map[child1].union(cluster_leaf_map[child2])

    def link_cluster_func(k):
        # cluster_id = k + n_samples
        leaf_indices = cluster_leaf_map[k]
        leaf_clusters = {cluster_names[i] for i in leaf_indices}
        
        if len(leaf_clusters) == 1:
            cluster_name = leaf_clusters.pop()
            return CLUSTER_COLORS[cluster_name]
        return 'gray'
    
    dendro_info = dendrogram(linked_weight, ax=axs[0], orientation='left',
                             link_color_func = link_cluster_func)
    original_clusters = hierarchy.fcluster(linked_weight, t=6, criterion='maxclust')
    plotted_clusters = original_clusters[dendro_info['leaves']]
    boundary_indices = []
    for i in range(len(plotted_clusters) - 1):
        if plotted_clusters[i] != plotted_clusters[i+1]:
            boundary_indices.append(i)
    for y_pos in boundary_indices:
        axs[1].axhline(y=y_pos * 10 + 10, color='white', ls='--', linewidth=1)
        axs[2].axhline(y=y_pos * 10 + 10, color='gray', ls=(5, (10, 3)), linewidth=1)

    ax = axs[0]
    ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='y', which='both', length=0, labelright=False, )
    ax.set_xticks([])
    ax.set_ylabel('Hierarchical Dendrogram')
    ax.set_xlabel('Ward Distance')        

    # set y ticks to cohort names
    tick_positions = np.arange(len(cohort_names) + 1) * 10 + 5
    face_colors = [CLUSTER_COLORS[cluster_names[i]] for i in dendro_info['leaves']]
    for tick_p, tick_c in zip(tick_positions, face_colors):
        rect_unclipped = mpatches.Rectangle(
            (0, tick_p - 5), -2, 10,
            linewidth=0, facecolor=tick_c, 
            clip_on=False  
        )
        ax.add_patch(rect_unclipped)
        
    sorted_idx = dendro_info['leaves']
    sns.heatmap(psth_matrix[sorted_idx, :].repeat(10, axis=0), 
                ax=axs[1], cmap='viridis', norm=SymLogNorm(linthresh=10., vmin=0, vmax=None),
                )
    ax = axs[1]
    ax.sharey(axs[0])
    ax.set_title('PSTH')
    ax.set_ylabel('')
    ax.set_xlabel("Time [s]")
    plot_tick_indices = np.searchsorted(bin_centers, (0, 0.5))
    ax.set_xticks(plot_tick_indices, [0, 0.5], rotation=0, )

    sns.heatmap(normalized_weight_matrix[sorted_idx, :].repeat(10, axis=0), 
                ax=axs[2], cmap='bwr', center=0.,
                )
    ax = axs[2]
    ax.sharey(axs[0])
    ax.set_ylabel('')
    ax.set_xticks(np.arange(n_components + 1) + 0.5, ["Baseline FR",] + [f"PC {i+1}" for i in range(n_components)], 
                    rotation=45, ha='right', fontsize=8)
    ax.set_title('Weights')
    
    # set y ticks to cohort names
    for ax, size in zip(axs[1:], (50, n_components + 1)):
        # set y ticks to cohort names
        tick_positions = np.arange(len(cohort_names) + 1) * 10 + 5
        face_colors = [COHORT_COLORS[cohort_names[i]] for i in sorted_idx]
        ax.tick_params(axis='y', which='both', length=0, labelleft=False)
        for tick_p, tick_c in zip(tick_positions, face_colors):
            rect_unclipped = mpatches.Rectangle(
                (0, tick_p - 5), -2 * size / 8, 10,
                linewidth=0, facecolor=tick_c, 
                clip_on=False  
            )
            ax.add_patch(rect_unclipped)

    save_path = path.join(get_saving_path(), "ClusteringAnalysis", f"Heatmap_{feature_space_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)






    """
    Figure :
    PSTH
    Heatmap
    For each single putative clusters
    """
    for cluster_id in range(n_clusters):
        for cohort_group, cohort_list in zip(("SST", "PV", "PYR", "ALL"), 
                                             (["SST_JUX", "SST_WC"], ["PV_JUX",], ["PYR_JUX",], ["SST_JUX", "SST_WC", "PV_JUX", "PYR_JUX"])):
            idx_in_cluster = [i for i, name in enumerate(cluster_names) if name == f"cluster {cluster_id}"]
            idx_in_cohort = [i for i, name in enumerate(cohort_names) if name in cohort_list]
            idx_in_both = list(set(idx_in_cluster).intersection(idx_in_cohort))
            if len(idx_in_both) > 0:
                fig, axs = plt.subplots(2, 1, figsize=(3, 3.5), constrained_layout=True, height_ratios=[1, 1.5])

                ax = axs[0]
                group_psth = AdvancedTimeSeries(
                    t=bin_centers, v=psth_matrix[idx_in_both, :].mean(axis=0), 
                    variance=psth_matrix[idx_in_both, :].std(axis=0), 
                    raw_array=psth_matrix[idx_in_both, :])
                oreo_plot(ax, group_psth, 0, 1,  
                          {"color": CLUSTER_COLORS[f"cluster {cluster_id}"], "lw": 2, "alpha": 1},
                          style_dicts.FILL_BETWEEN_STYLE | {"alpha": 0.4, })
                ax.axvspan(0, 0.5, alpha=0.5, color=color_scheme.PUFF_COLOR, lw=0, zorder=-10)
                ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
                ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5, zorder=-10)
                ax.set_xlim(-0.5, 1)
                ax.set_ylim(-10, 150)
                ax.set_xticks([])
                ax.set_yticks([])

                ax = axs[1]
                subset_psth_matrix = psth_matrix[idx_in_both, :]
                mask = (bin_centers > 0) & (bin_centers < 0.5)
                display_mask = (bin_centers >= -0.5) & (bin_centers <= 1)
                sorted_order = np.argsort(np.mean(subset_psth_matrix[:, mask], axis=1))
                sns.heatmap(subset_psth_matrix[sorted_order][:, display_mask],  
                            ax=ax, cbar=True, 
                            cmap='YlOrBr', norm=SymLogNorm(linthresh=10., vmin=0, vmax=150),)
                
                desired_ticks = [0, 0.5]
                tick_positions = [np.argmin(np.abs(bin_centers[display_mask] - tick)) for tick in desired_ticks]
                ax.set_xticks(tick_positions, desired_ticks, rotation=0)  

                ax.invert_yaxis()
                ax.set_xlabel("Time [s]")
                ax.set_yticks([0.5, len(idx_in_both)-0.5], 
                            ["1", f"{len(idx_in_both)}"], rotation=0)
                ax.spines[['right', 'top', 'left', 'bottom']].set_visible(True)

                
                save_path = path.join(get_saving_path(), "ClusteringAnalysis", 
                                      f"SubCluster_{feature_space_name}_{cohort_group}_cluster{cluster_id}.png")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, dpi=500, transparent=True)
                plt.close(fig)
                logger.info("Plot saved to " + save_path)
