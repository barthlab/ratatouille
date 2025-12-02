
import os.path as path
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import logging

from kitchen.operator.grouping import AdvancedTimeSeries
from kitchen.plotter import color_scheme, style_dicts
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path
from kitchen.plotter.utils.fill_plot import oreo_plot
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)
   
DEFAULT_UMAP_KWS = {'n_components': 2, 'random_state': 42, 'n_neighbors': 9}

def tick_formatter_cell_type(tick_text):
    a, b = tick_text.split("_")
    return fr'$\mathrm{{{a}}}_\mathrm{{{b}}}$'
def tick_formatter_cluster_type(tick_text):
    return tick_text.split("cluster ")[1]
    


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
    for x, y,z in zip(cohort_names, cluster_names, node_names):
        print(x, y, z)
    exit()
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

    df = pd.DataFrame({
        "Cohort": cohort_names,
        "Cluster": cluster_names,
    })
    cohort_per_cluster = df.groupby(['Cluster', 'Cohort']).size().unstack(fill_value=0)
    cohort_per_cluster = cohort_per_cluster.reindex(columns=["SST_JUX", "SST_WC", "PV_JUX", "PYR_JUX"], fill_value=0)
    cohort_per_cluster = cohort_per_cluster.reindex(index=[f"cluster {i}" for i in range(n_clusters)], fill_value=0)
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
    cluster_per_cohort = cluster_per_cohort.reindex(index=["SST_JUX", "SST_WC", "PV_JUX", "PYR_JUX"], fill_value=0)
    cluster_per_cohort = cluster_per_cohort.reindex(columns=[f"cluster {i}" for i in range(n_clusters)], fill_value=0)
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
    Overview figure of clustering composition.
    """

    Visualize_data_composition(cohort_per_cluster, cluster_per_cohort, feature_space_name)




    
    """
    Summary Heatmap with hierarchical dendrogram
    
    Weight shown aside
    """
    from scipy.cluster.hierarchy import linkage, dendrogram
    from matplotlib.colors import SymLogNorm
    import seaborn as sns
    from scipy.cluster import hierarchy

    fig, axs = plt.subplots(1, 3, figsize=(6, 4), constrained_layout=True, width_ratios=[0.3, 1, 0.2])

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
        axs[1].axhline(y=y_pos * 10 + 10, color='black', ls='--', linewidth=1)
        axs[2].axhline(y=y_pos * 10 + 10, color='black', ls=(5, (10, 3)), linewidth=1)

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
                ax=axs[1], cmap='YlOrBr', norm=SymLogNorm(linthresh=10., vmin=0, vmax=150),
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
    ax.set_xticks(np.arange(n_components + 1) + 0.5, ["Base FR",] + [f"PC {i+1}" for i in range(n_components)], 
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





def Visualize_data_composition(
        cohort_per_cluster: pd.DataFrame, 
        cluster_per_cohort: pd.DataFrame, 
        feature_space_name: str,
):
    n_clusters = len(cohort_per_cluster.index)
    n_cohorts = len(cluster_per_cohort.index)
    max_cols = max(n_clusters, n_cohorts)

    fig1, axs1 = plt.subplots(2, max_cols, figsize=(max_cols * 3, 2*3,), constrained_layout=True)

    # Row 1: Cohort Composition per Cluster
    for i, cluster_id in enumerate(cohort_per_cluster.index):
        data = cohort_per_cluster.loc[cluster_id]
        ax = axs1[0, i]
        
        # Filter out zero values to avoid clutter
        data = data[data > 0]
        
        if data.empty:
            continue
        def func(pct):
            absolute = int(round(pct/100.*data.sum()))
            return f"{pct:.1f}%\n({absolute})"
        pie_colors = [COHORT_COLORS[cohort_name] for cohort_name in data.index]
        ax.pie(data, 
            # labels=data.index, 
            colors=pie_colors, 
            autopct=func,
            shadow=False, 
            startangle=90, 
            pctdistance=0.8,
            wedgeprops={'edgecolor': 'black', 'lw': 1})
        
        centre_circle = plt.Circle((0,0),0.6,fc="white", ec="black", lw=1)
        ax.add_artist(centre_circle)
        ax.text(0, 0, cluster_id, ha='center', va='center', fontsize=25, 
                c=CLUSTER_COLORS[cluster_id], )
        
        # ax.set_title(f"Cluster {cluster_id}")
        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.

    # Row 2: Cluster Composition per Cohort
    for i, cohort_name in enumerate(cluster_per_cohort.index):
        data = cluster_per_cohort.loc[cohort_name]
        ax = axs1[1, i]

        # Filter out zero values
        data = data[data > 0]
        if data.empty:
            continue
        def func(pct):
            absolute = int(round(pct/100.*data.sum()))
            return f"{pct:.1f}%\n({absolute})"

        pie_colors = [CLUSTER_COLORS[cluster_id] for cluster_id in data.index]
        ax.pie(data, 
            # labels=data.index, 
            colors=pie_colors, 
            autopct=func,
            shadow=False, 
            startangle=90,
            pctdistance=0.8,
            wedgeprops={'edgecolor': 'black', 'lw': 1})
        
        # Donut chart
        centre_circle = plt.Circle((0,0),0.6,fc="white", ec="black", lw=1)
        ax.add_artist(centre_circle)
        ax.text(0, 0, tick_formatter_cell_type(cohort_name), ha='center', va='center', fontsize=25, 
                c=COHORT_COLORS[cohort_name], fontweight='bold')
        # ax.set_title(f"{cohort_name}")
        ax.axis('equal')

    # Hide unused subplots
    for i in range(n_clusters, max_cols):
        axs1[0, i].axis('off')
    for i in range(n_cohorts, max_cols):
        axs1[1, i].axis('off')

    save_path = path.join(get_saving_path(), "ClusteringAnalysis", 
                                        f"Composition_PiChart_{feature_space_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig1.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig1)
    logger.info("Plot saved to " + save_path)





    # === 2. Heatmap (Confusion Matrix) ===
    print("Generating Plot 2: Heatmap...")
    import seaborn as sns
    import matplotlib.patheffects as pe

    cluster_per_cohort_pct = cluster_per_cohort.div(cluster_per_cohort.sum(axis=1), axis=0) * 100
    cluster_per_cohort_pct = cluster_per_cohort_pct.fillna(0) # handle cohorts with 0 cells

    fig2, ax2 = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)

    annot_labels = cluster_per_cohort_pct.applymap(lambda x: f'{x:.0f}%' if x > 0 else '')  # type: ignore
    sns.heatmap(cluster_per_cohort, 
                annot=annot_labels, 
                fmt='s',           
                cmap='GnBu',      
                linewidths=0.5,
                linecolor='black',
                ax=ax2,
                square=True,
                cbar_kws={'label': 'Number of Cells'},
                annot_kws={"size": 8})

    # Apply custom tick formatting
    # Y-axis (Clusters)
    cluster_labels_orig = [tick.get_text() for tick in ax2.get_xticklabels()]
    cluster_labels_new = [tick_formatter_cluster_type(l) for l in cluster_labels_orig]
    ax2.set_xticklabels(cluster_labels_new, rotation=0)

    for tick_label in ax2.get_xticklabels():
        cluster_int = int(tick_label.get_text()) 
        tick_label.set_color(CLUSTER_COLORS[cluster_int])
        tick_label.set_fontweight('bold')
        # tick_label.set_path_effects([pe.withStroke(linewidth=0.3, foreground='black')])        

    # X-axis (Cohorts)
    cohort_labels_orig = [tick.get_text() for tick in ax2.get_yticklabels()]
    tick_colors = [COHORT_COLORS[label] for label in cohort_labels_orig]
    cohort_labels_new = [tick_formatter_cell_type(label) for label in cohort_labels_orig]
    ax2.set_yticklabels(cohort_labels_new, rotation=0)

    for tick_label, tick_color in zip(ax2.get_yticklabels(), tick_colors):
        tick_label.set_color(tick_color)
        tick_label.set_fontweight('bold')
        # tick_label.set_path_effects([pe.withStroke(linewidth=0.3, foreground='black')])

    ax2.set_ylabel("Cohort", )
    ax2.set_xlabel("Cluster",)
    save_path = path.join(get_saving_path(), "ClusteringAnalysis", 
                                            f"Composition_ConfusionMatrix_{feature_space_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig2.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig2)
    logger.info("Plot saved to " + save_path)


    # === 3. Percentage Stacked Bar Graphs ===
    print("Generating Plot 3: Percentage Bar Charts...")

    # Normalize the data
    # axis=0: divide each column
    # axis=1: divide each row
    cohort_per_cluster_pct = cohort_per_cluster.div(cohort_per_cluster.sum(axis=1), axis=0) * 100
    cluster_per_cohort_pct = cluster_per_cohort.div(cluster_per_cohort.sum(axis=1), axis=0) * 100

    fig3, axs3 = plt.subplots(1, 2, figsize=(3.5 * 2, 2.5), constrained_layout=True)

    # Plot 1: Cohort Composition per Cluster (Percentage)
    plot_colors_1 = [COHORT_COLORS[c] for c in cohort_per_cluster_pct.columns]
    cohort_per_cluster_pct.plot(kind='bar', 
                                stacked=True, 
                                ax=axs3[0], 
                                color=plot_colors_1,
                                edgecolor='black', 
                                legend=False,
                                lw=0.5)

    axs3[0].set_xlabel("Cluster")
    axs3[0].set_ylabel("Percentage (%)")
    axs3[0].spines[['right', 'top']].set_visible(False)
    axs3[0].set_ylim(0, 100) # Ensure Y-axis is 0-100

    # Apply cluster tick formatting
    new_xticks = [tick_formatter_cluster_type(tick.get_text()) for tick in axs3[0].get_xticklabels()]
    axs3[0].set_xticklabels(new_xticks, rotation=0)
    for tick_label in axs3[0].get_xticklabels():
        cluster_int = int(tick_label.get_text())
        tick_label.set_color(CLUSTER_COLORS[cluster_int])
        tick_label.set_fontweight('bold')
        # tick_label.set_path_effects([pe.withStroke(linewidth=0.3, foreground='black')])

    # Plot 2: Cluster Composition per Cohort (Percentage)
    plot_colors_2 = [CLUSTER_COLORS[c] for c in cluster_per_cohort_pct.columns]
    cluster_per_cohort_pct.plot(kind='bar', 
                                stacked=True, 
                                ax=axs3[1], 
                                color=plot_colors_2,
                                edgecolor='black', 
                                legend=False,
                                lw=0.5)

    axs3[1].set_xlabel("Cohort")
    axs3[1].set_ylabel("Percentage (%)")
    axs3[1].spines[['right', 'top']].set_visible(False)
    axs3[1].set_ylim(0, 100)

    # Apply cohort tick formatting
    tick_colors = [COHORT_COLORS[tick.get_text()] for tick in axs3[1].get_xticklabels()]
    new_xticks = [tick_formatter_cell_type(tick.get_text()) for tick in axs3[1].get_xticklabels()]
    axs3[1].set_xticklabels(new_xticks, rotation=0)
    for tick_label, tick_color in zip(axs3[1].get_xticklabels(), tick_colors):
        tick_label.set_color(tick_color)
        tick_label.set_fontweight('bold')
        # tick_label.set_path_effects([pe.withStroke(linewidth=0.3, foreground='black')])

    save_path = path.join(get_saving_path(), "ClusteringAnalysis", 
                                            f"Composition_Percentage_{feature_space_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig3.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig3)
    logger.info("Plot saved to " + save_path)

