import os.path as path
import os
import logging

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)





def Visualize_metric_score(all_feature_spaces, label_dict, metric_name: str):

    import matplotlib.pyplot as plt   
    import numpy as np
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    plt.rcParams["font.family"] = "Arial"

    x_offset = 0
    fig, axs = plt.subplots(len(label_dict), 1, figsize=(3, 5), constrained_layout=True)
    for ax, (label_name, (label_node_names, node_labels)) in zip(axs, label_dict.items()):
        ax.set_title(label_name)
        xticks, xtick_labels = [], []
        for feature_space_major_group_name, feature_space_minor_group in all_feature_spaces.items():
            for feature_space_name, (weight_matrix, node_names, cohort_names) in feature_space_minor_group.items():
                shared_node_names = np.intersect1d(label_node_names, node_names)
                sub_weight_matrix = weight_matrix[numpy_kit.reorder_indices(node_names, shared_node_names, _allow_leftover=True), :]
                sub_node_labels = node_labels[numpy_kit.reorder_indices(label_node_names, shared_node_names, _allow_leftover=True)]
                
                
                if ("PSTH" not in feature_space_name) or ("Waveform" in feature_space_name):
                    sub_weight_matrix = numpy_kit.zscore(sub_weight_matrix, axis=0) 


                if metric_name == "silhouette_score":
                    value = silhouette_score(sub_weight_matrix, sub_node_labels)
                elif metric_name == "calinski_harabasz_score":
                    value = calinski_harabasz_score(sub_weight_matrix, sub_node_labels)
                elif metric_name == "davies_bouldin_score":
                    value = davies_bouldin_score(sub_weight_matrix, sub_node_labels)
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
                    
                ax.bar(x_offset, value, width=0.5)
                xticks.append(x_offset)
                xtick_labels.append(feature_space_name)
                x_offset += 1
            x_offset += 0.5

        ax.set_xticks(xticks, xtick_labels, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.spines[['right', 'top']].set_visible(False)

    plt.show()
    plt.close(fig)
    return
    save_path = path.join(get_saving_path(), "FeatureSpace", "SilhouetteScore_bars.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=500, transparent=True)
    plt.close(fig)
    logger.info("Plot saved to " + save_path)



def Visualize_within_group_correlation(label_dict, list_of_feature_space_tuple):

    import matplotlib.pyplot as plt   
    import numpy as np
    from scipy.spatial.distance import pdist

    plt.rcParams["font.family"] = "Arial"

    fig, axs = plt.subplots(3, len(list_of_feature_space_tuple), constrained_layout=True)
    for axr, (feature_space_name, (weight_matrix, node_names, cohort_names)) in zip(axs.T, list_of_feature_space_tuple.items()):
        axr[0].set_title(feature_space_name)
        for clustering_name, (clustering_node_names, clustering_node_labels) in label_dict.items():
            shared_node_names = np.intersect1d(node_names, clustering_node_names)
            sub_weight_matrix = weight_matrix[numpy_kit.reorder_indices(node_names, shared_node_names, _allow_leftover=True), :]
            sub_node_labels = clustering_node_labels[numpy_kit.reorder_indices(clustering_node_names, shared_node_names, _allow_leftover=True)]

            within_group_correlation = []
            for group_id in np.unique(sub_node_labels):
                group_idx = sub_node_labels == group_id
                if np.sum(group_idx) > 1:
                    within_group_correlation.append(pdist(sub_weight_matrix[group_idx, :], metric="correlation"))
            within_group_correlation = np.mean(within_group_correlation)
            axr[0].bar(clustering_name, within_group_correlation, width=0.5)
        axr[0].set_ylabel("Within Group Correlation")
        
        for ax in axr:
            ax.spines[['right', 'top']].set_visible(False)
    
    plt.show()
    plt.close(fig)