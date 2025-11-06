import os.path as path
import os
import logging

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path
from kitchen.utils import numpy_kit

logger = logging.getLogger(__name__)


def Visualize_metric_score(all_feature_spaces, metric_name: str):

    import matplotlib.pyplot as plt    
    import seaborn as sns
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    plt.rcParams["font.family"] = "Arial"

    x_offset = 0
    fig, ax = plt.subplots(1, 1, figsize=(3, 2), constrained_layout=True)
    xticks, xtick_labels = [], []
    for feature_space_major_group_name, feature_space_minor_group in all_feature_spaces.items():
        for feature_space_name, (weight_matrix, node_names, cohort_names) in feature_space_minor_group.items():
            if ("PSTH" not in feature_space_name) or ("Waveform" in feature_space_name):
                weight_matrix = numpy_kit.zscore(weight_matrix, axis=0) 
            if metric_name == "silhouette_score":
                value = silhouette_score(weight_matrix, cohort_names)
            elif metric_name == "calinski_harabasz_score":
                value = calinski_harabasz_score(weight_matrix, cohort_names)
            elif metric_name == "davies_bouldin_score":
                value = davies_bouldin_score(weight_matrix, cohort_names)
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


