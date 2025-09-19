import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.select_trial_rules import PREDEFINED_PASSIVEPUFF_RULES
from kitchen.plotter.macros.basic_macros import session_overview, single_node_trial_avg_default, single_node_trial_parallel_default, subtree_summary_trial_avg_default
from kitchen.plotter.macros.jux_data_macros import dendrogram_plot, multiple_dataset_dendrogram_plot, raster_plot, single_cell_session_parallel_view_jux, subtree_summary_trial_avg_jux
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Mice, Session



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    sp_prefix = "heatmap_all_combined"
    dataset_jux_sst = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="SST_JUX", 
                                   recipe="default_ephys", name="SST_JUX")
    dataset_wc_sst = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="SST_WC", 
                                   recipe="default_ephys", name="SST_WC")
    dataset_jux_pv = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="PV_JUX", 
                                   recipe="default_ephys", name="PV_JUX")
    dataset_jux_pyr = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id="PYR_JUX", 
                                   recipe="default_ephys", name="PYR_JUX")
    all_datasets = [dataset_jux_sst, dataset_wc_sst, dataset_jux_pv, dataset_jux_pyr]
    
    for linkage_method in ("single", "complete", "average", "weighted"):
        for linkage_metric in ("cosine", "euclidean", "correlation"):
            multiple_dataset_dendrogram_plot(all_datasets, prefix_keyword="Short_" + linkage_metric + "_" + linkage_method + "_" + sp_prefix, 
                            LINKAGE_METHOD=linkage_method, LINKAGE_METRIC=linkage_metric,
                            FEATURE_RANGE = (-0.25, 0.75))
            multiple_dataset_dendrogram_plot(all_datasets, prefix_keyword="Medium_" + linkage_metric + "_" + linkage_method + "_" + sp_prefix, 
                            LINKAGE_METHOD=linkage_method, LINKAGE_METRIC=linkage_metric,
                            FEATURE_RANGE = (-1, 1.5))
            multiple_dataset_dendrogram_plot(all_datasets, prefix_keyword="Long_" + linkage_metric + "_" + linkage_method + "_" + sp_prefix, 
                            LINKAGE_METHOD=linkage_method, LINKAGE_METRIC=linkage_metric,
                            FEATURE_RANGE = (-2, 2.5))
    exit()
    sp_prefix = "heatmap"
    for dataset_name in ("PYR_JUX", "PV_JUX", "SST_JUX"):

        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys")
        for linkage_method in ("single", "complete", "average", "weighted"):
            for linkage_metric in ("cosine", "euclidean", "correlation"):
                dendrogram_plot(dataset, prefix_keyword="Short_" + linkage_metric + "_" + linkage_method + "_" + sp_prefix, 
                                LINKAGE_METHOD=linkage_method, LINKAGE_METRIC=linkage_metric,
                                FEATURE_RANGE = (-0.25, 0.75))
                dendrogram_plot(dataset, prefix_keyword="Medium_" + linkage_metric + "_" + linkage_method + "_" + sp_prefix, 
                                LINKAGE_METHOD=linkage_method, LINKAGE_METRIC=linkage_metric,
                                FEATURE_RANGE = (-1, 1.5))
                dendrogram_plot(dataset, prefix_keyword="Long_" + linkage_metric + "_" + linkage_method + "_" + sp_prefix, 
                                LINKAGE_METHOD=linkage_method, LINKAGE_METRIC=linkage_metric,
                                FEATURE_RANGE = (-2, 2.5))

if __name__ == "__main__":
    main()