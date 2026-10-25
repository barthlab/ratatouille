import logging

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import Visualize_ISI_distribution, Visualize_Waveform, VisualizeACG, get_all_physiology_features, get_combined_variability_physiology, get_weight_tuple_Decomposition_Weights, get_weight_tuple_PSTH, get_weight_tuple_Physiology_Fingerprint, get_weight_tuple_Waveform, simple_Landscope_feature_space, simple_pairwise_correlation_matrix, simple_pairwise_distance_distribution, simple_feature_correlation_matrix
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace_EarlySustain import VisualizeExample_Physiology_Fingerprint_EarlySpikeDistribution, VisualizeExample_Physiology_Fingerprint_EarlySpikeIndicatedTrailState, VisualizeExample_Physiology_Fingerprint_EarlyVsSustained, VisualizeExample_Physiology_Fingerprint_Vertical
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_StabilityTestSVD import stability_test_SVD, variability_visualize
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_StateDependency import Visualize_Variability_of_ephysiology
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_TemporalPattern import temporal_pattern_Early_spike, temporal_pattern_SVD, temporal_pattern_early_vs_sustained


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


def analyze_physiology_fingerprint():
    # VisualizeACG()
    # Visualize_ISI_distribution()
    # VisualizeExample_Physiology_Fingerprint_Vertical(bon_ferroni_adj=5)
    # VisualizeExample_Physiology_Fingerprint_EarlyVsSustained()
    # VisualizeExample_Physiology_Fingerprint_EarlySpikeDistribution()
    # VisualizeExample_Physiology_Fingerprint_EarlySpikeIndicatedTrailState()
    # exit()
    physiology_weight_matrix, node_name, cohort_name = get_weight_tuple_Physiology_Fingerprint(_drop_ACG_rise_time=False)

    # simple_Landscope_feature_space(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint")
    simple_pairwise_distance_distribution(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint", annotate=True)
    # simple_pairwise_correlation_matrix(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint")
    # simple_feature_correlation_matrix(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint")


def analyze_physiology_fingerprint_state_dependent():
    var_physiology_list = get_combined_variability_physiology()
    
    psth_weight_matrix, node_name, cohort_name = get_weight_tuple_PSTH("zscore", binsize=10/1000)

    selected_cells = [physio_dict for physio_dict in var_physiology_list if physio_dict.get("node_name", "") in node_name]
    print(len(selected_cells))
    n_sst, n_pv, n_pyr = 0, 0, 0
    for feature_dict in selected_cells:
        cohort_name = feature_dict.get("cohort_name", "")
        if cohort_name == "SST_JUX":
            n_sst += 1
        elif cohort_name == "PV_JUX":
            n_pv += 1
        elif cohort_name == "PYR_JUX":
            n_pyr += 1
    print(f"n_sst: {n_sst}, n_pv: {n_pv}, n_pyr: {n_pyr}")

    Visualize_Variability_of_ephysiology(selected_cells)


def analyze_psth():
    for variant in ["raw", "zscore", "log-scale"]:
        psth_weight_matrix, node_name, cohort_name = get_weight_tuple_PSTH(variant)
        simple_Landscope_feature_space(psth_weight_matrix, cohort_name, f"PSTH_{variant}", _normalize_all_dimension=False)
        simple_pairwise_distance_distribution(psth_weight_matrix, cohort_name, f"PSTH_{variant}", _normalize_all_dimension=False)

def analyze_waveform():
    Visualize_Waveform()
    # waveform_weight_matrix, node_name, cohort_name = get_weight_tuple_Waveform()
    # simple_Landscope_feature_space(waveform_weight_matrix, cohort_name, "Waveform", _normalize_all_dimension=False)
    # simple_pairwise_distance_distribution(waveform_weight_matrix, cohort_name, "Waveform", _normalize_all_dimension=False)


def analyze_decomposition_weights():

    # this is for clustering based on PSTH
    # for variant in ["zscore", "raw", ]:
    for variant in ["zscore", ]:
        for decomposition_method in ["SVD",]:
            # for n_components in [2, 3, 4, 5, 6, 10, 20]:
            for n_components in [5, ]:
                # for append_baseline in [True, False]:
                # for append_baseline in [True, ]:
                for append_baseline in [False, ]:
                    decomposition_result = get_weight_tuple_Decomposition_Weights(
                        variant, decomposition_method, n_components, append_baseline, 
                        _visualize=True, BINSIZE=10/1000) 
                    # if decomposition_result is None:
                    #     continue
                    component_projection, node_name, cohort_name = decomposition_result
                    # simple_Landscope_feature_space(
                    #     component_projection, cohort_name, 
                    #     f"Decomposition_{variant}_{decomposition_method}_{n_components}_{append_baseline}")
                    
                    simple_pairwise_distance_distribution(component_projection, cohort_name, 
                                                          f"Decomposition_{variant}_{decomposition_method}_{n_components}_{append_baseline}",
                                                           annotate=True)
                    # simple_pairwise_correlation_matrix(component_projection, cohort_name, 
                    #                                    f"Decomposition_{variant}_{decomposition_method}_{n_components}_{append_baseline}",)
                    # simple_feature_correlation_matrix(component_projection, cohort_name, 
                    #                                   f"Decomposition_{variant}_{decomposition_method}_{n_components}_{append_baseline}")

def decomposition_stability_test():   
    variability_visualize('raw', binsize=5/1000)
    # stability_test_SVD('zscore', binsize=10/1000)


def temporal_pattern():
    # temporal_pattern_SVD('raw', binsize=5/1000)
    # temporal_pattern_Early_spike()
    temporal_pattern_early_vs_sustained()

def main():
    # analyze_physiology_fingerprint_state_dependent()
    analyze_physiology_fingerprint()
    # analyze_psth()
    # analyze_waveform()
    analyze_decomposition_weights()
    # decomposition_stability_test()
    # temporal_pattern()


if __name__ == "__main__":
    main()