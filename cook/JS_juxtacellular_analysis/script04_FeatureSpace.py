import logging

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import Visualize_Waveform, VisualizeExample_Physiology_Fingerprint, get_weight_tuple_Decomposition_Weights, get_weight_tuple_PSTH, get_weight_tuple_Physiology_Fingerprint, get_weight_tuple_Waveform, simple_Landscope_feature_space, simple_pairwise_distance_distribution
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import COHORT_COLORS


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
    # VisualizeExample_Physiology_Fingerprint()
    physiology_weight_matrix, node_name, cohort_name = get_weight_tuple_Physiology_Fingerprint()

    simple_Landscope_feature_space(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint")
    simple_pairwise_distance_distribution(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint")

    physiology_weight_matrix, node_name, cohort_name = get_weight_tuple_Physiology_Fingerprint(with_spont_FR=False)
    simple_Landscope_feature_space(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint_noSpont")
    simple_pairwise_distance_distribution(physiology_weight_matrix, cohort_name, "Physiology_Fingerprint_noSpont")

def analyze_psth():
    for variant in ["raw", "zscore", "log-scale"]:
        psth_weight_matrix, node_name, cohort_name = get_weight_tuple_PSTH(variant)
        simple_Landscope_feature_space(psth_weight_matrix, cohort_name, f"PSTH_{variant}", _normalize_all_dimension=False)
        simple_pairwise_distance_distribution(psth_weight_matrix, cohort_name, f"PSTH_{variant}", _normalize_all_dimension=False)

def analyze_waveform():
    Visualize_Waveform()
    waveform_weight_matrix, node_name, cohort_name = get_weight_tuple_Waveform()
    simple_Landscope_feature_space(waveform_weight_matrix, cohort_name, "Waveform", _normalize_all_dimension=False)
    simple_pairwise_distance_distribution(waveform_weight_matrix, cohort_name, "Waveform", _normalize_all_dimension=False)


def analyze_decomposition_weights():
    for variant in ["zscore", "raw", ]:
        for decomposition_method in ["SVD",]:
            for n_components in [2, 3, 4, 5,]:
                for append_baseline in [True, False]:
                    decomposition_result = get_weight_tuple_Decomposition_Weights(
                        variant, decomposition_method, n_components, append_baseline, _visualize=True) 
                    if decomposition_result is None:
                        continue
                    component_projection, node_name, cohort_name = decomposition_result
                    simple_Landscope_feature_space(
                        component_projection, cohort_name, 
                        f"Decomposition_{variant}_{decomposition_method}_{n_components}_{append_baseline}")
                    # simple_pairwise_distance_distribution(component_projection, cohort_name, f"Decomposition_{variant}_{decomposition_method}_{n_components}_{append_baseline}", _normalize_all_dimension=False)
                    
def main():
    # analyze_physiology_fingerprint()
    # analyze_psth()
    # analyze_waveform()
    analyze_decomposition_weights()

if __name__ == "__main__":
    main()