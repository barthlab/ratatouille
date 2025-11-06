import logging

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_weight_tuple_Decomposition_Weights, get_weight_tuple_PSTH, get_weight_tuple_Physiology_Fingerprint, get_weight_tuple_Waveform
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpaceComparison import Visualize_metric_score


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)


def prepare_all_feature_spaces():
    all_feature_spaces = {
        "Singular Vector Space (w/o Baseline FR)":
        {
            "SVD_2_woBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 2, False),
            "SVD_3_woBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 3, False),
            "SVD_4_woBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 4, False),
            "SVD_5_woBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 5, False),
        },
        "Singular Vector Space (w/ Baseline FR)":
        {
            "SVD_2_wBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 2, True),
            "SVD_3_wBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 3, True),
            "SVD_4_wBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 4, True),
            "SVD_5_wBaseline": get_weight_tuple_Decomposition_Weights("zscore", "SVD", 5, True),
        },
        "Physiology Fingerprint": 
        {
            "Physiology_Fingerprint": get_weight_tuple_Physiology_Fingerprint(),
            "Physiology_Fingerprint_noSpont": get_weight_tuple_Physiology_Fingerprint(with_spont_FR=False),
        },
        "Waveform": 
        {
            "Waveform": get_weight_tuple_Waveform(),
        },
        "PSTH": 
        {
            "PSTH raw": get_weight_tuple_PSTH("raw", 10/1000),
            # "PSTH raw 20ms": get_weight_tuple_PSTH("raw", 20/1000),
            "PSTH zscore": get_weight_tuple_PSTH("zscore", 10/1000),
            # "PSTH zscore 20ms": get_weight_tuple_PSTH("zscore", 20/1000),
        },
    }
    return all_feature_spaces


def main():
    all_feature_spaces = prepare_all_feature_spaces()
    Visualize_metric_score(all_feature_spaces, "silhouette_score")
    Visualize_metric_score(all_feature_spaces, "calinski_harabasz_score")
    Visualize_metric_score(all_feature_spaces, "davies_bouldin_score")

if __name__ == "__main__":
    main()