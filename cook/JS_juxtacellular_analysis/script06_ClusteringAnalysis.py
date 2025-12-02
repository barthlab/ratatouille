import logging

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_ClusteringAnalysis import Visualize_ClusteringAnalysis
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import Visualize_Waveform_With_CLUSTER_LABEL, get_weight_tuple_Decomposition_Weights, get_weight_tuple_PSTH, get_weight_tuple_Physiology_Fingerprint, get_weight_tuple_Waveform, simple_Landscope_feature_space



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)




def main():
    PSTH_tuple = get_weight_tuple_PSTH("raw", 10/1000)

    # phy_fingerprint = get_weight_tuple_Physiology_Fingerprint()
    # Visualize_ClusteringAnalysis(phy_fingerprint, PSTH_tuple, feature_space_name="Physiology_Fingerprint")

    # svd_2_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", 2, True, _visualize=True)
    # Visualize_ClusteringAnalysis(svd_2_tuple, PSTH_tuple, feature_space_name="SVD2")
    
    svd_5_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", 5, True, _visualize=True)
    Visualize_ClusteringAnalysis(svd_5_tuple, PSTH_tuple, feature_space_name="SVD5")
    # Visualize_Waveform_With_CLUSTER_LABEL(svd_5_tuple, PSTH_tuple, feature_space_name="SVD5")
    

if __name__ == "__main__":
    main()