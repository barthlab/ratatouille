import logging

from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_ConvolutionAnalysis import Visualize_ConvolutedPSTH_InClsuters
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_weight_tuple_Decomposition_Weights, get_weight_tuple_PSTH



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
    PSTH_tuple = get_weight_tuple_PSTH("raw", 5/1000)    
    svd_5_tuple = get_weight_tuple_Decomposition_Weights("zscore", "SVD", 5, True, _visualize=False)
    Visualize_ConvolutedPSTH_InClsuters(svd_5_tuple, PSTH_tuple, "SVD5")
    

if __name__ == "__main__":
    main()