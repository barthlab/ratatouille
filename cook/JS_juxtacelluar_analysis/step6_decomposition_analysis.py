import logging
from os import path

from kitchen.plotter.macros.jux_data_macros import multiple_dataset_decomposition_plot



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

    
    for bin_size in (10/1000, ):
        for activity_period_name, activity_period in {"Medium": (-1, 1.5), }.items():
            for preprocessing_method in ( "z-score", ):
            # for preprocessing_method in ("log-scale", "raw", "z-score", "baseline-subtraction", "baseline-rescaling", "baseline-normalization",):
                for decomposition_method in ("SVD",  ):
                # for decomposition_method in ("PCA", "SVD", "FA", "SparsePCA", "NMF", "ICA", ):
                    for n_components in (2, 5):
                        decomposition_kwargs = {
                            "save_name": f"{activity_period_name}_{preprocessing_method}_{decomposition_method}_{n_components}_{bin_size}",
                            "FEATURE_RANGE": activity_period,
                            "PREPROCESSING_METHOD": preprocessing_method,
                            "DECOMPOSITION_METHOD": decomposition_method,
                            "n_components": n_components,
                            "BINSIZE": bin_size,
                        }
                        multiple_dataset_decomposition_plot(
                            prefix_keyword=f"ALL_{activity_period_name}_{preprocessing_method}",
                            dir_save_path= "tmp_42",
                            **decomposition_kwargs,
                        )

if __name__ == "__main__":
    main()