import sys
import os
import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES, PREDEFINED_TRIAL_RULES
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.macros.basic_macros import fov_overview, single_node_trial_avg_default
from kitchen.structure.hierarchical_data_structure import Fov


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING) 

# warnings.filterwarnings("ignore")



def main():
    for dataset_name in ("Ai148_ACC", "Ai148_SAT", "Ai148_PSE"):
        dataset = load_dataset(template_id="PassivePuff_FromMo", cohort_id=dataset_name, recipe="custom_first_puff_analysis")
        # dataset.status(save_path=path.join(path.dirname(__file__), f"{dataset_name}_status.xlsx"))
        
        plot_manual = PlotManual(fluorescence=True,)

        # # fov overview
        for fov_node in dataset.select(hash_key="fov"):
            assert isinstance(fov_node, Fov)
            # single_node_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_FOVTRIAL_RULES)
            # fov_summary_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_FOVTRIAL_RULES)
            # single_node_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_TRIAL_RULES)
            # fov_summary_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_TRIAL_RULES)
            fov_overview(fov_node, dataset, plot_manual=plot_manual)
       

        
if __name__ == "__main__":
    main()
    