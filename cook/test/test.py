import sys
import os
import os.path as path
import warnings

import kitchen.loader.hierarchical_loader as hier_loader
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES, PREDEFINED_TRIAL_RULES
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.plotter.macros.basic_macros import fov_overview, single_node_trial_avg_default, fov_summary_trial_avg_default
from kitchen.structure.hierarchical_data_structure import Fov

# warnings.filterwarnings("ignore")

def main():
    dataset = hier_loader.cohort_loader(template_id="RandPuff", cohort_id="MultiFallTTL_FromMatt") 
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    plot_manual = PlotManual(fluorescence=True, locomotion=True)

    # fov overview
    for fov_node in dataset.select(hash_key="fov"):
        assert isinstance(fov_node, Fov)
        single_node_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_FOVTRIAL_RULES)
        fov_summary_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_FOVTRIAL_RULES)
        # single_node_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_TRIAL_RULES)
        # fov_summary_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_TRIAL_RULES)
        fov_overview(fov_node, dataset, plot_manual=plot_manual)
        exit()

        
if __name__ == "__main__":
    main()
    