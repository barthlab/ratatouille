import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.select_trial_rules import PREDEFINED_PASSIVEPUFF_RULES
from kitchen.plotter.macros.basic_macros import session_overview, single_node_trial_avg_default, single_node_trial_parallel_default, subtree_summary_trial_avg_default
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Session



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)


def main():
    sp_prefix = "trial_wise"
    # for dataset_name in ("PV_JUX", "PYR_JUX", "SST_JUX", "SST_WC"):
    for dataset_name in ("PV_JUX", "PYR_JUX", ):

        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys")
        # dataset.status(save_path=path.join(path.dirname(__file__), f"status_report_{dataset_name}.xlsx"), row_level="cellsession")

        plot_manual_spike05Hz = PlotManual(potential=0.5,)
        plot_manual_spike300Hz = PlotManual(potential=300.)
        # for session_node in dataset.select(hash_key="cellsession"):
        #     print(session_node)
            # single_node_trial_avg_default(session_node, dataset, plot_manual=plot_manual_spike05Hz, 
            #                               prefix_keyword=sp_prefix+"spike05Hz", trial_rules=PREDEFINED_PASSIVEPUFF_RULES)
            # single_node_trial_parallel_default(session_node, dataset, plot_manual=plot_manual_spike05Hz, 
            #                                    prefix_keyword=sp_prefix+"spike05Hz", trial_rules=PREDEFINED_PASSIVEPUFF_RULES)
            # single_node_trial_avg_default(session_node, dataset, plot_manual=plot_manual_spike300Hz, 
            #                               prefix_keyword=sp_prefix+"spike300Hz", trial_rules=PREDEFINED_PASSIVEPUFF_RULES)
            # single_node_trial_parallel_default(session_node, dataset, plot_manual=plot_manual_spike300Hz, 
            #                                    prefix_keyword=sp_prefix+"spike300Hz", trial_rules=PREDEFINED_PASSIVEPUFF_RULES)
            # exit()
        
if __name__ == "__main__":
    main()