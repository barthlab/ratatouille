import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.select_trial_rules import PREDEFINED_PASSIVEPUFF_RULES
from kitchen.plotter.macros.basic_macros import session_overview, single_node_trial_avg_default, single_node_trial_parallel_default, subtree_summary_trial_avg_default
from kitchen.plotter.macros.jux_data_macros import single_cell_session_parallel_view_jux, subtree_summary_trial_avg_jux
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
    sp_prefix = "jux_analysis"
    for dataset_name in ("PYR_JUX", "PV_JUX", "SST_JUX", "SST_WC"):

        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys")
        # dataset.status(save_path=path.join(path.dirname(__file__), f"status_report_{dataset_name}.xlsx"), row_level="cellsession")

        for session_node in dataset.select(hash_key="cellsession"):
            print(session_node)
            single_cell_session_parallel_view_jux(session_node, dataset, prefix_keyword=sp_prefix)
        for mice_node in dataset.enumerate_by(Mice):
            print(mice_node)
            subtree_summary_trial_avg_jux(mice_node, dataset, prefix_keyword=sp_prefix)

if __name__ == "__main__":
    main()