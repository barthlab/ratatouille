import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.select_trial_rules import PREDEFINED_PASSIVEPUFF_RULES
from kitchen.plotter.macros.basic_macros import session_overview, single_node_trial_avg_default, single_node_trial_parallel_default, subtree_summary_trial_avg_default
from kitchen.plotter.macros.jux_data_macros import feature_overview, raster_plot, single_cell_session_parallel_view_jux, subtree_summary_trial_avg_jux, waveform_plot
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


def waveform_overview():
    sp_prefix = "Feature"
    datasets = []
    for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "SST_JUX",):        
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        datasets.append(dataset)
    waveform_plot(datasets, prefix_keyword=sp_prefix)

    
if __name__ == "__main__":
    waveform_overview()