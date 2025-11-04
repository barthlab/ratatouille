import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.basic_macros import dataset_overview, session_overview
from kitchen.plotter.macros.JS_juxta_data_macros import SPONT_FR_EVOKED_FR_SCATTER, AnwerOfEverything
from kitchen.plotter.plotting_manual import PlotManual

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
    # Plot all cell sessions    
    plot_manual_raw = PlotManual(potential='raw')
    plot_manual_spike4Hz = PlotManual(potential=4.)
    plot_manual_spike300Hz = PlotManual(potential=300.)
    plot_manual_conv = PlotManual(potential_conv=True)
    all_datasets = []
    for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "SST_JUX",):        
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        # dataset_overview(dataset, plot_manual=plot_manual_spike300Hz, prefix_keyword="spike300Hz")
        # dataset_overview(dataset, plot_manual=plot_manual_conv, prefix_keyword="conv")
        # AnwerOfEverything(dataset)
        all_datasets.append(dataset)
        # exit()
    SPONT_FR_EVOKED_FR_SCATTER(all_datasets)

if __name__ == "__main__":
    main()