import logging
from os import path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.basic_macros import dataset_overview, session_overview
from kitchen.plotter.macros.JS_juxta_data_macros import AnwerOfEverything
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



def preload_dataset():
    # preload all datasets and curate spike waveforms
    load_dataset(template_id="Slice_JuxtaCellular_FromJoe_202510", cohort_id="POm-VIP ABF", 
                    recipe="default_ephys_abf_sweep", name="POm-VIP ABF")

def main():
    # Plot all cell sessions    
    plot_manual_raw = PlotManual(potential='raw')
    plot_manual_spike4Hz = PlotManual(potential=4.)
    plot_manual_spike300Hz = PlotManual(potential=300.)
    plot_manual_conv = PlotManual(potential_conv=True)   
    dataset = load_dataset(template_id="Slice_JuxtaCellular_FromJoe_202510", cohort_id="POm-VIP ABF", 
                recipe="default_ephys_abf_sweep", name="POm-VIP ABF")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"), row_level="cellsession")
    for session_node in dataset.select(hash_key="cellsession"):
        print(session_node)
        session_overview(session_node, plot_manual=plot_manual_raw, prefix_keyword="raw")
        session_overview(session_node, plot_manual=plot_manual_spike4Hz, prefix_keyword="spike4Hz")
        session_overview(session_node, plot_manual=plot_manual_spike300Hz, prefix_keyword="spike300Hz")
            

if __name__ == "__main__":
    # preload_dataset()
    main()