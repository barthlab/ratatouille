import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.basic_macros import flat_view_default_macro
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
    for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "SST_JUX",):    
        load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                     recipe="default_ephys", name=dataset_name)

def main():
    # Plot all cell sessions    
    plot_manual_raw = PlotManual(potential='raw')
    plot_manual_spike4Hz = PlotManual(potential=4.)
    plot_manual_spike300Hz = PlotManual(potential=300.)
    plot_manual_conv = PlotManual(potential_conv=True)
    for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "SST_JUX",):        
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        for prefix_name, plot_manual in zip(
                ("raw", "spike4Hz", "spike300Hz", "conv",),
                (plot_manual_raw, plot_manual_spike4Hz, plot_manual_spike300Hz, plot_manual_conv,),
        ):
            flat_view_default_macro(dataset, node_level="cellsession", plot_manual=plot_manual, 
                                    prefix_keyword=prefix_name, unit_shape=(30, 1))
            for session_node in dataset.select(hash_key="cellsession"):
                flat_view_default_macro(dataset.subtree(session_node), node_level="cellsession", plot_manual=plot_manual, 
                                        prefix_keyword=prefix_name, unit_shape=(30, 2))

            

if __name__ == "__main__":
    # preload_dataset()
    main()