import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.JS_juxta_data_macros_SingleCell import SingleCell_4HZ_BeamView_Beautify, SingleCell_RasterPlot, SingleCell_4HZ_300HZ_BeamView, SingleCell_4HZ_ZOOMINOUT_StackView
from kitchen.plotter.macros.JS_juxta_data_macros_Settings import COHORT_COLORS
import kitchen.plotter.color_scheme as color_scheme
import kitchen.plotter.style_dicts as style_dicts



logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)



def main():  
    for dataset_name in ("SST_WC", ):     
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        for session_node in dataset.select(hash_key="cellsession"):
            print(session_node)
            SingleCell_4HZ_BeamView_Beautify(session_node, dataset)
    # exit()
    for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "SST_JUX",):     
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]
        for session_node in dataset.select(hash_key="cellsession"):
            print(session_node)
            SingleCell_4HZ_300HZ_BeamView(session_node, dataset)
            SingleCell_4HZ_ZOOMINOUT_StackView(session_node, dataset)
            SingleCell_RasterPlot(session_node, dataset)
            

if __name__ == "__main__":
    main()