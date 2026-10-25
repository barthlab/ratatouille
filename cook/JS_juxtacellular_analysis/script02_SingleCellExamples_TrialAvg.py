import logging
from os import path

from kitchen.configs.routing import default_data_path
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.HighFreqImaging_macros.posthoc_staining_analysis import cohens_d, visualize_staining_small
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SingleCell import SingleCell_4HZ_BeamView_Beautify, SingleCell_4HZ_JUST_LFP_AVG, SingleCell_RasterPlot, SingleCell_4HZ_300HZ_BeamView, SingleCell_4HZ_ZOOMINOUT_StackView
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import COHORT_COLORS
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


def register_posthoc_staining_result(mouse_node):
    import pandas as pd
    mouse_data_path = default_data_path(mouse_node, check_exist=False)
    pixel_value_file = path.join(mouse_data_path, "..", "..", f"{mouse_node.mice_id}.csv").replace("SST_WC", "sst_whole_cell_staining")
    assert path.exists(pixel_value_file), f"Pixel value file does not exist: {pixel_value_file}"
    pixel_values_df = pd.read_csv(pixel_value_file)
    print(pixel_values_df)

    soma_green = pixel_values_df.loc[pixel_values_df["region"] == "soma", "CH1"].to_numpy()
    soma_cyan = pixel_values_df.loc[pixel_values_df["region"] == "soma", "CH2"].to_numpy()
    soma_magenta = pixel_values_df.loc[pixel_values_df["region"] == "soma", "CH3"].to_numpy()

    neuropil_green = pixel_values_df.loc[pixel_values_df["region"] == "neuropil", "CH1"].to_numpy()
    neuropil_cyan = pixel_values_df.loc[pixel_values_df["region"] == "neuropil", "CH2"].to_numpy()
    neuropil_magenta = pixel_values_df.loc[pixel_values_df["region"] == "neuropil", "CH3"].to_numpy()

    save_path = path.join(path.dirname(pixel_value_file), f"pixels.png")
    visualize_staining_small(soma_magenta, soma_green, neuropil_magenta, neuropil_green, 
                                title=f"{mouse_node.mice_id} SST WC", save_path=save_path)

    cohen_d_green = cohens_d(soma_green, neuropil_green)
    cohen_d_cyan = cohens_d(soma_cyan, neuropil_cyan)
    cohen_d_magenta = cohens_d(soma_magenta, neuropil_magenta)
    print(f"{mouse_node.mice_id}: cohen_d_green={cohen_d_green:.3f}, cohen_d_cyan={cohen_d_cyan:.3f}, cohen_d_magenta={cohen_d_magenta:.3f}")


def main():  
    for dataset_name in ("SST_JUX", "SST_WC", "PV_JUX", "PYR_JUX", "PYR_WC"): 
    # for dataset_name in ("SST_WC", "PV_JUX", "PYR_JUX", "PYR_WC"): 
    # for dataset_name in ("SST_WC", ): 
        dataset = load_dataset(template_id="PassivePuff_JuxtaCellular_FromJS_202509", cohort_id=dataset_name, 
                               recipe="default_ephys", name=dataset_name)
        color_scheme.POTENTIAL_COLOR = COHORT_COLORS[dataset_name]
        style_dicts.POTENTIAL_TRACE_STYLE["color"] = COHORT_COLORS[dataset_name]

        for session_node in dataset.select(hash_key="cellsession"):          
            # register_posthoc_staining_result(session_node)
            # print(session_node)
            # SingleCell_4HZ_JUST_LFP_AVG(session_node, dataset)
            # SingleCell_4HZ_BeamView_Beautify(session_node, dataset)
            # SingleCell_4HZ_300HZ_BeamView(session_node, dataset, _first_k_trials=5)
            # SingleCell_4HZ_ZOOMINOUT_StackView(session_node, dataset, spike_mark=2)
            # SingleCell_4HZ_ZOOMINOUT_StackView(session_node, dataset, spike_mark=3)
            SingleCell_RasterPlot(session_node, dataset, _tiny=True)
            

if __name__ == "__main__":
    main()