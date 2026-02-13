import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.HighFreqImaging_macros.cellular_overview import visualize_celluar_activity_with_behavior, visualize_celluar_activity_with_deconv
from kitchen.plotter.macros.basic_macros import flat_view_default_macro, heatmap_view_default_macro, stack_view_default_macro
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
    dataset = load_dataset(template_id="PassivePuff_HighFreqImaging", cohort_id="HighFreqImaging_202602", 
                           recipe="default_two_photon_mes_parser", name="HFI_SST")
    # dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    plot_manual = PlotManual(whisker=True, fluorescence=True, locomotion=True)

    # visualize_celluar_activity_with_deconv(dataset)
    # visualize_celluar_activity_with_behavior(dataset)
    for mice_node in dataset.select("mice"):
        for session_node in dataset.subtree(mice_node).select("session"):
            flat_view_default_macro(dataset.subtree(session_node, "SessionSubtree"), node_level="session", plot_manual=plot_manual, 
                                    sharey=False, default_padding=0.5) 
            # stack_view_default_macro(dataset.subtree(session_node, "SessionSubtree"), node_level="cellsession", plot_manual=plot_manual,
            #                          _aligment_style="Aligned2Trial", unit_shape=(2, 2))
        # flat_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="session", plot_manual=plot_manual, 
        #                         sharey=False, default_padding=0.5)
        
        # stack_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="cellsession", plot_manual=plot_manual,
        #                         _aligment_style="Aligned2Trial", unit_shape=(2, 2))
        
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade,
        #                             _aligment_style="Aligned2Trial", _target_modality="locomotion")
        # exit()

if __name__ == "__main__":
    main()


