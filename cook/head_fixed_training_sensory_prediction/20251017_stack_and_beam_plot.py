import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.basic_macros import beam_view_default_macro, flat_view_default_macro, stack_view_default_macro
from kitchen.plotter.macros.behavior_macros import trace_view_delta_behavior_macro
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Fov, Session, Trial

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
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202510", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    plot_manual = PlotManual(locomotion=True, whisker=True, pupil=True)  
    
    trace_view_delta_behavior_macro(dataset.subset("RSS4 only", _self=lambda x: x.mice_id == "RSS4"), "pupil", "CuedBlank", (1, 3.0), (-1., -0),
                                    prefix_keyword="AfterStim", _aligment_style="Aligned2Stim", _to_percent=True)
    trace_view_delta_behavior_macro(dataset.subset("RSS4 only", _self=lambda x: x.mice_id == "RSS4"), "pupil", "CuedBlank", (1, 3.0), (-1., -0),
                                    prefix_keyword="AfterStim", _aligment_style="Aligned2Stim", _to_percent=True, _x_group_type=Trial)
    # trace_view_delta_behavior_macro(dataset.subset("RSS4 only", _self=lambda x: x.mice_id == "RSS4"), "whisker", "CuedBlank", (1, 3.0), (-1., -0), 
    #                                 prefix_keyword="AfterStim", _aligment_style="Aligned2Stim", _to_percent=True)

    
    # for mice_node in dataset.select("mice"):
    #     # flat_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="session", plot_manual=plot_manual)
    #     # stack_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="session", plot_manual=plot_manual,
    #     #                          _aligment_style="Aligned2Stim")

        
    #     stack_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="fov", plot_manual=plot_manual,
    #                                 _aligment_style="Aligned2Stim")
    # plot_manual = PlotManual(whisker=True, pupil=True)  
    # for session_node in dataset.select("session"):
    #     beam_view_default_macro(dataset.subtree(session_node, "SessionSubtree"), node_level="session", plot_manual=plot_manual,
    #                              _aligment_style="Aligned2Stim")

if __name__ == "__main__":
    main()


