import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.basic_macros import beam_view_default_macro, flat_view_default_macro, heatmap_view_default_macro, stack_view_default_macro, subtract_view_default_macro
from kitchen.plotter.macros.behavior_macros import half_stack_view_default_macro, trace_view_delta_behavior_macro
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
    dataset = load_dataset(template_id="HeadFixedTraining_FromJoe", cohort_id="SAT_202512", 
                           recipe="default_behavior_only_joe_data", name="JOE_DATA")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    plot_manual_saccade = PlotManual(locomotion=True, lick=True, whisker=True, pupil=True, saccade=True)  
    plot_manual_saccade_baseline = PlotManual(locomotion=True, lick=True, whisker=True, pupil=True, saccade=True, baseline_subtraction=(0, 1.7, False))  
    
    # plot_manual_saccade_presurprise = PlotManual(locomotion=True, whisker=True, pupil=True, saccade=True, 
    #                                              baseline_subtraction=(0, 1.7, False), amplitude_sorting=(4., 8., "day"))  
    # plot_manual_saccade_presurprise_session = PlotManual(locomotion=True, whisker=True, pupil=True, saccade=True, 
    #                                              baseline_subtraction=(0, 1.7, False), amplitude_sorting=(4., 8., "session"))      
    # plot_manual_saccade_sortonly = PlotManual(locomotion=True, whisker=True, pupil=True, saccade=True, amplitude_sorting=(4., 8., "day"))  

    for mice_node in dataset.select("mice"):
        # flat_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="session", plot_manual=plot_manual_saccade)

        stack_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="session", plot_manual=plot_manual_saccade_baseline,
                                 _aligment_style="Aligned2Reward")
        
        stack_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade_baseline,
                                    _aligment_style="Aligned2Reward")
        
        stack_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade_baseline,
                                    _aligment_style="Aligned2Reward")
        


        
        # subtract_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade_baseline,
        #                             _aligment_style="Aligned2Trial", _target_types="CuedPuff")
        
        # subtract_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade_baseline,
        #                             _aligment_style="Aligned2Trial", _target_types="CuedPuff")

        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade,
        #                             _aligment_style="Aligned2Trial", _target_modality="locomotion")




        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade,
        #                             _aligment_style="Aligned2Trial", _target_modality="whisker")
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade_sortonly,
        #                             _aligment_style="Aligned2Trial", _target_modality="whisker", _sort_row=True, _add_dummy=True)
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade,
        #                             _aligment_style="Aligned2Trial", _target_modality="whisker")
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade_sortonly,
        #                             _aligment_style="Aligned2Trial", _target_modality="whisker", _sort_row=True, _add_dummy=True)
        # exit()
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade,
        #                             _aligment_style="Aligned2Trial", _target_modality="saccade", _sort_row=False)
        
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade_presurprise_session,
        #                             _aligment_style="Aligned2Trial", _target_modality="pupil", _sort_row=True, _add_dummy=True)
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade_presurprise,
        #                             _aligment_style="Aligned2Trial", _target_modality="pupil", _sort_row=True, _add_dummy=True)
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade_presurprise,
        #                             _aligment_style="Aligned2Trial", _target_modality="pupil", _sort_row=True, _add_dummy=True)
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual_saccade_baseline,
        #                             _aligment_style="Aligned2Trial", _target_modality="pupil", _sort_row=False)
        # heatmap_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="mice", plot_manual=plot_manual_saccade_baseline,
        #                             _aligment_style="Aligned2Trial", _target_modality="pupil", _sort_row=False)
        
        
        # half_stack_view_default_macro(dataset.subtree(mice_node, "MiceSubtree"), node_level="day", plot_manual=plot_manual,
        #                              _aligment_style="Aligned2Trial", auto_title=False)
    # plot_manual = PlotManual(whisker=True, pupil=True)  
    # for session_node in dataset.select("session"):
    #     beam_view_default_macro(dataset.subtree(session_node, "SessionSubtree"), node_level="session", plot_manual=plot_manual,
    #                              _aligment_style="Aligned2Stim")

if __name__ == "__main__":
    main()


