import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.basic_macros import flat_view_default_macro, heatmap_view_default_macro, stack_view_default_macro
from kitchen.plotter.macros.longitudinal_recording_macros.cellular_overview import visualize_celluar_activity_in_heatmap, visualize_celluar_activity_in_heatmap_PSE_Expanded, visualize_celluar_activity_session_wise, visualize_celluar_activity_with_behavior, visualize_cellular_evoked_anticipatory_licking_in_great_details, visualize_cellular_evoked_in_great_details, visualize_cellular_evoked_in_great_details2, visualize_cellular_evoked_in_session_resolution_single_cell, visualize_cellular_evoked_in_session_resolution_single_cell2, visualize_cellular_evoked_overall_summary, visualize_daywise_correlation_summary, visualize_daywise_correlation_summary, visualize_daywise_performance, visualize_early_celluar_activity_summary, visualize_later_celluar_activity_summary, visualize_locomotion_in_fine_details, visualize_mouse_overall_licking_behavior, visualize_passive_reproduce_mo_figures, visualize_sessionwise_weight_distribution
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
    plot_manual = PlotManual(fluorescence=True, locomotion=True, lick=True)


    dataset = load_dataset(template_id="HeadFixedTraining_CalciumImaging", cohort_id="SAT_HFT_Combine", 
                           recipe="matt_two_photon_mes_parser", name="HFT_SAT")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report_SAT.xlsx"))
    
    # visualize_cellular_evoked_in_great_details(dataset, color="#D4E6F6", auc_range=(0, 2), line_color="deepskyblue")
    # visualize_cellular_evoked_in_great_details(dataset, color="#D4E6F6", auc_range=(0, 0.5), line_color="deepskyblue")

    # # visualize_cellular_evoked_in_session_resolution_single_cell(dataset, color="#D4E6F6", line_color="deepskyblue")
    # # visualize_cellular_evoked_in_session_resolution_single_cell2(dataset)

    # # visualize_mouse_overall_licking_behavior(dataset)
    # visualize_cellular_evoked_anticipatory_licking_in_great_details(
    #     dataset, auc_range=(0.7, 2),
    #     line_color_puff="deepskyblue", line_color_blank="skyblue", line_color_performance="dodgerblue",
    #     color="#D4E6F6", )
    
    # for normalize_flag in [True, False]:
    #     for split_flag in [True, ]:
    #         visualize_cellular_evoked_in_great_details2(dataset, color="#D4E6F6", auc_range=(0, 2), line_color="deepskyblue", 
    #                                                     _normalized_by_baseline=normalize_flag, _split_session_halves=split_flag)
    #         visualize_cellular_evoked_in_great_details2(dataset, color="#D4E6F6", auc_range=(0, 0.5), line_color="deepskyblue", 
    #                                                     _normalized_by_baseline=normalize_flag, _split_session_halves=split_flag)

    # # visualize_locomotion_in_fine_details(dataset, _element_trial_level="trial")
    visualize_passive_reproduce_mo_figures(dataset, theme_color="deepskyblue", _element_trial_level="trial", theme_scale=3.7, theme_name="SAT")

    # # visualize_sessionwise_weight_distribution(dataset, _element_trial_level="trial", colormap="coolwarm")
    # visualize_celluar_activity_session_wise(dataset, _element_trial_level="trial")

    # # visualize_daywise_performance(dataset, _element_trial_level="fovtrial", color="deepskyblue")
    # # visualize_cellular_evoked_overall_summary(dataset, color="deepskyblue", auc_range=(0, 0.5))
    # visualize_cellular_evoked_overall_summary(dataset, color="deepskyblue", auc_range=(0, 2))
    # # visualize_daywise_correlation_summary(dataset, _element_trial_level="trial")

    # # visualize_celluar_activity_with_behavior(dataset, _element_trial_level="trial")
    # visualize_celluar_activity_in_heatmap(dataset, _element_trial_level="trial", theme_color="deepskyblue")
    
    # # visualize_early_celluar_activity_summary(dataset, _element_trial_level="trial", early_time_range=(0., 0.5))
    # # visualize_early_celluar_activity_summary(dataset, _element_trial_level="trial", early_time_range=(0.5, 2))
    # # visualize_early_celluar_activity_summary(dataset, _element_trial_level="trial", early_time_range=(1., 2.))

    # # visualize_later_celluar_activity_summary(dataset, _element_trial_level="trial", later_time_range=(2.0, 2.5))
    # # visualize_later_celluar_activity_summary(dataset, _element_trial_level="trial", later_time_range=(2.5, 4.0))
    # # visualize_later_celluar_activity_summary(dataset, _element_trial_level="trial", later_time_range=(3.0, 4.0))


    exit()
    
    dataset = load_dataset(template_id="HeadFixedTraining_CalciumImaging", cohort_id="PSE_HFT_Combine", 
                           recipe="matt_two_photon_mes_parser", name="HFT_PSE")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report_PSE.xlsx"))
    
    # visualize_cellular_evoked_in_great_details(dataset, color="#FFD5D5", auc_range=(0, 2), line_color="orangered")
    # visualize_cellular_evoked_in_great_details(dataset, color="#FFD5D5", auc_range=(0, 0.5), line_color="orangered")

    # visualize_cellular_evoked_in_session_resolution_single_cell(dataset, color="#FFD5D5", line_color="orangered")
    # visualize_cellular_evoked_in_session_resolution_single_cell2(dataset)

    # visualize_mouse_overall_licking_behavior(dataset)
    # visualize_cellular_evoked_anticipatory_licking_in_great_details(
    #     dataset, auc_range=(0.7, 2), 
    #     line_color_puff="orangered", line_color_blank="coral", line_color_performance="firebrick",
    #     color="#FFD5D5", )
    
    # for normalize_flag in [True, False]:
    #     for split_flag in [True, ]:
    #         visualize_cellular_evoked_in_great_details2(dataset, color="#FFD5D5", auc_range=(0, 2), line_color="orangered", 
    #                                                     _normalized_by_baseline=normalize_flag, _split_session_halves=split_flag)
    #         visualize_cellular_evoked_in_great_details2(dataset, color="#FFD5D5", auc_range=(0, 0.5), line_color="orangered", 
    #                                                     _normalized_by_baseline=normalize_flag, _split_session_halves=split_flag)

    # visualize_locomotion_in_fine_details(dataset, _element_trial_level="trial")
    visualize_passive_reproduce_mo_figures(dataset, theme_color="orangered", _element_trial_level="trial", theme_scale=2., theme_name="PSE")

    # visualize_sessionwise_weight_distribution(dataset, _element_trial_level="trial", colormap="coolwarm")
    # visualize_celluar_activity_session_wise(dataset, _element_trial_level="trial")
    
    # visualize_daywise_performance(dataset, _element_trial_level="fovtrial", color="orangered")
    # visualize_cellular_evoked_overall_summary(dataset, color="orangered", auc_range=(0, 0.5))
    # visualize_cellular_evoked_overall_summary(dataset, color="orangered", auc_range=(0, 2))
    # visualize_daywise_correlation_summary(dataset, _element_trial_level="trial")

    # visualize_celluar_activity_with_behavior(dataset, _element_trial_level="trial")
    # visualize_celluar_activity_in_heatmap_PSE_Expanded(dataset, _element_trial_level="trial")
    # visualize_celluar_activity_in_heatmap(dataset, _element_trial_level="trial", theme_color="orangered")

    # visualize_early_celluar_activity_summary(dataset, _element_trial_level="trial", early_time_range=(0., 0.5))
    # visualize_early_celluar_activity_summary(dataset, _element_trial_level="trial", early_time_range=(0.5, 2))
    # visualize_early_celluar_activity_summary(dataset, _element_trial_level="trial", early_time_range=(1., 2.))

    # visualize_later_celluar_activity_summary(dataset, _element_trial_level="trial", later_time_range=(2.0, 2.5))
    # visualize_later_celluar_activity_summary(dataset, _element_trial_level="trial", later_time_range=(2.5, 4.0))
    # visualize_later_celluar_activity_summary(dataset, _element_trial_level="trial", later_time_range=(3.0, 4.0))

    
def main0():
    plot_manual = PlotManual(fluorescence=True, locomotion=True, lick=True, whisker=True)

    dataset = load_dataset(template_id="HeadFixedTraining_CalciumImaging", cohort_id="PFC_ablated_SAT_PSE_HFT", 
                           recipe="matt_two_photon_mes_parser", name="NewMice")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report_NEW.xlsx"))
    for mice_node in dataset.select("mice"):
        for session_node in dataset.subtree(mice_node).select("session"):
            flat_view_default_macro(dataset.subtree(session_node, "SessionSubtree"), node_level="session", plot_manual=plot_manual, 
                                    sharey=False, default_padding=0.5) 
            stack_view_default_macro(dataset.subtree(session_node, "SessionSubtree"), node_level="cellsession", plot_manual=plot_manual,
                                     _aligment_style="Aligned2Adaptive", unit_shape=(2, 2))


if __name__ == "__main__":
    main0()
    # main()


