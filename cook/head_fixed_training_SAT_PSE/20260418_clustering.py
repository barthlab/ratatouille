import os.path as path
import logging

from kitchen.configs import routing
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.longitudinal_recording_macros.cellular_clustering import get_dataset_snapshot, peak_reliability_clustering, store_dataset_snapshot, trial_response_shape_clustering
from kitchen.plotter.macros.longitudinal_recording_macros.cellular_overview import visualize_celluar_activity_session_wise, visualize_cellular_evoked_in_great_details2_with_given_labels
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

    # store_dataset_snapshot(dataset, save_path=path.join(path.dirname(__file__), "dataset_snapshot_SAT_cellday.pkl"), cell_level="cellday")
    store_dataset_snapshot(dataset, save_path=path.join(path.dirname(__file__), "dataset_snapshot_SAT_cellsession.pkl"), cell_level="cellsession")
    

    # for activity_range in ((0, 2), ):
    #     for responsive_percentile, axis_func_name in ((50, "peak"), (50, "auc")):
    #             peak_reliability_clustering(dataset, 
    #                                         activity_range=activity_range, responsive_percentile=responsive_percentile, axis_func_name=axis_func_name,
    #                                         _element_trial_level="trial", color="#D4E6F6", line_color="deepskyblue")



    dataset = load_dataset(template_id="HeadFixedTraining_CalciumImaging", cohort_id="PSE_HFT_Combine", 
                           recipe="matt_two_photon_mes_parser", name="HFT_PSE")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report_PSE.xlsx"))

    # store_dataset_snapshot(dataset, save_path=path.join(path.dirname(__file__), "dataset_snapshot_PSE_cellday.pkl"), cell_level="cellday")
    store_dataset_snapshot(dataset, save_path=path.join(path.dirname(__file__), "dataset_snapshot_PSE_cellsession.pkl"), cell_level="cellsession")

    # dataset = get_dataset_snapshot(save_path=path.join(path.dirname(__file__), "dataset_snapshot_PSE.pkl"))

    # for activity_range in ((0, 2), ):
    #     for responsive_percentile, axis_func_name in ((50, "peak"), (50, "auc")):
    #             peak_reliability_clustering(dataset, 
    #                                         activity_range=activity_range, responsive_percentile=responsive_percentile, axis_func_name=axis_func_name,
    #                                         _element_trial_level="trial", color="#FFD5D5", line_color="orangered")


def main2():

    sat_dataset = get_dataset_snapshot(save_path=path.join(path.dirname(__file__), "dataset_snapshot_SAT_cellday.pkl"))
    pse_dataset = get_dataset_snapshot(save_path=path.join(path.dirname(__file__), "dataset_snapshot_PSE_cellday.pkl"))

    # trial_response_shape_clustering(sat_dataset, pse_dataset, preprocess="raw")
    trial_response_shape_clustering(sat_dataset, pse_dataset, preprocess="downsample", cell_level="cellday", _include_blank=True)
    # trial_response_shape_clustering(sat_dataset, pse_dataset, preprocess="downsample", cell_level="cellday", _include_blank=False)


def main3():

    sat_dataset = get_dataset_snapshot(save_path=path.join(path.dirname(__file__), "dataset_snapshot_SAT_cellsession.pkl"))
    pse_dataset = get_dataset_snapshot(save_path=path.join(path.dirname(__file__), "dataset_snapshot_PSE_cellsession.pkl"))
    trial_response_shape_clustering(sat_dataset, pse_dataset, preprocess="downsample", cell_level="cellsession", _include_blank=False)
    optimal_labels = trial_response_shape_clustering(sat_dataset, pse_dataset, preprocess="downsample", cell_level="cellsession", _include_blank=False, _normalize=False)
    exit()
    

    dataset = load_dataset(template_id="HeadFixedTraining_CalciumImaging", cohort_id="SAT_HFT_Combine", 
                           recipe="matt_two_photon_mes_parser", name="HFT_SAT")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report_SAT.xlsx"))
    
    # visualize_celluar_activity_session_wise(dataset, _element_trial_level="trial",  _given_labels=optimal_labels)
    for normalize_flag in [True, False]:
        visualize_cellular_evoked_in_great_details2_with_given_labels(dataset, color="#D4E6F6", auc_range=(0, 2), line_color="deepskyblue", 
                                                                        _given_labels=optimal_labels,
                                                    _normalized_by_baseline=normalize_flag)

    dataset = load_dataset(template_id="HeadFixedTraining_CalciumImaging", cohort_id="PSE_HFT_Combine", 
                           recipe="matt_two_photon_mes_parser", name="HFT_PSE")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report_PSE.xlsx"))

    # visualize_celluar_activity_session_wise(dataset, _element_trial_level="trial", _given_labels=optimal_labels)
    for normalize_flag in [True, False]:
        visualize_cellular_evoked_in_great_details2_with_given_labels(dataset, color="#FFD5D5", auc_range=(0, 2), line_color="orangered", 
                                                                    _given_labels=optimal_labels,
                                                _normalized_by_baseline=normalize_flag)


if __name__ == "__main__":
    # main()
    # main2()
    main3()