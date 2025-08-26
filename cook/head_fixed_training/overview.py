import os.path as path
import warnings

from kitchen.plotter.macros.basic_macros import fov_overview, fov_trial_avg_default, node_trial_avg_default, session_overview
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Fov, Session
import kitchen.video.format_converter as format_converter
import kitchen.video.custom_extraction as custom_extraction
import kitchen.video.facemap_pupil_extraction as facemap_pupil_extraction
import kitchen.loader.two_photon_loader as hier_loader

warnings.filterwarnings("ignore")

def pre_conversion():
    hft_data_path = r"E:\Max_Behavior_Training"
    format_converter.video_convert(hft_data_path)
    
def whisker_extraction():
    data_set = hier_loader.naive_loader(template_id="ThreeStage_WaterOmission", cohort_id="HeadFixedTraining_202507")      
    custom_extraction.default_collection(data_set)

def pupil_extraction():
    data_set = hier_loader.naive_loader(template_id="ThreeStage_WaterOmission", cohort_id="HeadFixedTraining_202507")      
    facemap_pupil_extraction.default_collection(data_set)


def main():
    dataset = hier_loader.cohort_loader(template_id="ThreeStage_WaterOmission", cohort_id="HeadFixedTraining_202507") 
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    # # M002_PB_2420M
    plot_manual = PlotManual(lick=True, locomotion=True, whisker=True, pupil=True)
    
    for fov_node in dataset.select(
        hash_key="fov", mice_id="M002_PB_2420M"
    ):
        assert isinstance(fov_node, Fov)
        fov_trial_avg_default(fov_node, dataset, plot_manual=plot_manual)
        fov_overview(fov_node, dataset, plot_manual=plot_manual)

    for fovday_node in dataset.select(hash_key="fovday", mice_id="M002_PB_2420M", coordinate=lambda coordinate:(coordinate.day_id > "20250703")):
        for session_node in dataset.subtree(fovday_node).select(hash_key="session"):
            assert isinstance(session_node, Session)
            session_overview(session_node, plot_manual=plot_manual)
            node_trial_avg_default(session_node, dataset, plot_manual=plot_manual)
        node_trial_avg_default(fovday_node, dataset, plot_manual=plot_manual)

    # M003_QPV_5F
    plot_manual = PlotManual(lick=True, locomotion=True, whisker=True)

    for fov_node in dataset.select(
        hash_key="fov", mice_id="M003_QPV_5F",
    ):
        assert isinstance(fov_node, Fov)
        fov_trial_avg_default(fov_node, dataset, plot_manual=plot_manual)
        fov_overview(fov_node, dataset, plot_manual=plot_manual)
    
    for fovday_node in dataset.select(hash_key="fovday", mice_id="M003_QPV_5F", coordinate=lambda coordinate:(coordinate.day_id > "20250705")):
        for session_node in dataset.subtree(fovday_node).select(hash_key="session"):
            assert isinstance(session_node, Session)
            session_overview(session_node, plot_manual=plot_manual)
            node_trial_avg_default(session_node, dataset, plot_manual=plot_manual)
        node_trial_avg_default(fovday_node, dataset, plot_manual=plot_manual)
 

if __name__ == "__main__":
    # pre_conversion()
    # whisker_extraction()
    # pupil_extraction()
    main()


