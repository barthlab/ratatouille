import os.path as path
import warnings

from kitchen.plotter.macros.basic_macros import fov_overview, fov_trial_avg_default, node_trial_avg_default, session_overview
from kitchen.plotter.macros.water_omission_macros import mice_water_omission_overview, mice_water_omission_summary, water_omission_response_compare
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Fov, Session
import kitchen.video.format_converter as format_converter
import kitchen.video.custom_extraction as custom_extraction
import kitchen.loader.hierarchical_loader as hier_loader

warnings.filterwarnings("ignore")

def preprocessing():
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\HeadFixedTraining_SurpriseSignal"
    format_converter.video_convert(hft_data_path)
    data_set = hier_loader.naive_loader(template_id="WaterOmissionTemplate", cohort_id="HeadFixedTraining_SurpriseSignal")      
    custom_extraction.default_collection(data_set)


def main():
    dataset = hier_loader.cohort_loader(template_id="WaterOmissionTemplate", cohort_id="HeadFixedTraining_SurpriseSignal") 
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    plot_manual = PlotManual(lick=True, locomotion=True, whisker=True, pupil=True)  
    
    # water omission
    water_omission_response_compare(dataset, plot_manual)
    mice_water_omission_overview(dataset, plot_manual)
    mice_water_omission_summary(dataset, plot_manual)

    # fov overview
    for fov_node in dataset.select(hash_key="fov"):
        assert isinstance(fov_node, Fov)
        fov_trial_avg_default(fov_node, dataset, plot_manual=plot_manual)
        fov_overview(fov_node, dataset, plot_manual=plot_manual)

    # for fovday_node in dataset.select(hash_key="fovday", mice_id="M002_PB_2420M", coordinate=lambda coordinate:(coordinate.day_id > "20250703")):
    #     for session_node in dataset.subtree(fovday_node).select(hash_key="session"):
    #         assert isinstance(session_node, Session)
    #         session_overview(session_node, plot_manual=plot_manual)
    #         node_trial_avg_default(session_node, dataset, plot_manual=plot_manual)
    #     node_trial_avg_default(fovday_node, dataset, plot_manual=plot_manual)


if __name__ == "__main__":
    # preprocessing()
    main()


