import logging
import os.path as path

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.select_trial_rules import PREDEFINED_FOVTRIAL_RULES
from kitchen.plotter.macros.basic_macros import fov_overview, fov_summary_trial_avg_default, session_overview
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.video import format_converter, video_marker
from kitchen.video import custom_extraction


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').disabled = True


def label_videos():
    thermal_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\ThermalTraining\202508_Preliminary"
    format_converter.video_convert(thermal_data_path)
    # video_marker.marker_video_use_timeline(thermal_data_path)


def main():
    dataset = load_dataset(template_id="ThermalTraining", cohort_id="202508_Preliminary", recipe="custom_thermal_analysis")
    # custom_extraction.default_collection(dataset)
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    exit()
    plot_manual = PlotManual(lick=True, locomotion=True)
    # for session_node in dataset.select(hash_key="session"):
    #     session_overview(session_node, plot_manual=plot_manual)
    for fov_node in dataset.select(hash_key="fov"):
        fov_overview(fov_node, dataset, plot_manual=plot_manual)
        fov_summary_trial_avg_default(fov_node, dataset, plot_manual=plot_manual, trial_rules=PREDEFINED_FOVTRIAL_RULES)

    
if __name__ == "__main__":
    # label_videos()
    main()