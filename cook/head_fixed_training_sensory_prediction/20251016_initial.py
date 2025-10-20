import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.media import facemap_pupil_extraction, video_marker
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Fov, Session
import kitchen.media.format_converter as format_converter
import kitchen.media.custom_extraction as custom_extraction

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING) 

def preprocessing():
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\HeadFixedTraining\SensoryPrediction_202510"
    format_converter.video_convert(hft_data_path)
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202510", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    # custom_extraction.default_collection(dataset)
    # facemap_pupil_extraction.default_collection(dataset)

def label_videos():
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\HeadFixedTraining\SensoryPrediction_202510"
    # format_converter.video_convert(hft_data_path)
    video_marker.marker_video_use_timeline(hft_data_path)

def main():
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202510", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    plot_manual = PlotManual(lick=True, locomotion=True, whisker=True, pupil=True)  
    
    # # water omission
    # water_omission_response_compare(dataset, plot_manual)
    # mice_water_omission_overview(dataset, plot_manual)
    # mice_water_omission_summary(dataset, plot_manual)

    # # fov overview
    # for fov_node in dataset.select(hash_key="fov"):
    #     assert isinstance(fov_node, Fov)
    #     fov_overview(fov_node, dataset, plot_manual=plot_manual)


if __name__ == "__main__":
    # preprocessing()
    label_videos()
    # main()


