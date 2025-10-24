import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.media import facemap_pupil_extraction, video_marker
from kitchen.plotter.macros.basic_macros import flat_view_default_macro, stack_view_default_macro
from kitchen.plotter.plotting_manual import PlotManual
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
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\GRABSensor"
    format_converter.video_convert(hft_data_path)
    dataset = load_dataset(template_id="GRABSensor", cohort_id="NE_GRAB_202510", 
                           recipe="default_two_photon", name="NE_GRAB")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    custom_extraction.default_collection(dataset)
    # facemap_pupil_extraction.default_collection(dataset)

# def label_videos():
#     hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\GRABSensor"
#     # format_converter.video_convert(hft_data_path)
#     video_marker.marker_video_use_timeline(hft_data_path)

def main():
    dataset = load_dataset(template_id="GRABSensor", cohort_id="NE_GRAB_202510", 
                           recipe="default_two_photon", name="NE_GRAB")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    plot_manual = PlotManual(whisker=True, fluorescence=True, locomotion=True)  
    import matplotlib.pyplot as plt
    plt.rcParams["lines.linewidth"] = 0.5
    flat_view_default_macro(dataset, node_level="cellsession", plot_manual=plot_manual, sharey=False, unit_shape=(5, 5), 
                            node_num_flag=False, default_padding=0.3)

    
    # stack_view_default_macro(dataset, node_level="session", plot_manual=plot_manual, _aligment_style="Aligned2Stim", sharey=False)
    
    # stack_view_default_macro(dataset, node_level="session", plot_manual=plot_manual, _aligment_style="Aligned2Stim", _element_trial_level="fovtrial", sharey=False)

if __name__ == "__main__":
    # preprocessing()
    main()


