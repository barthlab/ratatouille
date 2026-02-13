import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.media import facemap_pupil_extraction, video_marker, meye_pupil_extraction
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
    hfi_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\PassivePuff_HighFreqImaging\HighFreqImaging_202602"
    format_converter.video_convert(hfi_data_path, src_format=".h264")
    dataset = load_dataset(template_id="PassivePuff_HighFreqImaging", cohort_id="HighFreqImaging_202602", 
                           recipe="default_two_photon_mes_parser", name="HFI_SST")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    custom_extraction.default_collection(dataset)
    # meye_pupil_extraction.default_collection(dataset)

def label_videos():
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\HeadFixedTraining\SensoryPrediction_202513"
    format_converter.video_convert(hft_data_path)
    video_marker.marker_video_use_timeline(hft_data_path)

def main():
    dataset = load_dataset(template_id="PassivePuff_HighFreqImaging", cohort_id="HighFreqImaging_202602", 
                           recipe="default_two_photon_mes_parser", name="HFI_SST")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))



if __name__ == "__main__":
    # preprocessing()
    # label_videos()
    main()


