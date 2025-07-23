import sys
import os
import os.path as path
import warnings


from kitchen.plotter.macros.basic_macros import fov_overview, fov_trial_avg_default, session_overview
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Fov, Session
import kitchen.video.custom_extraction as custom_extraction
import kitchen.loader.hierarchical_loader as hier_loader

warnings.filterwarnings("ignore")

def whisker_extraction():
    data_set = hier_loader.naive_loader(template_id="RandPuff", cohort_id="HighFreqImaging_202507")
    # format_converter.dataset_interface_h264_2_avi(data_set) 
    custom_extraction.default_collection(data_set)
   
def main():
    dataset = hier_loader.cohort_loader(template_id="RandPuff", cohort_id="HighFreqImaging_202507") 
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    
    plot_manual = PlotManual(fluorescence=True, locomotion=True, whisker=True)
    for fov_node in dataset.select(hash_key="fov"):
        assert isinstance(fov_node, Fov)
        fov_trial_avg_default(fov_node, dataset, plot_manual=plot_manual)
        fov_overview(fov_node, dataset, plot_manual=plot_manual)

    for session_node in dataset.select("session"):
        assert isinstance(session_node, Session)
        session_overview(session_node, plot_manual=plot_manual)


if __name__ == "__main__":
    main()


