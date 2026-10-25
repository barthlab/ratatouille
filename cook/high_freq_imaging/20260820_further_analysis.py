import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.macros.HighFreqImaging_macros.posthoc_staining_analysis import register_staining_info
from kitchen.plotter.macros.HighFreqImaging_macros.sorting_activity import sort_all_cell_activity
from kitchen.plotter.macros.HighFreqImaging_macros.state_dependent import visualize_single_cell_state_dependent
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
logging.getLogger('PIL').setLevel(logging.WARNING) 



def celltype_assignment_analysis():
    dataset = load_dataset(template_id="PassivePuff_HighFreqImaging", cohort_id="HighFreqImaging_Combine_ROIs", 
                           recipe="default_two_photon_mes_parser", name="HFI_SST")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    
    plot_manual = PlotManual(whisker=True, fluorescence=True, locomotion=True)

    register_staining_info(dataset, create_small_pic=False, create_big_pic=False)

    # dataset.status(save_path=path.join(path.dirname(__file__), "status_report_with_info.xlsx"), row_level="cellsession", add_info=True)



def state_dependent_analysis():
    dataset = load_dataset(template_id="PassivePuff_HighFreqImaging", cohort_id="HighFreqImaging_Combine_ROIs", 
                           recipe="default_two_photon_mes_parser", name="HFI_SST")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    
    plot_manual = PlotManual(whisker=True, fluorescence=True, locomotion=True)

    register_staining_info(dataset, create_small_pic=False, create_big_pic=False)
    # visualize_single_cell_state_dependent(dataset, plot_small_pic=False)

    

if __name__ == "__main__":
    sort_all_cell_activity()
    # deconv_clustering_analysis(_visualize=True)
    # celltype_assignment_analysis()
    # state_dependent_analysis()