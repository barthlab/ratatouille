import warnings
import kitchen.loader.two_photon_loader as hier_loader
from kitchen.plotter.macros.basic_macros import session_overview, single_node_trial_avg_default
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.video import custom_extraction, facemap_pupil_extraction, format_converter


warnings.filterwarnings("ignore")

def pre_conversion():
    hft_data_path = r"C:\Users\maxyc\PycharmProjects\Ratatouille\ingredients\dataset_template"
    format_converter.video_convert(hft_data_path)
    
def whisker_extraction():
    data_set = hier_loader.naive_loader(template_id="ThreeStage_WaterOmission", cohort_id="dataset_template")      
    custom_extraction.default_collection(data_set)

def pupil_extraction():
    data_set = hier_loader.naive_loader(template_id="ThreeStage_WaterOmission", cohort_id="dataset_template")      
    facemap_pupil_extraction.default_collection(data_set)
    
def main():
    dataset = hier_loader.cohort_loader(template_id="ThreeStage_WaterOmission", cohort_id="dataset_template")
    # dataset.status(save_path="template_status_report.xlsx")

    plot_manual = PlotManual(lick=True, locomotion=True)
    for session_node in dataset.select(hash_key="session"):
        # session_overview(session_node, plot_manual=plot_manual)  # type: ignore
        single_node_trial_avg_default(session_node, dataset, plot_manual=plot_manual)



if __name__ == "__main__":
    # pre_conversion()
    # whisker_extraction()
    # pupil_extraction()
    main()