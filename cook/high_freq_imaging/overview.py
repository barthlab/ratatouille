import sys
import os
import os.path as path
import warnings


import kitchen.video.custom_extraction as custom_extraction
import kitchen.loader.hierarchical_loader as hier_loader

# warnings.filterwarnings("ignore")

def whisker_extraction():
    data_set = hier_loader.naive_loader(template_id="RandPuff", cohort_id="HighFreqImaging_202507")
    # format_converter.dataset_interface_h264_2_avi(data_set) 
    custom_extraction.default_collection(data_set)
   
def main():
    dataset = hier_loader.cohort_loader(template_id="RandPuff", cohort_id="HighFreqImaging_202507") 
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    for session_node in dataset.select("session"):
        print(session_node)
    # for session_node in dataset.select("session"):
    #     node_flat_view(session_node, lick_flag=False, pupil_flag=False, save_name="Overview_{}.png")


if __name__ == "__main__":
    main()


