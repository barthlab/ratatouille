import sys
import os
import os.path as path
import warnings

from tqdm import tqdm

# from kitchen.plotter.data_viewer.datasets_view import datasets_avg_view
# from kitchen.plotter.data_viewer.nodes_view import node_avg_view, node_flat_view
from kitchen.structure.hierarchical_data_structure import Session
from kitchen.structure.neural_data_structure import Timeline
import kitchen.video.format_converter as format_converter
import kitchen.video.custom_extraction as custom_extraction
import kitchen.video.facemap_pupil_extraction as facemap_pupil_extraction
import kitchen.loader.hierarchical_loader as hier_loader

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



rule_w = {
    "hash_key": "trial",
    "timeline": lambda x: (len(x.filter("WaterOn")) > 0) & (len(x.filter("VerticalPuffOn")) == 0)
}

rule_nw = {
    "hash_key": "trial",
    "timeline": lambda x: (len(x.filter("NoWaterOn")) > 0) & (len(x.filter("BlankOn")) == 0)
}

rule_pw = {
    "hash_key": "trial",
    "timeline": lambda x: (len(x.filter("WaterOn")) > 0) & (len(x.filter("VerticalPuffOn")) > 0)
}

rule_pn = {
    "hash_key": "trial",
    "timeline": lambda x: (len(x.filter("NoWaterOn")) > 0) & (len(x.filter("VerticalPuffOn")) > 0)
}

rule_bn = {
    "hash_key": "trial",
    "timeline": lambda x: (len(x.filter("NoWaterOn")) > 0) & (len(x.filter("BlankOn")) > 0)
}

def main():
    dataset = hier_loader.cohort_loader(template_id="ThreeStage_WaterOmission", cohort_id="HeadFixedTraining_202507") 
    exit()
    # dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    # M002_PB_2420M

    for fovday_node in dataset.select(hash_key="fovday", mice_id="M002_PB_2420M", coordinate=lambda coordinate:(coordinate.day_id > "20250703")):
        for session_node in dataset.subtree(fovday_node).select(hash_key="session"):
            node_flat_view(session_node, fluorescence_flag=False, save_name="Overview_{}.png")

            datasets_avg_view(
                datasets=[dataset.subtree(session_node).select(**rule_w), 
                          dataset.subtree(session_node).select(**rule_nw),
                          dataset.subtree(session_node).select(**rule_pw),
                          dataset.subtree(session_node).select(**rule_bn),
                          dataset.subtree(session_node).select(**rule_pn)],
                sync_events=[("BuzzerOn",) for _ in range(5)],
                titles=[session_node.session_id + "_Water", 
                        session_node.session_id + "_NoWater",
                        session_node.session_id + "_PuffWater", 
                        session_node.session_id + "_BlankNoWater", 
                        session_node.session_id + "_PuffNoWater"],
                fluorescence_flag=False,
                save_name="TrialAvg_{}_buzzer_aligned.png"
            )

            datasets_avg_view(
                datasets=[dataset.subtree(session_node).select(**rule_w), 
                          dataset.subtree(session_node).select(**rule_nw),
                          dataset.subtree(session_node).select(**rule_pw),
                          dataset.subtree(session_node).select(**rule_bn),
                          dataset.subtree(session_node).select(**rule_pn)],
                sync_events=[("WaterOn", "NoWaterOn") for _ in range(5)],
                titles=[session_node.session_id + "_Water", 
                        session_node.session_id + "_NoWater",
                        session_node.session_id + "_PuffWater", 
                        session_node.session_id + "_BlankNoWater", 
                        session_node.session_id + "_PuffNoWater"],
                fluorescence_flag=False,
                save_name="TrialAvg_{}_water_aligned.png"
            )
    
    all_dataset, sync_events, titles = [], [], []
    for fovday_node in tqdm(dataset.select(hash_key="fovday", mice_id="M002_PB_2420M", coordinate=lambda coordinate:(coordinate.day_id > "20250703"))):

        all_dataset += [dataset.subtree(fovday_node).select(**rule_w), dataset.subtree(fovday_node).select(**rule_nw)]
        sync_events += [("BuzzerOn",) for _ in range(2)]
        titles += [fovday_node.day_id + "_Water", fovday_node.day_id + "_NoWater"]
        all_dataset += [dataset.subtree(fovday_node).select(**rule_pw), dataset.subtree(fovday_node).select(**rule_bn), dataset.subtree(fovday_node).select(**rule_pn)]
        sync_events += [("BuzzerOn",) for _ in range(3)]
        titles += [fovday_node.day_id + "_PuffWater", fovday_node.day_id + "_BlankNoWater", fovday_node.day_id + "_PuffNoWater"]
    datasets_avg_view(datasets=all_dataset, sync_events=sync_events, titles=titles, fluorescence_flag=False, save_name="TrialAvg_{}_buzzer_aligned.png")


    
    all_dataset, sync_events, titles = [], [], []
    for fovday_node in tqdm(dataset.select(hash_key="fovday", mice_id="M002_PB_2420M", coordinate=lambda coordinate:(coordinate.day_id > "20250703"))):
        all_dataset += [dataset.subtree(fovday_node).select(**rule_w), dataset.subtree(fovday_node).select(**rule_nw)]
        sync_events += [("WaterOn", "NoWaterOn") for _ in range(2)]
        titles += [fovday_node.day_id + "_Water", fovday_node.day_id + "_NoWater"]
        all_dataset += [dataset.subtree(fovday_node).select(**rule_pw), dataset.subtree(fovday_node).select(**rule_bn), dataset.subtree(fovday_node).select(**rule_pn)]
        sync_events += [("WaterOn", "NoWaterOn") for _ in range(3)]
        titles += [fovday_node.day_id + "_PuffWater", fovday_node.day_id + "_BlankNoWater", fovday_node.day_id + "_PuffNoWater"]
    datasets_avg_view(datasets=all_dataset, sync_events=sync_events, titles=titles, fluorescence_flag=False, save_name="TrialAvg_{}_water_aligned.png")



    # M003_QPV_5F
 
    for session_node in dataset.select(hash_key="session", mice_id="M003_QPV_5F", coordinate=lambda coordinate:(coordinate.day_id > "20250705")):
        node_flat_view(session_node, fluorescence_flag=False, pupil_flag=False, save_name="Overview_{}.png")

    all_dataset, sync_events, titles = [], [], []
    for fovday_node in tqdm(dataset.select(hash_key="fovday", mice_id="M003_QPV_5F", coordinate=lambda coordinate:(coordinate.day_id > "20250705"))):
        all_dataset += [dataset.subtree(fovday_node).select(**rule_w), dataset.subtree(fovday_node).select(**rule_nw)]
        sync_events += [("BuzzerOn",) for _ in range(2)]
        titles += [fovday_node.day_id + "_Water", fovday_node.day_id + "_NoWater"]
        all_dataset += [dataset.subtree(fovday_node).select(**rule_pw), dataset.subtree(fovday_node).select(**rule_bn)]
        sync_events += [("BuzzerOn",) for _ in range(2)]
        titles += [fovday_node.day_id + "_PuffWater", fovday_node.day_id + "_BlankNoWater"]
    datasets_avg_view(datasets=all_dataset, sync_events=sync_events, titles=titles, fluorescence_flag=False, pupil_flag=False, save_name="TrialAvg_{}_buzzer_aligned.png")


    
    all_dataset, sync_events, titles = [], [], []
    for fovday_node in tqdm(dataset.select(hash_key="fovday", mice_id="M003_QPV_5F", coordinate=lambda coordinate:(coordinate.day_id > "20250705"))):
        all_dataset += [dataset.subtree(fovday_node).select(**rule_w), dataset.subtree(fovday_node).select(**rule_nw)]
        sync_events += [("WaterOn", "NoWaterOn") for _ in range(2)]
        titles += [fovday_node.day_id + "_Water", fovday_node.day_id + "_NoWater"]
        all_dataset += [dataset.subtree(fovday_node).select(**rule_pw), dataset.subtree(fovday_node).select(**rule_bn), ]
        sync_events += [("WaterOn", "NoWaterOn") for _ in range(2)]
        titles += [fovday_node.day_id + "_PuffWater", fovday_node.day_id + "_BlankNoWater"]
    datasets_avg_view(datasets=all_dataset, sync_events=sync_events, titles=titles, fluorescence_flag=False, pupil_flag=False, save_name="TrialAvg_{}_water_aligned.png")


 

if __name__ == "__main__":
    # pre_conversion()
    # whisker_extraction()
    # pupil_extraction()
    main()


