import os.path as path
import logging

from kitchen.annotator.trial_type_history_annotator import meta_annotator_A_after_B
from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.plotter.macros.behavior_macros import trace_view_delta_behavior_macro
from kitchen.plotter.macros.sensory_prediction_macros.behavior_metric_summary import sensory_prediction_summary_behavior_macro
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import DataSet, Mice

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING) 


pupil_setting = {
    "range_baseline": (-2, 0),
    "range_compare": (2.5, 7.5),
    "baseline_setup": (-2, 0, False),
    "amplitude_setup": (2.5, 7.5, "day"),
    "yaxis_formattor": lambda x, _: f'{x * 1000:.0f}',
    "yaxis_label": r"$\Delta$ Pupil Area [$px^2$]",
    "yaxis_lim": (-0.02, 0.08, -0.03, 0.14),
}

whisker_setting = {
    "range_baseline": (-2, 0),
    "range_compare": (2.5, 7.5),
    "baseline_setup": None,
    "amplitude_setup": (2.5, 7.5, "day"),
    "yaxis_formattor": lambda x, _: x,
    "yaxis_label": r"$\Delta$ Whisker [a.u.]",
    "yaxis_lim": (-0.5, 0.5),
}

locomotion_setting = {
    "range_baseline": (-2, 0),
    "range_compare": (2.5, 7.5),
    "baseline_setup": None,
    "amplitude_setup": (2.5, 7.5, "day"),
    "yaxis_formattor": lambda x, _: x,
    "yaxis_label": r"$\Delta$ Locomotion [cm/s]",
    "yaxis_lim": (-0.05, 0.05),
}

saccade_setting = {
    "range_baseline": (-2, 0),
    "range_compare": (2.5, 7.5),
    "baseline_setup": None,
    "amplitude_setup": (2.5, 7.5, "day"),
    "yaxis_formattor": lambda x, _: x,
    "yaxis_label": r"$\Delta$ Saccade Velocity [px/s]",
    "yaxis_lim": (-0.2, 0.15),
}




strong_puff_query = {
    "left_day_trial_types": {"CuedPuff"}, 
    "right_day_trial_types": {"CuedPuff", "CueOnly"},
    "_additional_trial_types": {
        "CuedPuff Subseq.": meta_annotator_A_after_B("CuedPuff", "CueOnly"),
    },
    "prefix_keyword": "StrongPuff",
    "_right_day_cover": "olive",
}

BlueLED_query = {
    "left_day_trial_types": {"CuedBlueLED"}, 
    "right_day_trial_types": {"CuedBlueLED", "CueOnly"},
    "_additional_trial_types": {
        "CuedBlueLED Subseq.": meta_annotator_A_after_B("CuedBlueLED", "CueOnly"),
    },
    "prefix_keyword": "BlueLED",
    "_right_day_cover": "cyan",
}

pure_omission_query = {
    "left_day_trial_types": {"CuedPuff"}, 
    "right_day_trial_types": {"CuedPuff", "CueOnly"},
    "_additional_trial_types": {
        "CuedPuff Subseq.": meta_annotator_A_after_B("CuedPuff", "CueOnly"),
    },
    "prefix_keyword": "PureOmission",
    "_right_day_cover": "green",
}

BlankStim_query = {
    "left_day_trial_types": {"CuedPuff"}, 
    "right_day_trial_types": {"CuedPuff", "CuedBlank"},
    "_additional_trial_types": {
        "CuedPuff Subseq.": meta_annotator_A_after_B("CuedPuff", "CuedBlank"),
    },
    "prefix_keyword": "BlankStim",
    "_right_day_cover": "blue",
}

UnpredStim_query = {
    "left_day_trial_types": {"CuedPuff"}, 
    "right_day_trial_types": {"CuedPuff", "PuffOnly"},
    "_additional_trial_types": {
        "CuedPuff Subseq.": meta_annotator_A_after_B("CuedPuff", "PuffOnly"),
    },
    "prefix_keyword": "UnpredStim",
    "_right_day_cover": "orange",
}

DoublePuff_query = {
    "left_day_trial_types": {"DoublePuff"}, 
    "right_day_trial_types": {"DoublePuff", "PuffOnly"},
    "_additional_trial_types": {
        "DoublePuff Subseq.": meta_annotator_A_after_B("DoublePuff", "PuffOnly"),
    },
    "prefix_keyword": "DoublePuff",
    "_right_day_cover": "red",
}

PuffCue_query = {
    "left_day_trial_types": {"PuffCue"}, 
    "right_day_trial_types": {"PuffCue", "PuffOnly"},
    "_additional_trial_types": {
        "PuffCue Subseq.": meta_annotator_A_after_B("PuffCue", "PuffOnly"),
    },
    "prefix_keyword": "PuffCue",
    "_right_day_cover": "purple",
}


def main1():
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202512", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    
    for modality_name, setting in {
        "pupil": pupil_setting,
        # "whisker": whisker_setting,
        # "locomotion": locomotion_setting,
        # "saccade": saccade_setting,
    }.items():
        sensory_prediction_summary_behavior_macro(
            dataset, modality_name, 
            **strong_puff_query,
            **setting,
        )
        sensory_prediction_summary_behavior_macro(
            dataset, modality_name, 
            **BlueLED_query,
            **setting,
        )


def main2():
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202510-11", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    
    for modality_name, setting in {
        "pupil": pupil_setting,
        # "whisker": whisker_setting,
        # "locomotion": locomotion_setting,
        # "saccade": saccade_setting,
    }.items():
        for query in [
            pure_omission_query,
            BlankStim_query,
            UnpredStim_query,
            DoublePuff_query,
            PuffCue_query,
        ]:
            sensory_prediction_summary_behavior_macro(
                dataset.subset(mice_id=lambda mice_id: mice_id not in ("QYV5M", "SCE6F")), modality_name, 
                **query,
                **setting,
            )


def main3():
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202513", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    
    for modality_name, setting in {
        "pupil": pupil_setting,
        # "whisker": whisker_setting,
        # "locomotion": locomotion_setting,
        # "saccade": saccade_setting,
    }.items():
        # sensory_prediction_summary_behavior_macro(
        #     dataset, modality_name, 
        #     **strong_puff_query,
        #     **setting,
        # )
        sensory_prediction_summary_behavior_macro(
            dataset.subset(session_id=lambda session_id: (session_id is None) or ("WRCL" not in session_id)), modality_name, 
            **BlueLED_query,
            **setting,
        )
if __name__ == "__main__":
    main1()
    main2()
    main3()