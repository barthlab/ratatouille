import os.path as path
import logging

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.operator.split import split_dataset_by_trial_type
from kitchen.plotter.macros.behavior_macros import trace_view_delta_behavior_macro
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



def main():
    dataset = load_dataset(template_id="HeadFixedTraining", cohort_id="SensoryPrediction_202510-11", 
                           recipe="default_behavior_only", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))

    trace_view_delta_behavior_macro(
        dataset, "pupil", 
        left_day_trial_types={"CuedPuff"}, 
        right_day_trial_types={"CuedPuff", "CuedBlank"},

        range_compare=(4., 8.),
        range_baseline=(0, 1.7),

        prefix_keyword="CuedPuff_abs_CuedBlank",
        _right_day_cover="Green",

        yaxis_formattor=lambda x, _: f'{x * 1000:.0f}',
        yaxis_label=r"$\Delta$ Pupil Area [$px^2$]"
    )
    # exit()
    trace_view_delta_behavior_macro(
        dataset, "pupil", 
        left_day_trial_types={"CuedPuff"}, 
        right_day_trial_types={"CuedPuff", "CueOnly"},

        range_compare=(4., 8.),
        range_baseline=(0, 1.7),

        prefix_keyword="CuedPuff_abs_CueOnly",
        _right_day_cover="blue",

        yaxis_formattor=lambda x, _: f'{x * 1000:.0f}',
        yaxis_label=r"$\Delta$ Pupil Area [$px^2$]"
    )

    trace_view_delta_behavior_macro(
        dataset, "pupil", 
        left_day_trial_types={"CuedPuff"}, 
        right_day_trial_types={"CuedPuff", "PuffOnly"},

        range_compare=(4., 8.),
        range_baseline=(0, 1.7),

        prefix_keyword="CuedPuff_abs_PuffOnly",
        _right_day_cover="Red",

        yaxis_formattor=lambda x, _: f'{x * 1000:.0f}',
        yaxis_label=r"$\Delta$ Pupil Area [$px^2$]"
    )

if __name__ == "__main__":
    main()