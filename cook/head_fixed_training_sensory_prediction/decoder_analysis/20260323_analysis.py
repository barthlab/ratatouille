import os.path as path
import logging
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from kitchen.loader.general_loader_interface import load_dataset
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import Fov, Session, Trial
from kitchen.operator.packing import pack_neural_data
from kitchen.operator.sync_nodes import sync_nodes
from snippet_type_analysis import analyze_snippet_types
from plot_snippet_type_analysis import plot_analysis_dashboard


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler(path.join(path.dirname(__file__), "overview.log"), mode="w")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:\n%(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING) 



def _make_snippet(trial_node: Trial, ref_time: np.ndarray) -> Tuple[np.ndarray, int]:
    array_data = pack_neural_data(trial_node.data, 
        has_locomotion=True,
        has_whisker=True,
        has_pupil_area=True,
        has_pupil_center_x=True,
        has_pupil_center_y=True,
        has_saccade_velocity=True,
        _predefined_t=ref_time,
        interp_method="linear"
    ).v

    trial_type = trial_node.info.get("trial_type")
    if trial_type == "CueOnly":
        label = 1
    elif trial_type == "CuedBlueLED":
        label = 0
    else:
        raise ValueError(f"Unknown trial type {trial_type}")
    if np.any(np.isnan(array_data)):
        raise ValueError(f"NaN found in snippet {trial_node.coordinate}: {np.argwhere(np.isnan(array_data))}")
    return (array_data, label)
    
def main():
    dataset = load_dataset(template_id="HeadFixedTraining_Behavior", cohort_id="2026_03", 
                           recipe="default_behavior_only_long_trial", name="sensory_prediction")
    dataset.status(save_path=path.join(path.dirname(__file__), "status_report.xlsx"))
    snippets = []
    labels = []
    all_trial_nodes = dataset.select(hash_key="trial", _self=lambda x: x.info.get("trial_type") in ("CueOnly", "CuedBlueLED",), mice_id="SUI_4F")
    all_trial_nodes = sync_nodes(all_trial_nodes, ("TrialOn",), PlotManual())
    snippet_time = np.linspace(-10, 10, 600)
    for trial_node in all_trial_nodes.nodes:
        snippet, label = _make_snippet(trial_node, snippet_time)
        snippets.append(snippet)
        labels.append(label)

    results = analyze_snippet_types(snippets, labels, random_state=0)
    plot_analysis_dashboard(snippets, labels, results)
    plt.show()
    

if __name__ == "__main__":
    main()