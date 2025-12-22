from typing import Callable, Optional
import logging

from kitchen.structure.hierarchical_data_structure import FovTrial, Trial
from kitchen.settings.loaders import LOADER_STRICT_MODE


logger = logging.getLogger(__name__)


def trial_type_history_annotator(list_trial_nodes: list[Trial] | list[FovTrial], trial_type_history_annotator_name: Optional[str] = None):

    def io_default(list_trial_nodes: list[Trial] | list[FovTrial]):
        all_trial_types = [x.info.get("trial_type", None) for x in list_trial_nodes]
        
        for idx, trial_node in enumerate(list_trial_nodes):
            trial_node.info["prev_trial_types"] = all_trial_types[idx::-1]
            trial_node.info["post_trial_types"] = all_trial_types[idx:]


    trial_type_history_annotator_options = {
        "default": io_default,
    }
    if trial_type_history_annotator_name is None:
        logger.debug("No trial type history annotator specified, skip trial type history annotation")
        return
    annotator_to_use = trial_type_history_annotator_options.get(trial_type_history_annotator_name)
    if annotator_to_use is None:
        raise ValueError(f"Unknown trial type history annotator: {trial_type_history_annotator_name}. Available options: {trial_type_history_annotator_options.keys()}")
    
    try:
        annotator_to_use(list_trial_nodes)
    except Exception as e:
        if LOADER_STRICT_MODE:
            raise ValueError(f"Error annotating trial type history with {trial_type_history_annotator_name}: {e}")
        logger.debug(f"Error annotating trial type history with {trial_type_history_annotator_name}: {e}")




def meta_annotator_A_after_B(current_trial_type: str, prev_trial_type: str) -> Callable:
    def verifier(x: Trial | FovTrial):
        is_current_trial_type = x.info.get("trial_type") == current_trial_type
        after_prev_trial_type = (len(x.info.get("prev_trial_types", [])) > 1) and (x.info.get("prev_trial_types", [])[1] == prev_trial_type)
        return is_current_trial_type and after_prev_trial_type
    return verifier