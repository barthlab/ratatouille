from typing import Optional
import logging
import numpy as np

from kitchen.settings.timeline import AUDITORY_EVENTS_DEFAULT, REWARD_EVENTS_DEFAULT, STIMULUS_EVENTS_DEFAULT
from kitchen.settings.trials import DEFAULT_TRIAL_RANGE_FOR_TRIAL_TYPE_ANNOTATION, TRIAL_TYPE_ANNOTATION_WHISKER_RANGE, TRIAL_TYPE_ANNOTATION_WHISKER_THRESHOLD
from kitchen.structure.hierarchical_data_structure import FovTrial, Trial
from kitchen.settings.loaders import LOADER_STRICT_MODE


logger = logging.getLogger(__name__)


def trial_type_annotator(trial_node: Trial | FovTrial, trial_align: float, trial_type_annotator_name: Optional[str] = None):

    def io_default(trial_node: Trial | FovTrial, trial_align: float):
        assert trial_node.data.timeline is not None, f"Cannot find timeline in {trial_node}"

        # zoom into the core trial range
        core_trial_timeline = trial_node.data.timeline.aligned_to(trial_align).segment(*DEFAULT_TRIAL_RANGE_FOR_TRIAL_TYPE_ANNOTATION)

        # Buzzer mark
        if core_trial_timeline.includes("BuzzerOn"):
            buzzer_mark = 1
        elif not core_trial_timeline.includes(AUDITORY_EVENTS_DEFAULT):
            buzzer_mark = 0
        else:
            raise ValueError(f"Cannot determine buzzer mark in {trial_node.coordinate}")
        
        # Stim mark
        if core_trial_timeline.includes(("VerticalPuffOn", "Puff")):
            stim_mark = 1
        elif core_trial_timeline.includes(("BlankOn", "FakeRelayOn", "Blank")):
            stim_mark = -1
        elif core_trial_timeline.includes("PeltierLeftOn"):
            stim_mark = 2
        elif core_trial_timeline.includes("PeltierRightOn"):
            stim_mark = 3
        elif core_trial_timeline.includes("PeltierBothOn"):
            stim_mark = 4
        elif not core_trial_timeline.includes(STIMULUS_EVENTS_DEFAULT):
            stim_mark = 0
        else:
            raise ValueError(f"Cannot determine stim mark in {trial_node.coordinate}")
        
        # Reward mark
        if core_trial_timeline.includes("WaterOn"):
            reward_mark = 1
        elif core_trial_timeline.includes("NoWaterOn"):
            reward_mark = -1
        elif not core_trial_timeline.includes(REWARD_EVENTS_DEFAULT):
            reward_mark = 0
        else:
            raise ValueError(f"Cannot determine reward mark in {trial_node.coordinate}")

        core_marks = (buzzer_mark, stim_mark, reward_mark)
        if core_marks == (0, 0, 1):
            trial_node.info["trial_type"] = "WaterOnly"
        elif core_marks == (0, 0, -1):
            trial_node.info["trial_type"] = "NoWaterOnly"
        elif core_marks == (0, 1, 1):
            trial_node.info["trial_type"] = "PuffWater"
        elif core_marks == (0, 1, -1):
            trial_node.info["trial_type"] = "PuffNoWater"
        elif core_marks == (0, -1, 1):
            trial_node.info["trial_type"] = "BlankWater"
        elif core_marks == (0, -1, -1):
            trial_node.info["trial_type"] = "BlankNoWater"
        elif core_marks == (0, 1, 0):
            trial_node.info["trial_type"] = "PuffOnly"
        elif core_marks == (0, -1, 0):
            trial_node.info["trial_type"] = "BlankOnly"
        elif core_marks == (0, 2, 1):
            trial_node.info["trial_type"] = "PeltierLeftWater"
        elif core_marks == (0, 3, -1):
            trial_node.info["trial_type"] = "PeltierRightNoWater"
        elif core_marks == (0, 4, 1):
            trial_node.info["trial_type"] = "PeltierBothWater"
        elif core_marks == (1, 1, 0):
            trial_node.info["trial_type"] = "CuedPuff"
        elif core_marks == (1, -1, 0):
            trial_node.info["trial_type"] = "CuedBlank"
        elif core_marks == (0, 1, 0):
            trial_node.info["trial_type"] = "UncuedPuff"
        else:
            raise ValueError(f"Cannot determine trial type in {trial_node.timeline.v}, got core marks {core_marks}")

    def io_js_jux(trial_node: Trial | FovTrial, trial_align: float):
        assert trial_node.data.timeline is not None, f"Cannot find timeline in {trial_node}"

        # zoom into the core trial range
        core_trial_timeline = trial_node.data.timeline.aligned_to(trial_align).segment(*DEFAULT_TRIAL_RANGE_FOR_TRIAL_TYPE_ANNOTATION)

        puff_on_time = core_trial_timeline.filter("VerticalPuffOn")
        puff_off_time = core_trial_timeline.filter("VerticalPuffOff")
        assert len(puff_on_time) == 1, f"Expected 1 puff on, got {len(puff_on_time)}"
        assert len(puff_off_time) == 1, f"Expected 1 puff off, got {len(puff_off_time)}"
        puff_duration = puff_off_time.t[0] - puff_on_time.t[0]

        if 0.48 < puff_duration < 0.52:
            trial_node.info["trial_type"] = "500msPuff"
        elif 0.08 < puff_duration < 0.12:
            trial_node.info["trial_type"] = "100msPuff"
        else:
            raise ValueError(f"Cannot determine trial type in {trial_node.timeline.v}, got puff duration {puff_duration}")
        
    def io_stationary_vs_mobile(trial_node: Trial | FovTrial, trial_align: float):
        assert trial_node.data.timeline is not None, f"Cannot find timeline in {trial_node}"
        assert trial_node.data.whisker is not None, f"Cannot find whisker in {trial_node}"

        pre_stim_whisker = trial_node.data.whisker.aligned_to(trial_align).segment(*TRIAL_TYPE_ANNOTATION_WHISKER_RANGE)
        if pre_stim_whisker.v.mean() < TRIAL_TYPE_ANNOTATION_WHISKER_THRESHOLD:
            trial_node.info["trial_type"] = "Stationary"
        else:
            trial_node.info["trial_type"] = "Mobile"
        

    trial_type_annotator_options = {
        "default": io_default,
        "js_jux": io_js_jux,
        "stationary_vs_mobile": io_stationary_vs_mobile,
    }
    if trial_type_annotator_name is None:
        logger.debug("No trial type annotator specified, skip trial type annotation")
        return
    annotator_to_use = trial_type_annotator_options.get(trial_type_annotator_name)
    if annotator_to_use is None:
        raise ValueError(f"Unknown trial type annotator: {trial_type_annotator_name}. Available options: {trial_type_annotator_options.keys()}")
    
    try:
        annotator_to_use(trial_node, trial_align)
    except Exception as e:
        if LOADER_STRICT_MODE:
            raise ValueError(f"Error annotating trial type in {trial_node.coordinate} with {trial_type_annotator_name}: {e}")
        logger.debug(f"Error annotating trial type in {trial_node.coordinate} with {trial_type_annotator_name}: {e}")
