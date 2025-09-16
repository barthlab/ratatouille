from typing import Optional
import logging
import numpy as np

from kitchen.settings.potential import SPIKE_ANNOTATION_EARLY_WINDOW, SPIKE_STICKY_WINDOW
from kitchen.structure.hierarchical_data_structure import FovTrial, Trial
from kitchen.settings.loaders import LOADER_STRICT_MODE

logger = logging.getLogger(__name__)


def spike_annotation(trial_node: Trial | FovTrial, trial_align: float, spike_annotator_name: Optional[str] = None):

    def io_default(trial_node: Trial | FovTrial, trial_align: float):
        assert trial_node.data.potential is not None, f"Cannot find potential in {trial_node}"
        assert trial_node.data.timeline is not None, f"Cannot find timeline in {trial_node}"

        # find puff offset
        puff_offset = trial_node.data.timeline.filter("VerticalPuffOff")
        if len(puff_offset.t) == 0:
            raise ValueError(f"Cannot find puff offset in {trial_node}")
        puff_offset_t = puff_offset.t[0]

        spike_times = trial_node.data.potential.spikes.aligned_to(trial_align).t
        new_spike_v = []
        for spike_id, spike_t in enumerate(spike_times):
            if SPIKE_ANNOTATION_EARLY_WINDOW[0] <= spike_t <= SPIKE_ANNOTATION_EARLY_WINDOW[1]:
                new_spike_v.append("early_spike")
            elif SPIKE_ANNOTATION_EARLY_WINDOW[1] < spike_t <= puff_offset_t:
                new_spike_v.append("sustained_spike")
            elif (spike_id > 0) and (spike_times[spike_id - 1] + SPIKE_STICKY_WINDOW >= spike_t) and (new_spike_v[-1] == "sustained_spike"):
                new_spike_v.append("sustained_spike")
            else:
                new_spike_v.append("regular_spike")
        trial_node.data.potential.spikes.v = np.array(new_spike_v)

    spike_annotator_options = {
        "default": io_default,
    }
    if spike_annotator_name is None:
        logger.debug("No spike annotator specified, skip spike annotation")
        return
    annotator_to_use = spike_annotator_options.get(spike_annotator_name)
    if annotator_to_use is None:
        raise ValueError(f"Unknown spike annotator: {spike_annotator_name}. Available options: {spike_annotator_options.keys()}")
    
    try:
        annotator_to_use(trial_node, trial_align)
    except Exception as e:
        if LOADER_STRICT_MODE:
            raise ValueError(f"Error annotating spike in {trial_node.coordinate} with {spike_annotator_name}: {e}")
        logger.debug(f"Error annotating spike in {trial_node.coordinate} with {spike_annotator_name}: {e}")
