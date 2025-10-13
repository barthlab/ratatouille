from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Generator, Tuple
import logging
import pyabf
import os
import os.path as path

from kitchen.configs import routing
from kitchen.settings.loaders import LOADER_STRICT_MODE
from kitchen.settings.potential import JS_AIRPUFF_THRESHOLD, JS_CAM_THRESHOLD
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate
from kitchen.structure.neural_data_structure import Potential, TimeSeries, Timeline
from kitchen.structure.hierarchical_data_structure import Cohort


logger = logging.getLogger(__name__)


def potential_loader_from_cohort(cohort_node: Cohort, potential_loader_name: str = "default") -> \
    Generator[Tuple[TemporalObjectCoordinate, Timeline, Timeline, Potential], None, None]:
    """
    Load timeline and potential data from a cohort node.
    
    Args:
        cohort_node: Cohort node containing the path information
        
    Yields:
        Tuple[cell_coordinate, timeline, cam_timeline, potential]
    """
    
    def io_default(dir_path: str) -> Generator[Tuple[TemporalObjectCoordinate, Timeline, Timeline, Potential], None, None]:
        """Find all cell data pickle files and extract potential/timeline data."""
        
        # Search for all pickle files with the pattern *_datafile.pkl
        cell_data_filepaths = routing.search_pattern_file(
            pattern="*_datafile.pkl",
            search_dir=dir_path
        )
        for filepath in cell_data_filepaths:
            # Load pickle file
            with open(filepath, 'rb') as f:
                data = pd.read_pickle(f) 
                
            # Extract mice_name, day_name and cell_name
            mice_name = data.loc["mouse", 0]
            filename = data.loc["filename", 0]
            day_name, cell_name, session_name = filename.split('_')

            """Extract timeline"""
            # Stim information
            start_exp_t = float(data.loc["start_exp", 0])
            end_exp_t = float(data.loc["end_exp", 0])
            start_end_events = Timeline(
                v=np.array(["task start", "task end"]),
                t=np.array([start_exp_t, end_exp_t], dtype=np.float32)
            )
            stimulus_timeseries = TimeSeries(v=data.loc["airpuff", 0], t=data.loc["stim_time", 0]).segment(start_exp_t, end_exp_t)
            stimulus_events = stimulus_timeseries.threshold(JS_AIRPUFF_THRESHOLD, "VerticalPuffOn", "VerticalPuffOff").to_timeline()
            
            # composite timeline
            timeline = start_end_events + stimulus_events
            
            # camera timeline
            cam_timeseries = TimeSeries(v=data.loc["cam", 0], t=data.loc["stim_time", 0]).segment(start_exp_t, end_exp_t)
            cam_timeline = cam_timeseries.threshold(JS_CAM_THRESHOLD, "Frame", None).to_timeline()

            """Extract potential"""
            potential = Potential.create_master(
                vm=TimeSeries(v=data.loc["Vm1", 0], t=data.loc["vm_time", 0]).segment(start_exp_t, end_exp_t)
            )
            
            cell_coordinate = TemporalObjectCoordinate(
                temporal_uid=cohort_node.coordinate.temporal_uid.child_uid(day_id=day_name, session_id=filename),
                object_uid=cohort_node.coordinate.object_uid.child_uid(mice_id=mice_name, fov_id="only", cell_id=cell_name)
            )

            yield cell_coordinate, timeline, cam_timeline, potential
            
    
    def io_abf(dir_path: str) -> Generator[Tuple[TemporalObjectCoordinate, Timeline, Timeline, Potential], None, None]:
        """Find all cell data .abf sweep files and extract potential/timeline data."""
        JOE_OPTO_START = 2.0  # s
        JOE_PULSE_DURATION = 5/1000  # s
        JOE_PULSE_INTERVAL = 50/1000  # s
        JOE_NUM_PULSES = 5

        JOE_TRIAL_OFFSET = 5  # s
        
        # Search for all abf files with the pattern *.abf
        cell_data_filepaths = routing.search_pattern_file(
            pattern="*.abf",
            search_dir=dir_path
        )
        # load each single sweep
        coord_abf_map = defaultdict(dict)
        for filepath in cell_data_filepaths:
            abf = pyabf.ABF(filepath)
            dir_name, filename = path.basename(path.dirname(filepath)), path.basename(filepath)
            assert dir_name.startswith("JC") and dir_name.endswith(" ABF"), \
                f"Expected directory name to start with 'JC' and end with ' ABF', but got {dir_name}"
            day_name = dir_name[2:-5]
            mice_name = dir_name[:-5]
            cell_id = dir_name[2:-4]
            assert filename.startswith("JC"+cell_id) and filename.endswith(".ABF"), \
                f"Expected filename to start with 'JC{cell_id}' and end with '.ABF', but got {filename}"
            trial_id = int(filename.split("sweep")[1].split(".")[0])
            cell_coordinate = TemporalObjectCoordinate(
                temporal_uid=cohort_node.coordinate.temporal_uid.child_uid(day_id=day_name, session_id="JC"+cell_id),
                object_uid=cohort_node.coordinate.object_uid.child_uid(mice_id=mice_name, fov_id="only", cell_id=cell_id)
            )
            abf.setSweep(0)
            potential = Potential.create_master(
                vm=TimeSeries(v=abf.sweepY, t=abf.sweepX)
            )
            timeline_v = ["TrialOn", ] + ["StimOn", "StimOff"] * JOE_NUM_PULSES
            timeline_t = [JOE_OPTO_START, ] + [JOE_OPTO_START + i * (JOE_PULSE_DURATION + JOE_PULSE_INTERVAL) + j * JOE_PULSE_DURATION
                                               for i in range(JOE_NUM_PULSES) for j in range(2)]
            timeline = Timeline(v=np.array(timeline_v), t=np.array(timeline_t, dtype=np.float32))
            coord_abf_map[cell_coordinate][trial_id] = (timeline, potential)
        
        # concatenate sweeps into session
        for cell_coordinate, trial_id_timeline_potential_dict in coord_abf_map.items():
            sorted_trial_ids = sorted(trial_id_timeline_potential_dict.keys())
            concatenated_timeline = Timeline(
                v=np.concatenate([trial_id_timeline_potential_dict[trial_id][0].v for trial_id in sorted_trial_ids]),
                t=np.concatenate([trial_id_timeline_potential_dict[trial_id][0].t + each_trial_id * JOE_TRIAL_OFFSET 
                                  for each_trial_id, trial_id in enumerate(sorted_trial_ids)])
            )
            concatenated_potential = Potential.create_master(
                vm=TimeSeries(v=np.concatenate([trial_id_timeline_potential_dict[trial_id][1].vm.v for trial_id in sorted_trial_ids]),
                              t=np.concatenate([trial_id_timeline_potential_dict[trial_id][1].vm.t + each_trial_id * JOE_TRIAL_OFFSET 
                                                for each_trial_id, trial_id in enumerate(sorted_trial_ids)]))
            )
            yield cell_coordinate, concatenated_timeline, concatenated_timeline, concatenated_potential

    """Load timeline from fov node."""
    potential_loader_options = {
        "default": io_default,
        "abf": io_abf,
    }
    default_cohort_data_path = routing.default_data_path(cohort_node)
    
    if potential_loader_name is None:
        logger.debug("No potential loader specified, skip loading potential")
        return
    loader_to_use = potential_loader_options.get(potential_loader_name)
    if loader_to_use is None:
        raise ValueError(f"Unknown potential loader: {potential_loader_name}. Available options: {potential_loader_options.keys()}")
    
    try:
        yield from loader_to_use(default_cohort_data_path)
    except Exception as e:
        if LOADER_STRICT_MODE:
            raise ValueError(f"Error loading potential in {default_cohort_data_path} with {potential_loader_name}: {e}")
        logger.debug(f"Error loading potential in {default_cohort_data_path} with {potential_loader_name}: {e}")
