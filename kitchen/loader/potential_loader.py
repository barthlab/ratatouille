import pandas as pd
import numpy as np
from typing import Generator, Tuple
import logging

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
            
          
    """Load timeline from fov node."""
    potential_loader_options = {
        "default": io_default,
    }
    default_cohort_data_path = routing.default_data_path(cohort_node)
    
    if potential_loader_name is None:
        logger.info("No potential loader specified, skip loading potential")
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
