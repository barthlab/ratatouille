import os
import pandas as pd
import numpy as np
from typing import Generator, Tuple

from kitchen.configs import routing
from kitchen.settings.loaders import SPECIFIED_POTENTIAL_LOADER, io_enumerator
from kitchen.settings.potential import JS_AIRPUFF_THRESHOLD
from kitchen.structure.neural_data_structure import Events, Potential, TimeSeries, Timeline
from kitchen.structure.hierarchical_data_structure import Cohort


def potential_loader_from_cohort(cohort_node: Cohort) -> \
    Generator[Tuple[str, str, str, Timeline, Potential], None, None]:
    """
    Load timeline and potential data from a cohort node.
    
    Args:
        cohort_node: Cohort node containing the path information
        
    Yields:
        Tuple[cell_name, mice_name, day_name, timeline, potential]
    """
    
    def io_default(dir_path: str) -> Generator[Tuple[str, str, str, Timeline, Potential], None, None]:
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
            day_name, cell_name, _ = filename.split('_')

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

            """Extract potential"""
            potential = Potential(
                vm=TimeSeries(v=data.loc["Vm1", 0], t=data.loc["vm_time", 0]).segment(start_exp_t, end_exp_t),
                is_prime=True
            )

            yield cell_name, mice_name, day_name, timeline, potential
            
          
    
    """Load timeline from fov node."""
    default_cohort_data_path = routing.default_data_path(cohort_node)
    yield from io_enumerator(default_cohort_data_path, 
                             [io_default,], 
                             SPECIFIED_POTENTIAL_LOADER, strict_mode=True)
