import os.path as path
from typing import Any, Dict, Generator, Optional
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from kitchen.configs import routing
from kitchen.settings.behavior import LICK_INACTIVATE_WINDOW, LOCOMOTION_CIRCUMFERENCE, LOCOMOTION_NUM_TICKS, VIDEO_EXTRACTED_BEHAVIOR_MIN_MAX_PERCENTILE, VIDEO_EXTRACTED_BEHAVIOR_TYPES
from kitchen.settings.loaders import DATA_HODGEPODGE_MODE, SPECIFIED_BEHAVIOR_LOADER, io_enumerator
from kitchen.structure.hierarchical_data_structure import Node
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate
from kitchen.structure.neural_data_structure import Events, TimeSeries, Timeline
from kitchen.utils.numpy_kit import smooth_uniform
from kitchen.utils.sequence_kit import find_only_one



def behavior_loader_from_node(
        node: Node,
        timeline_dict: Dict[TemporalObjectCoordinate, Timeline]) -> \
            Generator[Dict[str, Any], None, None]:
    """
    Load behavioral data (lick, locomotion, pupil, etc.) for all sessions within a Node.
    
    All custom io function should follow the same strucutre:
    1. For each session, load all behavioral data.
    2. Yield a dictionary containing all behavioral data for the session.
    """
    
    def io_default(dir_path: str) -> Generator[Dict[str, Optional[TimeSeries | Events]], None, None]:
        for session_coordinate, timeline in timeline_dict.items():
            session_behavior = {}

            """load lick"""
            try:
                if DATA_HODGEPODGE_MODE:
                    lick_path = find_only_one(routing.search_pattern_file(
                        pattern=f"LICK_{session_coordinate.temporal_uid.session_id}.csv", search_dir=dir_path))
                else:
                    lick_path = path.join(dir_path, "lick", f"LICK_{session_coordinate.temporal_uid.session_id}.csv")                
        
                # check if lick file exists
                assert path.exists(lick_path), f"Cannot find lick path: {lick_path}"
                lick_data = pd.read_csv(lick_path, header=0).to_numpy()
                assert lick_data.shape[1] == 1, f"Cannot find 1 column in {lick_path}"

                # create lick events
                lick_events = Events(
                    v=np.ones_like(lick_data[:, 0], dtype=np.float32), 
                    t=np.array(lick_data[:, 0] / 1000, dtype=np.float32)).inactivation_window_filter(LICK_INACTIVATE_WINDOW)
                session_behavior["lick"] = lick_events
            except Exception as e:
                logger.warning(f"Cannot load lick from {dir_path}: {e}")

            """load locomotion"""                
            try:
                if DATA_HODGEPODGE_MODE:
                    locomotion_path = find_only_one(routing.search_pattern_file(
                        pattern=f"LOCOMOTION_{session_coordinate.temporal_uid.session_id}.csv", search_dir=dir_path))
                else:
                    locomotion_path = path.join(dir_path, "locomotion",
                                                f"LOCOMOTION_{session_coordinate.temporal_uid.session_id}.csv")
                    
                # check if locomotion file exists
                assert path.exists(locomotion_path), f"Cannot find locomotion path: {locomotion_path}"
                locomotion_data = pd.read_csv(locomotion_path, header=0).to_numpy()
                assert locomotion_data.shape[1] == 3, f"Cannot find 3 column in {locomotion_path}"

                # calculate delta distance            
                positions = np.array(locomotion_data[:, 1], dtype=np.float32)
                times = np.array(locomotion_data[:, 0] / 1000, dtype=np.float32)
                delta_dist = (positions[1:] - positions[:-1]) * LOCOMOTION_CIRCUMFERENCE / LOCOMOTION_NUM_TICKS

                # create position and locomotion events
                position_events = Events(v=(positions % LOCOMOTION_NUM_TICKS) / LOCOMOTION_NUM_TICKS, t=times)     
                locomotion_events = Events(v=delta_dist, t=times[1:])     
                session_behavior["position"] = position_events
                session_behavior["locomotion"] = locomotion_events
            except Exception as e:
                logger.warning(f"Cannot load locomotion from {dir_path}: {e}")

            """load video extracted behavior"""
            for behavior_type in VIDEO_EXTRACTED_BEHAVIOR_TYPES:                
                try:                    
                    if DATA_HODGEPODGE_MODE:
                        behavior_path = find_only_one(routing.search_pattern_file(
                            pattern=f"{behavior_type}_{session_coordinate.temporal_uid.session_id}.csv", search_dir=dir_path))
                    else:
                        behavior_path = path.join(dir_path, behavior_type.lower(),
                                                  f"{behavior_type}_{session_coordinate.temporal_uid.session_id}.csv")
                    
                    # check if behavior file exists
                    task_start, task_end = timeline.task_time()
                    assert path.exists(behavior_path), f"Cannot find {behavior_type} path: {behavior_path}"
                    behavior_data = pd.read_csv(behavior_path, header=0).to_numpy()
                    assert behavior_data.shape[1] == 2, f"Cannot find 2 column in {behavior_path}"

                    # create behavior timeseries
                    behavior_values = np.array(behavior_data[:, 1], dtype=np.float32)
                    down_value, up_value = np.nanpercentile(behavior_values, VIDEO_EXTRACTED_BEHAVIOR_MIN_MAX_PERCENTILE)
                    normalized_values = np.clip((behavior_values - down_value) / (up_value - down_value), 0, 1)
                    video_time = np.linspace(task_start, task_end, len(behavior_values), dtype=np.float32)
                    behavior_timeseries = TimeSeries(v=normalized_values, t=video_time)
                    session_behavior[behavior_type.lower()] = behavior_timeseries
                except Exception as e:
                    logger.warning(f"Cannot load {behavior_type} from {dir_path}: {e}")

            yield session_behavior
 
    def io_video_only(dir_path: str) -> Generator[Dict[str, Optional[TimeSeries | Events]], None, None]:
        for session_coordinate, timeline in timeline_dict.items():
            session_behavior = {}

            """load video extracted behavior"""
            for behavior_type in VIDEO_EXTRACTED_BEHAVIOR_TYPES:                
                try:                    
                    if DATA_HODGEPODGE_MODE:
                        behavior_path = find_only_one(routing.search_pattern_file(
                            pattern=f"{behavior_type}_{session_coordinate.temporal_uid.session_id}.csv", search_dir=dir_path))
                    else:
                        behavior_path = path.join(dir_path, behavior_type.lower(),
                                                  f"{behavior_type}_{session_coordinate.temporal_uid.session_id}.csv")
                    
                    # check if behavior file exists               
                    assert path.exists(behavior_path), f"Cannot find {behavior_type} path: {behavior_path}"
                    behavior_data = pd.read_csv(behavior_path, header=0).to_numpy()
                    assert behavior_data.shape[1] == 2, f"Cannot find 2 column in {behavior_path}"

                    # create behavior timeseries
                    behavior_values = np.array(behavior_data[:, 1], dtype=np.float32)

                    ## Test
                    behavior_values = smooth_uniform(behavior_values, window_len=5)

                    down_value, up_value = np.nanpercentile(behavior_values, VIDEO_EXTRACTED_BEHAVIOR_MIN_MAX_PERCENTILE)
                    normalized_values = np.clip((behavior_values - down_value) / (up_value - down_value), 0, 1)               
                    video_time = timeline.t[:len(behavior_values)]
                    behavior_timeseries = TimeSeries(v=normalized_values, t=video_time)
                    session_behavior[behavior_type.lower()] = behavior_timeseries
                except Exception as e:
                    logger.warning(f"Cannot load {behavior_type} from {dir_path}: {e}")

            yield session_behavior
 
    default_data_path = routing.default_data_path(node)

    """Load behavior from node."""
    yield from io_enumerator(default_data_path, 
                             [io_default, io_video_only ], 
                             SPECIFIED_BEHAVIOR_LOADER)
