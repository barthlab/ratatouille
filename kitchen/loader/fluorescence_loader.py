import os.path as path
from typing import Dict, Generator, Optional
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat


from kitchen.configs import routing
from kitchen.settings.fluorescence import CROP_BEGGINNG, DEFAULT_RECORDING_DURATION, FAST_MATCHING_MODE, NULL_TTL_OFFSET
from kitchen.settings.loaders import DATA_HODGEPODGE_MODE
from kitchen.settings.timeline import TTL_EVENT_DEFAULT
from kitchen.structure.hierarchical_data_structure import Fov
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate
from kitchen.structure.neural_data_structure import Fluorescence, TimeSeries, Timeline
from kitchen.utils.sequence_kit import find_only_one


def fluorescence_loader_from_fov(
        fov_node: Fov, timeline_dict: Dict[TemporalObjectCoordinate, Timeline]) -> Generator[Optional[Fluorescence], None, None]:

    def load_fall_mat(data_dir: str, session_duration: float) -> Generator[Fluorescence, None, None]:    
        """load fall mat file"""
        if DATA_HODGEPODGE_MODE:
            mat_path = find_only_one(routing.search_pattern_file(pattern="Fall.mat", search_dir=data_dir))
        else:
            mat_path = path.join(data_dir, "soma", "Fall.mat") 
        mat_dict = loadmat(mat_path)

        """get raw fluorescence"""
        F = mat_dict['F'].copy()
        Fneu = mat_dict['Fneu'].copy()
        is_cell = np.argwhere(mat_dict['iscell'][:, 0] == 1)[:, 0]
        raw_f = F[is_cell] - 0.7 * Fneu[is_cell]
        
        """get fov motion"""
        ops = mat_dict['ops'][0][0]
        fov_motion = np.concatenate([ops['xoff'], ops['yoff']], axis=0)

        """get cell position"""
        stat = mat_dict['stat']
        cell_position = np.stack([stat[0, cell_id][0, 0][6][0] for cell_id in is_cell], axis=0)

        """split into sessions"""
        num_cells, num_total_frames = raw_f.shape
        assert num_total_frames % n_session == 0, f"Cannot split {num_total_frames} frames into {n_session} sessions"
        num_frames_per_session = int(num_total_frames // n_session)

        raw_f = np.reshape(raw_f, (num_cells, num_frames_per_session, n_session), order='F')
        fov_motion = np.reshape(fov_motion, (2, num_frames_per_session, n_session), order='F')
        t = np.linspace(CROP_BEGGINNG, session_duration, num_frames_per_session)
        
        for session_id in range(n_session):
            yield Fluorescence(
                raw_f=TimeSeries(v=raw_f[..., session_id], t=t.copy()),
                fov_motion=TimeSeries(v=fov_motion[..., session_id], t=t.copy()),
                cell_position=cell_position,
                cell_idx=is_cell
            )

    def io_default(dir_path: str) -> Generator[Fluorescence, None, None]:
        ttl_file_path = path.join(dir_path, "ttl.xlsx")
        assert path.exists(ttl_file_path), f"Cannot find ttl file: {ttl_file_path}"

        """ calculate ttl to timeline offset """
        ttl_array = pd.ExcelFile(ttl_file_path)
        assert len(ttl_array.sheet_names) == n_session, f"Cannot find {n_session} sheets in {ttl_file_path}"
        for sheet_name, (session_coordinate, timeline), fluorescence in zip(
            ttl_array.sheet_names, 
            timeline_dict.items(), 
            load_fall_mat(dir_path, DEFAULT_RECORDING_DURATION)
        ):       
            """load ttl and align to timeline"""
            df = ttl_array.parse(sheet_name, header=0).to_numpy()
            assert df.shape[1] == 2, f"Cannot find 2 columns in {ttl_file_path} {sheet_name}"     
            ttl_t = df[:, 0]/1000

            # alignment happens here
            timeline_t = timeline.filter(TTL_EVENT_DEFAULT)

            """match ttl and timeline"""
            if FAST_MATCHING_MODE:
                ttl_to_timeline_offset = ttl_t[0] - timeline_t.t[0]
            else:
                assert len(ttl_t) == len(timeline_t.t), f"Cannot match ttl and timeline in {sheet_name}"
                ttl_to_timeline_offsets = ttl_t - timeline_t.t
                assert np.allclose(ttl_to_timeline_offsets, ttl_to_timeline_offsets[0]), \
                    f"Cannot match ttl and timeline in {sheet_name}"
                ttl_to_timeline_offset = ttl_to_timeline_offsets[0]
            
            """align fluorescence to timeline"""
            fluorescence.raw_f.t -= ttl_to_timeline_offset
            fluorescence.fov_motion.t -= ttl_to_timeline_offset
            yield fluorescence    

    def io_lost_ttl(dir_path: str) -> Generator[Fluorescence, None, None]:
        for (session_coordinate, timeline), fluorescence in zip(
            timeline_dict.items(), 
            load_fall_mat(dir_path, DEFAULT_RECORDING_DURATION)
        ):            
            """align fluorescence to timeline"""
            task_start, _ = timeline.task_time()
            ttl_to_timeline_offset = NULL_TTL_OFFSET - task_start  # align the fluorescence to the timeline by a fixed offset
            fluorescence.raw_f.t -= ttl_to_timeline_offset
            fluorescence.fov_motion.t -= ttl_to_timeline_offset
            yield fluorescence

    default_fov_data_path = routing.default_data_path(fov_node)
    n_session = len(timeline_dict)

    """Load fluorescence from fov node."""
    success_flag = False
    for io_func in [io_default, io_lost_ttl]:
        try:
            yield from io_func(default_fov_data_path)
            success_flag = True            
            break
        except Exception as e:
            warnings.warn(f"Cannot load fluorescence from {default_fov_data_path} via {io_func.__name__}: {e}")
    if not success_flag:
        for _ in range(n_session):
            yield None
