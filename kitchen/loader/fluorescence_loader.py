import os.path as path
from typing import Dict, Generator, Optional
import numpy as np
import pandas as pd
from scipy.io import loadmat
import logging

from kitchen.configs import routing
from kitchen.settings.fluorescence import CROP_BEGGINNG, DEFAULT_RECORDING_DURATION, FAST_MATCHING_MODE, NULL_TTL_OFFSET, NEUROPIL_SUBTRACTION_COEFFICIENT, DEFAULT_FRAMES_PER_SESSION
from kitchen.settings.loaders import DATA_HODGEPODGE_MODE, LOADER_STRICT_MODE
from kitchen.settings.timeline import TTL_EVENT_DEFAULT
from kitchen.structure.hierarchical_data_structure import Node
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate
from kitchen.structure.neural_data_structure import Fluorescence, TimeSeries, Timeline
from kitchen.utils.sequence_kit import find_only_one

logger = logging.getLogger(__name__)


def fluorescence_loader_from_node(
        node: Node, timeline_dict: Dict[TemporalObjectCoordinate, Timeline], fluorescence_loader_name: Optional[str] = None) \
            -> Generator[Optional[Fluorescence], None, None]:

    def load_fall_mat(data_dir: str, session_duration: float,
                      n_session: Optional[int] = None, num_frames_per_session: Optional[int] = None,
                      hodgepodge_mode: bool = DATA_HODGEPODGE_MODE) -> Generator[Fluorescence, None, None]:    
        """load fall mat file"""
        assert (n_session is not None) ^ (num_frames_per_session is not None), "Either n_session or frame_per_session must be specified"
        if hodgepodge_mode:
            mat_path = find_only_one(routing.search_pattern_file(pattern="Fall.mat", search_dir=data_dir))
        else:
            mat_path = path.join(data_dir, "soma", "Fall.mat") 
        mat_dict = loadmat(mat_path)

        """get raw fluorescence"""
        F = mat_dict['F'].copy()
        Fneu = mat_dict['Fneu'].copy()
        is_cell = np.argwhere(mat_dict['iscell'][:, 0] == 1)[:, 0]
        raw_f = F[is_cell] - NEUROPIL_SUBTRACTION_COEFFICIENT * Fneu[is_cell]
        
        """get fov motion"""
        ops = mat_dict['ops'][0][0]
        fov_motion = np.concatenate([ops['xoff'], ops['yoff']], axis=0)

        """get cell position"""
        stat = mat_dict['stat']
        cell_position = np.stack([stat[0, cell_id][0, 0][6][0] for cell_id in is_cell], axis=0)

        """split into sessions"""
        num_cells, num_total_frames = raw_f.shape
        if n_session is None:
            assert num_total_frames % num_frames_per_session == 0, f"Cannot split {num_total_frames} frames into {num_frames_per_session} frames per session"
            n_session = int(num_total_frames // num_frames_per_session)
        if num_frames_per_session is None:
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
        n_session = len(timeline_dict)
        ttl_file_path = path.join(dir_path, "ttl.xlsx")
        assert path.exists(ttl_file_path), f"Cannot find ttl file: {ttl_file_path}"

        """ calculate ttl to timeline offset """
        ttl_array = pd.ExcelFile(ttl_file_path)
        assert len(ttl_array.sheet_names) == n_session, f"Cannot find {n_session} sheets in {ttl_file_path}"
        for sheet_name, (session_coordinate, timeline), fluorescence in zip(
            ttl_array.sheet_names, 
            sorted(timeline_dict.items()), 
            load_fall_mat(dir_path, DEFAULT_RECORDING_DURATION, n_session)
        ):       
            """load ttl and align to timeline"""
            df = ttl_array.parse(sheet_name, header=0).to_numpy()
            assert df.shape[1] == 2, f"Cannot find 2 columns in {ttl_file_path} {sheet_name}"     
            ttl_t = df[:, 0]/1000

            # alignment happens here
            ttl_aligns = timeline.advanced_filter(TTL_EVENT_DEFAULT)

            """match ttl and timeline"""
            if FAST_MATCHING_MODE:
                ttl_to_timeline_offset = ttl_t[0] - ttl_aligns.t[0]
            else:
                assert len(ttl_t) == len(ttl_aligns.t), f"Cannot match ttl and timeline in {sheet_name}"
                ttl_to_timeline_offsets = ttl_t - ttl_aligns.t
                assert np.allclose(ttl_to_timeline_offsets, ttl_to_timeline_offsets[0]), \
                    f"Cannot match ttl and timeline in {sheet_name}"
                ttl_to_timeline_offset = ttl_to_timeline_offsets[0]
            
            """align fluorescence to timeline"""
            fluorescence.raw_f.t -= ttl_to_timeline_offset
            fluorescence.fov_motion.t -= ttl_to_timeline_offset
            yield fluorescence    

    def io_lost_ttl(dir_path: str) -> Generator[Fluorescence, None, None]:
        n_session = len(timeline_dict)
        for (session_coordinate, timeline), fluorescence in zip(
            timeline_dict.items(), 
            load_fall_mat(dir_path, DEFAULT_RECORDING_DURATION, n_session)
        ):            
            """align fluorescence to timeline"""
            task_start, _ = timeline.task_time()
            ttl_to_timeline_offset = NULL_TTL_OFFSET - task_start  # align the fluorescence to the timeline by a fixed offset
            fluorescence.raw_f.t -= ttl_to_timeline_offset
            fluorescence.fov_motion.t -= ttl_to_timeline_offset
            yield fluorescence

    def io_split_fall(dir_path: str) -> Generator[Fluorescence, None, None]:
        n_session = len(timeline_dict)
        ttl_file_path = path.join(dir_path, "ttl.xlsx")
        assert path.exists(ttl_file_path), f"Cannot find ttl file: {ttl_file_path}"

        """ calculate ttl to timeline offset """
        ttl_array = pd.ExcelFile(ttl_file_path)
        assert len(ttl_array.sheet_names) == 2 * n_session, f"Cannot find 2*{n_session} sheets in {ttl_file_path}"
        
        # load cell permutation matrix
        permutation_matrix = pd.read_csv(routing.search_pattern_file(pattern="*_roi_sheet_*.csv", search_dir=dir_path)[0], header=0)
        for session_coordinate in timeline_dict.keys():
            assert session_coordinate.day_id in permutation_matrix.columns, f"Cannot find {session_coordinate.day_id} in {permutation_matrix.columns}"
        for sheet_name, (session_coordinate, timeline) in zip(
            ttl_array.sheet_names[::2], 
            timeline_dict.items(), 
        ):      
            fluorescence = load_fall_mat(
                path.join(dir_path, "soma", session_coordinate.day_id),
                  DEFAULT_RECORDING_DURATION, 1, hodgepodge_mode=True).__next__()
            """load ttl and align to timeline"""
            df = ttl_array.parse(sheet_name, header=0).to_numpy()
            assert df.shape[1] == 1, f"Cannot find 2 columns in {ttl_file_path} {sheet_name}"     
            ttl_t = df[:, 0]/1000

            # find the all ttl align event
            ttl_aligns = timeline.advanced_filter(TTL_EVENT_DEFAULT)

            """match ttl and timeline"""
            if FAST_MATCHING_MODE:
                ttl_to_timeline_offset = ttl_t[0] - ttl_aligns.t[0]
            else:
                assert len(ttl_t) == len(ttl_aligns.t), f"Cannot match ttl and timeline in {sheet_name}"
                ttl_to_timeline_offsets = ttl_t - ttl_aligns.t
                assert np.allclose(ttl_to_timeline_offsets, ttl_to_timeline_offsets[0]), \
                    f"Cannot match ttl and timeline in {sheet_name}"
                ttl_to_timeline_offset = ttl_to_timeline_offsets[0]
            
            # permutate cell order
            permutation = permutation_matrix[session_coordinate.day_id].to_numpy()
            assert set(permutation) == set(fluorescence.cell_idx), \
                f"Cannot match roi sheet permutation and cell idx in {session_coordinate}: {permutation} vs {fluorescence.cell_idx}"
            new_order = [find_only_one(np.where(fluorescence.cell_idx == cell_perm_idx)) for cell_perm_idx in permutation]
            fluorescence.cell_order = np.array(new_order).squeeze(1)
            fluorescence.cell_idx = np.arange(len(fluorescence.cell_order))
            fluorescence.raw_f.v = fluorescence.raw_f.v[fluorescence.cell_order]
            fluorescence.cell_position = fluorescence.cell_position[fluorescence.cell_order]

            """align fluorescence to timeline"""
            fluorescence.raw_f.t -= ttl_to_timeline_offset
            fluorescence.fov_motion.t -= ttl_to_timeline_offset
            yield fluorescence    

    def io_classic(dir_path: str) -> Generator[Fluorescence, None, None]:
        n_session = len(timeline_dict) 
        all_fluorescence = list(load_fall_mat(dir_path, DEFAULT_RECORDING_DURATION, num_frames_per_session=DEFAULT_FRAMES_PER_SESSION, hodgepodge_mode=True))
        if len(all_fluorescence) == n_session:
            for fluorescence in all_fluorescence:
                yield fluorescence
        elif len(all_fluorescence) == n_session * 2:
            for fluorescence in all_fluorescence[::2]:
                yield fluorescence
        elif len(all_fluorescence) * 2 == n_session:
            for fluorescence in all_fluorescence:
                yield fluorescence
                yield fluorescence
        else:
            raise ValueError(f"Cannot match fluorescence and timeline in {dir_path}")
      

    """Load fluorescence from node."""
    fluorescence_loader_options = {
        "default": io_default,
        "lost_ttl": io_lost_ttl,
        "split_fall": io_split_fall,
        "classic": io_classic,
    }
    default_data_path = routing.default_data_path(node)
    
    if fluorescence_loader_name is None:
        logger.info("No fluorescence loader specified, skip loading fluorescence")
        return
    loader_to_use = fluorescence_loader_options.get(fluorescence_loader_name)
    if loader_to_use is None:
        raise ValueError(f"Unknown fluorescence loader: {fluorescence_loader_name}. Available options: {fluorescence_loader_options.keys()}")
    
    try:
        yield from loader_to_use(default_data_path)
    except Exception as e:
        if LOADER_STRICT_MODE:
            raise ValueError(f"Error loading fluorescence in {default_data_path} with {fluorescence_loader_name}: {e}")
        logger.debug(f"Error loading fluorescence in {default_data_path} with {fluorescence_loader_name}: {e}")
