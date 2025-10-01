import os
import re
import os.path as path
from typing import Generator, Tuple
import logging
import numpy as np
import pandas as pd

from kitchen.configs import routing
from kitchen.settings.loaders import DATA_HODGEPODGE_MODE, LOADER_STRICT_MODE, MATT_NAME_STYLE_FLAG
from kitchen.structure.hierarchical_data_structure import Fov
from kitchen.structure.neural_data_structure import Timeline


logger = logging.getLogger(__name__)

def timeline_loader_from_fov(fov_node: Fov, timeline_loader_name: str = "default") -> Generator[Tuple[str, str, Timeline], None, None]:
    """
    Load timeline from fov node.

    All custom io function should follow the same strucutre:
    1. Find all timeline files. [Implement DATA_HODGEPODGE_MODE if necessary]
    2. For each timeline file, extract day name, session name, and timeline data.
    3. Yield day name, session name, and timeline data.

    Note:
    day name and session name should be consistent with other data files, like behavior, fluorescence, etc.
    """
    
    def io_default(dir_path: str) -> Generator[Tuple[str, str, Timeline], None, None]:      
        """Find all timeline files.""" 
        if DATA_HODGEPODGE_MODE:
            timeline_filepaths = routing.search_pattern_file(pattern="TIMELINE_*.csv", search_dir=dir_path)
        else:
            timeline_filepaths = routing.search_pattern_file(pattern="TIMELINE_*.csv", search_dir=path.join(dir_path, "timeline"))
        
        assert len(timeline_filepaths) > 0, f"Cannot find timeline path: {dir_path}"            
            
        for filepath in timeline_filepaths:
            dirname, filename = path.split(filepath)
            if filename.startswith("TIMELINE_") and filename.endswith(".csv"):
                """Extract day name, session name, and timeline data."""
                data_array = pd.read_csv(path.join(dirname, filename), header=0)
                assert 'time' in data_array.columns and 'details' in data_array.columns, \
                      f"Cannot find 'time' and 'details' columns in {filename}"
                
                # extract the string between prefix "TIMELINE_" and postfix ".csv"
                session_name = re.search(r"TIMELINE_(.*)\.csv", filename)
                assert session_name is not None, f"Cannot extract session name from {filename}"
                session_name = session_name.group(1)
                if MATT_NAME_STYLE_FLAG:                    
                    day_name = filename.split("_")[2]
                    assert day_name.startswith("D"), f"Expected day name to start with 'D' in {filename}"
                    day_name = day_name[1:].zfill(2)
                else:
                    day_name = filename.split("_")[1]
                
                extracted_timeline = Timeline(
                    v=data_array['details'].to_numpy(),
                    t=data_array['time'].to_numpy(dtype=np.float32) / 1000
                )
                """Yield day name, session name, and timeline data."""
                yield day_name, session_name, extracted_timeline
        

    def io_old(dir_path: str) -> Generator[Tuple[str, str, Timeline], None, None]:        
        timeline_dir_path = path.join(dir_path, "timeline")
        assert path.exists(timeline_dir_path), f"Cannot find timeline path: {timeline_dir_path}"    

        for filename in os.listdir(timeline_dir_path):
            if filename.startswith("Arduino time point") and filename.endswith(".xlsx"):
                data_array = pd.ExcelFile(path.join(timeline_dir_path, filename))
                for sheet_id, sheet_name in enumerate(data_array.sheet_names):
                    df = data_array.parse(sheet_name, header=None).to_numpy()
                    assert df.shape[1] == 5, f"Cannot find 4 columns in {filename} {sheet_name}"
                    extracted_timeline = Timeline(
                        v=df[:, 0],
                        t=np.array(df[:, 4] / 1000, dtype=np.float32)
                    )
                    day_id = str(sheet_id // 2)
                    yield f"{day_id}", f"{sheet_id}", extracted_timeline


    def io_classic(dir_path: str) -> Generator[Tuple[str, str, Timeline], None, None]:
        timeline_type = pd.ExcelFile(path.join(dir_path, "Arduino.xlsx"))
        timeline_t = pd.ExcelFile(path.join(dir_path, "Arduino time point.xlsx"))
        
        def extract_session_id_list(sheet_name_list: list) -> list:
            session_id_list = []
            for sheet_name in sheet_name_list:
                if sheet_name.isdigit():
                    session_id_list.append(int(sheet_name))
                elif sheet_name.startswith("Sheet") and sheet_name[5:].isdigit():
                    session_id_list.append(int(sheet_name[5:]))
                elif sheet_name.startswith("ACC") and "_" in sheet_name:
                    day_id, session_id = sheet_name[3:].split("_")
                    session_id_list.append(2*(int(day_id) - 1) + int(session_id))
                elif sheet_name.startswith("SAT") and "_" in sheet_name:
                    day_id, session_id = sheet_name[3:].split("_")
                    session_id_list.append(2*(int(day_id) - 1) + int(session_id) + 12)
                elif sheet_name.startswith("PSE") and "_" in sheet_name:
                    day_id, session_id = sheet_name[3:].split("_")
                    session_id_list.append(2*(int(day_id) - 1) + int(session_id) + 12)
                else:
                    raise ValueError(f"Unknown sheet name {sheet_name} in {sheet_name_list}")
            return session_id_list
        
        timeline_type_session_id = extract_session_id_list(timeline_type.sheet_names)
        timeline_t_session_id = extract_session_id_list(timeline_t.sheet_names)
     
        # extract timeline type and t
        timeline_type_dict = {}
        for session_id, timeline_type_sheet_name in zip(timeline_type_session_id, timeline_type.sheet_names):
            df_type = timeline_type.parse(f"{timeline_type_sheet_name}", header=None).to_numpy()
            for i, type_value in enumerate(df_type[:, 0]):
                if type_value.lower() in ("puff", "real"):
                    df_type[i, 0] = "Puff"
                elif type_value.lower() in ("blank", "fake"):
                    df_type[i, 0] = "Blank"
                else:
                    raise ValueError(f"Unknown timeline type {type_value} in {timeline_type_sheet_name} at {dir_path}")
                timeline_type_dict[session_id] = df_type[:, 0]

        timeline_t_dict = {}
        for session_id, timeline_type_sheet_name, timeline_t_sheet_name in zip(timeline_t_session_id, timeline_type.sheet_names, timeline_t.sheet_names):
            df_t = timeline_t.parse(timeline_t_sheet_name, header=0).to_numpy()
            if df_t.shape[1] == 2:
                timeline_t_dict[session_id] = df_t[:, 0]
            else:
                timeline_t_dict[session_id] = np.array([])
                logger.debug(f"Cannot find timeline t in {timeline_t_sheet_name} at {dir_path}, got {df_t}, expected 2 columns")

        # duplication
        def duplication_if_need(timeline_data_dict: dict, data_name: str) -> dict:
            assert 0 not in timeline_data_dict.keys(), f"Expected session id start from 1, but got 0 in {data_name} at {dir_path}: {timeline_data_dict}"
            final_data_dict = {}
            if len(timeline_data_dict) > 16:
                # occassionly sheet_id will excceed 32, where we should renorm the first session id to 1
                if max(timeline_data_dict.keys()) > 32:
                    min_session_id = min(timeline_data_dict.keys())
                    timeline_data_dict = {k - min_session_id + 1: v for k, v in timeline_data_dict.items()}
                # no need for duplication, session start from 1
                for session_id, value in timeline_data_dict.items():
                    final_data_dict[(int((session_id - 1) // 2), session_id - 1)] = value
            else:
                # duplication
                for session_id, value in timeline_data_dict.items():
                    final_data_dict[(session_id - 1, (session_id - 1) * 2)] = value
                    final_data_dict[(session_id - 1, (session_id - 1) * 2 + 1)] = value
            return final_data_dict
        
        final_timeline_type = duplication_if_need(timeline_type_dict, "type")
        final_timeline_t = duplication_if_need(timeline_t_dict, "t")
        assert final_timeline_type.keys() == final_timeline_t.keys(), \
            f"Cannot match timeline type and timeline t in {dir_path}, got {final_timeline_type.keys()} and {final_timeline_t.keys()}"
        
        # construct timeline
        for (day_id, session_id), timeline_type in final_timeline_type.items():
            timeline_t = final_timeline_t[(day_id, session_id)]
            if abs(len(timeline_type) - len(timeline_t)) > 1:
                logger.debug(f"Cannot match shape of timeline type and timeline t in {dir_path} at day {day_id} session {session_id}, "\
                             f"got {len(timeline_type)} type: {timeline_type} and {len(timeline_t)} t: {timeline_t}")
                extracted_timeline = Timeline(v=np.array([]), t=np.array([]))
            else:
                min_length = min(len(timeline_type), len(timeline_t))
                extracted_timeline = Timeline(
                    v=timeline_type[:min_length],
                    t=np.array(timeline_t[:min_length] / 1000, dtype=np.float32)
                )
            yield f"{str(day_id)}".zfill(2), f"{str(session_id)}".zfill(2), extracted_timeline
                

    """Load timeline from fov node."""
    timeline_loader_options = {
        "default": io_default,
        "old": io_old,
        "classic": io_classic,
    }
    default_fov_data_path = routing.default_data_path(fov_node)

    loader_to_use = timeline_loader_options.get(timeline_loader_name)
    if loader_to_use is None:
        raise ValueError(f"Unknown timeline loader: {timeline_loader_name}. Available options: {timeline_loader_options.keys()}")
    
    yield from timeline_loader_options[timeline_loader_name](default_fov_data_path)
    