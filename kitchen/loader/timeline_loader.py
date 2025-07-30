import os
import re
import os.path as path
from typing import Generator, Tuple

import numpy as np
import pandas as pd

from kitchen.configs import routing
from kitchen.settings.loaders import DATA_HODGEPODGE_MODE, SPECIFIED_TIMELINE_LOADER, io_enumerator
from kitchen.structure.hierarchical_data_structure import Fov
from kitchen.structure.neural_data_structure import Timeline



def timeline_loader_from_fov(fov_node: Fov) -> Generator[Tuple[str, str, Timeline], None, None]:
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
                day_name = filename.split("_")[1]
                
                extracted_timeline = Timeline(
                    v=data_array['details'].to_numpy(),
                    t=data_array['time'].to_numpy(dtype=np.float32) / 1000
                )
                """Yield day name, session name, and timeline data."""
                yield day_name, session_name, extracted_timeline
        
    def io_matt_test(dir_path: str) -> Generator[Tuple[str, str, Timeline], None, None]:      
        """Find all timeline files."""  
        assert not DATA_HODGEPODGE_MODE, "Matt test mode only works in default data mode."
        timeline_filepaths = routing.search_pattern_file(pattern="puff_data_*.csv", search_dir=path.join(dir_path, "timeline"))
        assert len(timeline_filepaths) > 0, f"Cannot find timeline path: {dir_path}"            

        for filepath in timeline_filepaths:
            dirname, filename = path.split(filepath)
            if filename.startswith("puff_data_") and filename.endswith(".csv"):
                """Extract day name, session name, and timeline data."""
                data_array = pd.read_csv(path.join(dirname, filename), header=0)
                assert 'type' in data_array.columns
                assert 'solenoid on time' in data_array.columns and 'solenoid off time' in data_array.columns, \
                      f"Cannot find 'solenoid on time' and 'solenoid off time' columns in {filename}"
                v, t = [], []
                for _, row in data_array.iterrows():
                    if row['type'] == 'real':
                        v.append("VerticalPuffOn")
                        t.append(row['solenoid on time'])
                        v.append("VerticalPuffOff")
                        t.append(row['solenoid off time'])
                    elif row['type'] == 'fake':
                        v.append("BlankOn")
                        t.append(row['solenoid on time'])
                        v.append("BlankOff")
                        t.append(row['solenoid off time'])
                    else:
                        raise ValueError(f"Unknown type {row['type']} in {filename}")
                
                extracted_timeline = Timeline(
                    v=np.array(v),
                    t=np.array(t, dtype=np.float32) / 1000
                )

                session_name = re.search(r"puff_data_(.*)\.csv", filename)
                assert session_name is not None, f"Cannot extract session name from {filename}"
                session_name = session_name.group(1)

                day_name = filename.split("_")[3]

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
                    yield f"{day_id}",f"{sheet_id}", extracted_timeline
 

    """Load timeline from fov node."""
    default_fov_data_path = routing.default_data_path(fov_node)
    yield from io_enumerator(default_fov_data_path, 
                             [io_default, io_matt_test, io_old], 
                             SPECIFIED_TIMELINE_LOADER)
