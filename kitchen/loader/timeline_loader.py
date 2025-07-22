import os
import re
import os.path as path
from typing import Generator, Tuple

import numpy as np
import pandas as pd

from kitchen.configs import routing
from kitchen.settings.loaders import DATA_HODGEPODGE_MODE
from kitchen.structure.hierarchical_data_structure import Fov
from kitchen.structure.neural_data_structure import Timeline



def timeline_loader_from_fov(fov_node: Fov) -> Generator[Tuple[str, str, Timeline], None, None]:
    
    def io_default(dir_path: str) -> Generator[Tuple[str, str, Timeline], None, None]:       
        if DATA_HODGEPODGE_MODE:
            timeline_filepaths = routing.search_pattern_file(pattern="TIMELINE_*.csv", search_dir=dir_path)
        else:
            timeline_filepaths = routing.search_pattern_file(pattern="TIMELINE_*.csv", search_dir=path.join(dir_path, "timeline"))
        
        assert len(timeline_filepaths) > 0, f"Cannot find timeline path: {dir_path}"            
            
        for filepath in timeline_filepaths:
            dirname, filename = path.split(filepath)
            if filename.startswith("TIMELINE_") and filename.endswith(".csv"):
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

    if fov_node.coordinate.object_uid.cohort_id == "VIPcreAi148_FromMo":
        yield from io_old(default_fov_data_path)
    else:
        yield from io_default(default_fov_data_path)
