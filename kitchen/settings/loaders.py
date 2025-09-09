"""
Data file search mode configuration.

In strict search mode, data files must be organized in type-specific subdirectories under the session directory:
- Lick data -> 'lick/' directory
- Fluorescence data -> 'soma/' directory  
- Pupil data -> 'pupil/' directory
- etc.

In loose search mode (hodgepodge mode), data files can be located anywhere within the session directory structure.
This provides more flexibility but requires unique file naming conventions to identify data types.
"""


from typing import Callable, Dict, Generator, List
import inspect
import logging

logger = logging.getLogger(__name__)

DATA_HODGEPODGE_MODE = True 


# SPECIFIED_TIMELINE_LOADER = "io_default"  # io_default, io_old, io_classic
# SPECIFIED_FLUORESCENCE_LOADER = "io_default"  # io_default, io_lost_ttl, io_split_fall, io_classic
# SPECIFIED_BEHAVIOR_LOADER = "io_video_only"  # io_default, io_video_only
# SPECIFIED_POTENTIAL_LOADER = "io_default"  # io_default

MATT_NAME_STYLE_FLAG = False  # Flag for day_id detection from TIMELINE file, at position 2 start with D


LOADER_STRICT_MODE = False
