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
import warnings

DATA_HODGEPODGE_MODE = True 


SPECIFIED_TIMELINE_LOADER = "io_default"  # io_default, io_old, io_classic
SPECIFIED_FLUORESCENCE_LOADER = "io_default"  # io_default, io_lost_ttl, io_split_fall, io_classic
SPECIFIED_BEHAVIOR_LOADER = "io_default"  # io_default


MATT_NAME_STYLE_FLAG = False  # Flag for day_id detection from TIMELINE file, at position 2 start with D





def io_enumerator(dir_path: str, io_funcs: List[Callable[[str], Generator]], specified_io: str, strict_mode: bool = False) -> Generator:
    io_dispatch: Dict[str, Callable] = {f.__name__: f for f in io_funcs}
    io_func = io_dispatch.get(specified_io, None)

    if not io_func:
        error_msg = f"Cannot find specified I/O function: {specified_io} in {io_dispatch} during {inspect.stack()[1].function}"
        if strict_mode:
            raise ValueError(error_msg)
        warnings.warn(error_msg)
        return 

    try:
        yield from io_func(dir_path)
    except Exception as e:
        if strict_mode:
            raise 
        warnings.warn(f"Error executing {inspect.stack()[1].function} ->'{specified_io}': {e}")

