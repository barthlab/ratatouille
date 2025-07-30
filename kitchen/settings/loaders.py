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


from typing import Callable, Generator, List, Optional


DATA_HODGEPODGE_MODE = False 



SPECIFIED_TIMELINE_LOADER = "io_default"
SPECIFIED_FLUORESCENCE_LOADER = "io_default"
SPECIFIED_BEHAVIOR_LOADER = "io_default"




def io_enumerator(dir_path: str, io_funcs: List[Callable[[str], Generator]], specified_io: Optional[str] = None) -> Generator:
    try:
        assert specified_io is not None
        for io_func in io_funcs:
            if io_func.__name__ == specified_io:
                yield from io_func(dir_path)
    except Exception as e:
        print(f"Cannot load data via {specified_io}: {e}")
