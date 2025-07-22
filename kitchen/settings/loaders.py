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


DATA_HODGEPODGE_MODE = False 