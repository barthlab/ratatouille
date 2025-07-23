"""
Plotting overlapped data settings.

Controls how multiple data traces are plotted when overlapping them.

In harsh mode (PLOTTING_OVERLAP_HARSH_MODE = True):
    - Data traces can only be overlapped if ALL data points are valid
    - Ensures complete data integrity in overlapped visualizations
    - Useful for strict analysis requirements

In loose mode (PLOTTING_OVERLAP_HARSH_MODE = False):
    - Data traces can be overlapped if at least ONE valid data point exists
    - Allows for more flexible visualization with partial data
    - Suitable for exploratory analysis
"""

PLOTTING_OVERLAP_HARSH_MODE = True