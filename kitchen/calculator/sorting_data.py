
from typing import Optional
import numpy as np

from kitchen.operator.grouping import AdvancedTimeSeries


def get_amplitude_sorted_idxs(
        heatmap_array_adv_ts: AdvancedTimeSeries, 
        amplitude_range: tuple[float, float],
        _additional_level_refs: Optional[list] = None
):
    amplitude_range_in_frame = np.searchsorted(heatmap_array_adv_ts.t, amplitude_range)
    amplitudes = np.nanmean(heatmap_array_adv_ts.raw_array[:, amplitude_range_in_frame[0]:amplitude_range_in_frame[1]], axis=1)
    if _additional_level_refs is not None:
        assert len(_additional_level_refs) == len(amplitudes), "Sorting additional refs should have same length as amplitudes"
        """
        Given multiple sorting keys, lexsort returns an array of integer indices that describes the sort order by multiple keys.
        The last key in the sequence is used for the primary sort order, ties are broken by the second-to-last key, and so on.
        """
        sort_idxs = np.lexsort((amplitudes, np.array(_additional_level_refs)))
    else:
        sort_idxs = np.argsort(amplitudes)
    return sort_idxs