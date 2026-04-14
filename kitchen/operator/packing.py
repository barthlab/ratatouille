from kitchen.plotter.plotting_params import LOCOMOTION_BIN_SIZE, LICK_BIN_SIZE
from kitchen.structure.neural_data_structure import NeuralData, TimeSeries
from kitchen.operator.grouping import grouping_timeseries
from kitchen.utils.numpy_kit import smart_nan_removal

import numpy as np


def pack_neural_data(neural_data: NeuralData, 
    has_position: bool=False,
    has_locomotion: bool=False,
    has_whisker: bool=False,
    has_pupil_area: bool=False,
    has_pupil_center_x: bool=False,
    has_pupil_center_y: bool=False,
    has_saccade_velocity: bool=False,
    has_lick: bool=False,
    **kwargs,
) -> TimeSeries:
    """Pack data into a numpy array. Shape (D, T) where D is the number of data modalities and T is the number of time points."""
    all_data = []
    if has_position:
        assert neural_data.position is not None, "Cannot pack position data"
        all_data.append(neural_data.position.rate(bin_size=LOCOMOTION_BIN_SIZE))
    if has_locomotion:
        assert neural_data.locomotion is not None, "Cannot pack locomotion data"
        all_data.append(neural_data.locomotion.rate(bin_size=LOCOMOTION_BIN_SIZE))
    if has_whisker:
        assert neural_data.whisker is not None, "Cannot pack whisker data"
        all_data.append(neural_data.whisker)
    if has_pupil_area:
        assert neural_data.pupil is not None, "Cannot pack pupil data"
        all_data.append(neural_data.pupil.area_ts)
    if has_pupil_center_x:
        assert neural_data.pupil is not None, "Cannot pack pupil data"
        all_data.append(neural_data.pupil.center_x_ts)
    if has_pupil_center_y:
        assert neural_data.pupil is not None, "Cannot pack pupil data"
        all_data.append(neural_data.pupil.center_y_ts)
    if has_saccade_velocity:
        assert neural_data.pupil is not None, "Cannot pack pupil data"
        all_data.append(neural_data.pupil.saccade_velocity_ts)
    if has_lick:
        assert neural_data.lick is not None, "Cannot pack lick data"
        all_data.append(neural_data.lick.rate(bin_size=LICK_BIN_SIZE))
    for modality_idx, single_data in enumerate(all_data):
        single_data.v = smart_nan_removal(single_data.t, single_data.v, method="linear")
    adv_ts = grouping_timeseries(all_data, **kwargs)
    assert not np.any(np.isnan(adv_ts.raw_array)), f"NaN found in grouped data: {adv_ts}"
    return TimeSeries(v=adv_ts.raw_array, t=adv_ts.t)
    