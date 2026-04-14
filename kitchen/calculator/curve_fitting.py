from collections import defaultdict
from typing import Tuple

import numpy as np
from scipy.optimize import curve_fit

from kitchen.operator.sync_nodes import sync_nodes
from kitchen.plotter.plotting_manual import PlotManual
from kitchen.structure.hierarchical_data_structure import DataSet
from kitchen.structure.neural_data_structure import TimeSeries



def fit_evoked_trace(ts: TimeSeries):
    """
    Fit a 2-segment hinge model:
        y(t) = b0 + slope * max(0, t - tau)

    Parameters
    ----------
    ts : TimeSeries
        Time series data to fit.

    Returns
    -------
    result : dict
        {
            "intercept": float
            "latency": float
            "slope": float
            "fitted_trace": TimeSeries
        }
    """
    y = ts.v.copy()
    t = ts.t.copy()
    assert ts.v.ndim == 1, f"Expected 1D array for y, got {ts.v.shape}"

    def hinge(t, b0, tau, slope):
        return b0 + slope * np.maximum(0, t - tau)

    # Initial guesses
    b0_guess = y[0]
    tau_guess = np.median(t)
    slope_guess = max(0.05, np.nanmean((y - y[0]) / (t - t[0])))

    bounds = (
        [-np.inf, t[0], 0.05],
        [ np.inf, t[-2],  np.inf]
    )

    popt, _ = curve_fit(
        hinge, t, y,
        p0=[b0_guess, tau_guess, slope_guess],
        bounds=bounds,
        check_finite=True,
        maxfev=100
    )

    fit0 = hinge(t, *popt)

    return {
        "intercept": popt[0],
        "latency": popt[1],
        "slope": popt[2],
        "fitted_trace": TimeSeries(v=fit0, t=t),
    }


def fit_dataset_trial_fluo(
        dataset: DataSet,
        sync_events: Tuple[str],
        fit_segment: Tuple[float, float] = (0., 1.),
) -> DataSet:
    dataset_synced = sync_nodes(dataset, sync_events, PlotManual(fluorescence=True))
    fit_nodes = []
    for node in dataset_synced.select("trial"):
        if node.data.fluorescence is None:
            continue
        ts = node.data.fluorescence.df_f0.segment(*fit_segment)
        
        fit_result = fit_evoked_trace(ts.squeeze(0))

        new_node = node.shadow_clone()
        new_node.data.fluorescence.raw_f = TimeSeries(
            v=1 + fit_result["intercept"] + fit_result["slope"] * np.maximum(0, 
                node.data.fluorescence.df_f0.t - fit_result["latency"]),
            t=node.data.fluorescence.df_f0.t,
        ).unsqueeze(0)
        fit_nodes.append(new_node)
    return DataSet(name=dataset.name + "_fitted", nodes=fit_nodes)