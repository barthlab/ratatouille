import logging
import re
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Patch, Rectangle
import matplotlib.patheffects as pe
import numpy as np
import os.path as path
import os
import pandas as pd
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_FeatureSpace import get_trial_wise_spike_timing_features
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SPVtypes import get_grand_cluster_result
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_Settings import CLUSTER_COLORS, COHORT_COLORS
from kitchen.plotter.macros.JS_juxta_data_macros.JS_juxta_data_macros_SummaryMetric import get_saving_path

logger = logging.getLogger(__name__)


plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6



def _get_data_availability_table(node2label, trial_type):
    """Match trials by full node name, then sum sessions for each mouse/cell."""
    rows = []
    for node_name, cluster in node2label.items():
        fields = {}
        for field in ("Cohort", "Mice", "Cell", "Session"):
            match = re.search(rf"(?:^|[\s(]){field}_([^\s,)]+)", node_name)
            if match is None:
                raise ValueError(f"Cannot parse {field} from node name: {node_name}")
            fields[field] = match.group(1)
        if not fields["Cell"].isdigit():
            raise ValueError(f"Expected a numeric cell ID in {node_name}")
        rows.append({
            "node_name": node_name,
            "mouse": fields["Mice"],
            "cell_id": int(fields["Cell"]),
            "cohort": fields["Cohort"],
            "cluster": cluster,
            "session": fields["Session"],
        })
    if not rows:
        raise ValueError("No clustered cells available to plot.")
    data = pd.DataFrame(rows)
    labels_per_cell = data.groupby(["mouse", "cell_id"])[["cluster", "cohort"]].nunique()
    conflicting = labels_per_cell.gt(1).any(axis=1)
    if conflicting.any():
        raise ValueError("Cannot combine sessions with conflicting cluster or cohort labels: "
                         f"{labels_per_cell.index[conflicting].tolist()}")

    trial_counts = {}
    for cohort in sorted(data["cohort"].unique()):
        timing = get_trial_wise_spike_timing_features(cohort)
        names, spikes, types = (timing[key] for key in
                                ("all_node_name", "all_spikes", "all_trial_type"))
        if not (len(names) == len(spikes) == len(types)):
            raise ValueError(f"Misaligned trial-wise arrays for {cohort}")
        seen, cohort_counts = set(), {}
        for node_name, trials, puff_type in zip(names, spikes, types):
            key = (node_name, puff_type)
            if key in seen:
                raise ValueError(f"Duplicate trial-wise record: {key}")
            seen.add(key)
            cohort_counts.setdefault(node_name, 0)
            if trial_type is None or puff_type == trial_type:
                # Each element is one trial, including trials with zero spikes.
                cohort_counts[node_name] += len(trials)
        trial_counts.update(cohort_counts)
    data["n_trials"] = pd.array(data["node_name"].map(trial_counts), dtype="Int64")
    missing = data.loc[data["n_trials"].isna(), "node_name"]
    if not missing.empty:
        logger.warning("No trial-wise record for %d clustered cells:\n%s",
                       len(missing), missing.to_string(index=False))
    mixed_mice = data.groupby("mouse")["cohort"].nunique()
    if (mixed_mice > 1).any():
        logger.warning("Mice represented in multiple cohorts: %s",
                       mixed_mice.index[mixed_mice > 1].tolist())
    data = data.sort_values(["mouse", "cell_id", "session", "node_name"])
    data = data.groupby(["mouse", "cell_id"], as_index=False, sort=True).agg(
        node_name=("node_name", tuple),
        session=("session", tuple),
        cohort=("cohort", "first"),
        cluster=("cluster", "first"),
        # Preserve NA if any session lacks trial data instead of showing a partial total.
        n_trials=("n_trials", lambda counts: counts.sum(min_count=len(counts))),
        n_sessions=("node_name", "size"),
    )
    data["n_trials"] = data["n_trials"].astype("Int64")
    return data


def visualize_data_availability(trial_type=None):
    """Plot clustered cells by mouse and numeric cell ID, and return their counts.

    Count both 100msPuff and 500msPuff trials by default; pass either trial type
    to restrict counts. Gray slots mean absent from node2label, not necessarily
    unrecorded. A clustered cell without a trial-wise record displays 'X: NA'.
    Repeated sessions for the same mouse/cell are combined by summing trial counts.
    """
    if trial_type not in (None, "100msPuff", "500msPuff"):
        raise ValueError("trial_type must be None, '100msPuff', or '500msPuff'")
    node2label, _ = get_grand_cluster_result(_save_dataframe=True)
    data = _get_data_availability_table(node2label, trial_type)
    mice = sorted(data["mouse"].unique())
    # Include even IDs absent from every mouse, so gaps never shift columns.
    cell_ids = list(range(min(1, data["cell_id"].min()), data["cell_id"].max() + 1))
    mouse2row = {mouse: row for row, mouse in enumerate(mice)}
    cell2col = {cell_id: col for col, cell_id in enumerate(cell_ids)}
    missing_color = "#D0D0D0"
    fig, axs = plt.subplots(
        1, 2, figsize=(5.1, 3), constrained_layout=True,
    )
    try:
        for ax, color_by, palette, title in zip(
            (axs[1], axs[0]), ("cluster", "cohort"), (CLUSTER_COLORS, COHORT_COLORS),
            ("Colored by cluster", "Colored by cohort"),
        ):
            # Draw NA explicitly so it stays gray when saved with transparency.
            ax.imshow(np.full((len(mice), len(cell_ids), 3), to_rgb(missing_color)),
                      interpolation="nearest", aspect="auto")
            for entry in data.itertuples(index=False):
                row, col = mouse2row[entry.mouse], cell2col[entry.cell_id]
                # Pale fills keep colored cluster labels and black counts readable.
                fill = 0.25 * np.array(to_rgb(palette[getattr(entry, color_by)])) + 0.75
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       facecolor=fill, edgecolor="white", linewidth=0.5))
                count_text = "NA" if pd.isna(entry.n_trials) else str(entry.n_trials)
                cluster_text = TextArea(str(entry.cluster), textprops={
                    "color": CLUSTER_COLORS[entry.cluster], "fontsize": 6, "weight": "bold",
                    "path_effects": [pe.withStroke(linewidth=1., foreground="white")],
                })
                count = TextArea(f":{count_text}", textprops={"color": "black", "fontsize": 6})
                label = HPacker(children=[cluster_text, count], align="center", pad=0, sep=0)
                ax.add_artist(AnnotationBbox(label, (col, row), frameon=False, pad=0))

            ax.set_xlim(-0.5, len(cell_ids) - 0.5)
            ax.set_ylim(len(mice) - 0.5, -0.5)

            ax.set_xticks(range(len(cell_ids)), [f"{cell_id:02d}" for cell_id in cell_ids])
            ax.set_yticks(range(len(mice)), mice)

            ax.set_xticks(np.arange(len(cell_ids) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(mice) + 1) - 0.5, minor=True)

            ax.grid(which="minor", color="white", linewidth=0.5)

            ax.tick_params(which="both", length=0, pad=2)
            ax.set_xlabel("Cell index", labelpad=1)
            ax.set_ylabel("Mouse", labelpad=1)
            ax.set_title(title)
            
        trial_description = "100 + 500 ms puff" if trial_type is None else trial_type
        fig.suptitle(f"Clustered cell availability | label : trial count ({trial_description})")
        suffix = "" if trial_type is None else f"_{trial_type}"
        save_path = path.join(get_saving_path(), f"Dataavailability{suffix}.png")
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
    finally:
        plt.close(fig)
    logger.info("Plot saved to %s", save_path)
    return data





def _get_data_firing_rate_table(node2label, trial_type):
    """Pool selected trials across sessions and calculate mean rates in Hz.

    Spike times are in milliseconds. Use [-500, 0) for spontaneous activity and
    [0, puff duration) for evoked activity, including silent trials in the mean.
    """
    data = _get_data_availability_table(node2label, trial_type)
    evoked_end_ms = 100 if trial_type == "100msPuff" else 500
    node_spike_counts = {}
    for cohort in sorted(data["cohort"].unique()):
        timing = get_trial_wise_spike_timing_features(cohort)
        for node_name, trials, puff_type in zip(
            timing["all_node_name"], timing["all_spikes"], timing["all_trial_type"]
        ):
            if puff_type != trial_type:
                continue
            spontaneous_count, evoked_count = 0, 0
            for spikes in trials:
                spikes = np.asarray(spikes)
                spontaneous_count += np.count_nonzero((spikes >= -500) & (spikes < 0))
                evoked_count += np.count_nonzero((spikes >= 0) & (spikes < evoked_end_ms))
            node_spike_counts[node_name] = (spontaneous_count, evoked_count)

    spontaneous_rates, evoked_rates = [], []
    for entry in data.itertuples(index=False):
        if pd.isna(entry.n_trials) or entry.n_trials == 0:
            spontaneous_rates.append(np.nan)
            evoked_rates.append(np.nan)
            continue
        counts = np.array([node_spike_counts.get(node_name, (0, 0))
                           for node_name in entry.node_name]).sum(axis=0)
        spontaneous_rates.append(counts[0] / (entry.n_trials * 0.5))
        evoked_rates.append(counts[1] / (entry.n_trials * evoked_end_ms / 1000))
    data["spontaneous_fr_hz"] = spontaneous_rates
    data["evoked_fr_hz"] = evoked_rates
    return data


def visualize_data_habituation(trial_type=None):
    """Plot spontaneous and evoked mean firing rates for one puff trial type."""
    if trial_type not in ("100msPuff", "500msPuff"):
        raise ValueError("trial_type must be '100msPuff' or '500msPuff'")
    node2label, _ = get_grand_cluster_result(_save_dataframe=True)
    data = _get_data_firing_rate_table(node2label, trial_type)
    mice = sorted(data["mouse"].unique())
    # Include even IDs absent from every mouse, so gaps never shift columns.
    cell_ids = list(range(min(1, data["cell_id"].min()), data["cell_id"].max() + 1))
    mouse2row = {mouse: row for row, mouse in enumerate(mice)}
    cell2col = {cell_id: col for col, cell_id in enumerate(cell_ids)}
    missing_color = "#D0D0D0"
    cmap = plt.get_cmap("YlOrBr")
    norm = Normalize(vmin=0, vmax=50, clip=True)
    evoked_end_ms = 100 if trial_type == "100msPuff" else 500
    fig, axs = plt.subplots(
        1, 2, figsize=(5.1, 3), constrained_layout=True,
    )
    try:
        for ax, rate_column, title in zip(
            axs, ("spontaneous_fr_hz", "evoked_fr_hz"),
            ("Spontaneous (-500 to 0 ms)", f"Evoked (0 to {evoked_end_ms} ms)"),
        ):
            # Draw NA explicitly so it stays gray when saved with transparency.
            ax.imshow(np.full((len(mice), len(cell_ids), 3), to_rgb(missing_color)),
                      interpolation="nearest", aspect="auto")
            for entry in data.itertuples(index=False):
                row, col = mouse2row[entry.mouse], cell2col[entry.cell_id]
                rate = getattr(entry, rate_column)
                if not np.isfinite(rate):
                    continue
                ax.add_patch(Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       facecolor=cmap(norm(rate)), edgecolor="white", linewidth=0.5))
            ax.set_xlim(-0.5, len(cell_ids) - 0.5)
            ax.set_ylim(len(mice) - 0.5, -0.5)

            ax.set_xticks(range(len(cell_ids)), [f"{cell_id:02d}" for cell_id in cell_ids])
            ax.set_yticks(range(len(mice)), mice)

            ax.set_xticks(np.arange(len(cell_ids) + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(mice) + 1) - 0.5, minor=True)

            ax.grid(which="minor", color="white", linewidth=0.5)

            ax.tick_params(which="both", length=0, pad=2)
            ax.set_xlabel("Cell index", labelpad=1)
            ax.set_ylabel("Mouse", labelpad=1)
            ax.set_title(title)
                
        fig.suptitle(f"Mean firing rate ({trial_type})")
        save_path = path.join(get_saving_path(), f"Datahabituation_{trial_type}.png")
        os.makedirs(path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=500, transparent=True)
    finally:
        plt.close(fig)
    logger.info("Plot saved to %s", save_path)
    return data
