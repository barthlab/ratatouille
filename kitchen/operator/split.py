from functools import partial
import numpy as np

from kitchen.annotator.trial_type_annotator import INTRINSIC_TRIAL_TYPE_ORDER
from kitchen.plotter.plotting_manual import CHECK_PLOT_MANUAL, PlotManual
from kitchen.settings.trials import DEFAULT_TRIAL_RANGE_FOR_DUMMY_TRIAL
from kitchen.structure.hierarchical_data_structure import CellSession, DataSet, FovTrial, Node, Session, Trial
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate
from kitchen.utils.sequence_kit import select_from_value


def split_dataset_by_trial_type(dataset: DataSet, plot_manual: PlotManual, _element_trial_level: str = "trial", _add_dummy: bool = False) -> dict[str, DataSet]:
    all_trial_nodes = dataset.select(_element_trial_level, _empty_warning=False)
    type2dataset = select_from_value(
        all_trial_nodes.rule_based_group_by(lambda x: x.info.get("trial_type")),
        _self = partial(CHECK_PLOT_MANUAL, plot_manual=plot_manual)
    )
    # type2dataset = {k: v for k, v in type2dataset.items() if k in INTRINSIC_TRIAL_TYPE_ORDER}
    type2dataset = {k: v for k, v in sorted(type2dataset.items(), key=lambda x: INTRINSIC_TRIAL_TYPE_ORDER.index(x[0]))}
    if _add_dummy:
        type2dataset["Dummy"] = get_dummy_trials(dataset, _element_trial_level=_element_trial_level)
    return type2dataset


def get_dummy_trials(dataset: DataSet, _element_trial_level: str = "trial") -> DataSet:
    dummy_trial_nodes = RandomTrialSplitter(dataset, DEFAULT_TRIAL_RANGE_FOR_DUMMY_TRIAL, _element_trial_level=_element_trial_level)
    return DataSet(dataset.name + "_Dummy", dummy_trial_nodes)


def RandomTrialSplitter(dataset: DataSet, trial_range: tuple[float, float], num: int = 1000, 
                        _element_trial_level: str = "trial") -> list[Node]:
    """Split a dataset into multiple random trials."""
    selected_trial_nodes = []

    _element_session_level = "cellsession" if _element_trial_level == "trial" else "session"
    target_node_type = Trial if _element_trial_level == "trial" else FovTrial
    for session_level_node in dataset.select(_element_session_level):
        assert isinstance(session_level_node, (CellSession, Session))
        task_start, task_end = session_level_node.data.timeline.task_time()
        random_trial_aligns = np.random.uniform(task_start + np.abs(trial_range[0]), 
                                                task_end - np.abs(trial_range[1]), size=num)

        for trial_id, trial_t in enumerate(random_trial_aligns):
            trial_coordinate = TemporalObjectCoordinate(
                temporal_uid=session_level_node.coordinate.temporal_uid.child_uid(chunk_id=-1000000 + trial_id),
                object_uid=session_level_node.coordinate.object_uid)
            trial_neural_data = session_level_node.data.segment(
                start_t=trial_t + trial_range[0],
                end_t=trial_t + trial_range[1])
            new_trial_node = target_node_type(
                coordinate=trial_coordinate,
                data=trial_neural_data,
                info={"raw_ref_t": trial_t,}
            )
            new_trial_node.info["trial_type"] = "Dummy"
        
            selected_trial_nodes.append(new_trial_node)
    return selected_trial_nodes