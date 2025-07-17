from typing import List
import os
import os.path as path
from tqdm import tqdm

from kitchen.loader.fluorescence_loader import fluorescence_loader_from_fov
from kitchen.settings.timeline import TRIAL_ALIGN_EVENT_DEFAULT
from kitchen.settings.trials import TRIAL_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.loader.timeline_loader import timeline_loader_from_fov
from kitchen.loader.behavior_loader import behavior_loader_from_fov
from kitchen.structure.hierarchical_data_structure import DataSet, Cohort, FovDay, Mice, Fov, Session, CellSession, Trial
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate, TemporalUID, ObjectUID
from kitchen.structure.neural_data_structure import NeuralData
import kitchen.configs.routing as routing

"""
Hierarchical data loading and splitting functions.

Supports the complete experimental hierarchy:
Cohort -> Mice -> Fov -> Session -> CellSession -> Trial
                     FovDay </   Cell </

Each split function takes a parent node and creates child nodes by scanning
the file system structure and loading appropriate data.
"""

def SplitCohort2Mice(corhort_node: Cohort) -> List[Mice]:
    """Split a cohort node into multiple mice nodes."""
    default_cohort_data_path = routing.default_data_path(corhort_node)

    mice_id_list = [name for name in os.listdir(default_cohort_data_path) 
                    if path.isdir(path.join(default_cohort_data_path, name))]
    mice_nodes = []
    cur_temporal_uid = corhort_node.coordinate.temporal_uid
    cur_object_uid = corhort_node.coordinate.object_uid
    for mice_id in mice_id_list:
        mice_nodes.append(
            Mice(coordinate=TemporalObjectCoordinate(
                    temporal_uid=cur_temporal_uid,
                    object_uid=cur_object_uid.child_uid(mice_id=mice_id)),
                data=NeuralData(),)
        )
    return mice_nodes

def SplitMice2Fov(mice_node: Mice) -> List[Fov]:
    """Split a mice node into multiple fov nodes."""
    default_mice_data_path = routing.default_data_path(mice_node)

    fov_id_list = [name for name in os.listdir(default_mice_data_path) 
                    if path.isdir(path.join(default_mice_data_path, name))]
    fov_nodes = []
    cur_temporal_uid = mice_node.coordinate.temporal_uid
    cur_object_uid = mice_node.coordinate.object_uid
    for fov_id in fov_id_list:
        fov_nodes.append(
            Fov(coordinate=TemporalObjectCoordinate(
                    temporal_uid=cur_temporal_uid,
                    object_uid=cur_object_uid.child_uid(fov_id=fov_id)),
                data=NeuralData(),)
        )
    return fov_nodes

def SplitFov2Session(fov_node: Fov) -> List[Session]:
    """
    Split a fov node into multiple session nodes.
    All data are aligned to timeline files.
    """

    # Load timeline from fov node
    timeline_dict = {}
    for day_name, session_name, timeline in timeline_loader_from_fov(fov_node):
        session_coordinate = TemporalObjectCoordinate(
            temporal_uid=fov_node.coordinate.temporal_uid.child_uid(
                day_id=day_name, session_id=session_name),
            object_uid=fov_node.coordinate.object_uid)
        timeline_dict[session_coordinate] = timeline
    num_session = len(timeline_dict)
    assert num_session > 0, f"Cannot find any session in {fov_node}"

    """Load fluorescence and behavior from fov node."""
    session_nodes = []
    for fluorescence, behavior_dict, (session_coordinate, timeline) in zip(
        fluorescence_loader_from_fov(fov_node, timeline_dict),
        behavior_loader_from_fov(fov_node, timeline_dict),
        timeline_dict.items()
    ):
        session_nodes.append(
            Session(
                coordinate=session_coordinate,
                data=NeuralData(
                    fluorescence=fluorescence,
                    timeline=timeline,
                    **behavior_dict,
                ))
        )
    return session_nodes

def SplitSession2CellSession(session_node: Session) -> List[CellSession]: 
    """Split a session node into multiple cell session nodes."""
    cell_session_nodes = []
    if session_node.data.fluorescence is not None:
        for cell_idx, cell_name in enumerate(session_node.data.fluorescence.cell_idx):
            cell_session_fluorescence = session_node.data.fluorescence.extract_cell(cell_idx)
            cell_session_coordinate = TemporalObjectCoordinate(
                temporal_uid=session_node.coordinate.temporal_uid,
                object_uid=session_node.coordinate.object_uid.child_uid(cell_id=cell_name))
            cell_session_nodes.append(
                CellSession(
                    coordinate=cell_session_coordinate,
                    data=session_node.data.shadow_clone(fluorescence=cell_session_fluorescence)
                )
            )            
    else:
        cell_session_coordinate = TemporalObjectCoordinate(
            temporal_uid=session_node.coordinate.temporal_uid,
            object_uid=session_node.coordinate.object_uid.child_uid(cell_id=-1))
        cell_session_nodes.append(
            CellSession(
                coordinate=cell_session_coordinate,
                data=session_node.data.shadow_clone()
            )
        )
    return cell_session_nodes

def MergeSession2FovDay(dataset: DataSet) -> List[FovDay]:
    """Merge session nodes into fov day nodes."""
    fov_day_nodes = []
    # split sessions by day id
    dict_session_by_day = dataset.select("session").split_by("day_id")
    for day_id, session_nodes in dict_session_by_day.items():
        assert day_id is not None, f"Cannot find day id in {session_nodes}"
        # split sessions within same day by object_uid
        dict_same_day_sessions_by_fov = DataSet(session_nodes).split_by("object_uid")
        print(day_id, dict_same_day_sessions_by_fov.keys())
        for object_uid, same_fov_day_session_nodes in dict_same_day_sessions_by_fov.items():
            assert object_uid is not None, f"Cannot find object_uid in {same_fov_day_session_nodes}"
            example_session_node = same_fov_day_session_nodes[0]
            fov_day_coordinate = TemporalObjectCoordinate(
                temporal_uid=example_session_node.coordinate.temporal_uid.transit("day"),
                object_uid=example_session_node.coordinate.object_uid)            
            fov_day_nodes.append(
                FovDay(
                    coordinate=fov_day_coordinate,
                    data=NeuralData(),
                )
            )            
    return fov_day_nodes

def SplitCellSession2Trial(cell_session_node: CellSession) -> List[Trial]: 
    """Split a cell session node into multiple trial nodes."""

    def trial_splitter_default(cell_session_node: CellSession) -> List[Trial]:
        assert cell_session_node.data.timeline is not None, f"Cannot find timeline in {cell_session_node}"
        
        # find the all trial align event
        trial_aligns = None
        for trial_align_candidate in TRIAL_ALIGN_EVENT_DEFAULT:
            trial_aligns = cell_session_node.data.timeline.filter(trial_align_candidate)
            if len(trial_aligns) > 0:
                break
        assert trial_aligns is not None, f"Cannot find any trial align event in {cell_session_node}," \
            f" as its timeline is {cell_session_node.data.timeline}"

        # split the cell session into trials
        trial_nodes = []
        for trial_id, (trial_t, trial_v) in enumerate(trial_aligns):
            trial_coordinate = TemporalObjectCoordinate(
                temporal_uid=cell_session_node.coordinate.temporal_uid.child_uid(chunk_id=trial_id),
                object_uid=cell_session_node.coordinate.object_uid)
            trial_neural_data = cell_session_node.data.segment(
                start_t=trial_t + TRIAL_RANGE_RELATIVE_TO_ALIGNMENT[0],
                end_t=trial_t + TRIAL_RANGE_RELATIVE_TO_ALIGNMENT[1])
            trial_nodes.append(
                Trial(
                    coordinate=trial_coordinate,
                    data=trial_neural_data
                )
            )
        return trial_nodes
    
    return trial_splitter_default(cell_session_node)
        

def cohort_loader(template_id: str, cohort_id: str) -> DataSet:
    """
    Load data for a cohort, including all its children nodes.

    Loader follows such a hierarchy:
    Cohort -> Mice -> Fov -> Cell -> CellSession -> Trial
                          -> Session 
    Args:
        template_id (str): The template ID for initial cohort node.
        cohort_id (str): The cohort ID for initial cohort node.

    Returns:
        DataSet: A DataSet containing all the nodes for the cohort and its children.
    """
    # Initialize the cohort node
    init_cohort_node = Cohort(
        coordinate=TemporalObjectCoordinate(
            temporal_uid=TemporalUID(template_id=template_id), 
            object_uid=ObjectUID(cohort_id=cohort_id)),
        data=NeuralData(),)    
    loaded_data = DataSet(nodes=[init_cohort_node])

    # Cohort to Mice
    for cohort_node in loaded_data.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        loaded_data.add_node(SplitCohort2Mice(cohort_node))
    
    # Mice to Fov
    for mice_node in loaded_data.select("mice"):
        assert isinstance(mice_node, Mice)
        loaded_data.add_node(SplitMice2Fov(mice_node))

    # Fov to Session
    for fov_node in loaded_data.select("fov"):
        assert isinstance(fov_node, Fov)
        loaded_data.add_node(SplitFov2Session(fov_node))

    # Session to FovDay
    loaded_data.add_node(MergeSession2FovDay(loaded_data))

    # Session to CellSession
    for session_node in loaded_data.select("session"):
        assert isinstance(session_node, Session)
        loaded_data.add_node(SplitSession2CellSession(session_node)) 

    # CellSession to Trial
    for cell_session_node in tqdm(loaded_data.select("cellsession"), 
                                  desc="Splitting cell session to trials", unit="cell session"):
        assert isinstance(cell_session_node, CellSession)
        loaded_data.add_node(SplitCellSession2Trial(cell_session_node))

    return loaded_data


def naive_loader(template_id: str, cohort_id: str) -> DataSet:
    """
    Load data for a cohort to FOV level, for basic diagnosis & data manipulation.

    Loader follows such a hierarchy:
    Cohort -> Mice -> Fov 
    Args:
        template_id (str): The template ID for initial cohort node.
        cohort_id (str): The cohort ID for initial cohort node.

    Returns:
        DataSet: A DataSet containing all the nodes for the cohort.
    """
    # Initialize the cohort node
    init_cohort_node = Cohort(
        coordinate=TemporalObjectCoordinate(
            temporal_uid=TemporalUID(template_id=template_id), 
            object_uid=ObjectUID(cohort_id=cohort_id)),
        data=NeuralData(),)    
    loaded_data = DataSet(nodes=[init_cohort_node])

    # Cohort to Mice
    for cohort_node in loaded_data.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        loaded_data.add_node(SplitCohort2Mice(cohort_node))
    
    # Mice to Fov
    for mice_node in loaded_data.select("mice"):
        assert isinstance(mice_node, Mice)
        loaded_data.add_node(SplitMice2Fov(mice_node))

    return loaded_data