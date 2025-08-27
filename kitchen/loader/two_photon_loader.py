from itertools import zip_longest
from typing import List, Optional, overload
import os
import os.path as path
from tqdm import tqdm

from kitchen.loader.fluorescence_loader import fluorescence_loader_from_node
from kitchen.settings.timeline import TRIAL_ALIGN_EVENT_DEFAULT
from kitchen.settings.trials import TRIAL_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.loader.timeline_loader import timeline_loader_from_fov
from kitchen.loader.behavior_loader import behavior_loader_from_node
from kitchen.structure.hierarchical_data_structure import DataSet, Cohort, Day, FovDay, FovTrial, Mice, Fov, Session, CellSession, Trial
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate, TemporalUID, ObjectUID
from kitchen.structure.neural_data_structure import NeuralData
import kitchen.configs.routing as routing
from kitchen.utils.sequence_kit import split_by

"""
Hierarchical data loading and splitting functions.

Supports the complete experimental hierarchy:
                               /-> FovTrial
Cohort -> Mice -> Fov -> Session -> CellSession -> Trial
               Day <- FovDay </   Cell </  

Each split function takes a parent node and creates child nodes by scanning
the file system structure and loading appropriate data.
"""

def _SplitCohort2Mice(corhort_node: Cohort) -> List[Mice]:
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

def _SplitMice2Fov(mice_node: Mice) -> List[Fov]:
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

def _SplitFov2Session(fov_node: Fov) -> List[Session]:
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
    for fluorescence, behavior_dict, (session_coordinate, timeline) in zip_longest(
        fluorescence_loader_from_node(fov_node, timeline_dict),
        behavior_loader_from_node(fov_node, timeline_dict),
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

def _SplitSession2CellSession(session_node: Session) -> List[CellSession]: 
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

def _MergeSession2FovDay(dataset: DataSet) -> List[FovDay]:
    """Merge session nodes into fov day nodes."""
    fov_day_nodes = []
    
    # split sessions by day id
    dict_session_by_day = split_by(dataset.select("session"), attr_name= "temporal_uid.parent_uid")
    for day_id, session_nodes in dict_session_by_day.items():
        assert day_id is not None, f"Cannot find day id in {session_nodes}"

        # split sessions within same day by object_uid
        dict_same_day_sessions_by_fov = split_by(session_nodes, attr_name= "object_uid")
        for object_uid, same_fov_day_session_nodes in dict_same_day_sessions_by_fov.items():
            assert object_uid is not None, f"Cannot find object_uid in {same_fov_day_session_nodes}"
            example_session_node = same_fov_day_session_nodes[0]
            
            # create fov day node
            fov_day_coordinate = TemporalObjectCoordinate(
                temporal_uid=example_session_node.coordinate.temporal_uid.parent_uid,
                object_uid=object_uid)            
            fov_day_nodes.append(
                FovDay(
                    coordinate=fov_day_coordinate,
                    data=NeuralData(),
                )
            )            
    return fov_day_nodes

def _MergeFovDay2Day(dataset: DataSet) -> List[Day]:
    """Merge fov day nodes into day nodes."""
    day_nodes = []
    
    # split fov day by mice id
    dict_fov_day_by_mice = split_by(dataset.select("fovday"), attr_name= "object_uid.parent_uid")
    for mice_id, fov_day_nodes in dict_fov_day_by_mice.items():
        assert mice_id is not None, f"Cannot find mice id in {fov_day_nodes}"

        # split fov day within same mice by day id
        dict_same_mice_fov_day_by_day = split_by(fov_day_nodes, attr_name= "temporal_uid")
        for temporal_uid, same_day_fov_day_nodes in dict_same_mice_fov_day_by_day.items():
            assert temporal_uid is not None, f"Cannot find temporal_uid in {same_day_fov_day_nodes}"
            example_fov_day_node = same_day_fov_day_nodes[0]
            
            # create day node
            day_coordinate = TemporalObjectCoordinate(
                temporal_uid=temporal_uid,
                object_uid=example_fov_day_node.coordinate.object_uid.parent_uid)
            day_nodes.append(
                Day(
                    coordinate=day_coordinate,
                    data=NeuralData(),
                )
            )
    return day_nodes

@overload
def _trial_splitter_default(session_level_node: Session) -> List[FovTrial]: ...

@overload
def _trial_splitter_default(session_level_node: CellSession) -> List[Trial]: ...

def _trial_splitter_default(session_level_node: CellSession | Session) -> List[Trial] | List[FovTrial]:
    assert session_level_node.data.timeline is not None, f"Cannot find timeline in {session_level_node}"
    target_node_type = Trial if isinstance(session_level_node, CellSession) else FovTrial

    # find the all trial align event
    trial_aligns = session_level_node.data.timeline.advanced_filter(TRIAL_ALIGN_EVENT_DEFAULT)

    # split the session level node into trial level nodes
    trial_nodes = []
    for trial_id, (trial_t, trial_v) in enumerate(trial_aligns):
        trial_coordinate = TemporalObjectCoordinate(
            temporal_uid=session_level_node.coordinate.temporal_uid.child_uid(chunk_id=trial_id),
            object_uid=session_level_node.coordinate.object_uid)
        trial_neural_data = session_level_node.data.segment(
            start_t=trial_t + TRIAL_RANGE_RELATIVE_TO_ALIGNMENT[0],
            end_t=trial_t + TRIAL_RANGE_RELATIVE_TO_ALIGNMENT[1])
        trial_nodes.append(
            target_node_type(
                coordinate=trial_coordinate,
                data=trial_neural_data
            )
        )
    return trial_nodes

def _SplitCellSession2Trial(cell_session_node: CellSession) -> List[Trial]: 
    """Split a cell session node into multiple trial nodes."""
    return _trial_splitter_default(cell_session_node)

def _SplitSession2FovTrial(session_node: Session) -> List[FovTrial]: 
    """Split a session node into multiple fov trial nodes."""
    return _trial_splitter_default(session_node)


def cohort_loader(template_id: str, cohort_id: str, name: Optional[str] = None) -> DataSet:
    """
    Load data for a cohort, including all its children nodes.

    Loader follows such a hierarchy:    
    Cohort -> Mice -> Fov -> Session -> CellSession -> Trial
                     FovDay <-/     Cell <-/

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
    loaded_data = DataSet(name=name if name is not None else cohort_id, nodes=[init_cohort_node])

    # Cohort to Mice
    for cohort_node in loaded_data.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        loaded_data.add_node(_SplitCohort2Mice(cohort_node))
    
    # Mice to Fov
    for mice_node in loaded_data.select("mice"):
        assert isinstance(mice_node, Mice)
        loaded_data.add_node(_SplitMice2Fov(mice_node))

    # Fov to Session
    for fov_node in loaded_data.select("fov"):
        assert isinstance(fov_node, Fov)
        loaded_data.add_node(_SplitFov2Session(fov_node))

    # Session to FovDay
    loaded_data.add_node(_MergeSession2FovDay(loaded_data))

    # FovDay to Day
    loaded_data.add_node(_MergeFovDay2Day(loaded_data))

    # Session to CellSession
    for session_node in loaded_data.select("session"):
        assert isinstance(session_node, Session)
        loaded_data.add_node(_SplitSession2CellSession(session_node)) 

    # CellSession to Trial
    for cell_session_node in tqdm(loaded_data.select("cellsession"), 
                                  desc="Splitting cell session to trials", unit="cell session"):
        assert isinstance(cell_session_node, CellSession)
        loaded_data.add_node(_SplitCellSession2Trial(cell_session_node))
        
    # Session to FovTrial
    for session_node in tqdm(loaded_data.select("session"), 
                                  desc="Splitting session to fov trials", unit="session"):
        assert isinstance(session_node, Session)
        loaded_data.add_node(_SplitSession2FovTrial(session_node))

    return loaded_data


def naive_loader(template_id: str, cohort_id: str, name: Optional[str] = None) -> DataSet:
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
    loaded_data = DataSet(name=name if name is not None else cohort_id, nodes=[init_cohort_node])

    # Cohort to Mice
    for cohort_node in loaded_data.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        loaded_data.add_node(_SplitCohort2Mice(cohort_node))
    
    # Mice to Fov
    for mice_node in loaded_data.select("mice"):
        assert isinstance(mice_node, Mice)
        loaded_data.add_node(_SplitMice2Fov(mice_node))

    return loaded_data