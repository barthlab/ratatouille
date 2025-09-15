from itertools import zip_longest
from typing import Any, Dict, List, Optional, overload
import os
import os.path as path
from tqdm import tqdm
import logging

from kitchen.loader.fluorescence_loader import fluorescence_loader_from_node
from kitchen.settings.trials import TRIAL_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.loader.timeline_loader import timeline_loader_from_fov
from kitchen.loader.behavior_loader import behavior_loader_from_node
from kitchen.structure.hierarchical_data_structure import DataSet, Cohort, FovTrial, MergeFovDay2Day, MergeSession2FovDay, Mice, Fov, Session, CellSession, Trial
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate, TemporalUID, ObjectUID
from kitchen.structure.neural_data_structure import NeuralData
import kitchen.configs.routing as routing

logger = logging.getLogger(__name__)

"""
Hierarchical data loading and splitting functions.

Supports the complete experimental hierarchy:
                        object
                    Cohort      mice       fov        cell
                    ________________________________________
          template  | Cohort -> Mice -> Fov           Cell
 temporal           |                    |              ^
               day  |            Day <---|-- FovDay     |
                    |                    V    ^         |
           session  |                   Session -> CellSession
                    |                      V           V
             chunk  |                   FovTrial     Trial


                                /-> FovTrial
Cohort -> Mice -> Fov -> Session -> CellSession -> Trial
                 Day <- FovDay </       Cell </  

[Cohort] → [Mice] → [Fov] → [Session] ┬─→ [FovDay] ───→ [Day]
                                      │
                                      ├─→ [FovTrial]
                                      │
                                      └─→ [CellSession] ┬─→ [Trial]
                                                        │
                                                        └─→ [Cell]
Each split function takes a parent node and creates child nodes by scanning
the file system structure and loading appropriate data.
"""

def _SplitCohort2Mice(cohort_node: Cohort, **kwargs: Any) -> List[Mice]:
    """Split a cohort node into multiple mice nodes."""
    default_cohort_data_path = routing.default_data_path(cohort_node)

    mice_id_list = [name for name in os.listdir(default_cohort_data_path) 
                    if path.isdir(path.join(default_cohort_data_path, name))]
    mice_nodes = []
    cur_temporal_uid = cohort_node.coordinate.temporal_uid
    cur_object_uid = cohort_node.coordinate.object_uid
    for mice_id in mice_id_list:
        mice_nodes.append(
            Mice(coordinate=TemporalObjectCoordinate(
                    temporal_uid=cur_temporal_uid,
                    object_uid=cur_object_uid.child_uid(mice_id=mice_id)),
                data=NeuralData(),)
        )
    return mice_nodes

def _SplitMice2Fov(mice_node: Mice, **kwargs: Any) -> List[Fov]:
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

def _SplitFov2Session(fov_node: Fov, loaders: Dict[str, str], **kwargs: Any) -> List[Session]:
    """
    Split a fov node into multiple session nodes.
    All data are aligned to timeline files.
    """

    # Load timeline from fov node
    timeline_dict = {}
    assert 'timeline' in loaders, "Timeline loader must be specified"
    for day_name, session_name, timeline in timeline_loader_from_fov(fov_node, loaders['timeline']):
        session_coordinate = TemporalObjectCoordinate(
            temporal_uid=fov_node.coordinate.temporal_uid.child_uid(
                day_id=day_name, session_id=session_name),
            object_uid=fov_node.coordinate.object_uid)
        timeline_dict[session_coordinate] = timeline
    num_session = len(timeline_dict)
    assert num_session > 0, f"Cannot find any timeline in {fov_node}"

    """Load fluorescence and behavior from fov node."""
    session_nodes = []
    for fluorescence, behavior_dict, (session_coordinate, timeline) in zip_longest(
        fluorescence_loader_from_node(fov_node, timeline_dict, loaders.get('fluorescence')),
        behavior_loader_from_node(fov_node, timeline_dict, loaders.get('behavior')),
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

def _SplitSession2CellSession(session_node: Session, **kwargs: Any) -> List[CellSession]: 
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


@overload
def _trial_splitter(session_level_node: Session, split_by: list[str]) -> List[FovTrial]: ...

@overload
def _trial_splitter(session_level_node: CellSession, split_by: list[str]) -> List[Trial]: ...

def _trial_splitter(session_level_node: CellSession | Session, split_by: list[str]) -> List[Trial] | List[FovTrial]:
    assert session_level_node.data.timeline is not None, f"Cannot find timeline in {session_level_node}"
    target_node_type = Trial if isinstance(session_level_node, CellSession) else FovTrial

    # find the all trial align event
    trial_aligns = session_level_node.data.iterablize(*split_by)

    # split the session level node into trial level nodes
    trial_nodes = []
    for trial_id, trial_t in enumerate(trial_aligns):
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

def _SplitCellSession2Trial(cell_session_node: CellSession, **kwargs: Any) -> List[Trial]: 
    """Split a cell session node into multiple trial nodes."""
    return _trial_splitter(cell_session_node, **kwargs)

def _SplitSession2FovTrial(session_node: Session, **kwargs: Any) -> List[FovTrial]: 
    """Split a session node into multiple fov trial nodes."""
    return _trial_splitter(session_node, **kwargs)


def cohort_loader(template_id: str, cohort_id: str, recipe: dict, name: Optional[str] = None) -> DataSet:
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
    assert recipe['data_type'] == "two_photon", f"Expected two photon recipe, got {recipe['data_type']}"

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
    
    # Return if recipe doesn't specify trial spliter
    if "trial split" not in recipe or recipe['trial split'] == "none":
        logger.info("Recipe doesn't specify trial spliter, stop loading at fov level.")
        return loaded_data

    # Fov to Session
    assert "loaders" in recipe, "Recipe doesn't specify timeline loader"
    for fov_node in loaded_data.select("fov"):
        assert isinstance(fov_node, Fov)
        loaded_data.add_node(_SplitFov2Session(fov_node, loaders=recipe['loaders']))

    # Session to FovDay
    loaded_data.add_node(MergeSession2FovDay(loaded_data))

    # FovDay to Day
    loaded_data.add_node(MergeFovDay2Day(loaded_data))

    # Session to CellSession
    for session_node in loaded_data.select("session"):
        assert isinstance(session_node, Session)
        loaded_data.add_node(_SplitSession2CellSession(session_node)) 

    # CellSession to Trial
    for cell_session_node in tqdm(loaded_data.select("cellsession"), 
                                  desc="Splitting cell session to trials", unit="cell session"):
        assert isinstance(cell_session_node, CellSession)
        loaded_data.add_node(_SplitCellSession2Trial(cell_session_node, split_by=recipe['trial split']))
        
    # Session to FovTrial
    for session_node in tqdm(loaded_data.select("session"), 
                                  desc="Splitting session to fov trials", unit="session"):
        assert isinstance(session_node, Session)
        loaded_data.add_node(_SplitSession2FovTrial(session_node, split_by=recipe['trial split']))

    return loaded_data

