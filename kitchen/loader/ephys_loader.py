from typing import List, Optional, overload
import logging

from kitchen.loader.behavior_loader import behavior_loader_from_node
from kitchen.loader.potential_loader import potential_loader_from_cohort
from kitchen.loader.spike_type_annotator import spike_annotation
from kitchen.loader.spike_waveform_curator import node_spike_waveform_curation
from kitchen.settings.timeline import TRIAL_ALIGN_EVENT_DEFAULT
from kitchen.settings.trials import TRIAL_RANGE_RELATIVE_TO_ALIGNMENT
from kitchen.structure.hierarchical_data_structure import CellSession, Cohort, DataSet, FovTrial, MergeCellSession2Session, Session, Trial
from kitchen.structure.meta_data_structure import ObjectUID, TemporalObjectCoordinate, TemporalUID
from kitchen.structure.neural_data_structure import NeuralData

logger = logging.getLogger(__name__)

"""
Ephys loading loads data at cell-level
"""

def _SplitCohort2CellSession(cohort_node: Cohort, loaders: dict[str, str], 
                             spike_curation: bool, curation_overwrite: bool) -> List[CellSession]:
    """Split a cohort node into multiple cell session nodes."""
    cell_session_nodes = []
    for cell_coordinate, timeline, cam_timeline, potential in potential_loader_from_cohort(cohort_node): 
        logger.info(f"Building {cell_coordinate}...")
        
        # load behavior
        if loaders.get('behavior') is None:
            behavior_dict = {}
        else:
            behavior_dict = next(behavior_loader_from_node(cohort_node, {cell_coordinate: cam_timeline}, loaders.get('behavior')))        

        new_cell_session_node = CellSession(
            coordinate=cell_coordinate,
            data=NeuralData(timeline=timeline, potential=potential, **behavior_dict,)
        )
        if spike_curation:
            node_spike_waveform_curation(new_cell_session_node, overwrite=curation_overwrite)
        cell_session_nodes.append(new_cell_session_node)
    return cell_session_nodes

def _SplitCellSession2Trial(session_level_node: CellSession, spike_annotator_name: Optional[str] = None) -> List[Trial]:
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
        
        new_trial_node = target_node_type(
            coordinate=trial_coordinate,
            data=trial_neural_data
        )

        # annotate spikes in trial nodes
        if spike_annotator_name is not None:
            spike_annotation(new_trial_node, trial_align=trial_t, spike_annotator_name=spike_annotator_name)

        trial_nodes.append(new_trial_node)
    return trial_nodes



def cohort_loader(template_id: str, cohort_id: str, recipe: dict, name: Optional[str] = None) -> DataSet:
    """
    Load a specific cohort, including all its children nodes.

    Loader follows such a hierarchy:
    Cohort ------> Cell -> Trial 
            Mice <-/    -> Spike   

    Args:
        template_id (str): The template ID for initial cohort node.
        cohort_id (str): The cohort ID for initial cohort node.

    Returns:
        DataSet: A DataSet containing all the nodes for the cohort and its children. 
    """
    init_cohort_node = Cohort(
        coordinate=TemporalObjectCoordinate(
            temporal_uid=TemporalUID(template_id=template_id), 
            object_uid=ObjectUID(cohort_id=cohort_id)),
        data=NeuralData(),)    
    loaded_data = DataSet(name=name if name is not None else cohort_id, nodes=[init_cohort_node])

    # Cohort to Cell Session
    for cohort_node in loaded_data.select("cohort"):
        assert isinstance(cohort_node, Cohort)
        loaded_data.add_node(_SplitCohort2CellSession(
            cohort_node, loaders=recipe['loaders'], 
            spike_curation=recipe.get("spike curation", True),
            curation_overwrite=recipe.get("curation overwrite", False)))
    
    # Cell Session to Trial
    for cell_session_node in loaded_data.select("cellsession"):
        assert isinstance(cell_session_node, CellSession)
        
        # Spike type annotation
        trial_nodes = _SplitCellSession2Trial(cell_session_node, spike_annotator_name=recipe.get("spike annotation"))
        # Spike type inherit
        for trial_node in trial_nodes:
            cell_session_node.potential.spikes.update_value(trial_node.potential.spikes)
            
        # Add trial nodes
        loaded_data.add_node(trial_nodes)

    return loaded_data
