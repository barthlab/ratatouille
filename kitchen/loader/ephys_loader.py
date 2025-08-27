from typing import List, Optional

from kitchen.loader.behavior_loader import behavior_loader_from_node
from kitchen.loader.potential_loader import potential_loader_from_cohort
from kitchen.structure.hierarchical_data_structure import CellSession, Cohort, DataSet
from kitchen.structure.meta_data_structure import ObjectUID, TemporalObjectCoordinate, TemporalUID
from kitchen.structure.neural_data_structure import NeuralData


"""
Ephys loading loads data at cell-level
"""

def _SplitCohort2CellSession(cohort_node: Cohort) -> List[CellSession]:
    """Split a cohort node into multiple cell session nodes."""
    cell_session_nodes = []
    for cell_coordinate, timeline, cam_timeline, potential in potential_loader_from_cohort(cohort_node): 
        behavior_dict = next(behavior_loader_from_node(cohort_node, {cell_coordinate: cam_timeline}))        
        cell_session_nodes.append(
            CellSession(coordinate=cell_coordinate,
                        data=NeuralData(
                            timeline=timeline,
                            potential=potential,
                            **behavior_dict,
                        ))
        )
        print(f"Loaded {cell_session_nodes[-1]}")
    return cell_session_nodes

def cohort_loader(template_id: str, cohort_id: str, name: Optional[str] = None) -> DataSet:
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
        loaded_data.add_node(_SplitCohort2CellSession(cohort_node))

    return loaded_data
