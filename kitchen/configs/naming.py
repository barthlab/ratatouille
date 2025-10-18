"""
General naming function for Node objects with type-specific implementations.
"""

from functools import singledispatch

from kitchen.structure.hierarchical_data_structure import DataSet, Day, FovTrial, Node, Trial, CellSession, Session, Cell, FovDay, Fov, Mice, Cohort


@singledispatch
def get_node_name(node: Node) -> str:
    """
    Get a minimal yet descriptive string representation for a Node object.
    """
    return str(node)

@get_node_name.register
def _(node: FovTrial) -> str:
    chunk_id = node.coordinate.temporal_uid.chunk_id
    fov_id = node.coordinate.object_uid.fov_id
    return f"FOV_{fov_id}_Trial_{chunk_id}"


@get_node_name.register
def _(node: Trial) -> str:
    chunk_id = node.coordinate.temporal_uid.chunk_id
    cell_id = node.coordinate.object_uid.cell_id
    return f"Cell_{cell_id}_Trial_{chunk_id}"


@get_node_name.register
def _(node: CellSession) -> str:
    session_id = node.coordinate.temporal_uid.session_id
    cell_id = node.coordinate.object_uid.cell_id
    return f"{session_id}_Cell_{cell_id}"


@get_node_name.register
def _(node: Session) -> str:
    session_id = node.coordinate.temporal_uid.session_id
    return f"{session_id}"


@get_node_name.register
def _(node: Cell) -> str:
    cell_id = node.coordinate.object_uid.cell_id
    return f"Cell_{cell_id}"


@get_node_name.register
def _(node: FovDay) -> str:
    fov_id = node.coordinate.object_uid.fov_id
    day_id = node.coordinate.temporal_uid.day_id
    return f"FOV_{fov_id}_{day_id}"

@get_node_name.register
def _(node: Day) -> str:
    mice_id = node.coordinate.object_uid.mice_id
    day_id = node.coordinate.temporal_uid.day_id
    return f"{mice_id}_{day_id}"


@get_node_name.register
def _(node: Fov) -> str:
    fov_id = node.coordinate.object_uid.fov_id
    return f"FOV_{fov_id}"


@get_node_name.register
def _(node: Mice) -> str:
    mice_id = node.coordinate.object_uid.mice_id
    return f"{mice_id}"


@get_node_name.register
def _(node: Cohort) -> str:
    cohort_id = node.coordinate.object_uid.cohort_id
    return f"{cohort_id}"



def get_dataset_name(dataset: DataSet, break2line: bool = True) -> str:
    breakline = "\n" if break2line else " "
    return f"Dataset_{dataset.name}{breakline}{dataset.root_coordinate}"
