from glob import glob
import os
import os.path as path
from typing import List, Optional

from kitchen.structure.hierarchical_data_structure import DataSet, Node


ROOT_PATH = (r"C:\Users\maxyc\PycharmProjects\Ratatouille")
DATA_PATH = path.join(ROOT_PATH, "ingredients")
FIGURE_PATH = path.join(ROOT_PATH, "cuisine")
TEST_PATH = path.join(ROOT_PATH, "critic")



def robust_path_join(*args) -> str:
    """Safely join path components, filtering out None values."""
    assert len([arg for arg in args if arg is not None]) > 0, \
         f"at least one argument is required, got {args}"
    return path.join(*[str(arg) for arg in args if arg is not None])


def smart_path_append(prev_path, new_name: str) -> str:
    if new_name.startswith("_"):
        head, tail = os.path.split(prev_path)
        return robust_path_join(head, f"{tail}{new_name}")
    elif "{}" in new_name:
        head, tail = os.path.split(prev_path)
        return robust_path_join(head, new_name.replace("{}", tail))
    else:
        return robust_path_join(prev_path, new_name)


def default_data_path(node: Node, check_exist: bool = True) -> str:
    """Generate the default file system path for a node's data."""
    node_obj_uid = node.coordinate.object_uid
    node_data_path = robust_path_join(DATA_PATH, node.coordinate.temporal_uid.template_id, 
                                      *[value for name, value in node_obj_uid if value is not None])
    if check_exist:
        assert path.exists(node_data_path), f"Cannot find data path: {node_data_path}"
    return node_data_path


def default_fig_path(dataset: DataSet, fig_name: Optional[str] = None) -> str:
    """Generate the default file system path for a dataset's figures."""
    root_coordinate = dataset.root_coordinate
    fig_path_values = [
        root_coordinate.temporal_uid.template_id,
        root_coordinate.object_uid.cohort_id,
        root_coordinate.object_uid.mice_id,
        root_coordinate.object_uid.fov_id,
        root_coordinate.temporal_uid.day_id,
        root_coordinate.temporal_uid.session_id,
    ]
    fig_path = robust_path_join(FIGURE_PATH, *fig_path_values) 
    if fig_name is not None:
        fig_path = smart_path_append(fig_path, fig_name)
    return fig_path


def search_pattern_file(pattern: str, search_dir: str) -> List[str]:
    recursive_path = os.path.join(search_dir, '**', pattern)
    return list(glob(recursive_path, recursive=True))
