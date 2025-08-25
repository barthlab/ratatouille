"""
Hierarchical Data Structure Module

Defines Node classes for organizing neuroscience data in spatial-temporal hierarchies
and DataSet class for managing collections of nodes.

Hierarchical Organization:
- Spatial: Cohort → Mice → FOV → Cell
- Temporal: Template → Day → Session → Chunk → Spike

Data Organization Matrix:
                                    Time →
                           ┌───────────────────────────────────────────────────────────────────┐
                           │     Session 1     │     Session 2     │  ...  │     Session N     │
        ┌──────────────────┼───────────────────────────────────────────────────────────────────┤
        │ FOV 1  Cell 1~K  │                    ░░░░░░░ GCaMP activity (FOV 1) ░░░░░░░░        │
        │ FOV 2  Cell 1~K  │                                                                   │
        │ ...              │                                                                   │
        └──────────────────┴───────────────────────────────────────────────────────────────────┘
                                    │■■■■■■■■■■■■■■ Behavior Data (All FOVs) ■■■■■■■■■■■■■■│

Node Classes: Spike, Trial, CellSession, Session, Cell, Fov, Mice, Cohort
Each enforces appropriate coordinate constraints for its hierarchy level.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, List, Optional, Dict, Self, Sequence, Union
import warnings
import pandas as pd

from kitchen.settings.structure import NULL_STRUCTURE_NAME
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate
from kitchen.structure.neural_data_structure import NeuralData
from kitchen.utils.sequence_kit import filter_by
from kitchen.writer.excel_writer import write_boolean_dataframe


@dataclass(eq=False)  # eq=False is needed because we define a custom __eq__
class Node:
    """
    Base class for hierarchical data nodes with coordinate validation.

    Represents experimental data at a specific hierarchy level. Validates that
    coordinates match expected temporal/spatial levels for each node type.
    """
    coordinate: TemporalObjectCoordinate
    data: NeuralData

    _expected_temporal_uid_level: ClassVar[Optional[str]] = None
    _expected_object_uid_level: ClassVar[Optional[str]] = None

    def __post_init__(self):
        """Validate coordinate UID types match expected hierarchy levels."""
        self._validate_uid('temporal_uid', self._expected_temporal_uid_level)
        self._validate_uid('object_uid', self._expected_object_uid_level)

    def _validate_uid(self, uid_name: str, expected_level: Optional[str]):
        """Validate a single UID against its expected hierarchy level."""
        if expected_level is None:
            return

        actual_uid = getattr(self.coordinate, uid_name)
        if not actual_uid.level == expected_level:
            raise TypeError(
                f"For {self.hash_key}, expected {uid_name} @ {expected_level}-level, "
                f"but got {actual_uid.level}-level uid with value {actual_uid}."
            )

    def __getattr__(self, name: str) -> Any:
        """Look into coordinate and data for attributes."""
        if hasattr(self.coordinate, name):
            return getattr(self.coordinate, name)
        if hasattr(self.data, name):
            return getattr(self.data, name)
        raise AttributeError(f"'{self.__class__.__name__}' object [{str(self)}] has no attribute '{name}'")

    @property
    def hash_key(self) -> str:
        """Return lowercase class name for node type identification."""
        return self.__class__.__name__.lower()

    def __eq__(self, other: object) -> bool:
        """Check equality based on coordinates."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.coordinate == other.coordinate

    def __hash__(self) -> int:
        """Return hash based on coordinate."""
        return hash(self.coordinate)

    def __str__(self) -> str:
        """Return readable string representation."""
        return f"{self.hash_key} @{self.coordinate} \n with {self.data}"
    
    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return self.__str__()
    
    def aligned_to(self, align_time: float) -> Self:
        """Align all data to a specific time point."""
        return self.__class__(coordinate=self.coordinate, data=self.data.aligned_to(align_time))
    
        

# --- Specialized Node subclasses for different hierarchy levels ---

class Spike(Node):
    """Individual spikes within trials. Temporal: spike, Spatial: cell."""
    _expected_temporal_uid_level = 'spike'
    _expected_object_uid_level = 'cell'

class Trial(Node):
    """Individual trials/chunks within sessions. Temporal: chunk, Spatial: cell."""
    _expected_temporal_uid_level = 'chunk'
    _expected_object_uid_level = 'cell'

class FovTrial(Node):
    """Individual trials/chunks within sessions. Temporal: chunk, Spatial: fov."""
    _expected_temporal_uid_level = 'chunk'
    _expected_object_uid_level = 'fov'

class CellSession(Node):
    """Cell data aggregated at session level. Temporal: session, Spatial: cell."""
    _expected_temporal_uid_level = 'session'
    _expected_object_uid_level = 'cell'

class Session(Node):
    """Session-level data for entire FOVs. Temporal: session, Spatial: fov."""
    _expected_temporal_uid_level = 'session'
    _expected_object_uid_level = 'fov'

class Cell(Node):
    """Individual cell data across experimental template. Temporal: template, Spatial: cell."""
    _expected_temporal_uid_level = 'template'
    _expected_object_uid_level = 'cell'

class FovDay(Node):
    """FOV data for a specific day. Temporal: day, Spatial: fov."""
    _expected_temporal_uid_level = 'day'
    _expected_object_uid_level = 'fov'

class Fov(Node):
    """FOV data across experimental template. Temporal: template, Spatial: fov."""
    _expected_temporal_uid_level = 'template'
    _expected_object_uid_level = 'fov'

class Day(Node):
    """Day-level data for entire FOVs. Temporal: day, Spatial: mice."""
    _expected_temporal_uid_level = 'day'
    _expected_object_uid_level = 'mice'

class Mice(Node):
    """Mouse-level data across experimental template. Temporal: template, Spatial: mice."""
    _expected_temporal_uid_level = 'template'
    _expected_object_uid_level = 'mice'

class Cohort(Node):
    """Cohort-level data across experimental template. Temporal: template, Spatial: cohort."""
    _expected_temporal_uid_level = 'template'
    _expected_object_uid_level = 'cohort'


@dataclass
class DataSet:
    """
    Collection of hierarchical data nodes with efficient querying capabilities in a partially ordered set.

    Manages Node collections with fast lookup by type and functional filtering.
    Supports adding individual nodes, lists, or other DataSets with duplicate detection.
    """
    name: str
    nodes: List[Node] = field(default_factory=list)
    _fast_lookup: Dict[str, set[Node]] = field(repr=False, init=False)
    _root_coordinate: TemporalObjectCoordinate | int = field(repr=False, init=False)

    def __post_init__(self):
        """Initialize fast lookup dictionary for efficient querying."""
        self._fast_lookup = defaultdict(set)
        for node in self.nodes:
            self._fast_lookup[node.hash_key].add(node)
        self._root_coordinate = sum([node.coordinate for node in self.nodes], 0)

    def add_node(self, nodes: Union[Node, Sequence[Node], "DataSet"]):
        """Add nodes with duplicate detection. Supports single node, sequence of nodes, and DataSets."""
        if isinstance(nodes, DataSet):
            # Add all nodes from another DataSet
            for single_node in nodes:
                self.add_node(single_node)
        elif isinstance(nodes, Sequence) and not isinstance(nodes, str) and all(isinstance(single_node, Node) for single_node in nodes):
            # Add multiple nodes from sequence (excluding strings which are also sequences)
            for single_node in nodes:
                self.add_node(single_node)
        elif isinstance(nodes, Node):
            # Add single node with duplicate detection
            if nodes in self._fast_lookup[nodes.hash_key]:
                warnings.warn(f"Node {nodes} already exists in the dataset")
                return
            self.nodes.append(nodes)
            self._root_coordinate = self._root_coordinate + nodes.coordinate
            self._fast_lookup[nodes.hash_key].add(nodes)
        else:
            raise TypeError(f"Cannot add object of type {type(nodes)} to DataSet")

    def __iter__(self) -> Generator[Node, None, None]:
        """Iterate over all nodes."""
        for node in sorted(self.nodes, key=lambda node: node.coordinate):
            yield node
        
    def __len__(self) -> int:
        """Return number of nodes."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        """Return string representation."""
        return self.__str__()

    def __str__(self) -> str:
        """Return string representation """
        summary_str = f"Dataset {self.name} with {len(self)} nodes:\n"
        for node_name, node_set in self._fast_lookup.items():
            node_list = sorted(node_set, key=lambda node: node.coordinate)
            node_concat_string = "\n".join(str(node) for node in node_list[:3])
            summary_str += f"{len(node_list)} {node_name}: \n{node_concat_string}\n ... \n"
        return summary_str

    def __add__(self, other: "DataSet") -> "DataSet":
        """Combine two datasets into one."""
        new_dataset = DataSet(name=self.name + "_" + other.name, nodes=self.nodes + other.nodes)
        return new_dataset
    
    def __radd__(self, other: "DataSet") -> "DataSet":
        """Combine two datasets into one."""
        return self.__add__(other)
        
    def subset(self, _specified_name: str = NULL_STRUCTURE_NAME, _empty_warning: bool = True, **criterion: Any) -> "DataSet":
        """Filter all nodes by criterion function."""
        selected_nodes = filter_by(self.nodes, **criterion)
        if _empty_warning and len(selected_nodes) == 0:
            warnings.warn("Subset operation resulted in 0 nodes, check your criterion function")
        return DataSet(name=_specified_name, nodes=selected_nodes)

    def subtree(self, root_node: Node, _specified_name: str = NULL_STRUCTURE_NAME, _empty_warning: bool = True) -> "DataSet":
        """Extract a subtree rooted at a specific node."""
        selected_nodes = [node for node in self if root_node.coordinate.contains(node.coordinate)]
        if _empty_warning and len(selected_nodes) == 0:
            warnings.warn("Subtree operation resulted in 0 nodes, check your root node")
        return DataSet(name=_specified_name, nodes=selected_nodes)

    def select(self, hash_key: str, _specified_name: str = NULL_STRUCTURE_NAME, _empty_warning: bool = True, **criterion: Any) -> "DataSet":
        """Filter nodes of specific type by criterion. More efficient than subset()."""
        assert hash_key == hash_key.lower(), f"hash_key must be lower case, but got {hash_key}"
        selected_nodes = filter_by(self._fast_lookup[hash_key], **criterion)
        if _empty_warning and len(selected_nodes) == 0:
            warnings.warn("Select operation resulted in 0 nodes, check your criterion function")
        return DataSet(name=_specified_name, nodes=selected_nodes)
    
    def rule_based_selection(self, rule_dict: Dict[str, dict]) -> Dict[str, "DataSet"]:
        """Select nodes based on pre-defined rules."""    
        selected_nodes = {}    
        for name, rule_definition in rule_dict.items():
            this_type_nodes = self.select(**rule_definition)      
            if len(this_type_nodes) > 0:
                selected_nodes[name] = this_type_nodes      
        return selected_nodes


    def status(self, save_path: Optional[str] = None):
        """Return string representation of dataset status."""
        status_list = []
        for session_node in self.select("session"):
            assert isinstance(session_node, Session)
            status_list.append({"Cohort": session_node.coordinate.object_uid.cohort_id,
                                "Mice": session_node.coordinate.object_uid.mice_id,
                                "FOV": session_node.coordinate.object_uid.fov_id,
                                "Day": session_node.coordinate.temporal_uid.day_id,
                                "Session": session_node.coordinate.temporal_uid.session_id,
                                **session_node.data.status()})
        df = pd.DataFrame(status_list)
        
        save_path = "status_report.xlsx" if save_path is None else save_path
        write_boolean_dataframe(df, "Status Report", save_path)

    @property
    def root_coordinate(self) -> TemporalObjectCoordinate:
        """Return root coordinate of the dataset."""
        assert isinstance(self._root_coordinate, TemporalObjectCoordinate), f"Dataset is empty, with root: {self._root_coordinate}"
        return self._root_coordinate
