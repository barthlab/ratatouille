"""
Meta Data Structure Module

Core data structures for hierarchical experimental data organization:
- Hierarchical UID Classes: Spatial (cohort→mice→FOV→cell) and temporal (template→day→session→chunk→spike) identifiers
- Coordinate System: Combines spatial and temporal hierarchies for precise data localization
"""

from dataclasses import dataclass, field
from typing import ClassVar, Generator, Optional, Tuple, Any
from typing_extensions import Self


class HierarchicalUID:
    """
    Base class for hierarchical unique identifiers.

    Provides framework for hierarchical identifiers with flexible querying
    and containment relationships. Hierarchy defined by _HIERARCHY_FIELDS.
    """
    _HIERARCHY_FIELDS: ClassVar[Tuple[str, ...]]

    def __post_init__(self):
        """Validate first hierarchy field is provided."""
        # assert self.get_hier_value(self._HIERARCHY_FIELDS[0]) is not None, "first field should not be None"

    def __str__(self) -> str:
        """Return space-separated "Field_value" pairs for all non-None fields."""
        parts = []
        for name in self._HIERARCHY_FIELDS:
            value = self.get_hier_value(name)
            if value is not None:
                parts.append(f"{name.capitalize()}_{value}")
            else:
                break
        return " ".join(parts)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return self.__str__()
    
    def __iter__(self) -> Generator[Tuple[str, Any], None, None]:
        """Iterate over hierarchy fields and their values."""
        for name in self._HIERARCHY_FIELDS:
            yield name, self.get_hier_value(name)

    def __add__(self, other) -> Self:
        """Combine two UIDs into the root uid."""
        assert isinstance(other, self.__class__), f"Cannot add {other} to {self}"

        new_uid_dict = {}
        for name in self._HIERARCHY_FIELDS:
            if self.get_hier_value(name) != other.get_hier_value(name):                
                new_uid_dict[name + "_id"] = None
            else:
                new_uid_dict[name + "_id"] = self.get_hier_value(name)
        return self.__class__(**new_uid_dict)
    
    def __radd__(self, other) -> Self:
        """Combine two UIDs into the root uid."""
        return self.__add__(other)

    def get_hier_value(self, hier_name: str, default: Any = None) -> Any:
        """Get value for hierarchy field (without "_id" suffix)."""
        assert hier_name in self._HIERARCHY_FIELDS, f"{hier_name} is not a valid field for {self}"
        hier_value = getattr(self, hier_name + "_id", None)
        return hier_value if hier_value is not None else default

    @property
    def level(self) -> str:
        """Return most specific level (last non-None field in hierarchy)."""
        for i, name in enumerate(self._HIERARCHY_FIELDS):
            if self.get_hier_value(name) is None:
                return self._HIERARCHY_FIELDS[i-1]
        return self._HIERARCHY_FIELDS[-1]
    
    @property
    def level_index(self) -> int:
        """Return index of most specific level in hierarchy."""
        return self._HIERARCHY_FIELDS.index(self.level)
    
    @property
    def _comparison_tuple(self) -> tuple:
        """Returns a tuple of hierarchy values for comparison."""
        return tuple(str(self.get_hier_value(name, "")) for name in self._HIERARCHY_FIELDS )

    def __lt__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._comparison_tuple < other._comparison_tuple

    def contains(self, other: "HierarchicalUID") -> bool:
        """
        Check if this UID contains another UID in the hierarchy.

        Returns True if this UID is at a higher level and all specified fields match.
        """
        if not isinstance(other, self.__class__):
            return False

        for name in self._HIERARCHY_FIELDS:
            self_val = self.get_hier_value(name)
            other_val = other.get_hier_value(name)
            if self_val is None:
                return True  # This UID contains all UIDs at deeper levels
            if (other_val is None) or (self_val != other_val):
                return False  # Values don't match or other UID doesn't specify this level
        return True

    def is_ancestor_hierarchy(self, target_level: str) -> bool:
        """Check if this UID is at or below the specified level in the hierarchy."""
        assert target_level in self._HIERARCHY_FIELDS, f"{target_level} is not a valid level for {self}"
        return self._HIERARCHY_FIELDS.index(target_level) <= self._HIERARCHY_FIELDS.index(self.level)
    
    def transit(self, target_level: str) -> Self:
        """Create new UID at higher hierarchy level by removing more specific fields."""
        assert self.is_ancestor_hierarchy(target_level), f"cannot transit from {self.level} to {target_level}"

        new_uid_dict = {}
        for name in self._HIERARCHY_FIELDS:
            new_uid_dict[name + "_id"] = self.get_hier_value(name)
            if name == target_level:
                break
        return self.__class__(**new_uid_dict)

    def child_uid(self, **kwargs) -> Self:
        """Create new UID at lower hierarchy level by adding more specific fields."""
        assert all(getattr(self, name, None) is None for name in kwargs), f"cannot overwrite existing fields for {self}"
        new_uid_dict = {name + "_id": self.get_hier_value(name) for name in self._HIERARCHY_FIELDS}
        new_uid_dict.update(kwargs)
        return self.__class__(**new_uid_dict)

    @property
    def parent_uid(self) -> Self:
        """Create new UID at higher hierarchy level by removing the most specific field."""
        return self.transit(self._HIERARCHY_FIELDS[self.level_index - 1])
    

@dataclass(frozen=True)
class ObjectUID(HierarchicalUID):
    """
    Spatial hierarchy identifier: cohort → mice → FOV → cell.
    Allows flexible identification at different spatial scales.
    """
    cohort_id: Optional[str] = None
    mice_id: Optional[str] = None
    fov_id: Optional[str] = None
    cell_id: Optional[int] = None

    _HIERARCHY_FIELDS = ('cohort', 'mice', 'fov', 'cell')


@dataclass(frozen=True)
class TemporalUID(HierarchicalUID):
    """
    Temporal hierarchy identifier: template → day → session → chunk → spike.
    Allows flexible identification at different temporal scales.
    """
    template_id: Optional[str] = None
    day_id: Optional[str] = None
    session_id: Optional[str] = None
    chunk_id: Optional[int] = None
    spike_id: Optional[int] = None

    _HIERARCHY_FIELDS = ('template', 'day', 'session', 'chunk', 'spike')


@dataclass(frozen=True, order=True)
class TemporalObjectCoordinate:
    """
    Combines spatial and temporal hierarchical identifiers.

    Provides complete coordinate system for locating experimental data
    in both spatial and temporal dimensions.
    """
    object_uid: ObjectUID = field(default_factory=ObjectUID)
    temporal_uid: TemporalUID = field(default_factory=TemporalUID)

    def contains(self, other: "TemporalObjectCoordinate") -> bool:
        """
        Check if this coordinate contains another coordinate.

        Returns True if both object_uid and temporal_uid contain the other's UIDs.
        """
        return (self.object_uid.contains(other.object_uid) and
                self.temporal_uid.contains(other.temporal_uid))

    def transit(self, target_object_level: Optional[str] = None, target_temporal_level: Optional[str] = None) -> "TemporalObjectCoordinate":
        """Create new coordinate at higher hierarchy level."""
        return TemporalObjectCoordinate(
            object_uid=self.object_uid.transit(target_object_level if target_object_level is not None else self.object_uid.level),
            temporal_uid=self.temporal_uid.transit(target_temporal_level if target_temporal_level is not None else self.temporal_uid.level)
        )

    def is_ancestor_hierarchy(self, target_object_level: Optional[str] = None, target_temporal_level: Optional[str] = None) -> bool:
        """Check if this coordinate is at or below the specified level in the hierarchy."""
        target_object_level = target_object_level if target_object_level is not None else self.object_uid.level
        target_temporal_level = target_temporal_level if target_temporal_level is not None else self.temporal_uid.level
        return (self.object_uid.is_ancestor_hierarchy(target_object_level) and
                self.temporal_uid.is_ancestor_hierarchy(target_temporal_level))

    def __str__(self) -> str:
        """Return readable string representation."""
        return f"({self.object_uid}, {self.temporal_uid})"

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return self.__str__()
    
    def __add__(self, other) -> "TemporalObjectCoordinate":
        """Combine two coordinates into the root coordinate."""
        assert isinstance(other, TemporalObjectCoordinate), f"Cannot add {other} to {self}"
        return TemporalObjectCoordinate(self.object_uid + other.object_uid, self.temporal_uid + other.temporal_uid)
    
    def __radd__(self, other) -> "TemporalObjectCoordinate":
        """Combine two coordinates into the root coordinate."""
        if other == 0:
            return self
        return self.__add__(other)

    def __getattr__(self, name: str) -> Any:
        """Look into object_uid and temporal_uid for attributes."""
        if hasattr(self.object_uid, name):
            return getattr(self.object_uid, name)
        if hasattr(self.temporal_uid, name):
            return getattr(self.temporal_uid, name)
        raise AttributeError(f"'{self.__class__.__name__}' object [{str(self)}] has no attribute '{name}'")

    @property
    def level(self) -> Tuple[str, str]:
        """Return the most specific level of the coordinate."""
        return self.object_uid.level, self.temporal_uid.level
    
    @property
    def level_index(self) -> Tuple[int, int]:
        """Return the index of the most specific level of the coordinate."""
        return self.object_uid.level_index, self.temporal_uid.level_index
    
