import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen.structure.meta_data_structure import (
    TimeSeries, DenseTimeSeries, SparseTimeSeries,
    HierarchicalUID, ObjectUID, TemporalUID, TemporalObjectCoordinate
)


class TestTimeSeries:
    """Test cases for TimeSeries base class."""
    
    def test_valid_creation(self):
        """Test creating a valid TimeSeries."""
        values = [1, 2, 3, 4]
        times = np.array([0.0, 1.0, 2.0, 3.0])
        ts = TimeSeries(v=values, t=times)
        assert len(ts) == 4
        assert np.array_equal(ts.t, times)
        assert ts.v == values
    
    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise assertion error."""
        values = [1, 2, 3]
        times = np.array([0.0, 1.0])
        with pytest.raises(AssertionError, match="values, times have different length"):
            TimeSeries(v=values, t=times)
    
    def test_non_ascending_times(self):
        """Test that non-ascending times raise assertion error."""
        values = [1, 2, 3]
        times = np.array([0.0, 2.0, 1.0])  # Not ascending
        with pytest.raises(AssertionError, match="times should be in ascending order"):
            TimeSeries(v=values, t=times)
    
    def test_equal_times_allowed(self):
        """Test that equal consecutive times are allowed."""
        values = [1, 2, 3]
        times = np.array([0.0, 1.0, 1.0])  # Equal times allowed
        ts = TimeSeries(v=values, t=times)
        assert len(ts) == 3
    
    def test_segment_valid(self):
        """Test valid segmentation."""
        values = [1, 2, 3, 4, 5]
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ts = TimeSeries(v=values, t=times)
        
        segment = ts.segment(1.0, 3.0)
        assert len(segment) == 2
        assert segment.v == [2, 3]
        np.testing.assert_array_equal(segment.t, np.array([1.0, 2.0]))
    
    def test_segment_invalid_range(self):
        """Test segmentation with invalid range."""
        values = [1, 2, 3]
        times = np.array([0.0, 1.0, 2.0])
        ts = TimeSeries(v=values, t=times)
        
        with pytest.raises(AssertionError, match="start time .* should be earlier than end time"):
            ts.segment(2.0, 1.0)
    
    def test_segment_boundary_cases(self):
        """Test segmentation boundary cases."""
        values = [1, 2, 3, 4, 5]
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ts = TimeSeries(v=values, t=times)

        # Segment beyond range
        segment = ts.segment(5.0, 6.0)
        assert len(segment) == 0

        # Segment with very small range (start_t must be < end_t)
        # This will actually return one element at time 1.0
        segment = ts.segment(1.0, 1.1)
        assert len(segment) == 1

    def test_segment_equal_times_allowed(self):
        """Test that segment method correctly allows start_t == end_t."""
        values = [1, 2, 3, 4, 5]
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ts = TimeSeries(v=values, t=times)

        # Equal times should be allowed and return empty segment
        segment = ts.segment(1.0, 1.0)
        assert len(segment) == 0

        # Test that start_t > end_t still raises error
        with pytest.raises(AssertionError, match="start time .* should be earlier than end time"):
            ts.segment(2.0, 1.0)  # This should fail


class TestDenseTimeSeries:
    """Test cases for DenseTimeSeries."""
    
    def test_valid_creation(self):
        """Test creating a valid DenseTimeSeries."""
        values = np.array([1.0, 2.0, 3.0])
        times = np.array([0.0, 1.0, 2.0])
        dts = DenseTimeSeries(v=values, t=times)
        assert len(dts) == 3
        np.testing.assert_array_equal(dts.v, values)
    
    def test_invalid_values_type(self):
        """Test that non-numpy array values raise assertion error."""
        values = [1.0, 2.0, 3.0]  # List instead of numpy array
        times = np.array([0.0, 1.0, 2.0])
        with pytest.raises(AssertionError, match="values should be numpy array for DenseTimeSeries"):
            DenseTimeSeries(v=values, t=times)
    
    def test_invalid_dimensions(self):
        """Test that multi-dimensional arrays raise assertion error."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D array
        times = np.array([0.0, 1.0])
        with pytest.raises(AssertionError, match="values, times should be 1-d array"):
            DenseTimeSeries(v=values, t=times)
    
    def test_inheritance_from_timeseries(self):
        """Test that DenseTimeSeries inherits TimeSeries functionality."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        times = np.array([0.0, 1.0, 2.0, 3.0])
        dts = DenseTimeSeries(v=values, t=times)
        
        segment = dts.segment(1.0, 3.0)
        assert isinstance(segment, TimeSeries)
        assert len(segment) == 2


class TestSparseTimeSeries:
    """Test cases for SparseTimeSeries."""
    
    def test_valid_creation(self):
        """Test creating a valid SparseTimeSeries."""
        values = [1, 2, 3]
        times = np.array([0.0, 1.0, 2.0])
        sts = SparseTimeSeries(v=values, t=times)
        assert len(sts) == 3
        assert sts.v == values
        assert sts.event_type == int
    
    def test_invalid_values_type(self):
        """Test that non-list values raise assertion error."""
        values = np.array([1, 2, 3])  # Numpy array instead of list
        times = np.array([0.0, 1.0, 2.0])
        with pytest.raises(AssertionError, match="values should be list for SparseTimeSeries"):
            SparseTimeSeries(v=values, t=times)
    
    def test_event_type_detection(self):
        """Test that event_type is correctly detected."""
        # String events
        values = ["A", "B", "C"]
        times = np.array([0.0, 1.0, 2.0])
        sts = SparseTimeSeries(v=values, t=times)
        assert sts.event_type == str

        # Float events
        values = [1.5, 2.5, 3.5]
        times = np.array([0.0, 1.0, 2.0])
        sts = SparseTimeSeries(v=values, t=times)
        assert sts.event_type == float

    def test_empty_values_event_type_handling(self):
        """Test that SparseTimeSeries correctly handles empty values."""
        # This actually works correctly - the ternary operator handles empty case
        values = []
        times = np.array([])

        sts = SparseTimeSeries(v=values, t=times)
        assert sts.event_type is None  # Correctly set to None for empty values
        assert len(sts) == 0
    
    def test_inheritance_from_timeseries(self):
        """Test that SparseTimeSeries inherits TimeSeries functionality."""
        values = [1, 2, 3, 4]
        times = np.array([0.0, 1.0, 2.0, 3.0])
        sts = SparseTimeSeries(v=values, t=times)
        
        segment = sts.segment(1.0, 3.0)
        assert isinstance(segment, TimeSeries)
        assert len(segment) == 2


class TestHierarchicalUID:
    """Test cases for HierarchicalUID base class."""
    
    def test_abstract_nature(self):
        """Test that HierarchicalUID requires proper subclassing."""
        # HierarchicalUID can be instantiated but lacks _HIERARCHY_FIELDS
        # This demonstrates it's meant to be subclassed
        uid = HierarchicalUID()
        # The class doesn't have _HIERARCHY_FIELDS defined
        assert not hasattr(uid, '_HIERARCHY_FIELDS')


class TestObjectUID:
    """Test cases for ObjectUID."""
    
    def test_valid_creation_cohort_only(self):
        """Test creating ObjectUID with only cohort."""
        uid = ObjectUID(cohort_id="C57BL/6J")
        assert uid.cohort_id == "C57BL/6J"
        assert uid.mice_id is None
        assert uid.fov_id is None
        assert uid.cell_id is None
        assert uid.level == "cohort"
    
    def test_valid_creation_full_hierarchy(self):
        """Test creating ObjectUID with full hierarchy."""
        uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1", fov_id=1, cell_id=5)
        assert uid.cohort_id == "C57BL/6J"
        assert uid.mice_id == "Mouse1"
        assert uid.fov_id == 1
        assert uid.cell_id == 5
        assert uid.level == "cell"
    
    def test_invalid_creation_missing_cohort(self):
        """Test that missing cohort_id raises assertion error."""
        with pytest.raises(TypeError):
            ObjectUID()  # Missing required cohort_id
    
    def test_string_representation(self):
        """Test string representation of ObjectUID."""
        uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        expected = "Cohort_C57BL/6J Mice_Mouse1"
        assert str(uid) == expected
        # repr() uses the default dataclass representation, not the custom __str__
        assert "C57BL/6J" in repr(uid)
        assert "Mouse1" in repr(uid)
    
    def test_level_property(self):
        """Test level property for different hierarchies."""
        assert ObjectUID(cohort_id="C57BL/6J").level == "cohort"
        assert ObjectUID(cohort_id="C57BL/6J", mice_id="M1").level == "mice"
        assert ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1).level == "fov"
        assert ObjectUID(cohort_id="C57BL/6J", mice_id="M1", fov_id=1, cell_id=5).level == "cell"
    
    def test_get_hier_value(self):
        """Test get_hier_value method."""
        uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        assert uid.get_hier_value("cohort") == "C57BL/6J"
        assert uid.get_hier_value("mice") == "Mouse1"
        assert uid.get_hier_value("fov") is None
        assert uid.get_hier_value("cell") is None
    
    def test_contains_same_class(self):
        """Test contains method with same class instances."""
        parent = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        child = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1", fov_id=1)
        
        assert parent.contains(child)
        assert not child.contains(parent)
        assert parent.contains(parent)
    
    def test_contains_different_values(self):
        """Test contains method with different values."""
        uid1 = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        uid2 = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse2")
        
        assert not uid1.contains(uid2)
        assert not uid2.contains(uid1)
    
    def test_contains_different_class(self):
        """Test contains method with different class."""
        obj_uid = ObjectUID(cohort_id="C57BL/6J")
        temp_uid = TemporalUID(template_id="Template1")
        
        assert not obj_uid.contains(temp_uid)
    
    def test_transit_valid(self):
        """Test valid transit operations."""
        uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1", fov_id="1", cell_id=5)

        # The transit method actually works correctly
        result = uid.transit("mice")
        assert result.cohort_id == "C57BL/6J"
        assert result.mice_id == "Mouse1"
        assert result.fov_id is None
        assert result.cell_id is None

    def test_transit_method_works_correctly(self):
        """Test that transit method works correctly."""
        uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1", fov_id="1", cell_id=5)

        # Transit method works correctly
        result = uid.transit("mice")
        assert result.cohort_id == "C57BL/6J"
        assert result.mice_id == "Mouse1"
        assert result.fov_id is None
        assert result.cell_id is None

    def test_child_uid_method_works_correctly(self):
        """Test that child_uid method works correctly."""
        uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")

        # Adding new field works correctly
        result = uid.child_uid(fov_id="1")
        assert result.cohort_id == "C57BL/6J"
        assert result.mice_id == "Mouse1"
        assert result.fov_id == "1"

        # Trying to overwrite existing field correctly fails
        with pytest.raises(AssertionError, match="cannot overwrite existing fields"):
            uid.child_uid(mice_id="Mouse2")
    
    def test_transit_invalid_level(self):
        """Test transit with invalid level."""
        uid = ObjectUID(cohort_id="C57BL/6J")
        
        with pytest.raises(AssertionError, match="invalid_level is not a valid level"):
            uid.transit("invalid_level")
    
    def test_transit_lower_level(self):
        """Test transit to lower level (should fail)."""
        uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        
        with pytest.raises(AssertionError, match="cannot transit from mice to cell"):
            uid.transit("cell")
    
    def test_frozen_and_ordered(self):
        """Test that ObjectUID is frozen and ordered."""
        uid1 = ObjectUID(cohort_id="A")
        uid2 = ObjectUID(cohort_id="B")

        # Test frozen (immutable) - dataclass frozen=True prevents assignment
        with pytest.raises((AttributeError, TypeError)):
            uid1.cohort_id = "Modified"

        # Test ordered
        assert uid1 < uid2
        assert uid2 > uid1


class TestTemporalUID:
    """Test cases for TemporalUID."""
    
    def test_valid_creation_template_only(self):
        """Test creating TemporalUID with only template."""
        uid = TemporalUID(template_id="Template1")
        assert uid.template_id == "Template1"
        assert uid.day_id is None
        assert uid.session_id is None
        assert uid.chunk_id is None
        assert uid.level == "template"
    
    def test_valid_creation_full_hierarchy(self):
        """Test creating TemporalUID with full hierarchy."""
        uid = TemporalUID(template_id="Template1", day_id=1, session_id=2, chunk_id=3)
        assert uid.template_id == "Template1"
        assert uid.day_id == 1
        assert uid.session_id == 2
        assert uid.chunk_id == 3
        assert uid.level == "chunk"
    
    def test_hierarchy_fields(self):
        """Test that hierarchy fields are correct."""
        assert TemporalUID._HIERARCHY_FIELDS == ('template', 'day', 'session', 'chunk')
    
    def test_string_representation(self):
        """Test string representation of TemporalUID."""
        uid = TemporalUID(template_id="Template1", day_id=5)
        expected = "Template_Template1 Day_5"
        assert str(uid) == expected
    
    def test_contains_temporal_hierarchy(self):
        """Test contains method with temporal hierarchy."""
        parent = TemporalUID(template_id="Template1", day_id=1)
        child = TemporalUID(template_id="Template1", day_id=1, session_id=2)
        
        assert parent.contains(child)
        assert not child.contains(parent)
    
    def test_transit_temporal(self):
        """Test transit operations for temporal hierarchy."""
        uid = TemporalUID(template_id="Template1", day_id=1, session_id=2, chunk_id=3)

        # Test that the transit method works correctly
        try:
            result = uid.transit("session")
            assert result.template_id == "Template1"
            assert result.day_id == 1
            assert result.session_id == 2
            assert result.chunk_id is None
        except Exception as e:
            pytest.fail(f"transit method failed with: {e}")

    def test_temporal_uid_child_uid_works_correctly(self):
        """Test child_uid method works correctly for TemporalUID."""
        uid = TemporalUID(template_id="Template1", day_id=1)

        # Adding new field works correctly
        result = uid.child_uid(session_id=2)
        assert result.template_id == "Template1"
        assert result.day_id == 1
        assert result.session_id == 2

        # Trying to overwrite existing field correctly fails
        with pytest.raises(AssertionError, match="cannot overwrite existing fields"):
            uid.child_uid(day_id=2)


class TestTemporalObjectCoordinate:
    """Test cases for TemporalObjectCoordinate."""
    
    def test_valid_creation(self):
        """Test creating a valid TemporalObjectCoordinate."""
        obj_uid = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        temp_uid = TemporalUID(template_id="Template1", day_id=1)
        
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        assert coord.object_uid == obj_uid
        assert coord.temporal_uid == temp_uid
    
    def test_contains_both_dimensions(self):
        """Test contains method considering both object and temporal dimensions."""
        # Parent coordinate
        parent_obj = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        parent_temp = TemporalUID(template_id="Template1", day_id=1)
        parent_coord = TemporalObjectCoordinate(object_uid=parent_obj, temporal_uid=parent_temp)
        
        # Child coordinate (more specific in both dimensions)
        child_obj = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1", fov_id=1)
        child_temp = TemporalUID(template_id="Template1", day_id=1, session_id=2)
        child_coord = TemporalObjectCoordinate(object_uid=child_obj, temporal_uid=child_temp)
        
        assert parent_coord.contains(child_coord)
        assert not child_coord.contains(parent_coord)
    
    def test_contains_partial_mismatch(self):
        """Test contains method with partial mismatch."""
        coord1_obj = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse1")
        coord1_temp = TemporalUID(template_id="Template1")
        coord1 = TemporalObjectCoordinate(object_uid=coord1_obj, temporal_uid=coord1_temp)
        
        # Different object hierarchy
        coord2_obj = ObjectUID(cohort_id="C57BL/6J", mice_id="Mouse2")
        coord2_temp = TemporalUID(template_id="Template1", day_id=1)
        coord2 = TemporalObjectCoordinate(object_uid=coord2_obj, temporal_uid=coord2_temp)
        
        assert not coord1.contains(coord2)
    
    def test_string_representation(self):
        """Test string representation of TemporalObjectCoordinate."""
        obj_uid = ObjectUID(cohort_id="C57BL/6J")
        temp_uid = TemporalUID(template_id="Template1")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        
        expected = "(Cohort_C57BL/6J, Template_Template1)"
        assert str(coord) == expected
        assert repr(coord) == expected
    
    def test_frozen_and_ordered(self):
        """Test that TemporalObjectCoordinate is frozen and ordered."""
        obj_uid1 = ObjectUID(cohort_id="A")
        temp_uid1 = TemporalUID(template_id="A")
        coord1 = TemporalObjectCoordinate(object_uid=obj_uid1, temporal_uid=temp_uid1)

        obj_uid2 = ObjectUID(cohort_id="B")
        temp_uid2 = TemporalUID(template_id="B")
        coord2 = TemporalObjectCoordinate(object_uid=obj_uid2, temporal_uid=temp_uid2)

        # Test frozen (immutable) - dataclass frozen=True prevents assignment
        with pytest.raises((AttributeError, TypeError)):
            coord1.object_uid = obj_uid2

        # Test ordered
        assert coord1 < coord2
        assert coord2 > coord1


# Edge cases and error handling tests
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_values_timeseries(self):
        """Test TimeSeries with empty arrays."""
        values = []
        times = np.array([])
        ts = TimeSeries(v=values, t=times)
        assert len(ts) == 0

    def test_single_value_timeseries(self):
        """Test TimeSeries with single value."""
        values = [42]
        times = np.array([1.0])
        ts = TimeSeries(v=values, t=times)
        assert len(ts) == 1
        assert ts.v == [42]

    def test_uid_with_special_characters(self):
        """Test UID with special characters in string fields."""
        uid = ObjectUID(cohort_id="C57BL/6J-Special_Strain")
        assert uid.cohort_id == "C57BL/6J-Special_Strain"
        assert "C57BL/6J-Special_Strain" in str(uid)

    def test_uid_with_zero_ids(self):
        """Test UID with zero values for numeric fields."""
        uid = ObjectUID(cohort_id="Test", mice_id="M1", fov_id="0", cell_id=0)
        assert uid.fov_id == "0"
        assert uid.cell_id == 0
        assert uid.level == "cell"

    def test_uid_with_negative_ids(self):
        """Test UID with negative values for numeric fields."""
        uid = TemporalUID(template_id="Test", day_id=-1, session_id=-2)
        assert uid.day_id == -1
        assert uid.session_id == -2
        assert uid.level == "session"

    def test_level_property_edge_case(self):
        """Test edge case in level property when all fields are None except first."""
        # Test edge case handling in level property
        uid = ObjectUID(cohort_id="Test")  # Only first field set
        assert uid.level == "cohort"  # Should work fine

        # Test with TemporalUID
        temp_uid = TemporalUID(template_id="Test")
        assert temp_uid.level == "template"

    def test_type_annotation_flexibility(self):
        """Test that fov_id works with both strings and integers despite type annotation."""
        # Note: fov_id is typed as Optional[str] but works with integers at runtime
        # This demonstrates type annotation vs runtime behavior

        # Using string (correct according to type annotation)
        uid_str = ObjectUID(cohort_id="Test", mice_id="M1", fov_id="1", cell_id=5)
        assert uid_str.fov_id == "1"

        # Using integer (incorrect according to type annotation but works at runtime)
        # This should ideally be caught by type checkers
        uid_int = ObjectUID(cohort_id="Test", mice_id="M1", fov_id=1, cell_id=5)
        assert uid_int.fov_id == 1  # Works at runtime despite type annotation


if __name__ == "__main__":
    pytest.main([__file__])
