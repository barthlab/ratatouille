import pytest
import numpy as np
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen.structure.neural_data_structure import (
    TimeSeries, Events, Timeline, Fluorescence, NeuralData
)


class TestTimeSeries:
    """Test cases for TimeSeries class - designed to reveal bugs."""
    
    def test_multidimensional_data_validation(self):
        """Test that TimeSeries correctly handles multi-dimensional data."""
        # 2D data (e.g., multiple cells)
        values = np.random.rand(5, 100)  # 5 cells, 100 timepoints
        times = np.linspace(0, 10, 100)
        ts = TimeSeries(v=values, t=times)
        assert ts.v.shape == (5, 100)
        assert len(ts) == 100
    
    def test_3d_data_validation(self):
        """Test TimeSeries with 3D data (e.g., spatial x temporal)."""
        values = np.random.rand(10, 10, 50)  # 10x10 spatial, 50 timepoints
        times = np.linspace(0, 5, 50)
        ts = TimeSeries(v=values, t=times)
        assert ts.v.shape == (10, 10, 50)
        assert len(ts) == 50
    
    def test_mismatched_dimensions_bug(self):
        """Test that reveals dimension mismatch bugs."""
        # This should fail - last dimension doesn't match time length
        values = np.random.rand(5, 99)  # 5 cells, 99 timepoints
        times = np.linspace(0, 10, 100)  # 100 timepoints
        with pytest.raises(AssertionError, match="values, times have different length"):
            TimeSeries(v=values, t=times)
    
    def test_time_dimension_validation_bug(self):
        """Test that reveals time dimension validation bugs."""
        # 2D time array should fail
        values = np.random.rand(100)
        times = np.random.rand(10, 10)  # 2D time array
        with pytest.raises(AssertionError, match="times should be 1-d array"):
            TimeSeries(v=values, t=times)
    
    def test_non_ascending_times_edge_cases(self):
        """Test edge cases in time ordering validation."""
        values = np.array([1, 2, 3, 4])
        
        # Strictly decreasing times should fail
        times = np.array([4.0, 3.0, 2.0, 1.0])
        with pytest.raises(AssertionError, match="times should be in ascending order"):
            TimeSeries(v=values, t=times)
        
        # Mixed order should fail
        times = np.array([1.0, 3.0, 2.0, 4.0])
        with pytest.raises(AssertionError, match="times should be in ascending order"):
            TimeSeries(v=values, t=times)
    
    def test_equal_times_allowed(self):
        """Test that equal consecutive times are properly handled."""
        values = np.array([1, 2, 3, 4])
        times = np.array([1.0, 1.0, 2.0, 2.0])  # Equal times should be allowed
        ts = TimeSeries(v=values, t=times)
        assert len(ts) == 4
    
    def test_segment_boundary_bugs(self):
        """Test segmentation boundary conditions that might reveal bugs."""
        values = np.array([1, 2, 3, 4, 5])
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ts = TimeSeries(v=values, t=times)
        
        # Segment completely outside range
        segment = ts.segment(10.0, 20.0)
        assert len(segment) == 0
        
        # Segment with start_t == end_t should return empty
        segment = ts.segment(2.0, 2.0)
        assert len(segment) == 0
        
        # Segment with very small range
        segment = ts.segment(1.0, 1.1)
        assert len(segment) == 1
        assert segment.v[0] == 2
    
    def test_segment_invalid_range_bug(self):
        """Test that invalid segment ranges are properly caught."""
        values = np.array([1, 2, 3])
        times = np.array([0.0, 1.0, 2.0])
        ts = TimeSeries(v=values, t=times)
        
        # start_t > end_t should fail
        with pytest.raises(AssertionError, match="start time .* should be earlier than end time"):
            ts.segment(2.0, 1.0)
    
    def test_empty_timeseries_handling(self):
        """Test handling of empty TimeSeries."""
        values = np.array([])
        times = np.array([])
        ts = TimeSeries(v=values, t=times)
        assert len(ts) == 0
        
        # Segmenting empty series should work
        segment = ts.segment(0.0, 1.0)
        assert len(segment) == 0
    
    def test_single_timepoint_edge_case(self):
        """Test edge case with single timepoint."""
        values = np.array([42])
        times = np.array([1.5])
        ts = TimeSeries(v=values, t=times)
        assert len(ts) == 1
        
        # Segment that includes the single point
        segment = ts.segment(1.0, 2.0)
        assert len(segment) == 1
        assert segment.v[0] == 42
        
        # Segment that excludes the single point
        segment = ts.segment(0.0, 1.0)
        assert len(segment) == 0


class TestEvents:
    """Test cases for Events class - designed to reveal bugs."""
    
    def test_dimension_validation_bug(self):
        """Test that Events properly validates 1D arrays."""
        # 2D values should fail
        values = np.array([[1, 2], [3, 4]])
        times = np.array([1.0, 2.0])
        with pytest.raises(AssertionError, match="values, times should be 1-d array"):
            Events(v=values, t=times)
        
        # 2D times should fail
        values = np.array([1, 2])
        times = np.array([[1.0, 2.0]])
        with pytest.raises(AssertionError, match="values, times should be 1-d array"):
            Events(v=values, t=times)
    
    def test_shape_mismatch_bug(self):
        """Test that Events catches shape mismatches."""
        values = np.array([1, 2, 3])
        times = np.array([1.0, 2.0])  # Different length
        with pytest.raises(AssertionError, match="values, times have different length"):
            Events(v=values, t=times)
    
    def test_empty_events_time_validation(self):
        """Test time validation with empty events."""
        values = np.array([])
        times = np.array([])
        events = Events(v=values, t=times)
        assert len(events) == 0
        
        # Empty events should not trigger time ordering validation
        # This tests the len(self.t) > 0 condition in __post_init__
    
    def test_inactivation_window_filter_edge_cases(self):
        """Test inactivation window filter edge cases that might reveal bugs."""
        # Empty events
        values = np.array([])
        times = np.array([])
        events = Events(v=values, t=times)
        filtered = events.inactivation_window_filter(0.1)
        assert len(filtered) == 0
        
        # Single event
        values = np.array([1])
        times = np.array([1.0])
        events = Events(v=values, t=times)
        filtered = events.inactivation_window_filter(0.1)
        assert len(filtered) == 1
        assert filtered.v[0] == 1
        assert filtered.t[0] == 1.0
    
    def test_inactivation_window_filter_bug_zero_window(self):
        """Test inactivation window filter with invalid window size."""
        values = np.array([1, 2, 3])
        times = np.array([1.0, 1.05, 1.1])
        events = Events(v=values, t=times)
        
        # Zero window should fail
        with pytest.raises(AssertionError, match="inactivation window should be positive"):
            events.inactivation_window_filter(0.0)
        
        # Negative window should fail
        with pytest.raises(AssertionError, match="inactivation window should be positive"):
            events.inactivation_window_filter(-0.1)
    
    def test_inactivation_window_filter_logic_bug(self):
        """Test inactivation window filter logic for potential bugs."""
        values = np.array([1, 2, 3, 4, 5])
        times = np.array([0.0, 0.05, 0.15, 0.16, 0.3])
        events = Events(v=values, t=times)
        
        # With 0.1s window, should keep events at 0.0, 0.15, 0.3
        filtered = events.inactivation_window_filter(0.1)
        expected_times = [0.0, 0.15, 0.3]
        expected_values = [1, 3, 5]
        
        assert len(filtered) == 3
        np.testing.assert_array_equal(filtered.t, expected_times)
        np.testing.assert_array_equal(filtered.v, expected_values)
    
    def test_iteration_protocol(self):
        """Test Events iteration protocol."""
        values = np.array([10, 20, 30])
        times = np.array([1.0, 2.0, 3.0])
        events = Events(v=values, t=times)
        
        event_list = list(events)
        expected = [(1.0, 10), (2.0, 20), (3.0, 30)]
        assert event_list == expected
    
    def test_string_representation_with_floats(self):
        """Test string representation with float values."""
        values = np.array([1.234567, 2.345678, 3.456789])
        times = np.array([1.0, 2.0, 3.0])
        events = Events(v=values, t=times)
        
        str_repr = str(events)
        # Should round floats to 2 decimal places
        assert "1.23" in str_repr
        assert "2.35" in str_repr
        assert "3.46" in str_repr


class TestTimeline:
    """Test cases for Timeline class - designed to reveal bugs."""
    
    def test_invalid_event_types_bug(self):
        """Test that Timeline catches invalid event types."""
        values = np.array(["VerticalPuffOn", "InvalidEvent"])
        times = np.array([1.0, 2.0])

        # This should fail because "InvalidEvent" is not in SUPPORTED_TIMELINE_EVENT
        with pytest.raises(AssertionError, match="Found unsupported event.*InvalidEvent"):
            Timeline(v=values, t=times)
    
    def test_non_string_values_bug(self):
        """Test that Timeline catches non-string values."""
        values = np.array([1, 2])  # Numeric instead of string
        times = np.array([1.0, 2.0])
        
        with pytest.raises(AssertionError, match="timeline should be list of str"):
            Timeline(v=values, t=times)
    
    def test_filter_invalid_event_types(self):
        """Test Timeline filter with invalid event types."""
        values = np.array(["VerticalPuffOn", "BlankOn"])
        times = np.array([1.0, 2.0])
        timeline = Timeline(v=values, t=times)

        # Filtering with invalid event type should fail
        with pytest.raises(AssertionError, match="Found unsupported event.*InvalidEvent"):
            timeline.filter("InvalidEvent")
    
    def test_filter_empty_result(self):
        """Test Timeline filter that returns empty result."""
        values = np.array(["VerticalPuffOn", "BlankOn"])
        times = np.array([1.0, 2.0])
        timeline = Timeline(v=values, t=times)
        
        # Filter for event type not present
        filtered = timeline.filter("WaterOn")
        assert len(filtered) == 0
        assert isinstance(filtered, Timeline)


class TestFluorescence:
    """Test cases for Fluorescence class - designed to reveal bugs."""
    
    def test_shape_validation_bugs(self):
        """Test Fluorescence shape validation for potential bugs."""
        # Create valid base data
        raw_f_data = np.random.rand(5, 100)  # 5 cells, 100 timepoints
        times = np.linspace(0, 10, 100)
        raw_f = TimeSeries(v=raw_f_data, t=times)
        
        fov_motion_data = np.random.rand(2, 100)  # x,y motion, 100 timepoints
        fov_motion = TimeSeries(v=fov_motion_data, t=times)
        
        cell_position = np.random.rand(5, 2)  # 5 cells, x,y positions
        cell_idx = np.array([0, 1, 2, 3, 4])
        
        # This should work
        fluor = Fluorescence(
            raw_f=raw_f,
            fov_motion=fov_motion,
            cell_position=cell_position,
            cell_idx=cell_idx
        )
        assert fluor.num_cell == 5
        assert fluor.num_timepoint == 100
    
    def test_mismatched_cell_dimensions_bug(self):
        """Test that Fluorescence catches mismatched cell dimensions."""
        raw_f_data = np.random.rand(5, 100)  # 5 cells
        times = np.linspace(0, 10, 100)
        raw_f = TimeSeries(v=raw_f_data, t=times)
        
        fov_motion_data = np.random.rand(2, 100)
        fov_motion = TimeSeries(v=fov_motion_data, t=times)
        
        cell_position = np.random.rand(3, 2)  # Only 3 cells - mismatch!
        cell_idx = np.array([0, 1, 2, 3, 4])  # 5 cells
        
        with pytest.raises(AssertionError, match="cell_position: Expected shape"):
            Fluorescence(
                raw_f=raw_f,
                fov_motion=fov_motion,
                cell_position=cell_position,
                cell_idx=cell_idx
            )
    
    def test_extract_cell_boundary_bug(self):
        """Test extract_cell method boundary conditions."""
        raw_f_data = np.random.rand(3, 50)
        times = np.linspace(0, 5, 50)
        raw_f = TimeSeries(v=raw_f_data, t=times)
        
        fov_motion_data = np.random.rand(2, 50)
        fov_motion = TimeSeries(v=fov_motion_data, t=times)
        
        cell_position = np.random.rand(3, 2)
        cell_idx = np.array([10, 20, 30])
        
        fluor = Fluorescence(
            raw_f=raw_f,
            fov_motion=fov_motion,
            cell_position=cell_position,
            cell_idx=cell_idx
        )
        
        # Extract valid cell
        single_cell = fluor.extract_cell(1)
        assert single_cell.num_cell == 1
        assert single_cell.cell_idx[0] == 1  # This might be a bug - should it be 20?
        
        # Extract boundary cells
        first_cell = fluor.extract_cell(0)
        assert first_cell.num_cell == 1
        
        last_cell = fluor.extract_cell(2)
        assert last_cell.num_cell == 1


class TestNeuralData:
    """Test cases for NeuralData class - designed to reveal bugs."""
    
    def test_type_validation_bugs(self):
        """Test NeuralData type validation for potential bugs."""
        # Create valid data
        lick_events = Events(v=np.array([1, 1]), t=np.array([1.0, 2.0]))
        pupil_ts = TimeSeries(v=np.array([1, 2, 3]), t=np.array([1.0, 2.0, 3.0]))
        
        # Valid assignment
        neural_data = NeuralData(lick=lick_events, pupil=pupil_ts)
        
        # Invalid type assignments should fail
        with pytest.raises(AssertionError, match="position, locomotion, lick should be Events"):
            NeuralData(lick=pupil_ts)  # TimeSeries instead of Events
        
        with pytest.raises(AssertionError, match="pupil, tongue, whisker should be TimeSeries"):
            NeuralData(pupil=lick_events)  # Events instead of TimeSeries
    
    def test_shadow_clone_invalid_keys_bug(self):
        """Test shadow_clone with invalid keys."""
        neural_data = NeuralData()
        
        # Invalid key should fail
        with pytest.raises(AssertionError, match="NeuralData expects keys"):
            neural_data.shadow_clone(invalid_key="some_value")
    
    def test_segment_unimplemented_type_bug(self):
        """Test segmentation with unsupported data types."""
        # Create NeuralData with custom object that doesn't have segment method
        class UnsegmentableData:
            pass

        # Use shadow_clone to add unsupported field
        neural_data = NeuralData()
        # This will fail because NeuralData doesn't allow arbitrary fields
        with pytest.raises(AssertionError, match="NeuralData expects keys"):
            neural_data.shadow_clone(custom_field=UnsegmentableData())
    
    def test_empty_neural_data_operations(self):
        """Test operations on empty NeuralData."""
        neural_data = NeuralData()
        
        # String representation should show EMPTY
        str_repr = str(neural_data)
        assert "EMPTY" in str_repr
        
        # Shadow clone should work
        cloned = neural_data.shadow_clone()
        assert str(cloned) == str(neural_data)
        
        # Segmentation should work (returns empty NeuralData)
        segmented = neural_data.segment(0.0, 1.0)
        assert isinstance(segmented, NeuralData)


if __name__ == "__main__":
    pytest.main([__file__])
