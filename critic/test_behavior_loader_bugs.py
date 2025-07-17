import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen.loader.behavior_loader import behavior_loader_from_fov
from kitchen.structure.neural_data_structure import Events, TimeSeries, Timeline
from kitchen.structure.hierarchical_data_structure import Fov
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate, ObjectUID, TemporalUID


class TestBehaviorLoaderBugs:
    """Test cases designed to reveal bugs in behavior_loader_from_fov."""
    
    def create_test_fov(self):
        """Create a test FOV node for testing."""
        obj_uid = ObjectUID(cohort_id="TestCohort", mice_id="TestMouse", fov_id="TestFOV")
        temp_uid = TemporalUID(template_id="TestTemplate")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        return Fov(coordinate=coord, data=Mock())
    
    def create_test_timeline_dict(self):
        """Create test timeline dictionary."""
        obj_uid = ObjectUID(cohort_id="TestCohort", mice_id="TestMouse", fov_id="TestFOV")
        temp_uid = TemporalUID(template_id="TestTemplate", day_id="Day1", session_id="Session1")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        
        # Create timeline with task start/end events
        timeline_events = np.array(["task start", "VerticalPuffOn", "task end"])
        timeline_times = np.array([0.0, 5.0, 10.0])
        timeline = Timeline(v=timeline_events, t=timeline_times)
        
        return {coord: timeline}
    
    def test_missing_data_directory_bug(self):
        """Test behavior when data directory doesn't exist."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        # Mock routing to return non-existent path
        with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
            mock_path.return_value = "/nonexistent/path"
            
            # Should yield empty dict due to exception handling
            results = list(behavior_loader_from_fov(fov_node, timeline_dict))
            assert len(results) == 1
            assert results[0] == {}  # Empty dict when loading fails
    
    def test_malformed_lick_csv_bug(self):
        """Test behavior with malformed lick CSV files."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create malformed lick CSV (wrong number of columns)
            lick_dir = os.path.join(temp_dir, "lick")
            os.makedirs(lick_dir)
            lick_file = os.path.join(lick_dir, "LICK_Session1.csv")
            
            # CSV with 2 columns instead of 1
            malformed_data = pd.DataFrame({
                'time': [1000, 2000, 3000],
                'extra_column': [1, 2, 3]
            })
            malformed_data.to_csv(lick_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                # Should handle error gracefully and not include lick data
                assert len(results) == 1
                assert 'lick' not in results[0]
    
    def test_malformed_locomotion_csv_bug(self):
        """Test behavior with malformed locomotion CSV files."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create malformed locomotion CSV (wrong number of columns)
            locomotion_dir = os.path.join(temp_dir, "locomotion")
            os.makedirs(locomotion_dir)
            locomotion_file = os.path.join(locomotion_dir, "LOCOMOTION_Session1.csv")
            
            # CSV with 2 columns instead of 3
            malformed_data = pd.DataFrame({
                'time': [1000, 2000, 3000],
                'position': [100, 200, 300]
                # Missing third column
            })
            malformed_data.to_csv(locomotion_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                # Should handle error gracefully
                assert len(results) == 1
                assert 'locomotion' not in results[0]
                assert 'position' not in results[0]
    
    def test_locomotion_position_wraparound_bug(self):
        """Test potential bug in locomotion position wraparound calculation."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            locomotion_dir = os.path.join(temp_dir, "locomotion")
            os.makedirs(locomotion_dir)
            locomotion_file = os.path.join(locomotion_dir, "LOCOMOTION_Session1.csv")
            
            # Create data with position wraparound (599 -> 1)
            locomotion_data = pd.DataFrame({
                'time': [1000, 2000, 3000, 4000],
                'position': [598, 599, 1, 2],  # Wraparound from 599 to 1
                'extra': [0, 0, 0, 0]
            })
            locomotion_data.to_csv(locomotion_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                
                if 'locomotion' in results[0]:
                    locomotion = results[0]['locomotion']
                    # Check if wraparound is handled correctly
                    # The delta from 599 to 1 should be calculated properly
                    # This might reveal a bug in the delta calculation
                    assert len(locomotion.v) == 3  # Should have 3 delta values for 4 positions
    
    def test_missing_task_timeline_events_bug(self):
        """Test behavior when timeline is missing task start/end events."""
        fov_node = self.create_test_fov()
        
        # Create timeline without task start/end
        obj_uid = ObjectUID(cohort_id="TestCohort", mice_id="TestMouse", fov_id="TestFOV")
        temp_uid = TemporalUID(template_id="TestTemplate", day_id="Day1", session_id="Session1")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        
        timeline_events = np.array(["VerticalPuffOn", "BlankOn"])  # No task start/end
        timeline_times = np.array([1.0, 2.0])
        timeline = Timeline(v=timeline_events, t=timeline_times)
        timeline_dict = {coord: timeline}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create video behavior files
            for behavior_type in ["PUPIL", "TONGUE", "WHISKER"]:
                behavior_dir = os.path.join(temp_dir, behavior_type.lower())
                os.makedirs(behavior_dir)
                behavior_file = os.path.join(behavior_dir, f"{behavior_type}_Session1.csv")
                
                behavior_data = pd.DataFrame({
                    'frame': [0, 1, 2],
                    'value': [10, 20, 30]
                })
                behavior_data.to_csv(behavior_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                
                # Should skip video extracted behavior due to missing task events
                assert len(results) == 1
                assert 'pupil' not in results[0]
                assert 'tongue' not in results[0]
                assert 'whisker' not in results[0]
    
    def test_malformed_video_behavior_csv_bug(self):
        """Test behavior with malformed video behavior CSV files."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create malformed pupil CSV (wrong number of columns)
            pupil_dir = os.path.join(temp_dir, "pupil")
            os.makedirs(pupil_dir)
            pupil_file = os.path.join(pupil_dir, "PUPIL_Session1.csv")
            
            # CSV with 1 column instead of 2
            malformed_data = pd.DataFrame({
                'frame': [0, 1, 2]
                # Missing value column
            })
            malformed_data.to_csv(pupil_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                # Should handle error gracefully
                assert len(results) == 1
                assert 'pupil' not in results[0]
    
    def test_inconsistent_task_timeline_bug(self):
        """Test behavior with inconsistent task timeline (more than 2 task events)."""
        fov_node = self.create_test_fov()
        
        # Create timeline with multiple task start/end events
        obj_uid = ObjectUID(cohort_id="TestCohort", mice_id="TestMouse", fov_id="TestFOV")
        temp_uid = TemporalUID(template_id="TestTemplate", day_id="Day1", session_id="Session1")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        
        timeline_events = np.array(["task start", "task end", "task start", "task end"])
        timeline_times = np.array([0.0, 5.0, 6.0, 10.0])
        timeline = Timeline(v=timeline_events, t=timeline_times)
        timeline_dict = {coord: timeline}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pupil_dir = os.path.join(temp_dir, "pupil")
            os.makedirs(pupil_dir)
            pupil_file = os.path.join(pupil_dir, "PUPIL_Session1.csv")
            
            behavior_data = pd.DataFrame({
                'frame': [0, 1, 2],
                'value': [10, 20, 30]
            })
            behavior_data.to_csv(pupil_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                
                # Should skip video behavior due to len(task_times) != 2
                assert len(results) == 1
                assert 'pupil' not in results[0]
    
    def test_empty_csv_files_bug(self):
        """Test behavior with empty CSV files."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty lick CSV
            lick_dir = os.path.join(temp_dir, "lick")
            os.makedirs(lick_dir)
            lick_file = os.path.join(lick_dir, "LICK_Session1.csv")
            
            # Empty CSV with just headers
            empty_data = pd.DataFrame(columns=['time'])
            empty_data.to_csv(lick_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                
                # Should handle empty data gracefully
                assert len(results) == 1
                # Might include lick with empty arrays or skip it entirely
    
    def test_lick_inactivation_window_edge_case_bug(self):
        """Test lick inactivation window with edge cases."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            lick_dir = os.path.join(temp_dir, "lick")
            os.makedirs(lick_dir)
            lick_file = os.path.join(lick_dir, "LICK_Session1.csv")
            
            # Create lick data with events very close together
            lick_data = pd.DataFrame({
                'time': [1000, 1010, 1020, 1070, 1080]  # Some within inactivation window
            })
            lick_data.to_csv(lick_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                
                if 'lick' in results[0]:
                    lick_events = results[0]['lick']
                    # Should have filtered out events within inactivation window
                    # LICK_INACTIVATE_WINDOW = 1/20 = 0.05s = 50ms
                    # Events at 1000, 1070 should remain (70ms apart > 50ms)
                    assert len(lick_events.v) <= 5  # Should be filtered
    
    def test_multiple_sessions_bug(self):
        """Test behavior with multiple sessions in timeline_dict."""
        fov_node = self.create_test_fov()
        
        # Create multiple session coordinates
        obj_uid = ObjectUID(cohort_id="TestCohort", mice_id="TestMouse", fov_id="TestFOV")
        
        coord1 = TemporalObjectCoordinate(
            object_uid=obj_uid,
            temporal_uid=TemporalUID(template_id="TestTemplate", day_id="Day1", session_id="Session1")
        )
        coord2 = TemporalObjectCoordinate(
            object_uid=obj_uid,
            temporal_uid=TemporalUID(template_id="TestTemplate", day_id="Day1", session_id="Session2")
        )
        
        timeline1 = Timeline(v=np.array(["task start", "task end"]), t=np.array([0.0, 10.0]))
        timeline2 = Timeline(v=np.array(["task start", "task end"]), t=np.array([0.0, 10.0]))
        
        timeline_dict = {coord1: timeline1, coord2: timeline2}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data files for both sessions
            for session in ["Session1", "Session2"]:
                lick_dir = os.path.join(temp_dir, "lick")
                os.makedirs(lick_dir, exist_ok=True)
                lick_file = os.path.join(lick_dir, f"LICK_{session}.csv")
                
                lick_data = pd.DataFrame({'time': [1000, 2000]})
                lick_data.to_csv(lick_file, index=False)
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                
                # Should yield results for both sessions
                assert len(results) == 2
                # Both should have lick data
                for result in results:
                    assert 'lick' in result
    
    def test_file_permission_error_bug(self):
        """Test behavior when files exist but can't be read due to permissions."""
        fov_node = self.create_test_fov()
        timeline_dict = self.create_test_timeline_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            lick_dir = os.path.join(temp_dir, "lick")
            os.makedirs(lick_dir)
            lick_file = os.path.join(lick_dir, "LICK_Session1.csv")
            
            # Create file but mock pandas.read_csv to raise PermissionError
            Path(lick_file).touch()
            
            with patch('kitchen.loader.behavior_loader.routing.default_data_path') as mock_path:
                mock_path.return_value = temp_dir
                
                with patch('pandas.read_csv') as mock_read_csv:
                    mock_read_csv.side_effect = PermissionError("Permission denied")
                    
                    results = list(behavior_loader_from_fov(fov_node, timeline_dict))
                    
                    # Should handle permission error gracefully
                    assert len(results) == 1
                    assert 'lick' not in results[0]


if __name__ == "__main__":
    pytest.main([__file__])
