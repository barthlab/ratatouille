import pytest
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen.loader.behavior_loader import (
    spike_to_firing_rate, clean_time_series, parse_behavior_filename,
    process_locomotion_data, process_lick_data, process_pupil_data,
    process_tongue_data, process_whisker_data, behavior_loader_from_fov
)
from kitchen.structure.neural_data_structure import NeuralData, TimeSeries
from kitchen.structure.hierarchical_data_structure import Fov
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate, ObjectUID, TemporalUID


class TestSimplifiedBehaviorLoader:
    """Test the simplified behavior loader implementation."""
    
    def test_parse_behavior_filename_valid(self):
        """Test parsing valid behavior filename."""
        filename = "LOCOMOTION_20230101_M001_FOV1_S1_training"
        result = parse_behavior_filename(filename)
        assert result == "LOCOMOTION"
    
    def test_parse_behavior_filename_invalid_prefix(self):
        """Test parsing filename with invalid behavior type."""
        filename = "INVALID_20230101_M001_FOV1_S1_training"
        with pytest.raises(ValueError, match="not supported"):
            parse_behavior_filename(filename)
    
    def test_neural_data_with_behavior_modalities(self):
        """Test NeuralData class with behavior modalities."""
        locomotion_data = TimeSeries(v=np.array([1, 2, 3]), t=np.array([0.1, 0.2, 0.3]))
        lick_data = TimeSeries(v=np.array([0, 1]), t=np.array([0.1, 0.2]))
        
        neural_data = NeuralData(
            locomotion=locomotion_data,
            lick=lick_data,
            pupil=None,
            tongue=None,
            whisker=None
        )
        
        assert neural_data.locomotion is not None
        assert neural_data.lick is not None
        assert neural_data.pupil is None
        assert neural_data.tongue is None
        assert neural_data.whisker is None
    
    @patch('kitchen.configs.routing.default_data_path')
    @patch('os.walk')
    @patch('pandas.read_csv')
    def test_behavior_loader_from_fov_simplified(self, mock_read_csv, mock_walk, mock_path):
        """Test simplified behavior loader from FOV."""
        # Setup mocks
        mock_path.return_value = "/fake/path"
        mock_walk.return_value = [("/fake/path", [], [
            "LOCOMOTION_20230101_M001_FOV1_S1_training.csv",
            "LICK_20230101_M001_FOV1_S1_training.csv"
        ])]
        
        # Mock CSV data for different types
        def mock_csv_side_effect(filepath, **kwargs):
            mock_df = Mock()
            if "LOCOMOTION" in filepath:
                mock_df.to_numpy.return_value = np.array([[1000, 10.0, 0.0], [2000, 15.0, 0.0]])
            elif "LICK" in filepath:
                mock_df.to_numpy.return_value = np.array([[1000], [2000]])
            return mock_df
        
        mock_read_csv.side_effect = mock_csv_side_effect
        
        # Create mock FOV node
        fov_node = Mock(spec=Fov)
        fov_node.coordinate = Mock()
        
        # Test the loader
        result = behavior_loader_from_fov(fov_node)
        
        assert isinstance(result, dict)
        assert "locomotion" in result
        assert "lick" in result
        assert isinstance(result["locomotion"], TimeSeries)
        assert isinstance(result["lick"], TimeSeries)
    
    @patch('kitchen.configs.routing.default_data_path')
    @patch('os.walk')
    def test_behavior_loader_from_fov_no_data(self, mock_walk, mock_path):
        """Test behavior loader when no data files are found."""
        # Setup mocks for empty directory
        mock_path.return_value = "/fake/path"
        mock_walk.return_value = [("/fake/path", [], [])]
        
        # Create mock FOV node
        fov_node = Mock(spec=Fov)
        fov_node.coordinate = Mock()
        
        # Test the loader
        result = behavior_loader_from_fov(fov_node)
        
        assert isinstance(result, dict)
        assert len(result) == 0  # Should return empty dict when no data found
    
    def test_data_processing_functions(self):
        """Test that all data processing functions work correctly."""
        # Test locomotion
        locomotion_data = np.array([[1000, 10.0, 0.0], [2000, 15.0, 0.0]])
        locomotion_result = process_locomotion_data(locomotion_data)
        assert isinstance(locomotion_result, TimeSeries)
        
        # Test lick
        lick_data = np.array([[1000], [2000]])
        lick_result = process_lick_data(lick_data)
        assert isinstance(lick_result, TimeSeries)
        
        # Test pupil
        pupil_data = np.array([[1000, 50.0], [2000, 55.0]])
        pupil_result = process_pupil_data(pupil_data)
        assert isinstance(pupil_result, TimeSeries)
        
        # Test tongue
        tongue_data = np.array([[1000, 10.0], [2000, 12.0]])
        tongue_result = process_tongue_data(tongue_data)
        assert isinstance(tongue_result, TimeSeries)
        
        # Test whisker
        whisker_data = np.array([[1000, 5.0], [2000, 7.0]])
        whisker_result = process_whisker_data(whisker_data)
        assert isinstance(whisker_result, TimeSeries)


if __name__ == "__main__":
    pytest.main([__file__])
