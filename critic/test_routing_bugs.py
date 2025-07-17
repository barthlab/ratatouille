import pytest
import os
import tempfile
import sys
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen.configs.routing import robust_path_join, default_data_path, DATA_PATH
from kitchen.structure.hierarchical_data_structure import Node, Cohort, Mice, Fov
from kitchen.structure.meta_data_structure import TemporalObjectCoordinate, ObjectUID, TemporalUID
from kitchen.structure.neural_data_structure import NeuralData


class TestRobustPathJoinBugs:
    """Test cases designed to reveal bugs in robust_path_join."""
    
    def test_empty_arguments_bug(self):
        """Test behavior with no arguments."""
        with pytest.raises(AssertionError, match="at least one argument is required"):
            robust_path_join()
    
    def test_non_string_arguments_bug(self):
        """Test behavior with non-string arguments."""
        # Integer argument
        with pytest.raises(AssertionError, match="all arguments must be strings"):
            robust_path_join("path", 123, "file")
        
        # None argument mixed with strings
        with pytest.raises(AssertionError, match="all arguments must be strings"):
            robust_path_join("path", None, "file")
        
        # List argument
        with pytest.raises(AssertionError, match="all arguments must be strings"):
            robust_path_join("path", ["subpath"], "file")
        
        # Boolean argument
        with pytest.raises(AssertionError, match="all arguments must be strings"):
            robust_path_join("path", True, "file")
    
    def test_all_none_arguments_bug(self):
        """Test behavior when all arguments are None."""
        # This should fail because None values are filtered out,
        # leaving no arguments for os.path.join
        with pytest.raises(AssertionError, match="all arguments must be strings"):
            robust_path_join(None, None, None)
    
    def test_mixed_valid_invalid_arguments_bug(self):
        """Test behavior with mixed valid and invalid arguments."""
        # Mix of string and non-string
        with pytest.raises(AssertionError, match="all arguments must be strings"):
            robust_path_join("valid", "path", 42, "components")
    
    def test_empty_string_arguments(self):
        """Test behavior with empty string arguments."""
        # Empty strings should be valid
        result = robust_path_join("", "path", "", "file")
        expected = os.path.join("", "path", "", "file")
        assert result == expected
    
    def test_special_characters_in_paths(self):
        """Test behavior with special characters in path components."""
        # Paths with spaces
        result = robust_path_join("path with spaces", "file name")
        expected = os.path.join("path with spaces", "file name")
        assert result == expected
        
        # Paths with special characters
        result = robust_path_join("path@#$%", "file&*()!")
        expected = os.path.join("path@#$%", "file&*()!")
        assert result == expected
    
    def test_unicode_characters_bug(self):
        """Test behavior with unicode characters."""
        # Unicode path components
        result = robust_path_join("路径", "文件名", "测试")
        expected = os.path.join("路径", "文件名", "测试")
        assert result == expected
    
    def test_very_long_paths_bug(self):
        """Test behavior with very long path components."""
        # Very long path component
        long_component = "a" * 1000
        result = robust_path_join("short", long_component, "path")
        expected = os.path.join("short", long_component, "path")
        assert result == expected


class TestDefaultDataPathBugs:
    """Test cases designed to reveal bugs in default_data_path."""
    
    def create_test_node(self, cohort_id="TestCohort", mice_id=None, fov_id=None, cell_id=None):
        """Create a test node with specified hierarchy."""
        obj_uid = ObjectUID(cohort_id=cohort_id, mice_id=mice_id, fov_id=fov_id, cell_id=cell_id)
        temp_uid = TemporalUID(template_id="TestTemplate")
        coord = TemporalObjectCoordinate(object_uid=obj_uid, temporal_uid=temp_uid)
        return Node(coordinate=coord, data=NeuralData())
    
    def test_nonexistent_path_with_check_exist_true(self):
        """Test behavior when path doesn't exist and check_exist=True."""
        node = self.create_test_node()
        
        # Should raise AssertionError when path doesn't exist
        with pytest.raises(AssertionError, match="Cannot find data path"):
            default_data_path(node, check_exist=True)
    
    def test_nonexistent_path_with_check_exist_false(self):
        """Test behavior when path doesn't exist and check_exist=False."""
        node = self.create_test_node()
        
        # Should return path without checking existence
        result = default_data_path(node, check_exist=False)
        expected = os.path.join(DATA_PATH, "TestCohort")
        assert result == expected
    
    def test_hierarchical_path_construction_bug(self):
        """Test hierarchical path construction for potential bugs."""
        # Test different hierarchy levels
        cohort_node = self.create_test_node(cohort_id="C1")
        mice_node = self.create_test_node(cohort_id="C1", mice_id="M1")
        fov_node = self.create_test_node(cohort_id="C1", mice_id="M1", fov_id="F1")
        cell_node = self.create_test_node(cohort_id="C1", mice_id="M1", fov_id="F1", cell_id=1)
        
        # Get paths without checking existence
        cohort_path = default_data_path(cohort_node, check_exist=False)
        mice_path = default_data_path(mice_node, check_exist=False)
        fov_path = default_data_path(fov_node, check_exist=False)
        cell_path = default_data_path(cell_node, check_exist=False)
        
        # Verify hierarchical structure
        assert cohort_path == os.path.join(DATA_PATH, "C1")
        assert mice_path == os.path.join(DATA_PATH, "C1", "M1")
        assert fov_path == os.path.join(DATA_PATH, "C1", "M1", "F1")
        assert cell_path == os.path.join(DATA_PATH, "C1", "M1", "F1", "1")
    
    def test_none_values_in_hierarchy_bug(self):
        """Test behavior with None values in object UID hierarchy."""
        # This tests the filtering of None values in the path construction
        node = self.create_test_node(cohort_id="C1", mice_id="M1", fov_id=None, cell_id=None)
        
        result = default_data_path(node, check_exist=False)
        expected = os.path.join(DATA_PATH, "C1", "M1")
        assert result == expected
    
    def test_special_characters_in_uid_bug(self):
        """Test behavior with special characters in UID components."""
        # UIDs with special characters
        node = self.create_test_node(
            cohort_id="Cohort@#$%",
            mice_id="Mouse with spaces",
            fov_id="FOV_123"
        )
        
        result = default_data_path(node, check_exist=False)
        expected = os.path.join(DATA_PATH, "Cohort@#$%", "Mouse with spaces", "FOV_123")
        assert result == expected
    
    def test_numeric_vs_string_uid_bug(self):
        """Test potential issues with numeric vs string UID components."""
        # cell_id is typically numeric, others are strings
        node = self.create_test_node(
            cohort_id="C1",
            mice_id="M1", 
            fov_id="F1",
            cell_id=123  # Numeric cell_id
        )
        
        result = default_data_path(node, check_exist=False)
        expected = os.path.join(DATA_PATH, "C1", "M1", "F1", "123")
        assert result == expected
    
    def test_empty_string_uid_components_bug(self):
        """Test behavior with empty string UID components."""
        # Empty string components (but not None)
        node = self.create_test_node(
            cohort_id="C1",
            mice_id="",  # Empty string
            fov_id="F1"
        )
        
        result = default_data_path(node, check_exist=False)
        # Empty string should be included in path
        expected = os.path.join(DATA_PATH, "C1", "", "F1")
        assert result == expected
    
    def test_uid_iteration_bug(self):
        """Test potential bugs in UID iteration logic."""
        # The function uses: [value for name, value in node_obj_uid if value is not None]
        # This tests the iteration protocol of ObjectUID
        
        node = self.create_test_node(cohort_id="C1", mice_id="M1")
        
        # Manually check what the iteration yields
        uid_values = [value for name, value in node.coordinate.object_uid if value is not None]
        expected_values = ["C1", "M1"]
        assert uid_values == expected_values
        
        result = default_data_path(node, check_exist=False)
        expected = os.path.join(DATA_PATH, *expected_values)
        assert result == expected
    
    def test_path_exists_check_with_file_instead_of_directory(self):
        """Test behavior when path exists but is a file, not a directory."""
        node = self.create_test_node()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with the expected path name
            expected_path = os.path.join(temp_dir, "TestCohort")
            with open(expected_path, 'w') as f:
                f.write("test")
            
            with patch('kitchen.configs.routing.DATA_PATH', temp_dir):
                # Should this pass or fail? The path exists but it's a file
                # The current implementation only checks path.exists(), not path.isdir()
                result = default_data_path(node, check_exist=True)
                assert result == expected_path
    
    def test_symlink_path_bug(self):
        """Test behavior with symbolic links."""
        node = self.create_test_node()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create actual directory
            actual_dir = os.path.join(temp_dir, "actual")
            os.makedirs(actual_dir)
            
            # Create symlink
            symlink_path = os.path.join(temp_dir, "TestCohort")
            try:
                os.symlink(actual_dir, symlink_path)
                
                with patch('kitchen.configs.routing.DATA_PATH', temp_dir):
                    result = default_data_path(node, check_exist=True)
                    assert result == symlink_path
            except OSError:
                # Skip test if symlinks not supported (e.g., Windows without admin)
                pytest.skip("Symlinks not supported on this system")
    
    def test_very_deep_hierarchy_bug(self):
        """Test behavior with very deep hierarchy paths."""
        # Create a node with maximum hierarchy depth
        node = self.create_test_node(
            cohort_id="VeryLongCohortNameThatMightCauseIssues",
            mice_id="VeryLongMouseNameThatMightCauseIssues", 
            fov_id="VeryLongFOVNameThatMightCauseIssues",
            cell_id=999999
        )
        
        result = default_data_path(node, check_exist=False)
        
        # Should handle long paths without issues
        assert "VeryLongCohortNameThatMightCauseIssues" in result
        assert "VeryLongMouseNameThatMightCauseIssues" in result
        assert "VeryLongFOVNameThatMightCauseIssues" in result
        assert "999999" in result
    
    def test_concurrent_access_bug(self):
        """Test potential issues with concurrent access to paths."""
        # This is more of a stress test for potential race conditions
        node = self.create_test_node()
        
        # Multiple calls should return consistent results
        results = []
        for _ in range(100):
            result = default_data_path(node, check_exist=False)
            results.append(result)
        
        # All results should be identical
        assert all(r == results[0] for r in results)
    
    def test_node_coordinate_modification_bug(self):
        """Test behavior when node coordinate is modified after creation."""
        node = self.create_test_node()
        original_path = default_data_path(node, check_exist=False)
        
        # Node coordinates are frozen, so this should fail
        with pytest.raises((AttributeError, TypeError)):
            node.coordinate.object_uid.cohort_id = "Modified"
        
        # Path should remain unchanged
        new_path = default_data_path(node, check_exist=False)
        assert new_path == original_path


if __name__ == "__main__":
    pytest.main([__file__])
