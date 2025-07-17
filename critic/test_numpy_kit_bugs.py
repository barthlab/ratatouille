import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kitchen.utils.numpy_kit import numpy_percentile_filter


class TestNumpyPercentileFilterBugs:
    """Test cases designed to reveal bugs in numpy_percentile_filter."""
    
    def test_input_validation_bugs(self):
        """Test input validation edge cases."""
        # Non-numpy array input
        with pytest.raises(TypeError, match="Input must be a NumPy array"):
            numpy_percentile_filter([1, 2, 3, 4, 5], 3, 50.0)
        
        # Multi-dimensional array
        with pytest.raises(ValueError, match="Input array must be 1-dimensional"):
            numpy_percentile_filter(np.array([[1, 2], [3, 4]]), 3, 50.0)
        
        # Non-integer window size
        with pytest.raises(ValueError, match="Window size 's' must be a positive integer"):
            numpy_percentile_filter(np.array([1, 2, 3, 4, 5]), 3.5, 50.0)
        
        # Zero window size
        with pytest.raises(ValueError, match="Window size 's' must be a positive integer"):
            numpy_percentile_filter(np.array([1, 2, 3, 4, 5]), 0, 50.0)
        
        # Negative window size
        with pytest.raises(ValueError, match="Window size 's' must be a positive integer"):
            numpy_percentile_filter(np.array([1, 2, 3, 4, 5]), -1, 50.0)
        
        # Percentile out of range (negative)
        with pytest.raises(ValueError, match="Percentile 'q' must be between 0 and 100"):
            numpy_percentile_filter(np.array([1, 2, 3, 4, 5]), 3, -10.0)
        
        # Percentile out of range (> 100)
        with pytest.raises(ValueError, match="Percentile 'q' must be between 0 and 100"):
            numpy_percentile_filter(np.array([1, 2, 3, 4, 5]), 3, 150.0)
    
    def test_empty_array_handling(self):
        """Test handling of empty input arrays."""
        empty_array = np.array([])
        result = numpy_percentile_filter(empty_array, 3, 50.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        assert result.shape == (0,)
    
    def test_single_element_array_bug(self):
        """Test potential bug with single-element arrays."""
        single_element = np.array([42.0])
        result = numpy_percentile_filter(single_element, 3, 50.0)
        
        assert len(result) == 1
        assert result[0] == 42.0
    
    def test_window_larger_than_array_bug(self):
        """Test potential bug when window size is larger than array."""
        small_array = np.array([1, 2, 3])
        large_window = 10
        
        # This might cause issues with padding or window creation
        result = numpy_percentile_filter(small_array, large_window, 50.0)
        
        assert len(result) == 3  # Should return same length as input
        # All values might be the same due to large window
    
    def test_even_vs_odd_window_size_bug(self):
        """Test potential differences between even and odd window sizes."""
        test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Odd window size
        result_odd = numpy_percentile_filter(test_array, 3, 50.0)
        
        # Even window size
        result_even = numpy_percentile_filter(test_array, 4, 50.0)
        
        assert len(result_odd) == len(test_array)
        assert len(result_even) == len(test_array)
        
        # Results should be different due to different window sizes
        assert not np.array_equal(result_odd, result_even)
    
    def test_edge_padding_behavior_bug(self):
        """Test edge padding behavior for potential bugs."""
        # Array with distinct edge values
        test_array = np.array([100, 1, 2, 3, 4, 5, 200])
        window_size = 3
        
        result = numpy_percentile_filter(test_array, window_size, 50.0)
        
        # First element should be influenced by edge padding
        # The padding should replicate the edge value (100)
        # So the window for first element should be [100, 100, 1]
        # Median should be 100
        assert result[0] == 100
        
        # Last element should be influenced by edge padding
        # The window for last element should be [5, 200, 200]
        # Median should be 200
        assert result[-1] == 200
    
    def test_percentile_boundary_values_bug(self):
        """Test percentile boundary values (0th and 100th percentiles)."""
        test_array = np.array([5, 1, 9, 2, 8, 3, 7, 4, 6])
        window_size = 3
        
        # 0th percentile (minimum)
        result_min = numpy_percentile_filter(test_array, window_size, 0.0)
        
        # 100th percentile (maximum)
        result_max = numpy_percentile_filter(test_array, window_size, 100.0)
        
        assert len(result_min) == len(test_array)
        assert len(result_max) == len(test_array)
        
        # All values in result_min should be <= corresponding values in result_max
        assert np.all(result_min <= result_max)
    
    def test_stride_tricks_memory_bug(self):
        """Test potential memory issues with stride_tricks."""
        # Large array to test memory handling
        large_array = np.random.rand(10000)
        window_size = 101  # Large odd window
        
        result = numpy_percentile_filter(large_array, window_size, 50.0)
        
        assert len(result) == len(large_array)
        assert not np.any(np.isnan(result))  # No NaN values
        assert not np.any(np.isinf(result))  # No infinite values
    
    def test_data_type_preservation_bug(self):
        """Test that data types are preserved correctly."""
        # Integer array
        int_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        result_int = numpy_percentile_filter(int_array, 3, 50.0)
        
        # Float array
        float_array = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
        result_float = numpy_percentile_filter(float_array, 3, 50.0)
        
        # Results should maintain reasonable data types
        # Note: np.percentile might change data types
        assert len(result_int) == len(int_array)
        assert len(result_float) == len(float_array)
    
    def test_window_size_one_bug(self):
        """Test edge case with window size of 1."""
        test_array = np.array([1, 2, 3, 4, 5])
        result = numpy_percentile_filter(test_array, 1, 50.0)
        
        # With window size 1, each element should be its own percentile
        # So the result should be identical to the input
        np.testing.assert_array_equal(result, test_array)
    
    def test_identical_values_bug(self):
        """Test behavior with arrays containing identical values."""
        # All same values
        identical_array = np.array([7, 7, 7, 7, 7, 7, 7])
        result = numpy_percentile_filter(identical_array, 3, 50.0)
        
        # Result should be all 7s
        np.testing.assert_array_equal(result, identical_array)
    
    def test_extreme_values_bug(self):
        """Test behavior with extreme values."""
        # Array with very large and very small values
        extreme_array = np.array([1e-10, 1e10, 1e-10, 1e10, 1e-10])
        result = numpy_percentile_filter(extreme_array, 3, 50.0)
        
        assert len(result) == len(extreme_array)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_nan_values_bug(self):
        """Test behavior with NaN values in input."""
        # Array containing NaN values
        nan_array = np.array([1, 2, np.nan, 4, 5])
        
        # This might cause issues with percentile calculation
        result = numpy_percentile_filter(nan_array, 3, 50.0)
        
        # The result might contain NaN values or handle them in unexpected ways
        assert len(result) == len(nan_array)
        # The behavior with NaN is implementation-dependent
    
    def test_infinite_values_bug(self):
        """Test behavior with infinite values in input."""
        # Array containing infinite values
        inf_array = np.array([1, 2, np.inf, 4, 5])
        
        result = numpy_percentile_filter(inf_array, 3, 50.0)
        
        assert len(result) == len(inf_array)
        # The behavior with infinity is implementation-dependent
    
    def test_padding_calculation_bug(self):
        """Test potential bugs in padding calculation."""
        test_array = np.array([1, 2, 3, 4, 5, 6])
        
        # Test various window sizes to check padding calculation
        for window_size in [2, 3, 4, 5, 6, 7]:
            result = numpy_percentile_filter(test_array, window_size, 50.0)
            assert len(result) == len(test_array)
    
    def test_stride_calculation_bug(self):
        """Test potential bugs in stride calculation."""
        test_array = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        window_size = 4
        
        result = numpy_percentile_filter(test_array, window_size, 50.0)
        
        # Manually verify a few values to check stride calculation
        # For element at index 2 (value 3), window should be [1, 2, 3, 4]
        # Median should be 2.5
        expected_median = np.percentile([1, 2, 3, 4], 50.0)
        
        # Allow for small floating point differences
        assert abs(result[2] - expected_median) < 1e-10
    
    def test_memory_view_bug(self):
        """Test potential issues with memory views and stride_tricks."""
        # Create array from different sources
        list_array = np.array([1, 2, 3, 4, 5])
        slice_array = np.arange(10)[2:7]  # Creates a view
        copy_array = np.copy(slice_array)
        
        # All should work the same way
        result1 = numpy_percentile_filter(list_array, 3, 50.0)
        result2 = numpy_percentile_filter(slice_array, 3, 50.0)
        result3 = numpy_percentile_filter(copy_array, 3, 50.0)
        
        assert len(result1) == 5
        assert len(result2) == 5
        assert len(result3) == 5
        
        # Results should be similar (accounting for different input values)
        np.testing.assert_array_equal(result2, result3)
    
    def test_percentile_interpolation_bug(self):
        """Test potential issues with percentile interpolation."""
        # Array where percentile requires interpolation
        test_array = np.array([1, 2, 3, 4])
        window_size = 4
        
        # 25th percentile of [1, 2, 3, 4] requires interpolation
        result = numpy_percentile_filter(test_array, window_size, 25.0)
        
        assert len(result) == len(test_array)
        # The exact value depends on numpy's percentile interpolation method


if __name__ == "__main__":
    pytest.main([__file__])
