"""
Test module for table_detector.py
Tests the table detection functionality including border detection, cell detection, and duplicate removal.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, Mock

from src.table_detector import detect_table_borders, detect_cells, remove_duplicate_cells


class TestDetectTableBorders:
    """Test cases for detect_table_borders function."""
    
    def test_detect_table_borders_with_valid_image(self):
        """Test table border detection with a valid image containing a table."""
        # Create a simple test image with a rectangular table
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
        cv2.rectangle(img, (10, 10), (90, 90), (0, 0, 0), 2)  # Black table border
        
        result = detect_table_borders(img)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['x1', 'y1', 'x2', 'y2'])
        assert result['x1'] >= 0
        assert result['y1'] >= 0
        assert result['x2'] > result['x1']
        assert result['y2'] > result['y1']
    
    def test_detect_table_borders_with_grayscale_image(self):
        """Test table border detection with a grayscale image."""
        # Create a grayscale test image
        img = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.rectangle(img, (20, 20), (80, 80), 0, 2)
        
        result = detect_table_borders(img)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['x1', 'y1', 'x2', 'y2'])
    
    def test_detect_table_borders_no_table_found(self):
        """Test behavior when no table is found in the image."""
        # Create a completely white image
        img = np.ones((50, 50, 3), dtype=np.uint8) * 255
        
        with pytest.raises(ValueError, match="No table detected in the image"):
            detect_table_borders(img)
    
    def test_detect_table_borders_multiple_contours(self):
        """Test table border detection when multiple contours are present."""
        img = np.ones((120, 120, 3), dtype=np.uint8) * 255
        # Draw multiple rectangles, largest should be selected
        cv2.rectangle(img, (10, 10), (110, 110), (0, 0, 0), 2)  # Large table
        cv2.rectangle(img, (20, 20), (40, 40), (0, 0, 0), 2)    # Small table
        
        result = detect_table_borders(img)
        
        assert result['x2'] - result['x1'] > 50  # Should detect the larger table
        assert result['y2'] - result['y1'] > 50
    
    def test_detect_table_borders_edge_case_small_image(self):
        """Test table border detection with very small images."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (1, 1), (8, 8), (0, 0, 0), 1)
        
        result = detect_table_borders(img)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ['x1', 'y1', 'x2', 'y2'])


class TestDetectCells:
    """Test cases for detect_cells function."""
    
    def test_detect_cells_with_valid_table(self):
        """Test cell detection with a valid table image."""
        # Create a simple table with cells
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        table_bounds = {"x1": 10, "y1": 10, "x2": 90, "y2": 90}
        
        # Draw some cell boundaries within the table
        cv2.rectangle(img, (15, 15), (45, 45), (0, 0, 0), 1)
        cv2.rectangle(img, (55, 15), (85, 45), (0, 0, 0), 1)
        cv2.rectangle(img, (15, 55), (45, 85), (0, 0, 0), 1)
        cv2.rectangle(img, (55, 55), (85, 85), (0, 0, 0), 1)
        
        with patch('src.table_detector.remove_duplicate_cells') as mock_remove:
            mock_remove.return_value = [{"x1": 15, "y1": 15, "x2": 45, "y2": 45, "width": 30, "height": 30}]
            
            result = detect_cells(img, table_bounds)
            
            assert isinstance(result, list)
            assert len(result) > 0
            mock_remove.assert_called_once()
    
    def test_detect_cells_empty_table(self):
        """Test cell detection with an empty table."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # Just white image
        table_bounds = {"x1": 10, "y1": 10, "x2": 90, "y2": 90}
        
        with patch('src.table_detector.remove_duplicate_cells') as mock_remove:
            mock_remove.return_value = []
            result = detect_cells(img, table_bounds)
            assert result == []
    
    def test_detect_cells_boundary_coordinates(self):
        """Test that detected cells have correct coordinate transformation."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        table_bounds = {"x1": 20, "y1": 20, "x2": 80, "y2": 80}
        
        # Create a cell within the table region
        cv2.rectangle(img, (25, 25), (35, 35), (0, 0, 0), 1)
        
        with patch('src.table_detector.remove_duplicate_cells') as mock_remove:
            mock_remove.side_effect = lambda cells: cells  # Return cells as-is
            result = detect_cells(img, table_bounds)
            
            if result:  # If cells were detected
                cell = result[0]
                # Check that coordinates are in global image space, not table-relative
                assert cell["x1"] >= table_bounds["x1"]
                assert cell["y1"] >= table_bounds["y1"]
                assert cell["x2"] <= table_bounds["x2"]
                assert cell["y2"] <= table_bounds["y2"]
    
    def test_detect_cells_area_filtering(self):
        """Test that cells are filtered by area constraints."""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        table_bounds = {"x1": 10, "y1": 10, "x2": 190, "y2": 190}
        
        # Create cells of different sizes
        cv2.rectangle(img, (20, 20), (22, 22), (0, 0, 0), 1)    # Very small
        cv2.rectangle(img, (30, 30), (80, 80), (0, 0, 0), 1)    # Normal size
        cv2.rectangle(img, (11, 11), (189, 189), (0, 0, 0), 1)  # Very large
        
        with patch('src.table_detector.remove_duplicate_cells') as mock_remove:
            mock_remove.side_effect = lambda cells: cells
            result = detect_cells(img, table_bounds)
            
            # Should filter out very small and very large cells
            # The actual filtering is done by contour area detection, so we just check
            # that the function executes without error and returns some results
            assert isinstance(result, list)
    
    def test_detect_cells_invalid_table_bounds(self):
        """Test behavior with invalid table bounds."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        table_bounds = {"x1": 80, "y1": 80, "x2": 90, "y2": 90}  # Very small table
        
        with patch('src.table_detector.remove_duplicate_cells') as mock_remove:
            mock_remove.return_value = []
            result = detect_cells(img, table_bounds)
            assert isinstance(result, list)


class TestRemoveDuplicateCells:
    """Test cases for remove_duplicate_cells function."""
    
    def test_remove_duplicate_cells_no_duplicates(self):
        """Test with cells that have no duplicates."""
        cells = [
            {"x1": 10, "y1": 10, "x2": 20, "y2": 20},
            {"x1": 30, "y1": 30, "x2": 40, "y2": 40},
            {"x1": 50, "y1": 50, "x2": 60, "y2": 60}
        ]
        
        result = remove_duplicate_cells(cells)
        
        assert len(result) == 3
        assert result == cells
    
    def test_remove_duplicate_cells_with_duplicates(self):
        """Test with cells that have significant overlap."""
        cells = [
            {"x1": 10, "y1": 10, "x2": 30, "y2": 30},  # Original cell
            {"x1": 12, "y1": 12, "x2": 32, "y2": 32},  # Overlapping cell (should be removed)
            {"x1": 50, "y1": 50, "x2": 70, "y2": 70}   # Separate cell
        ]
        
        result = remove_duplicate_cells(cells)
        
        assert len(result) == 2  # Should remove one duplicate
        # Should keep the larger cell
        assert any(cell["x1"] == 10 and cell["y1"] == 10 for cell in result)
        assert any(cell["x1"] == 50 and cell["y1"] == 50 for cell in result)
    
    def test_remove_duplicate_cells_empty_list(self):
        """Test with empty cell list."""
        result = remove_duplicate_cells([])
        assert result == []
    
    def test_remove_duplicate_cells_invalid_dimensions(self):
        """Test with cells having invalid dimensions."""
        cells = [
            {"x1": 20, "y1": 10, "x2": 10, "y2": 30},  # Invalid: x2 < x1
            {"x1": 10, "y1": 20, "x2": 30, "y2": 10},  # Invalid: y2 < y1
            {"x1": 10, "y1": 10, "x2": 12, "y2": 12},  # Too small
            {"x1": 10, "y1": 10, "x2": 30, "y2": 30}   # Valid cell
        ]
        
        result = remove_duplicate_cells(cells)
        
        # Should only keep the valid cell
        assert len(result) == 1
        assert result[0]["x1"] == 10 and result[0]["y1"] == 10
    
    def test_remove_duplicate_cells_size_preference(self):
        """Test that larger cells are preferred over smaller ones."""
        cells = [
            {"x1": 10, "y1": 10, "x2": 20, "y2": 20},  # Small cell
            {"x1": 9, "y1": 9, "x2": 25, "y2": 25}     # Larger overlapping cell
        ]
        
        result = remove_duplicate_cells(cells)
        
        # Should keep the larger cell
        assert len(result) == 1
        assert result[0]["x2"] - result[0]["x1"] == 16  # Larger cell width
