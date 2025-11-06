"""
Test suite for TableAnalyzer class.
"""

import pytest
import cv2
import numpy as np
from src.table_analyzer import TableAnalyzer


class TestTableAnalyzer:
    """Test cases for TableAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a table analyzer instance for testing."""
        return TableAnalyzer(overlap_threshold=0.7, min_cell_size=5)
    
    @pytest.fixture
    def sample_table_image(self):
        """Create a sample table image."""
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Draw table border
        cv2.rectangle(img, (50, 50), (350, 250), (0, 0, 0), 3)
        
        # Draw some cells
        cv2.line(img, (50, 150), (350, 150), (0, 0, 0), 2)
        cv2.line(img, (200, 50), (200, 250), (0, 0, 0), 2)
        
        return img
    
    @pytest.fixture
    def blank_image(self):
        """Create a blank image without table."""
        return np.ones((200, 200, 3), dtype=np.uint8) * 255
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes with correct parameters."""
        analyzer = TableAnalyzer(overlap_threshold=0.8, min_cell_size=10)
        assert analyzer.overlap_threshold == 0.8
        assert analyzer.min_cell_size == 10
    
    def test_analyzer_default_initialization(self):
        """Test that analyzer uses default parameters when not specified."""
        analyzer = TableAnalyzer()
        assert analyzer.overlap_threshold == 0.7
        assert analyzer.min_cell_size == 5
    
    def test_detect_table_borders_with_table(self, analyzer, sample_table_image):
        """Test that table borders are detected in image with table."""
        result = analyzer.detect_table_borders(sample_table_image)
        
        # Should return a dictionary with boundary coordinates
        assert isinstance(result, dict)
        assert 'x1' in result
        assert 'y1' in result
        assert 'x2' in result
        assert 'y2' in result
        
        # Coordinates should be within image bounds
        assert 0 <= result['x1'] < sample_table_image.shape[1]
        assert 0 <= result['y1'] < sample_table_image.shape[0]
        assert result['x1'] < result['x2'] <= sample_table_image.shape[1]
        assert result['y1'] < result['y2'] <= sample_table_image.shape[0]
    
    def test_detect_table_borders_with_blank_image(self, analyzer, blank_image):
        """Test that blank images raise appropriate error."""
        with pytest.raises(ValueError, match="No table detected"):
            analyzer.detect_table_borders(blank_image)
    
    def test_detect_table_borders_with_grayscale(self, analyzer):
        """Test table detection works with grayscale images."""
        img = np.ones((200, 200), dtype=np.uint8) * 255
        cv2.rectangle(img, (30, 30), (170, 170), (0, 0, 0), 3)
        
        result = analyzer.detect_table_borders(img)
        assert isinstance(result, dict)
        assert all(key in result for key in ['x1', 'y1', 'x2', 'y2'])
    
    def test_remove_duplicate_cells_empty_list(self, analyzer):
        """Test that removing duplicates from empty list returns empty list."""
        result = analyzer.remove_duplicate_cells([])
        assert result == []
    
    def test_remove_duplicate_cells_no_duplicates(self, analyzer):
        """Test that non-overlapping cells are all kept."""
        cells = [
            {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10},
            {'x1': 20, 'y1': 0, 'x2': 30, 'y2': 10},
            {'x1': 0, 'y1': 20, 'x2': 10, 'y2': 30}
        ]
        result = analyzer.remove_duplicate_cells(cells)
        assert len(result) == 3
    
    def test_remove_duplicate_cells_with_duplicates(self, analyzer):
        """Test that overlapping cells are removed."""
        cells = [
            {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100},
            {'x1': 1, 'y1': 1, 'x2': 101, 'y2': 101},  # Almost identical
            {'x1': 200, 'y1': 200, 'x2': 300, 'y2': 300}
        ]
        result = analyzer.remove_duplicate_cells(cells)
        # Should keep only 2 cells (the duplicate should be removed)
        assert len(result) == 2
    
    def test_remove_duplicate_cells_filters_invalid(self, analyzer):
        """Test that cells with invalid dimensions are filtered out."""
        cells = [
            {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 100},  # Valid
            {'x1': 10, 'y1': 10, 'x2': 10, 'y2': 10},  # Zero width/height
            {'x1': 10, 'y1': 10, 'x2': 5, 'y2': 20},   # Negative width
            {'x1': 0, 'y1': 0, 'x2': 2, 'y2': 2}       # Too small
        ]
        result = analyzer.remove_duplicate_cells(cells)
        # Should keep only the valid cell
        assert len(result) == 1
        assert result[0]['x1'] == 0 and result[0]['x2'] == 100
    
    def test_detect_cells(self, analyzer, sample_table_image):
        """Test cell detection in a table."""
        table_bounds = analyzer.detect_table_borders(sample_table_image)
        cells = analyzer.detect_cells(sample_table_image, table_bounds)
        
        # Should return a list of cells
        assert isinstance(cells, list)
        
        # Each cell should have required keys
        for cell in cells:
            assert 'x1' in cell
            assert 'y1' in cell
            assert 'x2' in cell
            assert 'y2' in cell
            assert 'width' in cell
            assert 'height' in cell
    
    def test_organize_cells_into_grid_empty(self, analyzer):
        """Test organizing empty cell list into grid."""
        grid, num_rows, num_cols = analyzer.organize_cells_into_grid([])
        assert grid == []
        assert num_rows == 0
        assert num_cols == 0
    
    def test_organize_cells_into_grid_single_cell(self, analyzer):
        """Test organizing single cell into grid."""
        cells = [
            {
                'bounds': {'x1': 0, 'y1': 0, 'x2': 100, 'y2': 50},
                'text': 'Cell1'
            }
        ]
        grid, num_rows, num_cols = analyzer.organize_cells_into_grid(cells)
        
        assert num_rows >= 1
        assert num_cols >= 1
        assert len(grid) >= 1
    
    def test_organize_cells_into_grid_multiple_cells(self, analyzer):
        """Test organizing multiple cells into grid."""
        cells = [
            {'bounds': {'x1': 0, 'y1': 0, 'x2': 50, 'y2': 50}, 'text': 'A'},
            {'bounds': {'x1': 60, 'y1': 0, 'x2': 110, 'y2': 50}, 'text': 'B'},
            {'bounds': {'x1': 0, 'y1': 60, 'x2': 50, 'y2': 110}, 'text': 'C'},
            {'bounds': {'x1': 60, 'y1': 60, 'x2': 110, 'y2': 110}, 'text': 'D'}
        ]
        grid, num_rows, num_cols = analyzer.organize_cells_into_grid(cells)
        
        # Should create a grid with at least 2 rows
        assert num_rows >= 2
        assert num_cols >= 1
        assert len(grid) >= 2
        # Verify all cells have required attributes
        for row in grid:
            for cell in row:
                assert 'text' in cell
                assert 'rowspan' in cell
                assert 'colspan' in cell
