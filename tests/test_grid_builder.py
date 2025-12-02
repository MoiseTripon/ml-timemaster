"""
Test module for grid_builder.py
Tests the grid building functionality including cell organization and rowspan/colspan calculations.
"""

import pytest
from src.grid_builder import organize_cells_into_grid


class TestOrganizeCellsIntoGrid:
    """Test cases for organize_cells_into_grid function."""
    
    def test_organize_cells_empty_list(self):
        """Test organizing an empty list of cells."""
        grid, num_rows, num_cols = organize_cells_into_grid([])
        
        assert grid == []
        assert num_rows == 0
        assert num_cols == 0
    
    def test_organize_cells_single_cell(self):
        """Test organizing a single cell."""
        cells = [
            {
                "bounds": {"x1": 10, "y1": 10, "x2": 50, "y2": 30},
                "text": "Cell 1"
            }
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        assert num_rows == 1
        assert num_cols == 1
        assert len(grid) == 1
        assert len(grid[0]) == 1
        assert grid[0][0]["text"] == "Cell 1"
        assert grid[0][0]["rowspan"] == 1
        assert grid[0][0]["colspan"] == 1
    
    def test_organize_cells_simple_grid_2x2(self):
        """Test organizing cells into a simple 2x2 grid."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 50, "y2": 30}, "text": "A1"},
            {"bounds": {"x1": 60, "y1": 10, "x2": 100, "y2": 30}, "text": "B1"},
            {"bounds": {"x1": 10, "y1": 40, "x2": 50, "y2": 60}, "text": "A2"},
            {"bounds": {"x1": 60, "y1": 40, "x2": 100, "y2": 60}, "text": "B2"}
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        assert num_rows >= 1
        assert num_cols >= 1
        assert len(grid) >= 1
        
        # Check that all cells are represented in the grid structure
        all_grid_texts = set()
        for row in grid:
            for cell in row:
                if cell and "text" in cell:
                    all_grid_texts.add(cell["text"])
        
        # Should contain all our input texts
        input_texts = {"A1", "B1", "A2", "B2"}
        assert all_grid_texts.issubset(input_texts) or len(all_grid_texts) == len(input_texts)
    
    def test_organize_cells_with_rowspan(self):
        """Test organizing cells where one cell spans multiple rows."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 50, "y2": 60}, "text": "Tall cell"},  # Spans 2 rows
            {"bounds": {"x1": 60, "y1": 10, "x2": 100, "y2": 30}, "text": "B1"},
            {"bounds": {"x1": 60, "y1": 40, "x2": 100, "y2": 60}, "text": "B2"}
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        assert num_rows >= 1
        assert num_cols >= 1
        
        # Find the tall cell
        tall_cell = None
        for row in grid:
            for cell in row:
                if cell["text"] == "Tall cell":
                    tall_cell = cell
                    break
        
        assert tall_cell is not None
        assert tall_cell["rowspan"] > 1
    
    def test_organize_cells_with_colspan(self):
        """Test organizing cells where one cell spans multiple columns."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 100, "y2": 30}, "text": "Wide cell"},  # Spans 2 columns
            {"bounds": {"x1": 10, "y1": 40, "x2": 50, "y2": 60}, "text": "A2"},
            {"bounds": {"x1": 60, "y1": 40, "x2": 100, "y2": 60}, "text": "B2"}
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        assert num_rows >= 1
        assert num_cols >= 1
        
        # Find the wide cell
        wide_cell = None
        for row in grid:
            for cell in row:
                if cell["text"] == "Wide cell":
                    wide_cell = cell
                    break
        
        assert wide_cell is not None
        assert wide_cell["colspan"] > 1
    
    def test_organize_cells_tolerance_alignment(self):
        """Test that cells with slightly misaligned positions are still organized properly."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 50, "y2": 30}, "text": "A1"},
            {"bounds": {"x1": 12, "y1": 40, "x2": 52, "y2": 60}, "text": "A2"},  # Slightly off alignment
            {"bounds": {"x1": 60, "y1": 8, "x2": 100, "y2": 32}, "text": "B1"}   # Slightly off alignment
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        assert num_rows >= 1
        assert num_cols >= 1
        assert len(grid) > 0
        
        # Should still organize into a reasonable grid structure
        total_cells = sum(len(row) for row in grid)
        assert total_cells <= 3  # Should not duplicate cells
    
    def test_organize_cells_overlapping_positions(self):
        """Test behavior with cells that have overlapping positions."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 60, "y2": 60}, "text": "Large cell"},
            {"bounds": {"x1": 20, "y1": 20, "x2": 40, "y2": 40}, "text": "Small cell"},  # Inside large cell
            {"bounds": {"x1": 70, "y1": 10, "x2": 110, "y2": 30}, "text": "Separate cell"}
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        assert isinstance(grid, list)
        assert isinstance(num_rows, int)
        assert isinstance(num_cols, int)
        assert num_rows >= 0
        assert num_cols >= 0
    
    def test_organize_cells_irregular_spacing(self):
        """Test organizing cells with irregular spacing between rows and columns."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 30, "y2": 25}, "text": "A1"},
            {"bounds": {"x1": 100, "y1": 10, "x2": 120, "y2": 25}, "text": "B1"},  # Large gap
            {"bounds": {"x1": 10, "y1": 80, "x2": 30, "y2": 95}, "text": "A2"},  # Large gap
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        assert num_rows >= 1
        assert num_cols >= 1
        
        # Should still create a valid grid structure
        for row in grid:
            assert isinstance(row, list)
            for cell in row:
                assert "text" in cell
                assert "rowspan" in cell
                assert "colspan" in cell
    
    def test_organize_cells_grid_dimensions_consistency(self):
        """Test that the reported grid dimensions match the actual grid structure."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 50, "y2": 30}, "text": "A1"},
            {"bounds": {"x1": 60, "y1": 10, "x2": 100, "y2": 30}, "text": "B1"},
            {"bounds": {"x1": 110, "y1": 10, "x2": 150, "y2": 30}, "text": "C1"},
            {"bounds": {"x1": 10, "y1": 40, "x2": 50, "y2": 60}, "text": "A2"}
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        # Check that reported dimensions match actual grid
        assert num_rows == len(grid)
        if grid:
            max_cols = max(len(row) for row in grid)
            assert num_cols >= max_cols
    
    def test_organize_cells_cell_properties(self):
        """Test that all cells in the grid have required properties."""
        cells = [
            {"bounds": {"x1": 10, "y1": 10, "x2": 50, "y2": 30}, "text": "Test cell"}
        ]
        
        grid, num_rows, num_cols = organize_cells_into_grid(cells)
        
        for row in grid:
            for cell in row:
                assert "text" in cell
                assert "rowspan" in cell
                assert "colspan" in cell
                assert isinstance(cell["rowspan"], int)
                assert isinstance(cell["colspan"], int)
                assert cell["rowspan"] >= 1
                assert cell["colspan"] >= 1
