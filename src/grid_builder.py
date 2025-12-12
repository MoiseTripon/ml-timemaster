"""
Grid building module for ML Timemaster.
Contains the GridBuilder class for organizing detected cells into a structured grid.
"""

import logging
from typing import List, Dict, Tuple, Optional, Set


class GridBuilder:
    """Handles organization of cells into grid structure."""
    
    def __init__(self, tolerance: int = 15):
        """
        Initialize the GridBuilder.
        
        Args:
            tolerance (int): Tolerance for merging close positions
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)

    def organize_cells_into_grid(self, cells: List[Dict]) -> Tuple[List[List[Dict]], int, int]:
        """
        Organize cells into a 2D grid structure.
        
        Since cells are defined by their borders, we need to:
        1. Determine grid positions from cell boundaries
        2. Calculate rowspan/colspan based on cell size vs typical cell size
        3. Place cells in grid and handle overlaps
        """
        if not cells:
            return [], 0, 0

        # Sort cells by position
        sorted_cells = sorted(cells, key=lambda c: (c["bounds"]["y1"], c["bounds"]["x1"]))
        
        # Get unique row and column positions
        row_positions = self._get_unique_positions([c["bounds"]["y1"] for c in sorted_cells])
        row_positions.append(max(c["bounds"]["y2"] for c in sorted_cells))  # Add bottom edges
        row_positions = sorted(set(row_positions))
        
        col_positions = self._get_unique_positions([c["bounds"]["x1"] for c in sorted_cells])
        col_positions.append(max(c["bounds"]["x2"] for c in sorted_cells))  # Add right edges
        col_positions = sorted(set(col_positions))
        
        num_rows = len(row_positions) - 1
        num_cols = len(col_positions) - 1
        
        self.logger.info(f"Grid dimensions: {num_rows} rows x {num_cols} columns")
        
        if num_rows < 1 or num_cols < 1:
            return [], 0, 0
        
        # Calculate typical cell dimensions
        typical_width = self._calculate_typical_dimension(
            [c["bounds"]["x2"] - c["bounds"]["x1"] for c in sorted_cells]
        )
        typical_height = self._calculate_typical_dimension(
            [c["bounds"]["y2"] - c["bounds"]["y1"] for c in sorted_cells]
        )
        
        self.logger.info(f"Typical cell size: {typical_width:.1f} x {typical_height:.1f}")
        
        # Initialize grid
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        span_coverage: Set[Tuple[int, int]] = set()
        
        # Place cells in grid
        for cell in sorted_cells:
            row_idx = self._find_position_index(cell["bounds"]["y1"], row_positions)
            col_idx = self._find_position_index(cell["bounds"]["x1"], col_positions)
            
            if row_idx is None or col_idx is None:
                self.logger.warning(f"Could not place cell at ({cell['bounds']['x1']}, {cell['bounds']['y1']})")
                continue
            
            if row_idx >= num_rows or col_idx >= num_cols:
                continue
            
            if (row_idx, col_idx) in span_coverage:
                continue
            
            # Calculate spans based on cell boundaries
            rowspan = self._calculate_span(
                cell["bounds"]["y1"], cell["bounds"]["y2"],
                row_positions, typical_height
            )
            colspan = self._calculate_span(
                cell["bounds"]["x1"], cell["bounds"]["x2"],
                col_positions, typical_width
            )
            
            # Ensure spans are valid
            rowspan = max(1, min(rowspan, num_rows - row_idx))
            colspan = max(1, min(colspan, num_cols - col_idx))
            
            # Check for conflicts
            rowspan, colspan = self._adjust_for_conflicts(
                row_idx, col_idx, rowspan, colspan, span_coverage, num_rows, num_cols
            )
            
            cell_info = {
                "text": cell.get("text", ""),
                "rowspan": rowspan,
                "colspan": colspan
            }
            
            grid[row_idx][col_idx] = cell_info
            
            # Mark covered positions
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    span_coverage.add((r, c))
        
        # Fill gaps
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] is None:
                    grid[r][c] = {"text": "", "rowspan": 1, "colspan": 1}
        
        # Validate
        grid = self._validate_grid(grid, num_rows, num_cols)
        
        return grid, num_rows, num_cols

    def _get_unique_positions(self, positions: List[float]) -> List[float]:
        """Get unique positions, merging close ones."""
        if not positions:
            return []
        
        sorted_pos = sorted(set(positions))
        merged = []
        current_group = [sorted_pos[0]]
        
        for pos in sorted_pos[1:]:
            if pos - current_group[-1] <= self.tolerance:
                current_group.append(pos)
            else:
                merged.append(sum(current_group) / len(current_group))
                current_group = [pos]
        
        if current_group:
            merged.append(sum(current_group) / len(current_group))
        
        return merged

    def _calculate_typical_dimension(self, dimensions: List[float]) -> float:
        """Calculate typical dimension using lower quartile."""
        if not dimensions:
            return 50.0
        
        sorted_dims = sorted(dimensions)
        # Use lower quartile to get typical single-cell size
        idx = max(0, len(sorted_dims) // 4)
        return sorted_dims[idx]

    def _find_position_index(self, pos: float, positions: List[float]) -> Optional[int]:
        """Find index for a position."""
        for idx, ref_pos in enumerate(positions[:-1]):  # Exclude last position (it's the end boundary)
            if abs(pos - ref_pos) <= self.tolerance:
                return idx
        
        # Fallback: find closest
        min_dist = float('inf')
        best_idx = None
        for idx, ref_pos in enumerate(positions[:-1]):
            dist = abs(pos - ref_pos)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        return best_idx if min_dist <= self.tolerance * 2 else None

    def _calculate_span(self, start: float, end: float, 
                        positions: List[float], typical_size: float) -> int:
        """
        Calculate span based on how many grid positions the cell covers.
        """
        span = 0
        for i, pos in enumerate(positions[:-1]):
            next_pos = positions[i + 1]
            
            # Check if this grid cell is within the cell's boundaries
            if pos >= start - self.tolerance and next_pos <= end + self.tolerance:
                span += 1
            elif pos >= end:
                break
        
        return max(1, span)

    def _adjust_for_conflicts(self, row_idx: int, col_idx: int,
                               rowspan: int, colspan: int,
                               covered: Set[Tuple[int, int]],
                               num_rows: int, num_cols: int) -> Tuple[int, int]:
        """Adjust spans to avoid conflicts."""
        new_colspan = colspan
        while new_colspan > 1:
            conflict = False
            for r in range(row_idx, min(row_idx + rowspan, num_rows)):
                for c in range(col_idx, min(col_idx + new_colspan, num_cols)):
                    if (r, c) in covered:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                break
            new_colspan -= 1
        
        new_rowspan = rowspan
        while new_rowspan > 1:
            conflict = False
            for r in range(row_idx, min(row_idx + new_rowspan, num_rows)):
                for c in range(col_idx, min(col_idx + new_colspan, num_cols)):
                    if (r, c) in covered:
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                break
            new_rowspan -= 1
        
        return max(1, new_rowspan), max(1, new_colspan)

    def _validate_grid(self, grid: List[List[Optional[Dict]]], 
                       num_rows: int, num_cols: int) -> List[List[Dict]]:
        """Validate grid structure."""
        if not grid:
            return grid
        
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] is None:
                    grid[r][c] = {"text": "", "rowspan": 1, "colspan": 1}
                
                cell = grid[r][c]
                cell["colspan"] = max(1, min(cell.get("colspan", 1), num_cols - c))
                cell["rowspan"] = max(1, min(cell.get("rowspan", 1), num_rows - r))
        
        return grid