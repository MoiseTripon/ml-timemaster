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
        
        Cells come from the detector with grid information already computed.
        We need to:
        1. Transform the detector output to the expected bounds format
        2. Place cells in grid using their pre-computed positions and spans
        3. Handle any remaining gaps
        """
        if not cells:
            return [], 0, 0

        # Transform cells to have bounds dictionary format
        formatted_cells = []
        for cell in cells:
            # Check if cell already has bounds format
            if "bounds" in cell:
                formatted_cells.append(cell)
            else:
                # Transform from detector format (x1, y1, x2, y2) to bounds format
                formatted_cell = {
                    "bounds": {
                        "x1": cell.get("x1", 0),
                        "y1": cell.get("y1", 0),
                        "x2": cell.get("x2", 0),
                        "y2": cell.get("y2", 0)
                    },
                    "text": cell.get("text", ""),
                    "grid_row": cell.get("grid_row", 0),
                    "grid_col": cell.get("grid_col", 0),
                    "rowspan": cell.get("rowspan", 1),
                    "colspan": cell.get("colspan", 1)
                }
                formatted_cells.append(formatted_cell)

        # Sort cells by position
        sorted_cells = sorted(formatted_cells, key=lambda c: (
            c.get("grid_row", c["bounds"]["y1"]), 
            c.get("grid_col", c["bounds"]["x1"])
        ))
        
        # Determine grid dimensions from pre-computed grid positions
        if all("grid_row" in c and "grid_col" in c for c in sorted_cells):
            # Use pre-computed grid positions
            num_rows = max(c.get("grid_row", 0) + c.get("rowspan", 1) for c in sorted_cells)
            num_cols = max(c.get("grid_col", 0) + c.get("colspan", 1) for c in sorted_cells)
        else:
            # Fall back to computing from boundaries
            row_positions = self._get_unique_positions([c["bounds"]["y1"] for c in sorted_cells])
            row_positions.append(max(c["bounds"]["y2"] for c in sorted_cells))
            row_positions = sorted(set(row_positions))
            
            col_positions = self._get_unique_positions([c["bounds"]["x1"] for c in sorted_cells])
            col_positions.append(max(c["bounds"]["x2"] for c in sorted_cells))
            col_positions = sorted(set(col_positions))
            
            num_rows = len(row_positions) - 1
            num_cols = len(col_positions) - 1
        
        self.logger.info(f"Grid dimensions: {num_rows} rows x {num_cols} columns")
        
        if num_rows < 1 or num_cols < 1:
            return [], 0, 0
        
        # Initialize grid
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        span_coverage: Set[Tuple[int, int]] = set()
        
        # Place cells in grid - process in order to handle overlaps correctly
        for cell in sorted_cells:
            # Get position from pre-computed values or calculate
            if "grid_row" in cell and "grid_col" in cell:
                row_idx = cell["grid_row"]
                col_idx = cell["grid_col"]
                rowspan = cell.get("rowspan", 1)
                colspan = cell.get("colspan", 1)
            else:
                # Fall back to calculating from bounds
                if 'row_positions' not in locals():
                    row_positions = self._get_unique_positions([c["bounds"]["y1"] for c in sorted_cells])
                    row_positions.append(max(c["bounds"]["y2"] for c in sorted_cells))
                    row_positions = sorted(set(row_positions))
                    
                    col_positions = self._get_unique_positions([c["bounds"]["x1"] for c in sorted_cells])
                    col_positions.append(max(c["bounds"]["x2"] for c in sorted_cells))
                    col_positions = sorted(set(col_positions))
                
                row_idx = self._find_position_index(cell["bounds"]["y1"], row_positions)
                col_idx = self._find_position_index(cell["bounds"]["x1"], col_positions)
                
                if row_idx is None or col_idx is None:
                    self.logger.warning(f"Could not place cell at ({cell['bounds']['x1']}, {cell['bounds']['y1']})")
                    continue
                
                # Calculate spans
                rowspan = self._calculate_span(
                    cell["bounds"]["y1"], cell["bounds"]["y2"],
                    row_positions, None
                )
                colspan = self._calculate_span(
                    cell["bounds"]["x1"], cell["bounds"]["x2"],
                    col_positions, None
                )
            
            if row_idx >= num_rows or col_idx >= num_cols:
                continue
            
            if (row_idx, col_idx) in span_coverage:
                continue
            
            # Ensure spans are valid and within bounds
            rowspan = max(1, min(rowspan, num_rows - row_idx))
            colspan = max(1, min(colspan, num_cols - col_idx))
            
            # Check for conflicts and adjust spans
            rowspan, colspan = self._adjust_for_conflicts(
                row_idx, col_idx, rowspan, colspan, span_coverage, num_rows, num_cols
            )
            
            # Create cell with ONLY text, rowspan, colspan
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
        
        # Fill gaps - cells not covered by any span
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] is None and (r, c) not in span_coverage:
                    grid[r][c] = {"text": "", "rowspan": 1, "colspan": 1}
        
        # Set None for positions covered by spans (except origin cells)
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] is not None:
                    rowspan = grid[r][c].get("rowspan", 1)
                    colspan = grid[r][c].get("colspan", 1)
                    
                    # Mark spanned positions as None
                    for sr in range(r, min(r + rowspan, num_rows)):
                        for sc in range(c, min(c + colspan, num_cols)):
                            if sr != r or sc != c:
                                grid[sr][sc] = None
        
        # Final validation
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
                        positions: List[float], typical_size: Optional[float]) -> int:
        """
        Calculate span based on how many grid positions the cell covers.
        """
        span = 0
        for i, pos in enumerate(positions[:-1]):
            next_pos = positions[i + 1]
            
            # Check if this grid cell is within the cell's boundaries
            # Using a slightly more lenient check for long spanning cells
            cell_start = start - self.tolerance
            cell_end = end + self.tolerance
            
            if pos >= cell_start and next_pos <= cell_end:
                span += 1
            elif pos >= end:
                break
        
        return max(1, span)

    def _adjust_for_conflicts(self, row_idx: int, col_idx: int,
                               rowspan: int, colspan: int,
                               covered: Set[Tuple[int, int]],
                               num_rows: int, num_cols: int) -> Tuple[int, int]:
        """Adjust spans to avoid conflicts with already placed cells."""
        # First check if original spans work
        has_conflict = False
        for r in range(row_idx, min(row_idx + rowspan, num_rows)):
            for c in range(col_idx, min(col_idx + colspan, num_cols)):
                if (r, c) in covered:
                    has_conflict = True
                    break
            if has_conflict:
                break
        
        if not has_conflict:
            return rowspan, colspan
        
        # Try to find the best non-conflicting spans
        # Strategy: try reducing colspan first, then rowspan
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
        """Validate grid structure and ensure consistency."""
        if not grid:
            return grid
        
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] is not None:
                    cell = grid[r][c]
                    # Ensure only text, rowspan, colspan keys exist
                    validated_cell = {
                        "text": cell.get("text", ""),
                        "rowspan": max(1, min(cell.get("rowspan", 1), num_rows - r)),
                        "colspan": max(1, min(cell.get("colspan", 1), num_cols - c))
                    }
                    grid[r][c] = validated_cell
        
        return grid