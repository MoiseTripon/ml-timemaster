"""
Grid building module for ML Timemaster.
Contains the GridBuilder class for organizing detected cells into a structured grid.
"""

import logging


class GridBuilder:
    """Handles organization of cells into grid structure with rowspan and colspan calculations."""
    
    def __init__(self, tolerance=10):
        """
        Initialize the GridBuilder.
        
        Args:
            tolerance (int): Tolerance for merging close positions
        """
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)

    def organize_cells_into_grid(self, cells):
        """
        Organize cells into a grid structure and calculate rowspan and colspan values.
        
        Args:
            cells (list): List of cell dictionaries with 'bounds' and 'text' keys.
        
        Returns:
            tuple: (grid, num_rows, num_cols)
        """
        if not cells:
            return [], 0, 0

        # Sort cells by position (top-to-bottom, left-to-right)
        sorted_cells = sorted(cells, key=lambda c: (c["bounds"]["y1"], c["bounds"]["x1"]))
        
        # Find unique positions with better merging
        row_positions_raw = [cell["bounds"]["y1"] for cell in sorted_cells]
        col_positions_raw = [cell["bounds"]["x1"] for cell in sorted_cells]
        
        row_positions = self._merge_close_positions(sorted(set(row_positions_raw)))
        col_positions = self._merge_close_positions(sorted(set(col_positions_raw)))
        
        # Calculate typical dimensions
        typical_col_width = self._calculate_typical_col_width(col_positions)
        typical_row_height = self._calculate_typical_row_height(row_positions)

        num_rows = len(row_positions)
        num_cols = len(col_positions)

        # Initialize a full grid structure - every position must have a cell
        # Use None for cells that will be covered by spans
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        placed_cells = set()

        # Process each cell and place it in the grid
        for cell_idx, cell in enumerate(sorted_cells):
            # Find the starting row and column for this cell
            row_idx = self._find_row_index(cell["bounds"]["y1"], row_positions)
            col_idx = self._find_column_index(cell["bounds"]["x1"], col_positions)
            
            if row_idx is None or col_idx is None:
                continue
            
            # Skip if this position is already occupied
            if grid[row_idx][col_idx] is not None:
                continue
            
            # Calculate spans based on cell dimensions
            rowspan = self._calculate_rowspan(cell, row_idx, row_positions, typical_row_height)
            colspan = self._calculate_colspan(cell, col_idx, col_positions, typical_col_width)
            
            # Ensure we don't exceed grid bounds
            colspan = min(colspan, num_cols - col_idx)
            rowspan = min(rowspan, num_rows - row_idx)
            
            # Create cell info with only text, rowspan, colspan
            cell_info = {
                "text": cell["text"],
                "rowspan": rowspan,
                "colspan": colspan
            }
            
            # Place the cell in the grid
            grid[row_idx][col_idx] = cell_info
            placed_cells.add(cell_idx)
            
            # Mark all spanned positions as None (they will be skipped in rendering)
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    if r == row_idx and c == col_idx:
                        continue  # Skip the main cell
                    
                    if r < num_rows and c < num_cols:
                        grid[r][c] = None  # Mark as spanned

        # Fill any remaining empty cells with empty text cells
        for r in range(num_rows):
            for c in range(num_cols):
                if grid[r][c] is None:
                    # Check if this is part of a span by looking at nearby cells
                    is_spanned = False
                    
                    # Check cells above and to the left for spans that might cover this position
                    for check_r in range(max(0, r - 10), r + 1):
                        for check_c in range(max(0, c - 10), c + 1):
                            if check_r >= num_rows or check_c >= num_cols:
                                continue
                            cell = grid[check_r][check_c]
                            if cell and isinstance(cell, dict):
                                # Check if this cell's span covers our position
                                if (check_r + cell.get("rowspan", 1) > r and 
                                    check_c + cell.get("colspan", 1) > c and
                                    check_r <= r and check_c <= c):
                                    is_spanned = True
                                    break
                        if is_spanned:
                            break
                    
                    if not is_spanned:
                        # This is a truly empty cell, not a spanned one
                        grid[r][c] = {
                            "text": "",
                            "rowspan": 1,
                            "colspan": 1
                        }
        
        return grid, num_rows, num_cols

    def _merge_close_positions(self, positions):
        """Merge positions that are very close to each other."""
        if not positions:
            return []
        
        merged = []
        current_group = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_group[-1] <= self.tolerance:
                current_group.append(pos)
            else:
                merged.append(sum(current_group) / len(current_group))
                current_group = [pos]
        
        if current_group:
            merged.append(sum(current_group) / len(current_group))
        
        return merged

    def _calculate_typical_col_width(self, col_positions):
        """Calculate typical column width using median."""
        if len(col_positions) > 1:
            col_widths = [
                col_positions[i + 1] - col_positions[i]
                for i in range(len(col_positions) - 1)
            ]
            col_widths.sort()
            return col_widths[len(col_widths) // 2]
        return 100

    def _calculate_typical_row_height(self, row_positions):
        """Calculate typical row height using median."""
        if len(row_positions) > 1:
            row_heights = [
                row_positions[i + 1] - row_positions[i]
                for i in range(len(row_positions) - 1)
            ]
            row_heights.sort()
            return row_heights[len(row_heights) // 2]
        return 30

    def _find_row_index(self, y_pos, row_positions):
        """Find the best row index for a given y position."""
        if not row_positions:
            return None
        
        for idx, row_y in enumerate(row_positions):
            if abs(y_pos - row_y) <= self.tolerance:
                return idx
        
        return None

    def _find_column_index(self, x_pos, col_positions):
        """Find the best column index for a given x position."""
        if not col_positions:
            return None
        
        for idx, col_x in enumerate(col_positions):
            if abs(x_pos - col_x) <= self.tolerance:
                return idx
        
        return None

    def _calculate_rowspan(self, cell, row_idx, row_positions, typical_height):
        """Calculate rowspan based on cell height relative to typical row height."""
        if typical_height <= 0:
            return 1
        
        cell_height = cell["bounds"]["y2"] - cell["bounds"]["y1"]
        
        # Calculate how many typical rows this cell spans
        calculated_rowspan = max(1, round(cell_height / typical_height))
        
        # Verify against actual row positions
        actual_rowspan = 1
        cell_bottom = cell["bounds"]["y2"]
        
        for i in range(row_idx + 1, len(row_positions)):
            if row_positions[i] < cell_bottom - self.tolerance:
                actual_rowspan += 1
            else:
                break
        
        return max(calculated_rowspan, actual_rowspan)

    def _calculate_colspan(self, cell, col_idx, col_positions, typical_width):
        """Calculate colspan based on cell width relative to typical column width."""
        if typical_width <= 0:
            return 1
        
        cell_width = cell["bounds"]["x2"] - cell["bounds"]["x1"]
        
        # Calculate how many typical columns this cell spans
        calculated_colspan = max(1, round(cell_width / typical_width))
        
        # Verify against actual column positions
        actual_colspan = 1
        cell_right = cell["bounds"]["x2"]
        
        for i in range(col_idx + 1, len(col_positions)):
            if col_positions[i] < cell_right - self.tolerance:
                actual_colspan += 1
            else:
                break
        
        return max(calculated_colspan, actual_colspan)