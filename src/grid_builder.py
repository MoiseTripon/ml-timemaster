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
        self.logger.info("="*50)
        self.logger.info("Starting grid organization")
        self.logger.info(f"Input: {len(cells)} cells")
        
        if not cells:
            self.logger.warning("No cells to organize")
            return [], 0, 0

        # Sort cells by position (top-to-bottom, left-to-right)
        sorted_cells = sorted(cells, key=lambda c: (c["bounds"]["y1"], c["bounds"]["x1"]))
        
        # Log cell positions for debugging
        for i, cell in enumerate(sorted_cells):
            self.logger.debug(f"Cell {i}: y1={cell['bounds']['y1']}, x1={cell['bounds']['x1']}, text='{cell['text'][:30] if cell['text'] else ''}'")
        
        # Find unique positions with better merging
        row_positions_raw = [cell["bounds"]["y1"] for cell in sorted_cells]
        col_positions_raw = [cell["bounds"]["x1"] for cell in sorted_cells]
        
        row_positions = self._merge_close_positions(sorted(set(row_positions_raw)))
        col_positions = self._merge_close_positions(sorted(set(col_positions_raw)))
        
        self.logger.info(f"Grid positions: {len(row_positions)} rows x {len(col_positions)} cols")
        self.logger.debug(f"Row positions: {row_positions}")
        self.logger.debug(f"Column positions: {col_positions}")

        # Calculate typical dimensions
        typical_col_width = self._calculate_typical_col_width(col_positions)
        typical_row_height = self._calculate_typical_row_height(row_positions)

        # Create grid and track which cells have been placed
        grid = []
        occupied_positions = set()
        placed_cells = set()

        # Process each row
        for row_idx, row_y in enumerate(row_positions):
            current_row = []
            
            # Find all cells that belong to this row
            row_cells = []
            for cell_idx, cell in enumerate(sorted_cells):
                if cell_idx in placed_cells:
                    continue
                    
                # Check if cell starts in this row (with tolerance)
                if abs(cell["bounds"]["y1"] - row_y) <= self.tolerance:
                    row_cells.append((cell_idx, cell))
            
            # Sort row cells by x position
            row_cells.sort(key=lambda x: x[1]["bounds"]["x1"])
            
            # Process cells in this row
            for cell_idx, cell in row_cells:
                col_idx = self._find_column_index(cell["bounds"]["x1"], col_positions)
                
                # Calculate spans based on cell dimensions
                rowspan = self._calculate_rowspan(cell, row_idx, row_positions, typical_row_height)
                colspan = self._calculate_colspan(cell, col_idx, col_positions, typical_col_width)
                
                # Ensure we don't exceed grid bounds
                colspan = min(colspan, len(col_positions) - col_idx)
                rowspan = min(rowspan, len(row_positions) - row_idx)
                
                self.logger.debug(f"Cell at row {row_idx}, col {col_idx}: rowspan={rowspan}, colspan={colspan}, text='{cell['text'][:30] if cell['text'] else ''}'")
                
                cell_info = {
                    "text": cell["text"],
                    "rowspan": rowspan,
                    "colspan": colspan,
                }
                current_row.append(cell_info)
                placed_cells.add(cell_idx)
                
                # Mark positions as occupied
                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        occupied_positions.add((r, c))
            
            if current_row:
                grid.append(current_row)

        # Log unplaced cells
        unplaced = len(sorted_cells) - len(placed_cells)
        if unplaced > 0:
            self.logger.warning(f"Warning: {unplaced} cells were not placed in the grid")
            for i, cell in enumerate(sorted_cells):
                if i not in placed_cells:
                    self.logger.debug(f"Unplaced cell: {cell['text'][:30] if cell['text'] else ''}, bounds: {cell['bounds']}")

        num_rows = len(grid)
        num_cols = len(col_positions)

        self.logger.info(f"Final grid: {num_rows} rows with {len(placed_cells)} cells placed")
        
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

    def _find_column_index(self, x_pos, col_positions):
        """Find the best column index for a given x position."""
        if not col_positions:
            return 0
        
        min_dist = float('inf')
        best_idx = 0
        
        for idx, col_x in enumerate(col_positions):
            dist = abs(x_pos - col_x)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx
        
        return best_idx

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