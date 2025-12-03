"""
Grid building module for ML Timemaster.
Contains the GridBuilder class for organizing detected cells into a structured grid.
"""

import logging


class GridBuilder:
    """Handles organization of cells into grid structure with rowspan and colspan calculations."""
    
    def __init__(self, tolerance=5):
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

        # Sort cells by position
        sorted_cells = sorted(cells, key=lambda c: (c["bounds"]["y1"], c["bounds"]["x1"]))
        
        # Find unique positions
        row_positions_raw = [cell["bounds"]["y1"] for cell in sorted_cells]
        col_positions_raw = [cell["bounds"]["x1"] for cell in sorted_cells]
        
        row_positions = self._merge_close_positions(list(set(row_positions_raw)))
        col_positions = self._merge_close_positions(list(set(col_positions_raw)))
        
        self.logger.info(f"Grid positions: {len(row_positions)} rows x {len(col_positions)} cols")

        # Calculate typical column width
        typical_col_width = self._calculate_typical_col_width(col_positions)

        # Create grid
        temp_grid = []
        occupied_positions = set()

        # Process each row
        for row_idx, row_y in enumerate(row_positions):
            current_row = [None] * len(col_positions)
            has_cells = False

            for cell in sorted_cells:
                if abs(cell["bounds"]["y1"] - row_y) <= self.tolerance:
                    has_cells = True
                    col_idx = self._find_column_index(cell["bounds"]["x1"], col_positions)
                    
                    if col_idx >= len(col_positions):
                        continue

                    rowspan = self._calculate_rowspan(cell, row_idx, row_positions)
                    colspan = self._calculate_colspan(cell, col_idx, col_positions, typical_col_width)

                    if self._can_place_cell(row_idx, col_idx, rowspan, colspan, 
                                           occupied_positions, row_positions, col_positions):
                        cell_info = {
                            "text": cell["text"],
                            "rowspan": rowspan,
                            "colspan": colspan,
                        }
                        current_row[col_idx] = cell_info
                        self._mark_occupied(row_idx, col_idx, rowspan, colspan, 
                                          occupied_positions, row_positions, col_positions)

            if has_cells:
                has_colspan = any(
                    cell for cell in current_row
                    if cell is not None and cell.get("colspan", 1) > 1
                )
                if not has_colspan:
                    for i in range(len(current_row)):
                        if current_row[i] is None and (row_idx, i) not in occupied_positions:
                            current_row[i] = {"text": "", "rowspan": 1, "colspan": 1}

            temp_grid.append(current_row)

        # Filter out None values and empty rows
        final_grid = []
        for row in temp_grid:
            filtered_row = [cell for cell in row if cell is not None]
            if filtered_row:
                final_grid.append(filtered_row)

        num_rows = len(final_grid)
        num_cols = max((len(row) for row in final_grid), default=0)

        self.logger.info(f"Final grid: {num_rows} rows x {num_cols} cols")
        
        return final_grid, num_rows, num_cols

    def _merge_close_positions(self, positions):
        """Merge positions that are very close to each other."""
        if not positions:
            return []
        positions = sorted(positions)
        merged = [positions[0]]
        for pos in positions[1:]:
            if pos - merged[-1] > self.tolerance:
                merged.append(pos)
        return merged

    def _calculate_typical_col_width(self, col_positions):
        """Calculate typical column width."""
        if len(col_positions) > 1:
            col_widths = [
                col_positions[i + 1] - col_positions[i]
                for i in range(len(col_positions) - 1)
            ]
            return sum(col_widths) / len(col_widths)
        return 100

    def _find_column_index(self, x_pos, col_positions):
        """Find the column index for a given x position."""
        col_idx = 0
        while (col_idx < len(col_positions) and 
               col_positions[col_idx] + self.tolerance < x_pos):
            col_idx += 1
        return col_idx

    def _calculate_rowspan(self, cell, row_idx, row_positions):
        """Calculate rowspan for a cell."""
        rowspan = 1
        cell_bottom = cell["bounds"]["y2"]
        for next_row_y in row_positions[row_idx + 1:]:
            if next_row_y - self.tolerance <= cell_bottom:
                rowspan += 1
            else:
                break
        return rowspan

    def _calculate_colspan(self, cell, col_idx, col_positions, typical_col_width):
        """Calculate colspan for a cell."""
        cell_width = cell["bounds"]["x2"] - cell["bounds"]["x1"]
        colspan = 1
        remaining_width = cell_width
        next_col = col_idx

        while (next_col + 1 < len(col_positions) and 
               remaining_width > typical_col_width * 0.3):
            if next_col + 1 < len(col_positions):
                col_width = col_positions[next_col + 1] - col_positions[next_col]
            else:
                col_width = typical_col_width

            if remaining_width > col_width * 0.3:
                colspan += 1
                remaining_width -= col_width

            next_col += 1

        return colspan

    def _can_place_cell(self, row_idx, col_idx, rowspan, colspan, 
                       occupied_positions, row_positions, col_positions):
        """Check if a cell can be placed at the given position."""
        for r in range(row_idx, min(row_idx + rowspan, len(row_positions))):
            for c in range(col_idx, min(col_idx + colspan, len(col_positions))):
                if (r, c) in occupied_positions:
                    return False
        return True

    def _mark_occupied(self, row_idx, col_idx, rowspan, colspan, 
                      occupied_positions, row_positions, col_positions):
        """Mark positions as occupied by a cell."""
        for r in range(row_idx, min(row_idx + rowspan, len(row_positions))):
            for c in range(col_idx, min(col_idx + colspan, len(col_positions))):
                occupied_positions.add((r, c))