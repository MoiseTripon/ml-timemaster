"""
Grid building module for ML Timemaster.
Contains functions for organizing detected cells into a structured grid format with rowspan and colspan calculations.
"""

import logging

logger = logging.getLogger(__name__)


def organize_cells_into_grid(cells):
    """
    Organizes cells into a grid structure and calculates rowspan and colspan values.
    
    Args:
        cells (list): List of cell dictionaries, each containing 'bounds' and 'text' keys.
    
    Returns:
        tuple: (grid, num_rows, num_cols)
            - grid (list): 2D list of cell information with rowspan/colspan
            - num_rows (int): Number of rows in the grid
            - num_cols (int): Number of columns in the grid
    """
    logger.info("="*50)
    logger.info("Starting grid organization")
    logger.info(f"Input: {len(cells)} cells")
    
    if not cells:
        logger.warning("No cells to organize")
        return [], 0, 0

    # Log all input cells
    for i, cell in enumerate(cells):
        logger.debug(
            f"Input cell {i}: bounds={cell['bounds']}, text='{cell.get('text', '')[:30]}...'"
        )

    # Sort cells by position
    sorted_cells = sorted(cells, key=lambda c: (c["bounds"]["y1"], c["bounds"]["x1"]))
    
    # Find unique positions
    tolerance = 5
    
    row_positions_raw = [cell["bounds"]["y1"] for cell in sorted_cells]
    col_positions_raw = [cell["bounds"]["x1"] for cell in sorted_cells]
    
    logger.debug(f"Raw row positions: {sorted(set(row_positions_raw))}")
    logger.debug(f"Raw col positions: {sorted(set(col_positions_raw))}")

    row_positions = merge_close_positions(
        list(set(row_positions_raw)), tolerance
    )
    col_positions = merge_close_positions(
        list(set(col_positions_raw)), tolerance
    )
    
    logger.info(f"Merged row positions ({len(row_positions)}): {row_positions}")
    logger.info(f"Merged col positions ({len(col_positions)}): {col_positions}")


    # Calculate typical column width
    if len(col_positions) > 1:
        col_widths = [
            col_positions[i + 1] - col_positions[i]
            for i in range(len(col_positions) - 1)
        ]
        typical_col_width = sum(col_widths) / len(col_widths)
    else:
        typical_col_width = 100  # default value if only one column

    # Create a grid to store cell information
    temp_grid = []  # Use a temporary grid to store rows before filtering
    occupied_positions = set()  # Track which positions are already taken

    # Process each row
    for row_idx, row_y in enumerate(row_positions):
        current_row = [None] * len(col_positions)  # Initialize row with None values
        has_cells = False  # Track if this row has any cells

        # Process cells that start in this row
        for cell in sorted_cells:
            if abs(cell["bounds"]["y1"] - row_y) <= tolerance:
                has_cells = True
                # Find the starting column index for this cell
                col_idx = 0
                while (
                    col_idx < len(col_positions)
                    and col_positions[col_idx] + tolerance < cell["bounds"]["x1"]
                ):
                    col_idx += 1

                if col_idx >= len(col_positions):  # Safety check
                    continue

                # Calculate rowspan
                rowspan = 1
                cell_bottom = cell["bounds"]["y2"]
                for next_row_y in row_positions[row_idx + 1 :]:
                    if next_row_y - tolerance <= cell_bottom:
                        rowspan += 1
                    else:
                        break

                # Calculate colspan based on width coverage
                cell_right = cell["bounds"]["x2"]
                cell_width = cell_right - cell["bounds"]["x1"]
                colspan = 1

                # Calculate colspan by checking how many columns this cell spans
                remaining_width = cell_width
                next_col = col_idx
                while (
                    next_col + 1 < len(col_positions)
                    and remaining_width > typical_col_width * 0.3
                ):
                    # Get width of current column
                    if next_col + 1 < len(col_positions):
                        col_width = (
                            col_positions[next_col + 1] - col_positions[next_col]
                        )
                    else:
                        col_width = typical_col_width

                    # If we still have enough remaining width, increase colspan
                    if remaining_width > col_width * 0.3:
                        colspan += 1
                        remaining_width -= col_width

                    next_col += 1

                # Check if any of the positions this cell would occupy are already taken
                can_place = True
                for r in range(row_idx, min(row_idx + rowspan, len(row_positions))):
                    for c in range(col_idx, min(col_idx + colspan, len(col_positions))):
                        if (r, c) in occupied_positions:
                            can_place = False
                            break
                    if not can_place:
                        break

                if can_place:
                    # Create cell info
                    cell_info = {
                        "text": cell["text"],
                        "rowspan": rowspan,
                        "colspan": colspan,
                    }

                    # Place the cell and mark its positions as occupied
                    current_row[col_idx] = cell_info
                    for r in range(row_idx, min(row_idx + rowspan, len(row_positions))):
                        for c in range(
                            col_idx, min(col_idx + colspan, len(col_positions))
                        ):
                            occupied_positions.add((r, c))
                            if r == row_idx and c > col_idx:
                                current_row[c] = None  # Mark covered columns as None

        # Only add non-None values at the end of the row if there are no cells with colspan
        if has_cells:
            # Check if there's a cell with colspan in this row
            has_colspan = any(
                cell
                for cell in current_row
                if cell is not None and cell.get("colspan", 1) > 1
            )
            if not has_colspan:
                # Fill in empty cells only if there's no colspan
                for i in range(len(current_row)):
                    if (
                        current_row[i] is None
                        and (row_idx, i) not in occupied_positions
                    ):
                        current_row[i] = {"text": "", "rowspan": 1, "colspan": 1}

        temp_grid.append(current_row)

    # Filter out None values and empty rows
    final_grid = []
    for row in temp_grid:
        # Remove None values from the row
        filtered_row = [cell for cell in row if cell is not None]
        # Only add the row if it contains at least one cell
        if filtered_row:
            final_grid.append(filtered_row)

    # Calculate grid dimensions
    num_rows = len(final_grid)
    num_cols = max((len(row) for row in final_grid), default=0)

    logger.info(f"Final grid: {num_rows} rows x {num_cols} cols")
    
    # Log grid structure
    for row_idx, row in enumerate(final_grid):
        logger.debug(f"Row {row_idx}: {len(row)} cells")
        for col_idx, cell in enumerate(row):
            if cell:
                logger.debug(
                    f"  [{row_idx},{col_idx}]: text='{cell.get('text', '')[:20]}...', "
                    f"rowspan={cell.get('rowspan', 1)}, colspan={cell.get('colspan', 1)}"
                )

    return final_grid, num_rows, num_cols

def merge_close_positions(positions, tolerance):
    """Merge positions that are very close to each other."""
    if not positions:
        return []
    positions = sorted(positions)
    merged = [positions[0]]
    for pos in positions[1:]:
        if pos - merged[-1] > tolerance:
            merged.append(pos)
        else:
            logger.debug(f"Merging position {pos} with {merged[-1]} (diff={pos - merged[-1]})")
    return merged