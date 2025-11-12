"""
Table analysis module for ML Timemaster.
Contains the TableAnalyzer class for detecting and organizing table cells.
"""

import cv2


class TableAnalyzer:
    """
    Handles table detection, cell detection, and grid organization.
    """
    
    def __init__(self, overlap_threshold=0.7, min_cell_size=5):
        """
        Initialize the TableAnalyzer.
        
        Args:
            overlap_threshold (float): Threshold for considering cells as duplicates (0-1).
            min_cell_size (int): Minimum width/height in pixels for valid cells.
        """
        self.overlap_threshold = overlap_threshold
        self.min_cell_size = min_cell_size
    
    def detect_table_borders(self, img):
        """
        Detects the main table borders in the image.
        
        Args:
            img (numpy.ndarray): Input image.
            
        Returns:
            dict: Table boundary coordinates {'x1', 'y1', 'x2', 'y2'}.
            
        Raises:
            ValueError: If no table is detected in the image.
        """
        # Convert to grayscale if not already
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            3
        )
        
        # Detect edges
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No table detected in the image")
        
        # Find the largest contour (assumed to be the table)
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        table_bounds = {'x1': x, 'y1': y, 'x2': x + w, 'y2': y + h}
        
        print(f"Detected table boundaries: {table_bounds}")
        return table_bounds
    
    def detect_cells(self, img, table_bounds):
        """
        Detects individual cells within the table.
        
        Args:
            img (numpy.ndarray): Input image.
            table_bounds (dict): Table boundary coordinates.
            
        Returns:
            list: List of cell dictionaries with coordinates, bounds, and text.
        """
        # Extract table region
        x = table_bounds['x1']
        y = table_bounds['y1']
        w = table_bounds['x2'] - table_bounds['x1']
        h = table_bounds['y2'] - table_bounds['y1']
        table_region = img[y:y+h, x:x+w]
        
        # Convert to grayscale if needed
        if len(table_region.shape) == 3:
            gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = table_region.copy()
        
        # Apply multiple preprocessing techniques to catch different types of cells
        preprocessed_images = []
        
        # Method 1: Standard Otsu's thresholding with different parameters
        _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_images.append(binary1)
        
        # Method 2: Multiple adaptive thresholding attempts with different parameters
        binary2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
        )
        preprocessed_images.append(binary2)
        
        binary3 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4
        )
        preprocessed_images.append(binary3)
        
        # Method 3: Edge detection with different parameters
        edges1 = cv2.Canny(gray, 50, 150)
        edges2 = cv2.Canny(gray, 30, 200)
        preprocessed_images.extend([edges1, edges2])
        
        # Method 4: Combination of methods
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary4 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        preprocessed_images.append(binary4)
        
        # Find contours in each preprocessed image
        all_contours = []
        for processed in preprocessed_images:
            contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        # Filter and process contours
        cells = []
        min_area = (w * h) * 0.0001  # Minimum area in pixels
        max_area = (w * h) * 0.99  # Maximum area (99% of table area)
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                x1, y1, width, height = cv2.boundingRect(contour)
                # Convert coordinates back to original image space
                cell = {
                    # Flat coordinates (for backward compatibility)
                    'x1': x1 + table_bounds['x1'],
                    'y1': y1 + table_bounds['y1'],
                    'x2': x1 + width + table_bounds['x1'],
                    'y2': y1 + height + table_bounds['y1'],
                    'width': width,
                    'height': height,
                    # Bounds dict (for organize_cells_into_grid)
                    'bounds': {
                        'x1': x1 + table_bounds['x1'],
                        'y1': y1 + table_bounds['y1'],
                        'x2': x1 + width + table_bounds['x1'],
                        'y2': y1 + height + table_bounds['y1']
                    },
                    # Empty text field
                    'text': ''
                }
                cells.append(cell)
        
        # Remove duplicates
        cells = self.remove_duplicate_cells(cells)
        
        return cells
    
    def remove_duplicate_cells(self, cells):
        """
        Remove duplicate cells that overlap significantly.
        
        Args:
            cells (list): List of cell dictionaries.
            
        Returns:
            list: List of unique cells.
        """
        if not cells:
            return []
        
        def calculate_overlap(cell1, cell2):
            """Calculate the overlap area between two cells."""
            # Calculate cell areas first
            cell1_area = (cell1['x2'] - cell1['x1']) * (cell1['y2'] - cell1['y1'])
            cell2_area = (cell2['x2'] - cell2['x1']) * (cell2['y2'] - cell2['y1'])
            
            # If either cell has zero or negative area, they can't overlap
            if cell1_area <= 0 or cell2_area <= 0:
                return 0.0
            
            # Calculate overlap coordinates
            x_left = max(cell1['x1'], cell2['x1'])
            y_top = max(cell1['y1'], cell2['y1'])
            x_right = min(cell1['x2'], cell2['x2'])
            y_bottom = min(cell1['y2'], cell2['y2'])
            
            # If there's no overlap, return 0
            if x_right <= x_left or y_bottom <= y_top:
                return 0.0
            
            # Calculate overlap area
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Return overlap ratio based on the smaller cell
            return float(overlap_area) / min(cell1_area, cell2_area)
        
        def cells_are_similar(cell1, cell2):
            """Check if two cells are similar in position and size."""
            # First check if either cell has invalid dimensions
            if (cell1['x2'] <= cell1['x1'] or cell1['y2'] <= cell1['y1'] or
                cell2['x2'] <= cell2['x1'] or cell2['y2'] <= cell2['y1']):
                return False
                
            overlap = calculate_overlap(cell1, cell2)
            return overlap > self.overlap_threshold
        
        # Filter out cells with invalid dimensions first
        valid_cells = [
            cell for cell in cells 
            if (cell['x2'] > cell['x1'] and 
                cell['y2'] > cell['y1'] and 
                (cell['x2'] - cell['x1']) >= self.min_cell_size and
                (cell['y2'] - cell['y1']) >= self.min_cell_size)
        ]
        
        # Sort cells by area (largest first)
        valid_cells.sort(
            key=lambda c: (c['x2'] - c['x1']) * (c['y2'] - c['y1']), 
            reverse=True
        )
        
        unique_cells = []
        for cell in valid_cells:
            is_duplicate = False
            for unique_cell in unique_cells:
                if cells_are_similar(cell, unique_cell):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_cells.append(cell)
        
        print(f"Removed {len(valid_cells) - len(unique_cells)} duplicate cells")
        return unique_cells
    
    def organize_cells_into_grid(self, cells):
        """
        Organizes cells into a grid structure and calculates rowspan and colspan values.
        
        Args:
            cells (list): List of cell dictionaries with 'bounds' and 'text' keys.
            
        Returns:
            tuple: (grid, num_rows, num_cols)
                - grid (list): 2D list of cell information
                - num_rows (int): Number of rows in the grid
                - num_cols (int): Number of columns in the grid
        """
        if not cells:
            return [], 0, 0
        
        # Normalize input cells to ensure they have 'bounds' and 'text' keys
        normalized_cells = []
        for cell in cells:
            normalized_cell = {}
            
            # Handle bounds
            if 'bounds' in cell:
                normalized_cell['bounds'] = cell['bounds']
            elif all(key in cell for key in ['x1', 'y1', 'x2', 'y2']):
                # Create bounds from flat coordinates
                normalized_cell['bounds'] = {
                    'x1': cell['x1'],
                    'y1': cell['y1'],
                    'x2': cell['x2'],
                    'y2': cell['y2']
                }
            else:
                # Skip cells without proper coordinates
                continue
            
            # Handle text
            normalized_cell['text'] = cell.get('text', '')
            
            normalized_cells.append(normalized_cell)
        
        cells = normalized_cells
        
        # Sort cells by position (top to bottom, left to right)
        sorted_cells = sorted(cells, key=lambda c: (c['bounds']['y1'], c['bounds']['x1']))
        
        # Find unique row and column positions with tolerance
        tolerance = 5  # 5 pixels tolerance for row/column alignment
        
        def merge_close_positions(positions, tolerance):
            if not positions:
                return []
            positions = sorted(positions)
            merged = [positions[0]]
            for pos in positions[1:]:
                if pos - merged[-1] > tolerance:
                    merged.append(pos)
            return merged
        
        # Get unique positions and merge those that are very close
        row_positions = merge_close_positions(
            list(set(cell['bounds']['y1'] for cell in sorted_cells)), 
            tolerance
        )
        col_positions = merge_close_positions(
            list(set(cell['bounds']['x1'] for cell in sorted_cells)), 
            tolerance
        )
        
        # Calculate typical column width
        if len(col_positions) > 1:
            col_widths = [col_positions[i+1] - col_positions[i] for i in range(len(col_positions)-1)]
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
                if abs(cell['bounds']['y1'] - row_y) <= tolerance:
                    has_cells = True
                    # Find the starting column index for this cell
                    col_idx = 0
                    while col_idx < len(col_positions) and col_positions[col_idx] + tolerance < cell['bounds']['x1']:
                        col_idx += 1
                    
                    if col_idx >= len(col_positions):  # Safety check
                        continue
                    
                    # Calculate rowspan
                    rowspan = 1
                    cell_bottom = cell['bounds']['y2']
                    for next_row_y in row_positions[row_idx + 1:]:
                        if next_row_y - tolerance <= cell_bottom:
                            rowspan += 1
                        else:
                            break
                    
                    # Calculate colspan based on width coverage
                    cell_right = cell['bounds']['x2']
                    cell_width = cell_right - cell['bounds']['x1']
                    colspan = 1
                    
                    # Calculate colspan by checking how many columns this cell spans
                    remaining_width = cell_width
                    next_col = col_idx
                    while next_col + 1 < len(col_positions) and remaining_width > typical_col_width * 0.3:
                        # Get width of current column
                        if next_col + 1 < len(col_positions):
                            col_width = col_positions[next_col + 1] - col_positions[next_col]
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
                            'text': cell['text'],
                            'rowspan': rowspan,
                            'colspan': colspan
                        }
                        
                        # Place the cell and mark its positions as occupied
                        current_row[col_idx] = cell_info
                        for r in range(row_idx, min(row_idx + rowspan, len(row_positions))):
                            for c in range(col_idx, min(col_idx + colspan, len(col_positions))):
                                occupied_positions.add((r, c))
                                if r == row_idx and c > col_idx:
                                    current_row[c] = None  # Mark covered columns as None
            
            # Only add non-None values at the end of the row if there are no cells with colspan
            if has_cells:
                # Check if there's a cell with colspan in this row
                has_colspan = any(cell for cell in current_row if cell is not None and cell.get('colspan', 1) > 1)
                if not has_colspan:
                    # Fill in empty cells only if there's no colspan
                    for i in range(len(current_row)):
                        if current_row[i] is None and (row_idx, i) not in occupied_positions:
                            current_row[i] = {'text': '', 'rowspan': 1, 'colspan': 1}
            
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
        
        return final_grid, num_rows, num_cols