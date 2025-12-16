"""
Table detection module for ML Timemaster.
Contains the TableDetector class for detecting table borders and cells.
"""

import logging
import cv2
import numpy as np
import os
from typing import List, Dict, Tuple, Set


class TableDetector:
    """Handles detection of table borders and individual cells using grid-based approach."""
    
    def __init__(self, debug_mode=False, debug_output_dir="debug_output"):
        """
        Initialize the TableDetector.
        
        Args:
            debug_mode (bool): Enable debug visualizations
            debug_output_dir (str): Directory for debug output
        """
        self.debug_mode = debug_mode
        self.debug_output_dir = debug_output_dir
        self.logger = logging.getLogger(__name__)
        
        if self.debug_mode and not os.path.exists(self.debug_output_dir):
            os.makedirs(self.debug_output_dir)

    def save_debug_image(self, img, name, step=""):
        """Save an image for debugging purposes."""
        if not self.debug_mode:
            return
        
        filename = f"{self.debug_output_dir}/{step}_{name}.png"
        cv2.imwrite(filename, img)
        self.logger.debug(f"Saved debug image: {filename}")

    def detect_table_borders(self, img):
        """
        Step 1: Detect the main table region in the image.
        """
        self.logger.info("="*50)
        self.logger.info("STEP 1: Detecting table region")
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        self.save_debug_image(gray, "grayscale", "01")
        
        img_h, img_w = gray.shape[:2]
        morph_kernel_size = max(3, min(img_w, img_h) // 200)
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        morphed = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
        self.save_debug_image(morphed, "morphed_gradient", "02")
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        edges1 = cv2.Canny(blurred, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(morphed, 30, 100, apertureSize=3)
        edges = cv2.bitwise_or(edges1, edges2)
        
        dilation_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, dilation_kernel, iterations=1)
        edges = cv2.erode(edges, dilation_kernel, iterations=1)
        
        self.save_debug_image(edges, "edges_combined", "03")
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"Found {len(contours)} contours")
        
        if not contours:
            self.logger.warning("No contours found with morphology, trying without")
            edges_fallback = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.error("No contours found in the image")
                raise ValueError("No table detected in the image")
        
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        min_area_ratio = 0.1
        if (w * h) < (img_w * img_h * min_area_ratio):
            self.logger.warning("Detected table seems too small, looking for larger contour")
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in sorted_contours:
                x_tmp, y_tmp, w_tmp, h_tmp = cv2.boundingRect(contour)
                if (w_tmp * h_tmp) >= (img_w * img_h * min_area_ratio):
                    x, y, w, h = x_tmp, y_tmp, w_tmp, h_tmp
                    break
        
        table_bounds = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
        self.logger.info(f"Detected table boundaries: {table_bounds}")
        
        if self.debug_mode:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            self.save_debug_image(debug_img, "table_bounds", "04")
        
        return table_bounds

    def detect_cells(self, img, table_bounds):
        """
        Main method to detect cells using the 5-step approach.
        Returns cells with bounds that will be transformed by grid_builder.
        """
        self.logger.info("="*50)
        self.logger.info("Starting cell detection using grid-based approach")
        
        x, y = table_bounds["x1"], table_bounds["y1"]
        w = table_bounds["x2"] - table_bounds["x1"]
        h = table_bounds["y2"] - table_bounds["y1"]
        
        table_region = img[y:y+h, x:x+w]
        self.save_debug_image(table_region, "table_region", "05")
        
        if len(table_region.shape) == 3:
            binary_img = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        else:
            binary_img = table_region.copy()
        
        # Step 2: Detect horizontal and vertical ruling lines
        h_lines, v_lines = self._detect_ruling_lines(binary_img, w, h)
        
        # Step 3: Build fine-grained grid including ALL lines (even partial ones)
        h_grid_lines, v_grid_lines = self._build_complete_grid(h_lines, v_lines, w, h)
        
        # Step 3.5: Filter out tiny peripheral grid lines
        h_grid_lines, v_grid_lines = self._filter_peripheral_gridlines(h_grid_lines, v_grid_lines, w, h)
        
        # Step 4: Detect which grid boundaries have actual lines (barriers)
        h_barriers, v_barriers = self._identify_line_barriers(
            h_grid_lines, v_grid_lines, h_lines, v_lines, w, h
        )
        
        # Step 5: Create cells considering all grid positions and merge where no barriers exist
        cells = self._create_cells_with_spans(
            h_grid_lines, v_grid_lines, h_barriers, v_barriers, table_bounds, w, h
        )
        
        self.logger.info(f"Found {len(cells)} cells after processing")
        
        # Debug visualization
        if self.debug_mode:
            self._save_debug_visualization(
                binary_img, cells, table_bounds, h_grid_lines, v_grid_lines,
                h_barriers, v_barriers
            )
        
        return cells

    def _detect_ruling_lines(self, binary_img, w, h):
        """
        Step 2: Detect ALL horizontal and vertical ruling lines in the table.
        Including partial lines that might define individual cells.
        """
        self.logger.info("STEP 2: Detecting all ruling lines")
        
        h_kernel_width = min(max(int(w * 0.05), 20), 100)
        v_kernel_height = min(max(int(h * 0.05), 20), 100)
        
        _, thresh = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        self.save_debug_image(thresh, "threshold", "06")
        
        # Detect horizontal lines with lower threshold to catch all lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
        h_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)  # Reduced iterations
        self.save_debug_image(h_lines_img, "horizontal_lines", "07a")
        
        # Detect vertical lines with lower threshold to catch all lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
        v_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=1)  # Reduced iterations
        self.save_debug_image(v_lines_img, "vertical_lines", "07b")
        
        # Extract line segments with lower minimum length to catch cell dividers
        h_lines = self._extract_all_line_segments(h_lines_img, 'horizontal', w, h)
        v_lines = self._extract_all_line_segments(v_lines_img, 'vertical', w, h)
        
        self.logger.info(f"Detected {len(h_lines)} horizontal and {len(v_lines)} vertical line segments")
        
        return h_lines, v_lines

    def _extract_all_line_segments(self, line_img, orientation, w, h):
        """Extract ALL line segments including small ones that might be cell dividers."""
        segments = []
        # Lower minimum length to catch cell dividers
        min_length = max(10, min(w, h) * 0.01)  # Just 1% of dimension or 10 pixels
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            line_img, connectivity=8
        )
        
        for i in range(1, num_labels):
            comp_x = stats[i, cv2.CC_STAT_LEFT]
            comp_y = stats[i, cv2.CC_STAT_TOP]
            comp_w = stats[i, cv2.CC_STAT_WIDTH]
            comp_h = stats[i, cv2.CC_STAT_HEIGHT]
            
            if orientation == 'horizontal':
                if comp_w >= min_length and comp_h < comp_w * 0.5:
                    segments.append({
                        'y': int(centroids[i][1]),
                        'x1': comp_x,
                        'x2': comp_x + comp_w,
                        'length': comp_w,
                        'thickness': comp_h,
                        'extent_ratio': comp_w / w
                    })
            else:
                if comp_h >= min_length and comp_w < comp_h * 0.5:
                    segments.append({
                        'x': int(centroids[i][0]),
                        'y1': comp_y,
                        'y2': comp_y + comp_h,
                        'length': comp_h,
                        'thickness': comp_w,
                        'extent_ratio': comp_h / h
                    })
        
        return segments

    def _build_complete_grid(self, h_lines, v_lines, w, h):
        """
        Step 3: Build a complete grid using ALL detected lines.
        This creates the finest possible grid to detect individual cells.
        """
        self.logger.info("STEP 3: Building complete grid from all lines")
        
        tolerance = 15
        
        # Get ALL unique horizontal positions
        h_positions = []
        for line in h_lines:
            h_positions.append(line['y'])
        
        # Always add table borders
        h_positions.append(0)
        h_positions.append(h)
        
        # Merge only very close positions (likely duplicates)
        h_grid_lines = self._merge_close_positions(sorted(set(h_positions)), tolerance)
        
        # Get ALL unique vertical positions
        v_positions = []
        for line in v_lines:
            v_positions.append(line['x'])
        
        # Always add table borders
        v_positions.append(0)
        v_positions.append(w)
        
        # Merge only very close positions (likely duplicates)
        v_grid_lines = self._merge_close_positions(sorted(set(v_positions)), tolerance)
        
        self.logger.info(f"Complete grid: {len(h_grid_lines)} horizontal x {len(v_grid_lines)} vertical lines")
        
        return h_grid_lines, v_grid_lines

    def _merge_close_positions(self, positions, tolerance):
        """Merge only positions that are very close (likely the same line detected multiple times)."""
        if not positions:
            return []
        
        merged = []
        current_group = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_group[-1] <= tolerance:
                current_group.append(pos)
            else:
                # Add the average of the group
                merged.append(int(sum(current_group) / len(current_group)))
                current_group = [pos]
        
        if current_group:
            merged.append(int(sum(current_group) / len(current_group)))
        
        return merged

    def _filter_peripheral_gridlines(self, h_grid_lines, v_grid_lines, w, h):
        """
        Step 3.5: Filter out tiny peripheral grid lines that would create cells smaller than 10px.
        Only removes grid lines at the edges that are too close to borders.
        """
        self.logger.info("STEP 3.5: Filtering peripheral grid lines")
        
        min_cell_size = 20  # Minimum cell dimension in pixels
        
        # Filter horizontal grid lines
        filtered_h_lines = []
        for i, line in enumerate(h_grid_lines):
            # Keep first and last (borders)
            if i == 0 or i == len(h_grid_lines) - 1:
                filtered_h_lines.append(line)
                continue
            
            # Check distance to previous line
            prev_dist = line - filtered_h_lines[-1] if filtered_h_lines else h
            
            # Check distance to borders if at edges
            if i == 1:  # Second line (first after top border)
                if prev_dist < min_cell_size:
                    self.logger.debug(f"Removing horizontal grid line at {line} (too close to top border)")
                    continue
            elif i == len(h_grid_lines) - 2:  # Second to last (before bottom border)
                next_dist = h_grid_lines[i + 1] - line
                if next_dist < min_cell_size:
                    self.logger.debug(f"Removing horizontal grid line at {line} (too close to bottom border)")
                    continue
            
            filtered_h_lines.append(line)
        
        # Filter vertical grid lines
        filtered_v_lines = []
        for i, line in enumerate(v_grid_lines):
            # Keep first and last (borders)
            if i == 0 or i == len(v_grid_lines) - 1:
                filtered_v_lines.append(line)
                continue
            
            # Check distance to previous line
            prev_dist = line - filtered_v_lines[-1] if filtered_v_lines else w
            
            # Check distance to borders if at edges
            if i == 1:  # Second line (first after left border)
                if prev_dist < min_cell_size:
                    self.logger.debug(f"Removing vertical grid line at {line} (too close to left border)")
                    continue
            elif i == len(v_grid_lines) - 2:  # Second to last (before right border)
                next_dist = v_grid_lines[i + 1] - line
                if next_dist < min_cell_size:
                    self.logger.debug(f"Removing vertical grid line at {line} (too close to right border)")
                    continue
            
            filtered_v_lines.append(line)
        
        # Additional pass: remove any internal lines that create tiny cells
        final_h_lines = [filtered_h_lines[0]]  # Start with first line
        for i in range(1, len(filtered_h_lines)):
            if filtered_h_lines[i] - final_h_lines[-1] >= min_cell_size or i == len(filtered_h_lines) - 1:
                final_h_lines.append(filtered_h_lines[i])
            else:
                self.logger.debug(f"Removing horizontal grid line at {filtered_h_lines[i]} (creates tiny cell)")
        
        final_v_lines = [filtered_v_lines[0]]  # Start with first line
        for i in range(1, len(filtered_v_lines)):
            if filtered_v_lines[i] - final_v_lines[-1] >= min_cell_size or i == len(filtered_v_lines) - 1:
                final_v_lines.append(filtered_v_lines[i])
            else:
                self.logger.debug(f"Removing vertical grid line at {filtered_v_lines[i]} (creates tiny cell)")
        
        self.logger.info(f"Filtered grid: {len(final_h_lines)} horizontal x {len(final_v_lines)} vertical lines")
        self.logger.info(f"Removed {len(h_grid_lines) - len(final_h_lines)} horizontal and {len(v_grid_lines) - len(final_v_lines)} vertical lines")
        
        return final_h_lines, final_v_lines

    def _identify_line_barriers(self, h_grid_lines, v_grid_lines, h_lines, v_lines, w, h):
        """
        Step 4: Identify which grid boundaries have actual line barriers.
        A barrier exists where there's an actual line in the image.
        """
        self.logger.info("STEP 4: Identifying actual line barriers")
        
        tolerance = 15
        # Lower coverage threshold to detect partial lines as barriers
        min_coverage = 0.3  # Line must cover at least 30% of the cell boundary
        
        # Identify horizontal barriers (lines between rows)
        h_barriers = set()
        for i in range(len(h_grid_lines) - 1):
            y_pos = h_grid_lines[i]
            
            # Check if there's any horizontal line at this position
            for line in h_lines:
                if abs(line['y'] - y_pos) <= tolerance:
                    # This grid line has an actual line, mark all cells it crosses
                    for j in range(len(v_grid_lines) - 1):
                        x_start = v_grid_lines[j]
                        x_end = v_grid_lines[j + 1]
                        
                        # Check if the line covers this cell boundary
                        overlap_start = max(x_start, line['x1'])
                        overlap_end = min(x_end, line['x2'])
                        overlap = max(0, overlap_end - overlap_start)
                        coverage = overlap / (x_end - x_start) if x_end > x_start else 0
                        
                        if coverage >= min_coverage:
                            # Barrier between row i-1 and row i at column j
                            if i > 0:
                                h_barriers.add((i-1, j))
        
        # Identify vertical barriers (lines between columns)
        v_barriers = set()
        for j in range(len(v_grid_lines) - 1):
            x_pos = v_grid_lines[j]
            
            # Check if there's any vertical line at this position
            for line in v_lines:
                if abs(line['x'] - x_pos) <= tolerance:
                    # This grid line has an actual line, mark all cells it crosses
                    for i in range(len(h_grid_lines) - 1):
                        y_start = h_grid_lines[i]
                        y_end = h_grid_lines[i + 1]
                        
                        # Check if the line covers this cell boundary
                        overlap_start = max(y_start, line['y1'])
                        overlap_end = min(y_end, line['y2'])
                        overlap = max(0, overlap_end - overlap_start)
                        coverage = overlap / (y_end - y_start) if y_end > y_start else 0
                        
                        if coverage >= min_coverage:
                            # Barrier between column j-1 and column j at row i
                            if j > 0:
                                v_barriers.add((i, j-1))
        
        self.logger.info(f"Found {len(h_barriers)} horizontal and {len(v_barriers)} vertical barriers")
        
        return h_barriers, v_barriers

    def _create_cells_with_spans(self, h_grid_lines, v_grid_lines, h_barriers, v_barriers,
                                 table_bounds, w, h):
        """
        Step 5: Create cells for each grid position and calculate spans.
        Cells span multiple grid positions when there are no barriers between them.
        """
        self.logger.info("STEP 5: Creating cells with proper spans")
        
        num_rows = len(h_grid_lines) - 1
        num_cols = len(v_grid_lines) - 1
        
        if num_rows == 0 or num_cols == 0:
            self.logger.warning("No grid cells to process")
            return []
        
        # Track which grid positions have been assigned to cells
        assigned = [[False] * num_cols for _ in range(num_rows)]
        cells = []
        
        # Process each grid position
        for row in range(num_rows):
            for col in range(num_cols):
                if assigned[row][col]:
                    continue
                
                # Find how far this cell extends (no barriers means it spans multiple positions)
                max_row, max_col = self._find_cell_extent(
                    row, col, num_rows, num_cols, h_barriers, v_barriers, assigned
                )
                
                # Calculate cell boundaries
                x1 = v_grid_lines[col]
                y1 = h_grid_lines[row]
                x2 = v_grid_lines[max_col + 1]
                y2 = h_grid_lines[max_row + 1]
                
                # No need to filter here since we already filtered the grid lines
                # All cells created from the filtered grid will be >= 10px
                
                # Calculate spans
                rowspan = max_row - row + 1
                colspan = max_col - col + 1
                
                # Create cell
                cell = {
                    "x1": x1 + table_bounds["x1"],
                    "y1": y1 + table_bounds["y1"],
                    "x2": x2 + table_bounds["x1"],
                    "y2": y2 + table_bounds["y1"],
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1),
                    "grid_row": row,
                    "grid_col": col,
                    "rowspan": rowspan,
                    "colspan": colspan
                }
                
                cells.append(cell)
                
                # Mark all covered grid positions as assigned
                for r in range(row, max_row + 1):
                    for c in range(col, max_col + 1):
                        assigned[r][c] = True
        
        # Log statistics
        single_cells = [c for c in cells if c['rowspan'] == 1 and c['colspan'] == 1]
        multi_cells = [c for c in cells if c['rowspan'] > 1 or c['colspan'] > 1]
        self.logger.info(f"Created {len(cells)} cells from {num_rows}x{num_cols} grid")
        self.logger.info(f"  - Single cells: {len(single_cells)}")
        self.logger.info(f"  - Multi-span cells: {len(multi_cells)}")
        if multi_cells:
            max_rowspan = max(c['rowspan'] for c in cells)
            max_colspan = max(c['colspan'] for c in cells)
            self.logger.info(f"  - Max rowspan: {max_rowspan}, Max colspan: {max_colspan}")
        
        return cells

    def _find_cell_extent(self, row, col, num_rows, num_cols, h_barriers, v_barriers, assigned):
        """
        Find how far a cell extends from (row, col) position.
        A cell extends until it hits a barrier or an already assigned position.
        """
        max_row = row
        max_col = col
        
        # First, expand right as far as possible (check for vertical barriers)
        for c in range(col + 1, num_cols):
            # Check if already assigned
            if assigned[row][c]:
                break
            # Check if there's a vertical barrier between c-1 and c
            if (row, c-1) in v_barriers:
                break
            max_col = c
        
        # Then, expand down as far as possible (check for horizontal barriers)
        can_expand_down = True
        for r in range(row + 1, num_rows):
            # Check all columns in the current span
            for c in range(col, max_col + 1):
                # Check if already assigned
                if assigned[r][c]:
                    can_expand_down = False
                    break
                # Check for horizontal barrier between r-1 and r
                if (r-1, c) in h_barriers:
                    can_expand_down = False
                    break
            
            if not can_expand_down:
                break
            
            # Also verify no vertical barriers split the cell at this row
            for c in range(col, max_col):
                if (r, c) in v_barriers:
                    can_expand_down = False
                    break
            
            if not can_expand_down:
                break
            
            max_row = r
        
        return max_row, max_col

    def _save_debug_visualization(self, binary_img, cells, table_bounds,
                                   h_grid_lines, v_grid_lines,
                                   h_barriers, v_barriers):
        """Save debug visualization."""
        # Grid visualization
        grid_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        
        # Draw all grid lines (thin)
        for y in h_grid_lines:
            cv2.line(grid_img, (0, y), (grid_img.shape[1], y), (200, 200, 200), 1)
        
        for x in v_grid_lines:
            cv2.line(grid_img, (x, 0), (x, grid_img.shape[0]), (200, 200, 200), 1)
        
        # Draw barriers (thick)
        for (row, col) in h_barriers:
            if row + 1 < len(h_grid_lines) and col < len(v_grid_lines) - 1:
                y = h_grid_lines[row + 1]
                x1 = v_grid_lines[col]
                x2 = v_grid_lines[col + 1]
                cv2.line(grid_img, (x1, y), (x2, y), (0, 0, 255), 2)
        
        for (row, col) in v_barriers:
            if col + 1 < len(v_grid_lines) and row < len(h_grid_lines) - 1:
                x = v_grid_lines[col + 1]
                y1 = h_grid_lines[row]
                y2 = h_grid_lines[row + 1]
                cv2.line(grid_img, (x, y1), (x, y2), (0, 0, 255), 2)
        
        self.save_debug_image(grid_img, "grid_with_barriers", "08")
        
        # Cells visualization
        cells_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        
        # Sort cells by area to draw smaller ones last (on top)
        sorted_cells = sorted(cells, key=lambda c: c['area'], reverse=True)
        
        for i, cell in enumerate(sorted_cells):
            x1 = cell['x1'] - table_bounds['x1']
            y1 = cell['y1'] - table_bounds['y1']
            x2 = cell['x2'] - table_bounds['x1']
            y2 = cell['y2'] - table_bounds['y1']
            
            # Use different colors for cells with different spans
            if cell['rowspan'] > 1 or cell['colspan'] > 1:
                # Multi-span cells in blue shades
                color = (255 - i*10 % 100, 100, 100)
            else:
                # Single cells in green shades
                color = (100, 255 - i*10 % 100, 100)
            
            cv2.rectangle(cells_img, (x1+1, y1+1), (x2-1, y2-1), color, 2)
            
            # Add cell info
            info = f"r{cell['grid_row']}c{cell['grid_col']}: {cell['rowspan']}x{cell['colspan']}"
            font_scale = 0.3
            thickness = 1
            cv2.putText(cells_img, info, (x1+3, y1+12),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        
        self.save_debug_image(cells_img, "detected_cells", "09")