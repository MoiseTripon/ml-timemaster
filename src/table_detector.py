"""
Table detection module for ML Timemaster.
Contains the TableDetector class for detecting table borders and cells.
"""

import logging
import cv2
import numpy as np
import os


class TableDetector:
    """Handles detection of table borders and individual cells."""
    
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
        Detect the main table borders in the image.
        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            dict: Table bounds with keys x1, y1, x2, y2
        """
        self.logger.info("="*50)
        self.logger.info("Starting table border detection")
        
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        self.save_debug_image(gray, "grayscale", "01")
        
        # Scale kernel size based on image dimensions (smaller kernel for border detection)
        img_h, img_w = gray.shape[:2]
        morph_kernel_size = max(3, min(img_w, img_h) // 200)
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        # Apply light morphology BEFORE Canny to clean up the image
        morphed = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
        
        self.save_debug_image(morphed, "morphed_gradient", "02")
        
        # Apply Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple edge detection approaches and combine results
        edges1 = cv2.Canny(blurred, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(morphed, 30, 100, apertureSize=3)
        
        # Combine edges from both methods
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Apply dilation to connect nearby edges
        dilation_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, dilation_kernel, iterations=1)
        edges = cv2.erode(edges, dilation_kernel, iterations=1)
        
        self.save_debug_image(edges, "edges_combined", "03")
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"Found {len(contours)} contours")
        
        if not contours:
            self.logger.warning("No contours found with morphology, trying without")
            edges_fallback = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.error("No contours found in the image")
                raise ValueError("No table detected in the image")
        
        # Find the largest contour (assume it's the table)
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Validate the detected bounds
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
        Detect individual cells within the table, including merged cells (colspan/rowspan).
        
        Args:
            img (numpy.ndarray): Input image
            table_bounds (dict): Table boundaries
            
        Returns:
            list: List of cell dictionaries with bounds and dimensions
        """
        self.logger.info("="*50)
        self.logger.info("Starting cell detection")
        
        # Extract table region
        x, y = table_bounds["x1"], table_bounds["y1"]
        w = table_bounds["x2"] - table_bounds["x1"]
        h = table_bounds["y2"] - table_bounds["y1"]
        
        table_region = img[y:y+h, x:x+w]
        self.save_debug_image(table_region, "table_region", "05")
        
        # Convert to grayscale if needed
        if len(table_region.shape) == 3:
            binary_img = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        else:
            binary_img = table_region.copy()
        
        # Detect cells using actual contours (handles merged cells)
        cells = self._detect_cells_with_merged(binary_img, table_bounds, w, h)
        self.logger.info(f"Cell detection found {len(cells)} cells (including merged)")
        
        # If no cells found, try fallback methods
        if len(cells) < 1:
            self.logger.info("Falling back to contour-based detection")
            cells = self._detect_cells_by_contours(binary_img, table_bounds, w, h)
            cells = self._remove_duplicate_cells(cells)
        
        # Merge adjacent cells that should be one (no line between them)
        cells = self._merge_adjacent_cells(cells, binary_img, table_bounds)
        
        self.logger.info(f"Total cells detected after merging: {len(cells)}")
        
        # Save debug visualization
        if self.debug_mode:
            debug_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            for i, cell in enumerate(cells):
                x1 = cell['x1'] - table_bounds['x1']
                y1 = cell['y1'] - table_bounds['y1']
                x2 = cell['x2'] - table_bounds['x1']
                y2 = cell['y2'] - table_bounds['y1']
                
                color = ((i * 37) % 255, (i * 67) % 255, (i * 97) % 255)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(debug_img, str(i), (x1+5, y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            self.save_debug_image(debug_img, "detected_cells", "08")
        
        return cells

    def _detect_cells_with_merged(self, binary_img, table_bounds, w, h):
        """
        Detect cells including merged cells (colspan/rowspan) by finding actual cell contours.
        
        Args:
            binary_img: Grayscale image of the table region
            table_bounds: Dictionary with table boundary coordinates
            w: Width of the table region
            h: Height of the table region
            
        Returns:
            list: List of detected cell dictionaries
        """
        # First detect actual lines that exist in the image
        h_lines, v_lines = self._detect_actual_lines(binary_img, w, h)
        
        self.logger.info(f"Detected {len(h_lines)} horizontal and {len(v_lines)} vertical line segments")
        
        # Filter out weak/short lines that might be noise
        h_lines = self._filter_strong_lines(h_lines, 'horizontal', w, h)
        v_lines = self._filter_strong_lines(v_lines, 'vertical', w, h)
        
        self.logger.info(f"After filtering: {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
        
        # Build a line image from detected segments
        line_img = np.zeros_like(binary_img)
        
        # Draw horizontal lines
        for line in h_lines:
            cv2.line(line_img, (line['x1'], line['y']), (line['x2'], line['y']), 255, 2)
        
        # Draw vertical lines
        for line in v_lines:
            cv2.line(line_img, (line['x'], line['y1']), (line['x'], line['y2']), 255, 2)
        
        self.save_debug_image(line_img, "detected_line_segments", "06c")
        
        # Find cell contours from the line image
        cells = self._find_cells_from_lines(line_img, binary_img, table_bounds, w, h)
        
        return cells

    def _filter_strong_lines(self, lines, orientation, table_w, table_h):
        """
        Filter out weak lines that are likely noise or internal cell divisions.
        Only keep lines that span a significant portion of the table.
        
        Args:
            lines: List of detected line segments
            orientation: 'horizontal' or 'vertical'
            table_w, table_h: Table dimensions
            
        Returns:
            list: Filtered lines
        """
        if not lines:
            return lines
        
        filtered = []
        
        if orientation == 'horizontal':
            # Keep lines that span at least 30% of table width or are near edges
            min_length = table_w * 0.3
            edge_tolerance = table_h * 0.05
            
            for line in lines:
                line_length = line['x2'] - line['x1']
                is_edge = line['y'] < edge_tolerance or line['y'] > table_h - edge_tolerance
                
                if line_length >= min_length or is_edge:
                    filtered.append(line)
                    
        else:  # vertical
            # Keep lines that span at least 30% of table height or are near edges
            min_length = table_h * 0.3
            edge_tolerance = table_w * 0.05
            
            for line in lines:
                line_length = line['y2'] - line['y1']
                is_edge = line['x'] < edge_tolerance or line['x'] > table_w - edge_tolerance
                
                if line_length >= min_length or is_edge:
                    filtered.append(line)
        
        return filtered

    def _detect_actual_lines(self, binary_img, w, h):
        """
        Detect actual line segments that exist in the image.
        
        Returns:
            tuple: (horizontal_lines, vertical_lines)
        """
        # Use larger kernel sizes to detect more prominent lines
        horizontal_kernel_width = min(max(int(w * 0.08), 30), 150)
        vertical_kernel_height = min(max(int(h * 0.08), 30), 150)
        
        # Threshold the image
        _, thresh = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if needed (lines should be white)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        # Create kernels for line detection
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (horizontal_kernel_width, 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (1, vertical_kernel_height)
        )
        
        # Detect horizontal lines with stronger morphology
        horizontal_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_lines_img = cv2.dilate(horizontal_lines_img, horizontal_kernel, iterations=1)
        self.save_debug_image(horizontal_lines_img, "horizontal_lines_raw", "06a")
        
        # Detect vertical lines with stronger morphology
        vertical_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_lines_img = cv2.dilate(vertical_lines_img, vertical_kernel, iterations=1)
        self.save_debug_image(vertical_lines_img, "vertical_lines_raw", "06b")
        
        # Extract line segments
        h_segments = self._extract_line_segments(horizontal_lines_img, 'horizontal', w, h)
        v_segments = self._extract_line_segments(vertical_lines_img, 'vertical', w, h)
        
        return h_segments, v_segments

    def _extract_line_segments(self, line_img, orientation, w, h):
        """
        Extract line segments from a binary image containing detected lines.
        
        Args:
            line_img: Binary image with detected lines
            orientation: 'horizontal' or 'vertical'
            w, h: Width and height of the image
            
        Returns:
            list: List of line segment dictionaries
        """
        segments = []
        # Increase minimum line length to filter out noise
        min_line_length = max(20, min(w, h) * 0.05)
        
        if orientation == 'horizontal':
            # Use connected components for more robust line detection
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_img, connectivity=8)
            
            for i in range(1, num_labels):  # Skip background
                x_start = stats[i, cv2.CC_STAT_LEFT]
                y_center = int(centroids[i][1])
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Only consider thin horizontal components
                if width >= min_line_length and height < width * 0.3:
                    segments.append({
                        'y': y_center,
                        'x1': x_start,
                        'x2': x_start + width,
                        'orientation': 'horizontal'
                    })
        
        else:  # vertical
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(line_img, connectivity=8)
            
            for i in range(1, num_labels):  # Skip background
                x_center = int(centroids[i][0])
                y_start = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Only consider thin vertical components
                if height >= min_line_length and width < height * 0.3:
                    segments.append({
                        'x': x_center,
                        'y1': y_start,
                        'y2': y_start + height,
                        'orientation': 'vertical'
                    })
        
        # Merge nearby parallel segments
        segments = self._merge_nearby_segments(segments, orientation)
        
        return segments

    def _merge_nearby_segments(self, segments, orientation, tolerance=8):
        """
        Merge line segments that are close to each other.
        
        Args:
            segments: List of line segments
            orientation: 'horizontal' or 'vertical'
            tolerance: Maximum distance between segments to merge
            
        Returns:
            list: Merged line segments
        """
        if not segments:
            return segments
        
        merged = []
        
        if orientation == 'horizontal':
            segments.sort(key=lambda s: s['y'])
            
            current = segments[0].copy()
            for seg in segments[1:]:
                if abs(seg['y'] - current['y']) <= tolerance:
                    current['x1'] = min(current['x1'], seg['x1'])
                    current['x2'] = max(current['x2'], seg['x2'])
                    current['y'] = (current['y'] + seg['y']) // 2
                else:
                    merged.append(current)
                    current = seg.copy()
            merged.append(current)
        
        else:  # vertical
            segments.sort(key=lambda s: s['x'])
            
            current = segments[0].copy()
            for seg in segments[1:]:
                if abs(seg['x'] - current['x']) <= tolerance:
                    current['y1'] = min(current['y1'], seg['y1'])
                    current['y2'] = max(current['y2'], seg['y2'])
                    current['x'] = (current['x'] + seg['x']) // 2
                else:
                    merged.append(current)
                    current = seg.copy()
            merged.append(current)
        
        return merged

    def _find_cells_from_lines(self, line_img, original_img, table_bounds, w, h):
        """
        Find individual cells (including merged cells) from the detected line segments.
        
        Args:
            line_img: Binary image with drawn line segments
            original_img: Original grayscale image
            table_bounds: Table boundary coordinates
            w, h: Width and height of table region
            
        Returns:
            list: List of detected cells
        """
        # Invert the line image to have cells as white regions
        inverted = cv2.bitwise_not(line_img)
        
        # Close small gaps that might split cells
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Open to remove small noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        
        self.save_debug_image(opened, "cells_as_regions", "07")
        
        # Find contours of cells
        contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        min_cell_area = max(200, (w * h) * 0.002)  # Increased minimum
        max_cell_area = (w * h) * 0.95
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if area < min_cell_area or area > max_cell_area:
                continue
            
            x1, y1, width, height = cv2.boundingRect(contour)
            
            if width < 15 or height < 15:
                continue
            
            # Check if this contour is inside another
            if hierarchy[0][i][3] != -1:
                parent_area = cv2.contourArea(contours[hierarchy[0][i][3]])
                if min_cell_area <= parent_area <= max_cell_area:
                    continue
            
            cell = {
                "x1": x1 + table_bounds["x1"],
                "y1": y1 + table_bounds["y1"],
                "x2": x1 + width + table_bounds["x1"],
                "y2": y1 + height + table_bounds["y1"],
                "width": width,
                "height": height,
                "area": area,
                "is_merged": False
            }
            
            cells.append(cell)
        
        # Sort cells by position
        cells.sort(key=lambda c: (c['y1'], c['x1']))
        
        # Identify merged cells
        cells = self._identify_merged_cells(cells, w, h)
        
        return cells

    def _merge_adjacent_cells(self, cells, binary_img, table_bounds):
        """
        Merge adjacent cells that don't have a line between them.
        This handles cases where merged cells were incorrectly split.
        
        Args:
            cells: List of detected cells
            binary_img: Grayscale image of the table region
            table_bounds: Table boundary coordinates
            
        Returns:
            list: Cells with adjacent ones merged
        """
        if len(cells) < 2:
            return cells
        
        # Threshold for line detection between cells
        _, thresh = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        merged = True
        while merged:
            merged = False
            new_cells = []
            used = set()
            
            for i, cell1 in enumerate(cells):
                if i in used:
                    continue
                
                current_cell = cell1.copy()
                
                for j, cell2 in enumerate(cells):
                    if j <= i or j in used:
                        continue
                    
                    # Check if cells are horizontally adjacent (same row)
                    if self._cells_same_row(current_cell, cell2):
                        # Check if there's a line between them
                        if not self._has_line_between(current_cell, cell2, thresh, table_bounds, 'vertical'):
                            # Merge cells
                            current_cell = self._merge_two_cells(current_cell, cell2)
                            used.add(j)
                            merged = True
                    
                    # Check if cells are vertically adjacent (same column)
                    elif self._cells_same_column(current_cell, cell2):
                        if not self._has_line_between(current_cell, cell2, thresh, table_bounds, 'horizontal'):
                            current_cell = self._merge_two_cells(current_cell, cell2)
                            used.add(j)
                            merged = True
                
                new_cells.append(current_cell)
                used.add(i)
            
            cells = new_cells
        
        # Re-sort and re-identify merged cells
        cells.sort(key=lambda c: (c['y1'], c['x1']))
        
        return cells

    def _cells_same_row(self, cell1, cell2, tolerance=10):
        """Check if two cells are in the same row (horizontally adjacent)."""
        # Check vertical overlap
        y_overlap = min(cell1['y2'], cell2['y2']) - max(cell1['y1'], cell2['y1'])
        min_height = min(cell1['y2'] - cell1['y1'], cell2['y2'] - cell2['y1'])
        
        if y_overlap < min_height * 0.5:
            return False
        
        # Check if horizontally adjacent (small gap)
        gap = min(abs(cell1['x2'] - cell2['x1']), abs(cell2['x2'] - cell1['x1']))
        return gap < tolerance

    def _cells_same_column(self, cell1, cell2, tolerance=10):
        """Check if two cells are in the same column (vertically adjacent)."""
        # Check horizontal overlap
        x_overlap = min(cell1['x2'], cell2['x2']) - max(cell1['x1'], cell2['x1'])
        min_width = min(cell1['x2'] - cell1['x1'], cell2['x2'] - cell2['x1'])
        
        if x_overlap < min_width * 0.5:
            return False
        
        # Check if vertically adjacent (small gap)
        gap = min(abs(cell1['y2'] - cell2['y1']), abs(cell2['y2'] - cell1['y1']))
        return gap < tolerance

    def _has_line_between(self, cell1, cell2, thresh_img, table_bounds, orientation):
        """
        Check if there's a significant line between two cells.
        
        Args:
            cell1, cell2: Two cell dictionaries
            thresh_img: Thresholded image (lines are white)
            table_bounds: Table boundary coordinates
            orientation: 'horizontal' or 'vertical' - the line orientation to check
            
        Returns:
            bool: True if there's a line between cells
        """
        # Convert to local coordinates
        x1_1 = cell1['x1'] - table_bounds['x1']
        y1_1 = cell1['y1'] - table_bounds['y1']
        x2_1 = cell1['x2'] - table_bounds['x1']
        y2_1 = cell1['y2'] - table_bounds['y1']
        
        x1_2 = cell2['x1'] - table_bounds['x1']
        y1_2 = cell2['y1'] - table_bounds['y1']
        x2_2 = cell2['x2'] - table_bounds['x1']
        y2_2 = cell2['y2'] - table_bounds['y1']
        
        if orientation == 'vertical':
            # Check for vertical line between horizontally adjacent cells
            if cell1['x2'] <= cell2['x1']:
                x_check = (x2_1 + x1_2) // 2
            else:
                x_check = (x2_2 + x1_1) // 2
            
            y_start = max(y1_1, y1_2)
            y_end = min(y2_1, y2_2)
            
            if y_end <= y_start or x_check < 0 or x_check >= thresh_img.shape[1]:
                return True  # Can't check, assume there's a line
            
            # Sample the column between cells
            column = thresh_img[y_start:y_end, max(0, x_check-2):min(thresh_img.shape[1], x_check+3)]
            if column.size == 0:
                return True
            
            # Check if enough white pixels (line present)
            line_pixels = np.sum(column > 200)
            total_pixels = column.size
            line_ratio = line_pixels / total_pixels if total_pixels > 0 else 0
            
            return line_ratio > 0.3
        
        else:  # horizontal
            # Check for horizontal line between vertically adjacent cells
            if cell1['y2'] <= cell2['y1']:
                y_check = (y2_1 + y1_2) // 2
            else:
                y_check = (y2_2 + y1_1) // 2
            
            x_start = max(x1_1, x1_2)
            x_end = min(x2_1, x2_2)
            
            if x_end <= x_start or y_check < 0 or y_check >= thresh_img.shape[0]:
                return True
            
            # Sample the row between cells
            row = thresh_img[max(0, y_check-2):min(thresh_img.shape[0], y_check+3), x_start:x_end]
            if row.size == 0:
                return True
            
            line_pixels = np.sum(row > 200)
            total_pixels = row.size
            line_ratio = line_pixels / total_pixels if total_pixels > 0 else 0
            
            return line_ratio > 0.3

    def _merge_two_cells(self, cell1, cell2):
        """Merge two cells into one."""
        return {
            "x1": min(cell1['x1'], cell2['x1']),
            "y1": min(cell1['y1'], cell2['y1']),
            "x2": max(cell1['x2'], cell2['x2']),
            "y2": max(cell1['y2'], cell2['y2']),
            "width": max(cell1['x2'], cell2['x2']) - min(cell1['x1'], cell2['x1']),
            "height": max(cell1['y2'], cell2['y2']) - min(cell1['y1'], cell2['y1']),
            "area": (max(cell1['x2'], cell2['x2']) - min(cell1['x1'], cell2['x1'])) * 
                    (max(cell1['y2'], cell2['y2']) - min(cell1['y1'], cell2['y1'])),
            "is_merged": True
        }

    def _identify_merged_cells(self, cells, table_w, table_h):
        """
        Identify which cells are merged based on their size.
        
        Args:
            cells: List of detected cells
            table_w, table_h: Table dimensions
            
        Returns:
            list: Cells with updated merge information
        """
        if len(cells) < 2:
            return cells
        
        # Use smaller cells to estimate typical size
        widths = sorted([c['width'] for c in cells])
        heights = sorted([c['height'] for c in cells])
        
        percentile = 0.25
        typical_width = widths[int(len(widths) * percentile)]
        typical_height = heights[int(len(heights) * percentile)]
        
        self.logger.info(f"Typical cell size: {typical_width}x{typical_height}")
        
        for cell in cells:
            colspan = max(1, round(cell['width'] / typical_width)) if typical_width > 0 else 1
            rowspan = max(1, round(cell['height'] / typical_height)) if typical_height > 0 else 1
            
            if colspan > 1 or rowspan > 1:
                cell['is_merged'] = True
                cell['colspan'] = colspan
                cell['rowspan'] = rowspan
                self.logger.debug(f"Merged cell: colspan={colspan}, rowspan={rowspan}")
            else:
                cell['colspan'] = 1
                cell['rowspan'] = 1
        
        return cells

    def _detect_cells_by_contours(self, binary_img, table_bounds, w, h):
        """Fallback: Detect cells using contour detection."""
        cells = []
        
        morph_kernel_size = max(3, min(w, h) // 50)
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        morphed = cv2.morphologyEx(binary_img, cv2.MORPH_GRADIENT, morph_kernel)
        
        preprocessing_methods = [
            ("original", binary_img),
            ("morphed", morphed),
            ("inverted", cv2.bitwise_not(binary_img)),
            ("edges_50_150", cv2.Canny(binary_img, 50, 150)),
            ("edges_30_200", cv2.Canny(morphed, 30, 200)),
        ]
        
        min_area = (w * h) * 0.002
        max_area = (w * h) * 0.95
        
        for method_name, processed in preprocessing_methods:
            contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x1, y1, width, height = cv2.boundingRect(contour)
                    cell = {
                        "x1": x1 + table_bounds["x1"],
                        "y1": y1 + table_bounds["y1"],
                        "x2": x1 + width + table_bounds["x1"],
                        "y2": y1 + height + table_bounds["y1"],
                        "width": width,
                        "height": height,
                    }
                    cells.append(cell)
        
        return cells

    def _remove_duplicate_cells(self, cells, overlap_threshold=0.7):
        """Remove duplicate cells that overlap significantly."""
        if not cells:
            return []

        def calculate_overlap(cell1, cell2):
            cell1_area = (cell1["x2"] - cell1["x1"]) * (cell1["y2"] - cell1["y1"])
            cell2_area = (cell2["x2"] - cell2["x1"]) * (cell2["y2"] - cell2["y1"])

            if cell1_area <= 0 or cell2_area <= 0:
                return 0.0

            x_left = max(cell1["x1"], cell2["x1"])
            y_top = max(cell1["y1"], cell2["y1"])
            x_right = min(cell1["x2"], cell2["x2"])
            y_bottom = min(cell1["y2"], cell2["y2"])

            if x_right <= x_left or y_bottom <= y_top:
                return 0.0

            overlap_area = (x_right - x_left) * (y_bottom - y_top)
            return float(overlap_area) / min(cell1_area, cell2_area)

        valid_cells = [
            cell for cell in cells
            if (cell["x2"] > cell["x1"] and cell["y2"] > cell["y1"]
                and (cell["x2"] - cell["x1"]) >= 10
                and (cell["y2"] - cell["y1"]) >= 10)
        ]

        valid_cells.sort(
            key=lambda c: (c["x2"] - c["x1"]) * (c["y2"] - c["y1"]), 
            reverse=True
        )

        unique_cells = []
        for cell in valid_cells:
            is_duplicate = False
            for unique_cell in unique_cells:
                overlap = calculate_overlap(cell, unique_cell)
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_cells.append(cell)

        return unique_cells