"""
Table detection module for ML Timemaster.
Contains the TableDetector class for detecting table borders and cells.
"""

import logging
import cv2
import numpy as np
import os


class TableDetector:
    """Handles detection of table borders and individual cells using border-based approach."""
    
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
        """
        self.logger.info("="*50)
        self.logger.info("Starting table border detection")
        
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
        Detect cells using border-based approach.
        
        A cell is defined by the 4 lines that surround it:
        - Top horizontal line
        - Bottom horizontal line
        - Left vertical line
        - Right vertical line
        
        This approach naturally handles merged cells without falsely splitting them.
        """
        self.logger.info("="*50)
        self.logger.info("Starting cell detection using border-based approach")
        
        x, y = table_bounds["x1"], table_bounds["y1"]
        w = table_bounds["x2"] - table_bounds["x1"]
        h = table_bounds["y2"] - table_bounds["y1"]
        
        table_region = img[y:y+h, x:x+w]
        self.save_debug_image(table_region, "table_region", "05")
        
        if len(table_region.shape) == 3:
            binary_img = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        else:
            binary_img = table_region.copy()
        
        # Step 1: Detect all horizontal and vertical lines
        h_lines, v_lines = self._detect_lines(binary_img, w, h)
        self.logger.info(f"Detected {len(h_lines)} horizontal and {len(v_lines)} vertical lines")
        
        # Step 2: Ensure table borders exist
        h_lines = self._ensure_border_lines(h_lines, 'horizontal', w, h)
        v_lines = self._ensure_border_lines(v_lines, 'vertical', w, h)
        
        # Step 3: Find all possible cell rectangles defined by line borders
        cells = self._find_cells_by_borders(h_lines, v_lines, binary_img, table_bounds, w, h)
        
        self.logger.info(f"Found {len(cells)} cells")
        
        # Step 4: Remove duplicates and validate
        cells = self._remove_duplicate_cells(cells)
        
        self.logger.info(f"After deduplication: {len(cells)} cells")
        
        # Debug visualization
        if self.debug_mode:
            self._save_debug_visualization(
                binary_img, cells, table_bounds, h_lines, v_lines
            )
        
        return cells

    def _detect_lines(self, binary_img, w, h):
        """Detect horizontal and vertical lines in the image."""
        h_kernel_width = min(max(int(w * 0.05), 20), 100)
        v_kernel_height = min(max(int(h * 0.05), 20), 100)
        
        _, thresh = cv2.threshold(binary_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        
        self.save_debug_image(thresh, "threshold", "06")
        
        # Detect horizontal lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
        h_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        h_lines_img = cv2.dilate(h_lines_img, h_kernel, iterations=1)
        self.save_debug_image(h_lines_img, "horizontal_lines", "07a")
        
        # Detect vertical lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
        v_lines_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)
        v_lines_img = cv2.dilate(v_lines_img, v_kernel, iterations=1)
        self.save_debug_image(v_lines_img, "vertical_lines", "07b")
        
        h_lines = self._extract_line_segments(h_lines_img, 'horizontal', w, h)
        v_lines = self._extract_line_segments(v_lines_img, 'vertical', w, h)
        
        return h_lines, v_lines

    def _extract_line_segments(self, line_img, orientation, w, h):
        """Extract line segments from binary image."""
        segments = []
        min_length = max(10, min(w, h) * 0.02)
        
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
                        'length': comp_w
                    })
            else:
                if comp_h >= min_length and comp_w < comp_h * 0.5:
                    segments.append({
                        'x': int(centroids[i][0]),
                        'y1': comp_y,
                        'y2': comp_y + comp_h,
                        'length': comp_h
                    })
        
        segments = self._merge_segments(segments, orientation)
        return segments

    def _merge_segments(self, segments, orientation, tolerance=12):
        """Merge nearby parallel line segments."""
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
                    current['length'] = current['x2'] - current['x1']
                else:
                    merged.append(current)
                    current = seg.copy()
            merged.append(current)
        else:
            segments.sort(key=lambda s: s['x'])
            current = segments[0].copy()
            
            for seg in segments[1:]:
                if abs(seg['x'] - current['x']) <= tolerance:
                    current['y1'] = min(current['y1'], seg['y1'])
                    current['y2'] = max(current['y2'], seg['y2'])
                    current['x'] = (current['x'] + seg['x']) // 2
                    current['length'] = current['y2'] - current['y1']
                else:
                    merged.append(current)
                    current = seg.copy()
            merged.append(current)
        
        return merged

    def _ensure_border_lines(self, lines, orientation, w, h):
        """Ensure table border lines exist."""
        tolerance = 15
        
        if orientation == 'horizontal':
            has_top = any(line['y'] < tolerance for line in lines)
            if not has_top:
                lines.append({'y': 0, 'x1': 0, 'x2': w, 'length': w})
            
            has_bottom = any(line['y'] > h - tolerance for line in lines)
            if not has_bottom:
                lines.append({'y': h, 'x1': 0, 'x2': w, 'length': w})
        else:
            has_left = any(line['x'] < tolerance for line in lines)
            if not has_left:
                lines.append({'x': 0, 'y1': 0, 'y2': h, 'length': h})
            
            has_right = any(line['x'] > w - tolerance for line in lines)
            if not has_right:
                lines.append({'x': w, 'y1': 0, 'y2': h, 'length': h})
        
        return lines

    def _find_cells_by_borders(self, h_lines, v_lines, binary_img, table_bounds, table_w, table_h):
        """
        Find cells by looking for rectangles formed by bordering lines.
        
        For each combination of 4 lines (top, bottom, left, right):
        - Check if they form a valid rectangle
        - Verify that these lines actually border a cell region
        - Ensure the cell is not split by internal lines
        """
        cells = []
        
        # Sort lines for easier processing
        h_lines_sorted = sorted(h_lines, key=lambda l: l['y'])
        v_lines_sorted = sorted(v_lines, key=lambda l: l['x'])
        
        self.logger.info(f"Searching for cells among {len(h_lines_sorted)} h-lines and {len(v_lines_sorted)} v-lines")
        
        # Try each pair of horizontal lines (top and bottom)
        for i, top_line in enumerate(h_lines_sorted):
            for bottom_line in h_lines_sorted[i+1:]:
                top_y = top_line['y']
                bottom_y = bottom_line['y']
                
                # Skip if too small
                cell_height = bottom_y - top_y
                if cell_height < 10:
                    continue
                
                # Try each pair of vertical lines (left and right)
                for j, left_line in enumerate(v_lines_sorted):
                    for right_line in v_lines_sorted[j+1:]:
                        left_x = left_line['x']
                        right_x = right_line['x']
                        
                        # Skip if too small
                        cell_width = right_x - left_x
                        if cell_width < 10:
                            continue
                        
                        # Check if these 4 lines actually border a cell
                        if self._is_valid_cell_border(
                            top_line, bottom_line, left_line, right_line,
                            top_y, bottom_y, left_x, right_x,
                            h_lines_sorted, v_lines_sorted,
                            binary_img, table_w, table_h
                        ):
                            cell = {
                                "x1": left_x + table_bounds["x1"],
                                "y1": top_y + table_bounds["y1"],
                                "x2": right_x + table_bounds["x1"],
                                "y2": bottom_y + table_bounds["y1"],
                                "width": cell_width,
                                "height": cell_height,
                                "area": cell_width * cell_height,
                            }
                            cells.append(cell)
        
        return cells

    def _is_valid_cell_border(self, top_line, bottom_line, left_line, right_line,
                               top_y, bottom_y, left_x, right_x,
                               all_h_lines, all_v_lines,
                               binary_img, table_w, table_h):
        """
        Check if 4 lines form a valid cell border.
        
        Criteria:
        1. The lines must actually overlap the cell boundaries (coverage check)
        2. There must be NO internal horizontal line that fully crosses the cell
        3. There must be NO internal vertical line that fully crosses the cell
        4. The region should not be completely empty (optional)
        """
        tolerance = 15
        coverage_threshold = 0.5  # Line must cover at least 50% of the border
        
        cell_width = right_x - left_x
        cell_height = bottom_y - top_y
        
        # Check 1: Top line must cover the top border
        top_coverage = self._calculate_line_coverage(
            top_line, left_x, right_x, 'horizontal'
        )
        if top_coverage < coverage_threshold:
            return False
        
        # Check 2: Bottom line must cover the bottom border
        bottom_coverage = self._calculate_line_coverage(
            bottom_line, left_x, right_x, 'horizontal'
        )
        if bottom_coverage < coverage_threshold:
            return False
        
        # Check 3: Left line must cover the left border
        left_coverage = self._calculate_line_coverage(
            left_line, top_y, bottom_y, 'vertical'
        )
        if left_coverage < coverage_threshold:
            return False
        
        # Check 4: Right line must cover the right border
        right_coverage = self._calculate_line_coverage(
            right_line, top_y, bottom_y, 'vertical'
        )
        if right_coverage < coverage_threshold:
            return False
        
        # Check 5: No internal horizontal line should fully split the cell
        for h_line in all_h_lines:
            if h_line['y'] <= top_y + tolerance or h_line['y'] >= bottom_y - tolerance:
                continue  # This is the border or outside
            
            # This is an internal horizontal line
            # Check if it fully crosses the cell
            line_left = h_line['x1']
            line_right = h_line['x2']
            
            # Calculate how much of the cell width this line covers
            overlap_left = max(left_x, line_left)
            overlap_right = min(right_x, line_right)
            overlap_width = max(0, overlap_right - overlap_left)
            
            coverage = overlap_width / cell_width if cell_width > 0 else 0
            
            # If the line covers most of the cell width, it splits the cell
            if coverage > 0.7:
                return False
        
        # Check 6: No internal vertical line should fully split the cell
        for v_line in all_v_lines:
            if v_line['x'] <= left_x + tolerance or v_line['x'] >= right_x - tolerance:
                continue  # This is the border or outside
            
            # This is an internal vertical line
            line_top = v_line['y1']
            line_bottom = v_line['y2']
            
            overlap_top = max(top_y, line_top)
            overlap_bottom = min(bottom_y, line_bottom)
            overlap_height = max(0, overlap_bottom - overlap_top)
            
            coverage = overlap_height / cell_height if cell_height > 0 else 0
            
            if coverage > 0.7:
                return False
        
        # All checks passed
        return True

    def _calculate_line_coverage(self, line, start, end, orientation):
        """
        Calculate how much of a border segment a line covers.
        
        Args:
            line: The line dictionary
            start: Start of the border segment (x for horizontal, y for vertical)
            end: End of the border segment
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        border_length = end - start
        if border_length <= 0:
            return 0.0
        
        if orientation == 'horizontal':
            line_start = line['x1']
            line_end = line['x2']
        else:
            line_start = line['y1']
            line_end = line['y2']
        
        # Calculate overlap
        overlap_start = max(start, line_start)
        overlap_end = min(end, line_end)
        overlap = max(0, overlap_end - overlap_start)
        
        coverage = overlap / border_length
        return coverage

    def _remove_duplicate_cells(self, cells, overlap_threshold=0.8):
        """
        Remove duplicate cells that overlap significantly.
        
        When we have nested rectangles (e.g., a merged cell and the sub-cells
        it should contain), keep the appropriate ones based on area and coverage.
        """
        if not cells:
            return []
        
        # Sort by area (smaller first, so we prefer larger cells when there's overlap)
        cells_sorted = sorted(cells, key=lambda c: c['area'])
        
        unique_cells = []
        
        for cell in cells_sorted:
            is_duplicate = False
            
            for unique_cell in unique_cells:
                overlap = self._calculate_cell_overlap(cell, unique_cell)
                
                # If this cell is almost entirely contained in an existing cell, skip it
                if overlap > overlap_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_cells.append(cell)
        
        return unique_cells

    def _calculate_cell_overlap(self, cell1, cell2):
        """Calculate overlap ratio between two cells."""
        x_left = max(cell1['x1'], cell2['x1'])
        y_top = max(cell1['y1'], cell2['y1'])
        x_right = min(cell1['x2'], cell2['x2'])
        y_bottom = min(cell1['y2'], cell2['y2'])
        
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        
        overlap_area = (x_right - x_left) * (y_bottom - y_top)
        smaller_area = min(cell1['area'], cell2['area'])
        
        return overlap_area / smaller_area if smaller_area > 0 else 0.0

    def _save_debug_visualization(self, binary_img, cells, table_bounds,
                                   h_lines, v_lines):
        """Save debug visualization."""
        # Lines visualization
        lines_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        
        for line in h_lines:
            cv2.line(lines_img, (line['x1'], line['y']), (line['x2'], line['y']), 
                    (255, 0, 0), 2)
        
        for line in v_lines:
            cv2.line(lines_img, (line['x'], line['y1']), (line['x'], line['y2']), 
                    (0, 255, 0), 2)
        
        self.save_debug_image(lines_img, "detected_lines", "08")
        
        # Cells visualization
        cells_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        
        for i, cell in enumerate(cells):
            x1 = cell['x1'] - table_bounds['x1']
            y1 = cell['y1'] - table_bounds['y1']
            x2 = cell['x2'] - table_bounds['x1']
            y2 = cell['y2'] - table_bounds['y1']
            
            # Random color for each cell
            color = ((i * 37) % 256, (i * 67) % 256, (i * 97) % 256)
            
            cv2.rectangle(cells_img, (x1+2, y1+2), (x2-2, y2-2), color, 2)
            cv2.putText(cells_img, str(i), (x1+5, y1+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        self.save_debug_image(cells_img, "detected_cells", "09")