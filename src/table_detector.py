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
        
        # Scale kernel size based on image dimensions (conservative sizing)
        img_h, img_w = gray.shape[:2]
        morph_kernel_size = max(2, min(img_w, img_h) // 150)
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        # Apply gentle morphology to reduce noise while preserving edges
        # Use only closing with single iteration to fill small gaps without removing lines
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        self.save_debug_image(morphed, "morphed", "02")
        
        # Detect edges after morphology
        edges = cv2.Canny(morphed, 50, 150, apertureSize=3)
        self.save_debug_image(edges, "edges", "03")
        
        # Dilate edges slightly to connect nearby edge fragments
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)
        self.save_debug_image(edges, "edges_dilated", "03a")
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"Found {len(contours)} contours")
        
        if not contours:
            # Fallback: try without morphology
            self.logger.warning("No contours found with morphology, trying without")
            edges_fallback = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.error("No contours found in the image")
                raise ValueError("No table detected in the image")
        
        # Find the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        table_bounds = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
        
        self.logger.info(f"Detected table boundaries: {table_bounds}")
        
        # Save debug visualization
        if self.debug_mode:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            self.save_debug_image(debug_img, "table_bounds", "04")
        
        return table_bounds

    def detect_cells(self, img, table_bounds):
        """
        Detect individual cells within the table.
        
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
        
        # Try line-intersection method first (primary)
        cells = self._detect_cells_by_line_intersection(binary_img, table_bounds, w, h)
        self.logger.info(f"Line intersection method found {len(cells)} cells")
        
        # If line intersection didn't find enough cells, try contour method as fallback
        if len(cells) < 2:
            self.logger.info("Line intersection found insufficient cells, trying contour method")
            cells_fallback = self._detect_cells_by_contours(binary_img, table_bounds, w, h)
            self.logger.info(f"Contour method found {len(cells_fallback)} cells")
            cells.extend(cells_fallback)
            cells = self._remove_duplicate_cells(cells)
        
        self.logger.info(f"Total cells detected: {len(cells)}")
        
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
            self.save_debug_image(debug_img, "detected_cells", "09")
        
        return cells

    def _detect_cells_by_line_intersection(self, binary_img, table_bounds, w, h):
        """
        Detect cells using horizontal and vertical line detection with line-intersection.
        
        This method finds horizontal and vertical lines separately, determines their
        positions, and creates cells from the intersections of consecutive lines.
        
        Args:
            binary_img: Grayscale image of the table region
            table_bounds: Dictionary with table boundary coordinates
            w: Width of the table region
            h: Height of the table region
            
        Returns:
            list: List of detected cell dictionaries
        """
        # Scale kernel sizes with table dimensions
        # Use 3-8% of dimension, with reasonable min/max bounds
        horizontal_kernel_width = max(20, min(int(w * 0.05), w // 3))
        vertical_kernel_height = max(20, min(int(h * 0.05), h // 3))
        
        self.logger.debug(f"Table dimensions: {w}x{h}")
        self.logger.debug(f"Using horizontal kernel width: {horizontal_kernel_width}")
        self.logger.debug(f"Using vertical kernel height: {vertical_kernel_height}")
        
        # Scale morphology cleanup kernel (small, for noise reduction)
        cleanup_kernel_size = max(2, min(w, h) // 200)
        cleanup_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (cleanup_kernel_size, cleanup_kernel_size)
        )
        
        # Apply gentle morphology to clean up the image
        cleaned = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, cleanup_kernel, iterations=1)
        self.save_debug_image(cleaned, "cleaned", "06")
        
        # Try to detect lines with adaptive thresholding
        # This works better for varying lighting conditions
        thresh_adaptive = cv2.adaptiveThreshold(
            cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 2
        )
        
        # Also try OTSU thresholding
        _, thresh_otsu = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Try both and use the one that gives better results
        h_positions, v_positions = self._detect_lines_from_threshold(
            thresh_adaptive, horizontal_kernel_width, vertical_kernel_height, w, h, "adaptive"
        )
        
        if len(h_positions) < 2 or len(v_positions) < 2:
            self.logger.debug("Adaptive threshold didn't find enough lines, trying OTSU")
            h_pos_otsu, v_pos_otsu = self._detect_lines_from_threshold(
                thresh_otsu, horizontal_kernel_width, vertical_kernel_height, w, h, "otsu"
            )
            
            # Use whichever found more lines
            if len(h_pos_otsu) > len(h_positions):
                h_positions = h_pos_otsu
            if len(v_pos_otsu) > len(v_positions):
                v_positions = v_pos_otsu
        
        # If still not enough, try with non-inverted threshold
        if len(h_positions) < 2 or len(v_positions) < 2:
            self.logger.debug("Inverted threshold didn't work, trying non-inverted")
            _, thresh_normal = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h_pos_normal, v_pos_normal = self._detect_lines_from_threshold(
                thresh_normal, horizontal_kernel_width, vertical_kernel_height, w, h, "normal"
            )
            
            if len(h_pos_normal) > len(h_positions):
                h_positions = h_pos_normal
            if len(v_pos_normal) > len(v_positions):
                v_positions = v_pos_normal
        
        self.logger.info(f"Found {len(h_positions)} horizontal lines at positions: {h_positions}")
        self.logger.info(f"Found {len(v_positions)} vertical lines at positions: {v_positions}")
        
        # Ensure table boundaries are included
        h_positions = self._ensure_boundary_lines(h_positions, h)
        v_positions = self._ensure_boundary_lines(v_positions, w)
        
        # Save debug visualization of detected lines
        if self.debug_mode:
            debug_lines = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
            for y_pos in h_positions:
                cv2.line(debug_lines, (0, y_pos), (w, y_pos), (0, 255, 0), 2)
            for x_pos in v_positions:
                cv2.line(debug_lines, (x_pos, 0), (x_pos, h), (255, 0, 0), 2)
            # Draw intersection points
            for y_pos in h_positions:
                for x_pos in v_positions:
                    cv2.circle(debug_lines, (x_pos, y_pos), 4, (0, 0, 255), -1)
            self.save_debug_image(debug_lines, "line_intersections", "08")
        
        # Create cells from line intersections
        cells = self._create_cells_from_intersections(
            h_positions, v_positions, table_bounds, w, h
        )
        
        return cells

    def _detect_lines_from_threshold(self, thresh_img, h_kernel_width, v_kernel_height, w, h, method_name):
        """
        Detect horizontal and vertical lines from a thresholded image.
        
        Args:
            thresh_img: Thresholded binary image
            h_kernel_width: Width of horizontal line detection kernel
            v_kernel_height: Height of vertical line detection kernel
            w: Width of image
            h: Height of image
            method_name: Name for debug output
            
        Returns:
            tuple: (horizontal_positions, vertical_positions)
        """
        self.save_debug_image(thresh_img, f"threshold_{method_name}", "07")
        
        # Create kernels for line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        self.save_debug_image(horizontal_lines, f"horizontal_lines_{method_name}", "07a")
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        self.save_debug_image(vertical_lines, f"vertical_lines_{method_name}", "07b")
        
        # Find line positions
        h_positions = self._find_line_positions(horizontal_lines, axis='horizontal', dimension=h)
        v_positions = self._find_line_positions(vertical_lines, axis='vertical', dimension=w)
        
        return h_positions, v_positions

    def _find_line_positions(self, line_img, axis='horizontal', dimension=None):
        """
        Find the positions of detected lines using projection profile analysis.
        
        Args:
            line_img: Binary image with detected lines
            axis: 'horizontal' or 'vertical'
            dimension: Size along the perpendicular axis for threshold calculation
            
        Returns:
            list: Sorted list of line positions (y-coordinates for horizontal, 
                  x-coordinates for vertical)
        """
        positions = []
        
        if axis == 'horizontal':
            # Sum along rows to find horizontal lines (project onto y-axis)
            projection = np.sum(line_img, axis=1).astype(np.float64)
        else:
            # Sum along columns to find vertical lines (project onto x-axis)
            projection = np.sum(line_img, axis=0).astype(np.float64)
        
        max_val = np.max(projection)
        if max_val == 0:
            return positions
        
        # Normalize projection
        projection = projection / max_val
        
        # Use a low threshold to catch faint lines
        threshold = 0.05
        
        # Find indices above threshold
        line_indices = np.where(projection > threshold)[0]
        
        if len(line_indices) == 0:
            return positions
        
        # Group consecutive indices (lines might be multiple pixels thick)
        # Use adaptive tolerance based on image dimension
        tolerance = max(3, (dimension or len(projection)) // 50)
        
        groups = self._group_consecutive_indices(line_indices, tolerance)
        
        # Take the median of each group as the line position
        positions = [int(np.median(group)) for group in groups]
        
        return sorted(positions)

    def _group_consecutive_indices(self, indices, tolerance):
        """
        Group consecutive indices within a tolerance.
        
        Args:
            indices: Array of indices
            tolerance: Maximum gap between consecutive indices in same group
            
        Returns:
            list: List of grouped indices
        """
        if len(indices) == 0:
            return []
        
        groups = []
        current_group = [indices[0]]
        
        for idx in indices[1:]:
            if idx - current_group[-1] <= tolerance:
                current_group.append(idx)
            else:
                groups.append(current_group)
                current_group = [idx]
        groups.append(current_group)
        
        return groups

    def _ensure_boundary_lines(self, positions, max_dimension, margin_ratio=0.02):
        """
        Ensure that boundary lines (at 0 and max_dimension) are included.
        
        Args:
            positions: List of line positions
            max_dimension: Maximum coordinate (width or height)
            margin_ratio: Ratio of dimension to consider as margin
            
        Returns:
            list: Updated positions with boundaries included
        """
        positions = list(positions)  # Make a copy
        margin = max(5, int(max_dimension * margin_ratio))
        
        # Add start boundary if not present
        if len(positions) == 0 or positions[0] > margin:
            positions.insert(0, 0)
        
        # Add end boundary if not present
        if len(positions) == 0 or positions[-1] < max_dimension - margin:
            positions.append(max_dimension)
        
        return sorted(set(positions))  # Remove duplicates and sort

    def _create_cells_from_intersections(self, h_positions, v_positions, table_bounds, w, h):
        """
        Create cell rectangles from the intersections of horizontal and vertical lines.
        
        Args:
            h_positions: List of y-coordinates for horizontal lines
            v_positions: List of x-coordinates for vertical lines
            table_bounds: Dictionary with table boundary coordinates
            w: Width of table region
            h: Height of table region
            
        Returns:
            list: List of cell dictionaries
        """
        cells = []
        
        # Calculate minimum cell dimensions (0.5% of table size or 5px minimum)
        min_cell_width = max(5, int(w * 0.005))
        min_cell_height = max(5, int(h * 0.005))
        
        # Iterate through consecutive line pairs to form cells
        for row_idx in range(len(h_positions) - 1):
            for col_idx in range(len(v_positions) - 1):
                y1 = h_positions[row_idx]
                y2 = h_positions[row_idx + 1]
                x1 = v_positions[col_idx]
                x2 = v_positions[col_idx + 1]
                
                cell_width = x2 - x1
                cell_height = y2 - y1
                
                # Filter cells that are too small
                if cell_width >= min_cell_width and cell_height >= min_cell_height:
                    cell = {
                        "x1": x1 + table_bounds["x1"],
                        "y1": y1 + table_bounds["y1"],
                        "x2": x2 + table_bounds["x1"],
                        "y2": y2 + table_bounds["y1"],
                        "width": cell_width,
                        "height": cell_height,
                        "row": row_idx,
                        "col": col_idx,
                    }
                    cells.append(cell)
        
        return cells

    def _detect_cells_by_contours(self, binary_img, table_bounds, w, h):
        """Detect cells using contour detection with multiple preprocessing (fallback method)."""
        cells = []
        
        # Scale morphology kernel with table size
        morph_kernel_size = max(2, min(w, h) // 100)
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        # Apply gentle morphology before processing
        morphed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
        
        preprocessing_methods = [
            ("morphed", morphed),
            ("inverted", cv2.bitwise_not(morphed)),
            ("edges_50_150", cv2.Canny(morphed, 50, 150)),
            ("edges_30_200", cv2.Canny(morphed, 30, 200)),
        ]
        
        # Scale area thresholds with table size
        min_area = max(100, (w * h) * 0.001)
        max_area = (w * h) * 0.95
        
        # Scale minimum dimensions
        min_width = max(10, int(w * 0.01))
        min_height = max(10, int(h * 0.01))
        
        for method_name, processed in preprocessing_methods:
            contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x1, y1, width, height = cv2.boundingRect(contour)
                    
                    if width >= min_width and height >= min_height:
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

        # Filter valid cells
        valid_cells = [
            cell for cell in cells
            if (cell["x2"] > cell["x1"] and cell["y2"] > cell["y1"]
                and (cell["x2"] - cell["x1"]) >= 5
                and (cell["y2"] - cell["y1"]) >= 5)
        ]

        # Sort by area (largest first)
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