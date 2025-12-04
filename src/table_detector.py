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
            binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            binary = img.copy()
        
        self.save_debug_image(binary, "grayscale", "01")
        
        # Apply morphological operations BEFORE edge detection
        # This helps clean up the image and enhance edges
        kernel_size = max(3, min(binary.shape) // 100)  # Scale with image size
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        
        # Close operation to fill small gaps
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        self.save_debug_image(morphed, "morphology_close", "02a")
        
        # Gradient to enhance edges
        gradient = cv2.morphologyEx(morphed, cv2.MORPH_GRADIENT, kernel)
        self.save_debug_image(gradient, "morphology_gradient", "02b")
        
        # Now apply Canny on the preprocessed image
        edges = cv2.Canny(gradient, 50, 150, apertureSize=3)
        self.save_debug_image(edges, "edges", "02c")
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"Found {len(contours)} contours")
        
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
            debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            self.save_debug_image(debug_img, "table_bounds", "03")
        
        return table_bounds

    def detect_cells(self, img, table_bounds):
        """
        Detect individual cells within the table using line intersection method.
        
        Args:
            img (numpy.ndarray): Input image
            table_bounds (dict): Table boundaries
            
        Returns:
            list: List of cell dictionaries with bounds and dimensions
        """
        self.logger.info("="*50)
        self.logger.info("Starting cell detection using line intersection")
        
        # Extract table region
        x, y = table_bounds["x1"], table_bounds["y1"]
        w = table_bounds["x2"] - table_bounds["x1"]
        h = table_bounds["y2"] - table_bounds["y1"]
        
        table_region = img[y:y+h, x:x+w]
        self.save_debug_image(table_region, "table_region", "04")
        
        # Convert to grayscale if needed
        if len(table_region.shape) == 3:
            binary_img = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        else:
            binary_img = table_region.copy()
        
        # Detect cells using line intersection method
        cells = self._detect_cells_by_line_intersection(binary_img, table_bounds, w, h)
        self.logger.info(f"Line intersection method found {len(cells)} cells")
        
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
            self.save_debug_image(debug_img, "detected_cells", "06")
        
        return cells

    def _detect_cells_by_line_intersection(self, binary_img, table_bounds, w, h):
        """
        Detect cells using horizontal and vertical line detection with intersection.
        Lines are found, then intersected to form rectangular cells.
        """
        # Calculate adaptive kernel sizes based on table dimensions
        # Kernels should be proportional to table size
        horizontal_kernel_width = max(40, w // 10)  # Scale with table width
        vertical_kernel_height = max(40, h // 10)   # Scale with table height
        
        self.logger.info(f"Using adaptive kernel sizes - Horizontal: {horizontal_kernel_width}, Vertical: {vertical_kernel_height}")
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_width, 1))
        horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        self.save_debug_image(horizontal_lines, "horizontal_lines", "05a")
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_height))
        vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        self.save_debug_image(vertical_lines, "vertical_lines", "05b")
        
        # Extract line positions
        horizontal_positions = self._extract_line_positions(horizontal_lines, axis='horizontal')
        vertical_positions = self._extract_line_positions(vertical_lines, axis='vertical')
        
        self.logger.info(f"Found {len(horizontal_positions)} horizontal lines and {len(vertical_positions)} vertical lines")
        
        # Add table boundaries if not detected
        if 0 not in horizontal_positions:
            horizontal_positions.insert(0, 0)
        if h not in horizontal_positions:
            horizontal_positions.append(h)
        if 0 not in vertical_positions:
            vertical_positions.insert(0, 0)
        if w not in vertical_positions:
            vertical_positions.append(w)
        
        # Sort positions
        horizontal_positions = sorted(set(horizontal_positions))
        vertical_positions = sorted(set(vertical_positions))
        
        # Create cells from line intersections
        cells = []
        for i in range(len(horizontal_positions) - 1):
            for j in range(len(vertical_positions) - 1):
                y1 = horizontal_positions[i]
                y2 = horizontal_positions[i + 1]
                x1 = vertical_positions[j]
                x2 = vertical_positions[j + 1]
                
                # Filter out very small cells (likely noise)
                min_cell_width = max(10, w // 50)   # Adaptive minimum size
                min_cell_height = max(10, h // 50)  # Adaptive minimum size
                
                if (x2 - x1) >= min_cell_width and (y2 - y1) >= min_cell_height:
                    cell = {
                        "x1": x1 + table_bounds["x1"],
                        "y1": y1 + table_bounds["y1"],
                        "x2": x2 + table_bounds["x1"],
                        "y2": y2 + table_bounds["y1"],
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                    cells.append(cell)
        
        return cells

    def _extract_line_positions(self, line_image, axis='horizontal', min_gap=5):
        """
        Extract positions of lines from a binary image.
        
        Args:
            line_image: Binary image with detected lines
            axis: 'horizontal' or 'vertical'
            min_gap: Minimum gap between consecutive lines to consider them separate
            
        Returns:
            list: Sorted list of line positions
        """
        positions = []
        
        if axis == 'horizontal':
            # Sum pixels across columns for horizontal lines
            projection = np.sum(line_image, axis=1)
        else:
            # Sum pixels across rows for vertical lines
            projection = np.sum(line_image, axis=0)
        
        # Find peaks in projection (line positions)
        threshold = np.max(projection) * 0.1  # Consider positions with at least 10% of max intensity
        line_detected = projection > threshold
        
        # Find continuous regions
        in_line = False
        start_pos = 0
        
        for i, is_line in enumerate(line_detected):
            if is_line and not in_line:
                in_line = True
                start_pos = i
            elif not is_line and in_line:
                in_line = False
                # Use the middle of the line region
                middle_pos = (start_pos + i - 1) // 2
                
                # Check if this line is far enough from the previous one
                if not positions or abs(middle_pos - positions[-1]) >= min_gap:
                    positions.append(middle_pos)
        
        # Handle case where line extends to the edge
        if in_line:
            middle_pos = (start_pos + len(line_detected) - 1) // 2
            if not positions or abs(middle_pos - positions[-1]) >= min_gap:
                positions.append(middle_pos)
        
        return sorted(positions)

    def _remove_duplicate_cells(self, cells, tolerance=10):
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

        # Sort by area
        valid_cells.sort(key=lambda c: (c["x2"] - c["x1"]) * (c["y2"] - c["y1"]), reverse=True)

        unique_cells = []
        for cell in valid_cells:
            is_duplicate = False
            for unique_cell in unique_cells:
                overlap = calculate_overlap(cell, unique_cell)
                if overlap > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_cells.append(cell)

        return unique_cells