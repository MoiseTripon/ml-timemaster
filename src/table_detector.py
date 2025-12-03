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
        
        # Detect edges
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        self.save_debug_image(edges, "edges", "02")
        
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
        self.save_debug_image(table_region, "table_region", "04")
        
        # Convert to grayscale if needed
        if len(table_region.shape) == 3:
            binary_img = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        else:
            binary_img = table_region.copy()
        
        # Detect cells using multiple methods
        all_cells = []
        
        cells_method1 = self._detect_cells_by_lines(binary_img, table_bounds, w, h)
        self.logger.info(f"Line method found {len(cells_method1)} cells")
        all_cells.extend(cells_method1)
        
        cells_method2 = self._detect_cells_by_contours(binary_img, table_bounds, w, h)
        self.logger.info(f"Contour method found {len(cells_method2)} cells")
        all_cells.extend(cells_method2)
        
        # Remove duplicates
        cells = self._remove_duplicate_cells(all_cells)
        self.logger.info(f"Total cells after deduplication: {len(cells)}")
        
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

    def _detect_cells_by_lines(self, binary_img, table_bounds, w, h):
        """Detect cells using horizontal and vertical line detection."""
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
        _, table_structure = cv2.threshold(table_structure, 0, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        min_area = 100
        max_area = (w * h) * 0.5
        
        for contour in contours:
            area = cv2.contourArea(contour)
            x1, y1, width, height = cv2.boundingRect(contour)
            
            if min_area <= area <= max_area and width >= 10 and height >= 10:
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

    def _detect_cells_by_contours(self, binary_img, table_bounds, w, h):
        """Detect cells using contour detection with multiple preprocessing."""
        cells = []
        
        preprocessing_methods = [
            ("original", binary_img),
            ("inverted", cv2.bitwise_not(binary_img)),
            ("edges_50_150", cv2.Canny(binary_img, 50, 150)),
            ("edges_30_200", cv2.Canny(binary_img, 30, 200)),
        ]
        
        min_area = (w * h) * 0.0001
        max_area = (w * h) * 0.99
        
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