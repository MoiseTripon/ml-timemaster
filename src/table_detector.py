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
        
        # Scale kernel size based on image dimensions
        img_h, img_w = gray.shape[:2]
        morph_kernel_size = max(3, min(img_w, img_h) // 200)
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        # Apply morphology BEFORE Canny to enhance borders
        morphed = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
        
        self.save_debug_image(morphed, "morphed_gradient", "02")
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Multiple edge detection approaches
        edges1 = cv2.Canny(blurred, 50, 150, apertureSize=3)
        edges2 = cv2.Canny(morphed, 30, 100, apertureSize=3)
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Connect nearby edges
        dilation_kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, dilation_kernel, iterations=1)
        edges = cv2.erode(edges, dilation_kernel, iterations=1)
        
        self.save_debug_image(edges, "edges_combined", "03")
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.logger.info(f"Found {len(contours)} contours")
        
        if not contours:
            # Fallback without morphology
            self.logger.warning("No contours found with morphology, trying without")
            edges_fallback = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.logger.error("No contours found in the image")
                raise ValueError("No table detected in the image")
        
        # Find the largest contour
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        
        # Validate detected bounds
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
        
        # Save debug visualization
        if self.debug_mode:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
            self.save_debug_image(debug_img, "table_bounds", "04")
        
        return table_bounds

    def detect_cells(self, img, table_bounds):
        """
        Detect individual cells within the table, supporting colspan/rowspan.
        
        Args:
            img (numpy.ndarray): Input image
            table_bounds (dict): Table boundaries
            
        Returns:
            list: List of cell dictionaries with bounds and dimensions
        """
        self.logger.info("="*50)
        self.logger.info("Starting cell detection (with colspan/rowspan support)")
        
        # Extract table region
        x, y = table_bounds["x1"], table_bounds["y1"]
        w = table_bounds["x2"] - table_bounds["x1"]
        h = table_bounds["y2"] - table_bounds["y1"]
        
        table_region = img[y:y+h, x:x+w]
        self.save_debug_image(table_region, "table_region", "05")
        
        # Convert to grayscale if needed
        if len(table_region.shape) == 3:
            gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = table_region.copy()
        
        # Detect cells using line structure analysis
        cells = self._detect_cells_from_line_structure(gray, table_bounds, w, h)
        self.logger.info(f"Line structure method found {len(cells)} cells")
        
        # Fallback to contour method if needed
        if len(cells) < 2:
            self.logger.info("Falling back to contour-based detection")
            cells_fallback = self._detect_cells_by_contours(gray, table_bounds, w, h)
            cells.extend(cells_fallback)
            cells = self._remove_duplicate_cells(cells)
        
        self.logger.info(f"Total cells detected: {len(cells)}")
        
        # Save debug visualization
        if self.debug_mode:
            debug_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for i, cell in enumerate(cells):
                x1 = cell['x1'] - table_bounds['x1']
                y1 = cell['y1'] - table_bounds['y1']
                x2 = cell['x2'] - table_bounds['x1']
                y2 = cell['y2'] - table_bounds['y1']
                
                color = ((i * 37) % 255, (i * 67) % 255, (i * 97) % 255)
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
                # Add cell number
                cv2.putText(debug_img, str(i), (x1 + 5, y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            self.save_debug_image(debug_img, "detected_cells", "10")
        
        return cells

    def _detect_cells_from_line_structure(self, gray, table_bounds, w, h):
        """
        Detect cells by finding actual line structure and extracting cell contours.
        This method respects colspan/rowspan by only detecting lines that actually exist.
        
        Args:
            gray: Grayscale image of the table region
            table_bounds: Dictionary with table boundary coordinates
            w: Width of the table region
            h: Height of the table region
            
        Returns:
            list: List of detected cell dictionaries
        """
        # Scale kernel sizes with table dimensions
        horizontal_kernel_width = min(max(int(w * 0.05), 20), 100)
        vertical_kernel_height = min(max(int(h * 0.05), 20), 100)
        
        self.logger.debug(f"Using horizontal kernel width: {horizontal_kernel_width}")
        self.logger.debug(f"Using vertical kernel height: {vertical_kernel_height}")
        
        # Try different preprocessing methods
        best_cells = []
        
        # Method 1: OTSU threshold
        _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        for idx, thresh in enumerate([thresh1, thresh2]):
            # Ensure lines are white on black background
            if np.mean(thresh) > 127:
                thresh_inv = cv2.bitwise_not(thresh)
            else:
                thresh_inv = thresh
            
            self.save_debug_image(thresh_inv, f"threshold_method_{idx}", f"06_{idx}")
            
            # Detect horizontal and vertical lines
            horizontal_lines, vertical_lines = self._detect_lines(
                thresh_inv, horizontal_kernel_width, vertical_kernel_height
            )
            
            self.save_debug_image(horizontal_lines, f"horizontal_lines_{idx}", f"06a_{idx}")
            self.save_debug_image(vertical_lines, f"vertical_lines_{idx}", f"06b_{idx}")
            
            # Combine lines to form table structure
            table_structure = cv2.add(horizontal_lines, vertical_lines)
            
            # Clean up the structure
            cleanup_kernel = np.ones((3, 3), np.uint8)
            table_structure = cv2.dilate(table_structure, cleanup_kernel, iterations=1)
            table_structure = cv2.morphologyEx(table_structure, cv2.MORPH_CLOSE, cleanup_kernel)
            
            self.save_debug_image(table_structure, f"table_structure_{idx}", f"07_{idx}")
            
            # Find cells as contours in the inverted structure
            cells = self._extract_cells_from_structure(
                table_structure, table_bounds, w, h
            )
            
            self.logger.info(f"Method {idx}: Found {len(cells)} cells")
            
            if len(cells) > len(best_cells):
                best_cells = cells
        
        return best_cells

    def _detect_lines(self, binary_img, horizontal_kernel_width, vertical_kernel_height):
        """
        Detect horizontal and vertical lines in the image.
        
        Args:
            binary_img: Binary image with lines as white
            horizontal_kernel_width: Width of horizontal detection kernel
            vertical_kernel_height: Height of vertical detection kernel
            
        Returns:
            tuple: (horizontal_lines, vertical_lines) as binary images
        """
        # Create kernels
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (horizontal_kernel_width, 1)
        )
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (1, vertical_kernel_height)
        )
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        horizontal_lines = cv2.morphologyEx(horizontal_lines, cv2.MORPH_CLOSE, horizontal_kernel, iterations=1)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        vertical_lines = cv2.morphologyEx(vertical_lines, cv2.MORPH_CLOSE, vertical_kernel, iterations=1)
        
        return horizontal_lines, vertical_lines

    def _extract_cells_from_structure(self, table_structure, table_bounds, w, h):
        """
        Extract individual cells from the table structure using contour detection.
        This naturally handles merged cells (colspan/rowspan) because they won't
        have internal dividing lines.
        
        Args:
            table_structure: Binary image with table lines
            table_bounds: Dictionary with table boundary coordinates
            w: Width of table region
            h: Height of table region
            
        Returns:
            list: List of detected cell dictionaries
        """
        cells = []
        
        # Invert the structure to find cells as white regions
        structure_inv = cv2.bitwise_not(table_structure)
        
        self.save_debug_image(structure_inv, "structure_inverted", "08")
        
        # Find contours of cells
        contours, hierarchy = cv2.findContours(
            structure_inv, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours to find actual cells
        min_cell_area = (w * h) * 0.001  # At least 0.1% of table area
        max_cell_area = (w * h) * 0.95   # At most 95% of table area
        min_cell_width = max(10, int(w * 0.01))
        min_cell_height = max(10, int(h * 0.01))
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            
            if min_cell_area <= area <= max_cell_area:
                x1, y1, cell_w, cell_h = cv2.boundingRect(contour)
                
                # Filter by minimum dimensions
                if cell_w >= min_cell_width and cell_h >= min_cell_height:
                    # Verify this is a valid cell (rectangular-ish shape)
                    rect_area = cell_w * cell_h
                    rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    # Accept cells that are reasonably rectangular (> 70% fill)
                    if rectangularity > 0.7:
                        cell = {
                            "x1": x1 + table_bounds["x1"],
                            "y1": y1 + table_bounds["y1"],
                            "x2": x1 + cell_w + table_bounds["x1"],
                            "y2": y1 + cell_h + table_bounds["y1"],
                            "width": cell_w,
                            "height": cell_h,
                            "area": area,
                        }
                        cells.append(cell)
        
        # Sort cells by position (top to bottom, left to right)
        cells.sort(key=lambda c: (c["y1"], c["x1"]))
        
        # Assign row and column indices based on position clustering
        cells = self._assign_grid_positions(cells)
        
        return cells

    def _assign_grid_positions(self, cells, tolerance_ratio=0.3):
        """
        Assign row and column indices to cells based on their positions.
        This handles merged cells by assigning based on top-left corner.
        
        Args:
            cells: List of cell dictionaries
            tolerance_ratio: Ratio of average cell dimension for clustering tolerance
            
        Returns:
            list: Cells with 'row' and 'col' indices added
        """
        if not cells:
            return cells
        
        # Get unique y positions (rows) and x positions (columns)
        y_positions = [c["y1"] for c in cells]
        x_positions = [c["x1"] for c in cells]
        
        # Calculate tolerances based on cell sizes
        avg_height = np.mean([c["height"] for c in cells]) if cells else 20
        avg_width = np.mean([c["width"] for c in cells]) if cells else 20
        
        y_tolerance = avg_height * tolerance_ratio
        x_tolerance = avg_width * tolerance_ratio
        
        # Cluster y positions into rows
        row_positions = self._cluster_positions(y_positions, y_tolerance)
        col_positions = self._cluster_positions(x_positions, x_tolerance)
        
        # Assign row and column to each cell
        for cell in cells:
            # Find closest row
            cell["row"] = self._find_closest_cluster(cell["y1"], row_positions)
            cell["col"] = self._find_closest_cluster(cell["x1"], col_positions)
        
        return cells

    def _cluster_positions(self, positions, tolerance):
        """
        Cluster positions that are within tolerance of each other.
        
        Args:
            positions: List of positions
            tolerance: Maximum distance for same cluster
            
        Returns:
            list: Sorted list of cluster centers
        """
        if not positions:
            return []
        
        sorted_positions = sorted(set(positions))
        clusters = []
        current_cluster = [sorted_positions[0]]
        
        for pos in sorted_positions[1:]:
            if pos - current_cluster[-1] <= tolerance:
                current_cluster.append(pos)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [pos]
        clusters.append(np.mean(current_cluster))
        
        return sorted(clusters)

    def _find_closest_cluster(self, value, clusters):
        """
        Find the index of the closest cluster to a value.
        
        Args:
            value: Position value
            clusters: List of cluster centers
            
        Returns:
            int: Index of closest cluster
        """
        if not clusters:
            return 0
        
        distances = [abs(value - c) for c in clusters]
        return distances.index(min(distances))

    def _detect_cells_by_contours(self, gray, table_bounds, w, h):
        """
        Fallback method: Detect cells using contour detection with multiple preprocessing.
        
        Args:
            gray: Grayscale image of table region
            table_bounds: Dictionary with table boundary coordinates
            w: Width of table region
            h: Height of table region
            
        Returns:
            list: List of detected cell dictionaries
        """
        cells = []
        
        # Scale morphology kernel with table size
        morph_kernel_size = max(3, min(w, h) // 50)
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, 
            (morph_kernel_size, morph_kernel_size)
        )
        
        # Apply morphology before edge detection
        morphed = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
        
        preprocessing_methods = [
            ("original", gray),
            ("morphed", morphed),
            ("inverted", cv2.bitwise_not(gray)),
            ("edges_50_150", cv2.Canny(gray, 50, 150)),
            ("edges_30_200", cv2.Canny(morphed, 30, 200)),
        ]
        
        # Scale area thresholds with table size
        min_area = (w * h) * 0.001
        max_area = (w * h) * 0.95
        
        for method_name, processed in preprocessing_methods:
            contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area <= area <= max_area:
                    x1, y1, cell_w, cell_h = cv2.boundingRect(contour)
                    
                    # Check rectangularity
                    rect_area = cell_w * cell_h
                    rectangularity = area / rect_area if rect_area > 0 else 0
                    
                    if rectangularity > 0.5:  # More lenient for fallback
                        cell = {
                            "x1": x1 + table_bounds["x1"],
                            "y1": y1 + table_bounds["y1"],
                            "x2": x1 + cell_w + table_bounds["x1"],
                            "y2": y1 + cell_h + table_bounds["y1"],
                            "width": cell_w,
                            "height": cell_h,
                        }
                        cells.append(cell)
        
        return cells

    def _remove_duplicate_cells(self, cells, overlap_threshold=0.7):
        """
        Remove duplicate cells that overlap significantly.
        
        Args:
            cells: List of cell dictionaries
            overlap_threshold: Overlap ratio threshold for considering duplicates
            
        Returns:
            list: Deduplicated list of cells
        """
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

        # Re-sort by position
        unique_cells.sort(key=lambda c: (c.get("row", c["y1"]), c.get("col", c["x1"])))

        return unique_cells