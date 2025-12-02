"""
Table detection module for ML Timemaster.
Contains functions for detecting table borders, individual cells, and removing duplicate cells.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_table_borders(img):
    """
    Detects the main table borders in the image.
    
    Args:
        img (numpy.ndarray): Input grayscale or binary image containing the table.
    
    Returns:
        dict: Table boundary coordinates with keys 'x1', 'y1', 'x2', 'y2'.
        
    Raises:
        ValueError: If no table is detected in the image.
    """
    # Convert to grayscale if needed (for backward compatibility with tests)
    if len(img.shape) == 3:
        binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        binary = img.copy()

    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No table detected in the image")

    # Find the largest contour (assumed to be the table)
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    table_bounds = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}

    logger.info(f"Detected table boundaries: {table_bounds}")
    return table_bounds

def detect_cells(img, table_bounds):
    """
    Detects individual cells within the table using line detection.
    """
    # Extract table region
    x, y, w, h = (
        table_bounds["x1"],
        table_bounds["y1"],
        table_bounds["x2"] - table_bounds["x1"],
        table_bounds["y2"] - table_bounds["y1"],
    )
    table_region = img[y : y + h, x : x + w]

    # Convert to grayscale if needed
    if len(table_region.shape) == 3:
        gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = table_region.copy()

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Find horizontal lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Find vertical lines
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # Combine lines
    table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
    _, table_structure = cv2.threshold(table_structure, 0, 255, cv2.THRESH_BINARY)
    
    # Find contours of cells
    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter and process contours
    cells = []
    min_area = 100  # Minimum area in pixels
    max_area = (w * h) * 0.5  # Maximum area (50% of table area)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x1, y1, width, height = cv2.boundingRect(contour)
            # Skip very thin cells (likely lines, not actual cells)
            if width < 10 or height < 10:
                continue
                
            # Convert coordinates back to original image space
            cell = {
                "x1": x1 + table_bounds["x1"],
                "y1": y1 + table_bounds["y1"],
                "x2": x1 + width + table_bounds["x1"],
                "y2": y1 + height + table_bounds["y1"],
                "width": width,
                "height": height,
            }
            cells.append(cell)
    
    # If no cells found, try alternative method
    if len(cells) < 2:
        cells = detect_cells_by_connected_components(table_region, table_bounds)
    
    # Remove duplicates
    cells = remove_duplicate_cells(cells)
    
    logger.info(f"Detected {len(cells)} cells")
    return cells

def detect_cells_by_connected_components(table_region, table_bounds):
    """
    Alternative cell detection using connected components.
    """
    if len(table_region.shape) == 3:
        gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
    else:
        gray = table_region.copy()
    
    # Invert the image (make text/lines white, background black)
    inverted = cv2.bitwise_not(gray)
    
    # Apply threshold
    _, binary = cv2.threshold(inverted, 128, 255, cv2.THRESH_BINARY)
    
    # Apply morphological operations to connect text within cells
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    
    cells = []
    min_area = 500  # Minimum area for a cell
    
    for i in range(1, num_labels):  # Skip background (label 0)
        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area >= min_area and width >= 20 and height >= 20:
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

def remove_duplicate_cells(cells, tolerance=10):
    """
    Remove duplicate cells that overlap significantly.
    
    Args:
        cells (list): List of cell dictionaries.
        tolerance (int): Tolerance for cell similarity comparison.
    
    Returns:
        list: List of unique cells without significant overlaps.
    """
    if not cells:
        return []

    def calculate_overlap(cell1, cell2):
        """Calculate the overlap area between two cells."""
        # Calculate cell areas first
        cell1_area = (cell1["x2"] - cell1["x1"]) * (cell1["y2"] - cell1["y1"])
        cell2_area = (cell2["x2"] - cell2["x1"]) * (cell2["y2"] - cell2["y1"])

        # If either cell has zero or negative area, they can't overlap
        if cell1_area <= 0 or cell2_area <= 0:
            return 0.0

        # Calculate overlap coordinates
        x_left = max(cell1["x1"], cell2["x1"])
        y_top = max(cell1["y1"], cell2["y1"])
        x_right = min(cell1["x2"], cell2["x2"])
        y_bottom = min(cell1["y2"], cell2["y2"])

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
        if (
            cell1["x2"] <= cell1["x1"]
            or cell1["y2"] <= cell1["y1"]
            or cell2["x2"] <= cell2["x1"]
            or cell2["y2"] <= cell2["y1"]
        ):
            return False

        overlap = calculate_overlap(cell1, cell2)
        return overlap > 0.7  # 70% overlap threshold

    # Filter out cells with invalid dimensions first
    valid_cells = [
        cell
        for cell in cells
        if (
            cell["x2"] > cell["x1"]
            and cell["y2"] > cell["y1"]
            and (cell["x2"] - cell["x1"]) >= 5
            and (cell["y2"] - cell["y1"]) >= 5  # Minimum width of 5 pixels
        )  # Minimum height of 5 pixels
    ]

    # Sort cells by area (largest first)
    valid_cells.sort(
        key=lambda c: (c["x2"] - c["x1"]) * (c["y2"] - c["y1"]), reverse=True
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

    logger.info(f"Removed {len(valid_cells) - len(unique_cells)} duplicate cells")
    return unique_cells
