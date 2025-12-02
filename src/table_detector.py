"""
Table detection module for ML Timemaster.
"""

import logging
import cv2
import numpy as np
import os

logger = logging.getLogger(__name__)

# Debug settings
DEBUG_MODE = True  # Set to True to enable debug visualizations
DEBUG_OUTPUT_DIR = "debug_output"


def save_debug_image(img, name, step=""):
    """Save an image for debugging purposes."""
    if not DEBUG_MODE:
        return
    
    if not os.path.exists(DEBUG_OUTPUT_DIR):
        os.makedirs(DEBUG_OUTPUT_DIR)
    
    filename = f"{DEBUG_OUTPUT_DIR}/{step}_{name}.png"
    cv2.imwrite(filename, img)
    logger.debug(f"Saved debug image: {filename}")


def detect_table_borders(img):
    """
    Detects the main table borders in the image.
    """
    logger.info("="*50)
    logger.info("Starting table border detection")
    logger.info(f"Input image shape: {img.shape}")
    logger.info(f"Input image dtype: {img.dtype}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted BGR to grayscale")
    else:
        binary = img.copy()
        logger.debug("Image already grayscale")
    
    save_debug_image(binary, "grayscale", "01")
    
    # Detect edges
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    save_debug_image(edges, "edges", "02")
    logger.debug(f"Edge detection complete. Non-zero pixels: {cv2.countNonZero(edges)}")
    
    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.info(f"Found {len(contours)} contours")
    
    if not contours:
        logger.error("No contours found in the image")
        raise ValueError("No table detected in the image")
    
    # Log contour areas
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        logger.debug(f"Contour {i}: area={area}")
    
    # Find the largest contour
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    table_bounds = {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
    
    logger.info(f"Detected table boundaries: {table_bounds}")
    logger.info(f"Table dimensions: {w}x{h} pixels")
    
    # Save debug visualization of table bounds
    if DEBUG_MODE:
        debug_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        save_debug_image(debug_img, "table_bounds", "03")
    
    return table_bounds


def detect_cells(img, table_bounds):
    """
    Detects individual cells within the table.
    """
    logger.info("="*50)
    logger.info("Starting cell detection")
    logger.info(f"Table bounds: {table_bounds}")
    
    # Extract table region
    x, y, w, h = (
        table_bounds["x1"],
        table_bounds["y1"],
        table_bounds["x2"] - table_bounds["x1"],
        table_bounds["y2"] - table_bounds["y1"],
    )
    logger.debug(f"Extracting table region: x={x}, y={y}, w={w}, h={h}")
    
    table_region = img[y : y + h, x : x + w]
    logger.debug(f"Table region shape: {table_region.shape}")
    
    save_debug_image(table_region, "table_region", "04")
    
    # Convert to grayscale if needed
    if len(table_region.shape) == 3:
        binary_img = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)
    else:
        binary_img = table_region.copy()
    
    save_debug_image(binary_img, "binary_img", "05")
    
    # Track all detected cells from different methods
    all_cells = []
    
    # Method 1: Line-based detection
    logger.info("Method 1: Line-based detection")
    cells_method1 = detect_cells_by_lines(binary_img, table_bounds, w, h)
    logger.info(f"Method 1 found {len(cells_method1)} cells")
    all_cells.extend(cells_method1)
    
    # Method 2: Contour-based detection
    logger.info("Method 2: Contour-based detection")
    cells_method2 = detect_cells_by_contours(binary_img, table_bounds, w, h)
    logger.info(f"Method 2 found {len(cells_method2)} cells")
    all_cells.extend(cells_method2)
    
    # Remove duplicates
    logger.info(f"Total cells before deduplication: {len(all_cells)}")
    cells = remove_duplicate_cells(all_cells)
    logger.info(f"Total cells after deduplication: {len(cells)}")
    
    # Log each detected cell
    for i, cell in enumerate(cells):
        logger.debug(
            f"Cell {i}: x1={cell['x1']}, y1={cell['y1']}, "
            f"x2={cell['x2']}, y2={cell['y2']}, "
            f"width={cell['width']}, height={cell['height']}"
        )
    
    # Save debug visualization
    if DEBUG_MODE:
        debug_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for i, cell in enumerate(cells):
            # Adjust coordinates to table region
            x1 = cell['x1'] - table_bounds['x1']
            y1 = cell['y1'] - table_bounds['y1']
            x2 = cell['x2'] - table_bounds['x1']
            y2 = cell['y2'] - table_bounds['y1']
            
            color = ((i * 37) % 255, (i * 67) % 255, (i * 97) % 255)
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug_img, str(i), (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        save_debug_image(debug_img, "detected_cells", "06")
    
    return cells


def detect_cells_by_lines(binary_img, table_bounds, w, h):
    """Detect cells using horizontal and vertical line detection."""
    logger.debug("Detecting cells by line detection method")
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    save_debug_image(horizontal_lines, "horizontal_lines", "05a")
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    save_debug_image(vertical_lines, "vertical_lines", "05b")
    
    # Combine lines
    table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)
    _, table_structure = cv2.threshold(table_structure, 0, 255, cv2.THRESH_BINARY)
    save_debug_image(table_structure, "table_structure", "05c")
    
    # Find contours
    contours, _ = cv2.findContours(table_structure, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Line method found {len(contours)} contours")
    
    cells = []
    min_area = 100
    max_area = (w * h) * 0.5
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        x1, y1, width, height = cv2.boundingRect(contour)
        
        logger.debug(f"Line contour {i}: area={area}, x={x1}, y={y1}, w={width}, h={height}")
        
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
            logger.debug(f"  -> Added as cell")
        else:
            logger.debug(f"  -> Rejected (area or size constraints)")
    
    return cells


def detect_cells_by_contours(binary_img, table_bounds, w, h):
    """Detect cells using contour detection with multiple preprocessing."""
    logger.debug("Detecting cells by contour detection method")
    
    cells = []
    
    # Try different preprocessing methods
    preprocessing_methods = [
        ("original", binary_img),
        ("inverted", cv2.bitwise_not(binary_img)),
        ("edges_50_150", cv2.Canny(binary_img, 50, 150)),
        ("edges_30_200", cv2.Canny(binary_img, 30, 200)),
    ]
    
    min_area = (w * h) * 0.0001
    max_area = (w * h) * 0.99
    
    for method_name, processed in preprocessing_methods:
        logger.debug(f"Trying preprocessing method: {method_name}")
        save_debug_image(processed, f"preprocess_{method_name}", "05d")
        
        contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"  Found {len(contours)} contours")
        
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


def remove_duplicate_cells(cells, tolerance=10):
    """Remove duplicate cells that overlap significantly."""
    logger.debug(f"Removing duplicates from {len(cells)} cells")
    
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

    # Filter out cells with invalid dimensions
    valid_cells = [
        cell for cell in cells
        if (cell["x2"] > cell["x1"] and cell["y2"] > cell["y1"]
            and (cell["x2"] - cell["x1"]) >= 5
            and (cell["y2"] - cell["y1"]) >= 5)
    ]
    logger.debug(f"Valid cells after size filter: {len(valid_cells)}")

    # Sort by area (largest first)
    valid_cells.sort(key=lambda c: (c["x2"] - c["x1"]) * (c["y2"] - c["y1"]), reverse=True)

    unique_cells = []
    for cell in valid_cells:
        is_duplicate = False
        for unique_cell in unique_cells:
            overlap = calculate_overlap(cell, unique_cell)
            if overlap > 0.7:
                logger.debug(f"Cell {cell} is duplicate of {unique_cell} (overlap={overlap:.2f})")
                is_duplicate = True
                break
        if not is_duplicate:
            unique_cells.append(cell)

    logger.info(f"Removed {len(valid_cells) - len(unique_cells)} duplicate cells")
    return unique_cells