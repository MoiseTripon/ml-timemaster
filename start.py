import datetime
import logging
import os

import cv2
import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from src.cell_ocr import CellOCR
from src.grid_builder import organize_cells_into_grid
from src.image_preprocessor import ImagePreprocessor
from src.table_detector import detect_cells, detect_table_borders

logger = logging.getLogger(__name__)

def process_schedule_file(file_path):
    """
    Processes a schedule file (PDF or image) and extracts table data.
    
    Args:
        file_path (str): Path to the PDF or image file to process.
    
    Returns:
        dict: Comprehensive JSON structure with all table information including:
            - metadata: File information and processing timestamp
            - table: Table bounds, dimensions, grid structure, and individual cells
            - visualization: Information about the generated visualization file
    
    Raises:
        ValueError: If the file cannot be processed or no table is detected.
        Exception: For other processing errors.
    """
    preprocessor = ImagePreprocessor()
    ocr_processor = CellOCR()
    
    try:
        # Step 1: Preprocess the file
        processed_path, processed_img, original_img = preprocessor.preprocess(file_path)
        logger.info(f"Processed image size: {processed_img.shape}")

        # Step 2: Detect table borders (using processed image)
        table_bounds = detect_table_borders(processed_img)

        # Step 3: Detect individual cells (using processed image)
        cells = detect_cells(processed_img, table_bounds)

        cells = detect_cells(processed_img, table_bounds)
        logger.info(f"Detected {len(cells)} cells")

        # Debug visualization
        debug_visualize_detection(original_img, table_bounds, cells)

        if len(cells) == 0:
            logger.error("No cells detected in the table")
            raise ValueError("No cells were detected in the table")
        elif len(cells) == 1:
            logger.warning("Only one cell detected - this might be the entire table")
       
        # Step 4: Extract text from each cell
        result = {"table_bounds": table_bounds, "cells": []}

        for idx, cell in enumerate(cells):
            try:
                # Extract text from cell using OCR processor (using original image for better OCR)
                text = ocr_processor.extract_cell_text(original_img, cell)

                # Add cell to result
                cell_info = {
                    "id": idx,
                    "bounds": {
                        "x1": cell["x1"],
                        "y1": cell["y1"],
                        "x2": cell["x2"],
                        "y2": cell["y2"],
                    },
                    "dimensions": {"width": cell["width"], "height": cell["height"]},
                    "text": text,
                }
                result["cells"].append(cell_info)

            except Exception as ocr_error:
                logger.error(f"Error processing cell {idx}: {str(ocr_error)}")

        # Step 5: Organize cells into grid structure
        grid, num_rows, num_cols = organize_cells_into_grid(result["cells"])

        # Step 6: Create comprehensive response object
        response = {
            "metadata": {
                "file_name": os.path.basename(file_path),
                "file_type": "pdf" if file_path.lower().endswith(".pdf") else "image",
                "image_size": {
                    "width": original_img.shape[1],
                    "height": original_img.shape[0],
                },
                "processing_timestamp": str(datetime.datetime.now()),
            },
            "table": {
                "bounds": table_bounds,
                "dimensions": {"rows": num_rows, "columns": num_cols},
                "grid": grid,
                "cells": result["cells"],
            },
            "visualization": {
                "output_file": f"detected_table_{os.path.splitext(os.path.basename(file_path))[0]}.png"
            },
        }

        # Step 7: Create a visualization image (using original image for better visualization)
        vis_img = original_img.copy()

        # Draw table boundary in red
        cv2.rectangle(
            vis_img,
            (table_bounds["x1"], table_bounds["y1"]),
            (table_bounds["x2"], table_bounds["y2"]),
            (0, 0, 255),
            3,
        )

        # Generate random colors for cells
        np.random.seed(42)  # For reproducible colors
        colors = np.random.randint(0, 255, size=(len(result["cells"]), 3)).tolist()

        # Draw cells with random colors
        for cell, color in zip(result["cells"], colors):
            cv2.rectangle(
                vis_img,
                (cell["bounds"]["x1"], cell["bounds"]["y1"]),
                (cell["bounds"]["x2"], cell["bounds"]["y2"]),
                color,
                2,
            )

        # Save visualization
        cv2.imwrite(response["visualization"]["output_file"], vis_img)
        logger.info(f"Saved visualization to {response['visualization']['output_file']}")

        # Display processed image
        plt.figure(figsize=(15, 12))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title(f"Detected Table and Cells ({num_rows}x{num_cols} grid)")
        plt.show()

        return response

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        if 'processed_path' in locals() and os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except OSError:
                pass

def debug_visualize_detection(img, table_bounds, cells):
    """Create a debug visualization to see what's being detected."""
    debug_img = img.copy()
    
    # Draw table bounds in red
    cv2.rectangle(
        debug_img,
        (table_bounds["x1"], table_bounds["y1"]),
        (table_bounds["x2"], table_bounds["y2"]),
        (0, 0, 255),
        3
    )
    
    # Draw each cell in a different color
    for i, cell in enumerate(cells):
        color = (
            (i * 37) % 255,
            (i * 67) % 255,
            (i * 97) % 255
        )
        cv2.rectangle(
            debug_img,
            (cell["x1"], cell["y1"]),
            (cell["x2"], cell["y2"]),
            color,
            2
        )
        # Add cell number
        cv2.putText(
            debug_img,
            str(i),
            (cell["x1"] + 5, cell["y1"] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    cv2.imwrite("debug_detection.png", debug_img)
    logger.info(f"Saved debug visualization with {len(cells)} cells")

# Example usage
if __name__ == "__main__":
    file_path = "i_a.pdf"  
    try:
        result = process_schedule_file(file_path)
        if not result["table"]["grid"]:
            print("No cells were detected.")
            exit(1)

        print("\nExtracted Table Data:")
        print(f"File: {result['metadata']['file_name']}")
        print(
            f"Image size: {result['metadata']['image_size']['width']}x{result['metadata']['image_size']['height']}"
        )
        print(
            f"Table dimensions: {result['table']['dimensions']['rows']}x{result['table']['dimensions']['columns']}"
        )
        print("\nTable structure:")
        for row in result["table"]["grid"]:
            print(row)

    except ValueError as ve:
        print(f"Error: {str(ve)}")
        if "pdf" in str(ve).lower():
            print("\nEnsure the following for PDFs:")
            print("1. Poppler is installed")
            print("2. The PDF is not corrupted")
            print("3. The file is accessible")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
