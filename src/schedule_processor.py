"""
Schedule processing module for ML Timemaster.
Contains the ScheduleProcessor class for orchestrating the entire processing pipeline.
"""

import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .cell_ocr import CellOCR
from .image_preprocessor import ImagePreprocessor
from .table_analyzer import TableAnalyzer


class ScheduleProcessor:
    """
    Main processor that orchestrates the entire schedule extraction pipeline.
    Combines preprocessing, table detection, OCR, and visualization.
    """

    def __init__(
        self,
        dpi=500,
        overlap_threshold=0.7,
        min_cell_size=5,
        rotation_confidence_threshold=70.0,
        minimum_confidence_threshold=50.0,
        high_confidence_threshold=90.0,
    ):
        """
        Initialize the ScheduleProcessor.

        Args:
            dpi (int): DPI resolution for PDF to image conversion.
            overlap_threshold (float): Threshold for considering cells as duplicates.
            min_cell_size (int): Minimum width/height in pixels for valid cells.
            rotation_confidence_threshold (float): Threshold below which rotations are attempted.
            minimum_confidence_threshold (float): Minimum confidence required to accept text.
            high_confidence_threshold (float): Threshold above which text is immediately accepted.
        """
        self.preprocessor = ImagePreprocessor(dpi=dpi)
        self.table_analyzer = TableAnalyzer(
            overlap_threshold=overlap_threshold, min_cell_size=min_cell_size
        )
        self.ocr = CellOCR(
            rotation_confidence_threshold=rotation_confidence_threshold,
            minimum_confidence_threshold=minimum_confidence_threshold,
            high_confidence_threshold=high_confidence_threshold,
        )

    def process(self, file_path):
        """
        Processes a schedule file (PDF or image) and extracts table data.

        Args:
            file_path (str): Path to the PDF or image file.

        Returns:
            dict: Comprehensive JSON structure with all table information including:
                - metadata: File information and processing details
                - table: Table bounds, dimensions, grid structure, and cells
                - visualization: Output file information

        Raises:
            Exception: If any step of the processing fails.
        """
        processed_path = None
        try:
            # Step 1: Preprocess the file
            processed_path, original_img = self.preprocessor.preprocess(file_path)
            print(f"Original image size: {original_img.shape}")

            # Step 2: Detect table borders
            table_bounds = self.table_analyzer.detect_table_borders(original_img)

            # Step 3: Detect individual cells
            cells = self.table_analyzer.detect_cells(original_img, table_bounds)

            # Step 4: Extract text from each cell
            result = {"table_bounds": table_bounds, "cells": []}

            for idx, cell in enumerate(cells):
                try:
                    # Extract text from cell
                    text = self.ocr.extract_cell_text(original_img, cell)

                    # Add cell to result
                    cell_info = {
                        "id": idx,
                        "bounds": {
                            "x1": cell["x1"],
                            "y1": cell["y1"],
                            "x2": cell["x2"],
                            "y2": cell["y2"],
                        },
                        "dimensions": {
                            "width": cell["width"],
                            "height": cell["height"],
                        },
                        "text": text,
                    }
                    result["cells"].append(cell_info)

                except Exception as ocr_error:
                    print(f"Error processing cell {idx}: {str(ocr_error)}")

            # Step 5: Organize cells into grid structure
            grid, num_rows, num_cols = self.table_analyzer.organize_cells_into_grid(
                result["cells"]
            )

            # Step 6: Create comprehensive response object
            response = {
                "metadata": {
                    "file_name": os.path.basename(file_path),
                    "file_type": "pdf"
                    if file_path.lower().endswith(".pdf")
                    else "image",
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

            # Step 7: Create a visualization image
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
            print(f"Saved visualization to {response['visualization']['output_file']}")

            # Display processed image
            plt.figure(figsize=(15, 12))
            plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"Detected Table and Cells ({num_rows}x{num_cols} grid)")
            plt.show()

            return response

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            if processed_path and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except:
                    pass
