import datetime
import logging
import os
import sys
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.performance_logger import PerformanceHandler, timed_operation
from src.image_preprocessor import ImagePreprocessor
from src.table_detector import TableDetector
from src.cell_ocr import CellOCR
from src.grid_builder import GridBuilder


def setup_logging(debug=False, log_level=logging.INFO):
    """
    Configure logging for the entire application - called once at startup.
    
    Args:
        debug (bool): Enable debug mode with file logging
        log_level (int): Logging level
        
    Returns:
        logging.Logger: Configured root logger
    """
    # Create logs directory if needed
    if debug and not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear any existing handlers
    root_logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Console handler with performance optimization
    console_handler = PerformanceHandler(
        base_handler=logging.StreamHandler(sys.stdout),
        batch_size=100 if not debug else 1,  # No batching in debug mode
        batch_timeout=1.0
    )
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for debug mode
    if debug:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(f"logs/table_detector_{timestamp}.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

class TableOCRProcessor:
    def __init__(self, debug=False, api_mode=False):
        """
        Initialize the Table OCR Processor.
        """
        self.debug = debug
        self.api_mode = api_mode
        self.logger = logging.getLogger(__name__)
        
        # Make sure api_mode is NOT passed to these components
        # They might be affected by the parameter
        self.preprocessor = ImagePreprocessor()
        self.table_detector = TableDetector(debug_mode=debug)  # NOT api_mode
        self.ocr_processor = CellOCR(verbose_logging=debug)   # NOT api_mode
        self.grid_builder = GridBuilder()

    def process_file(self, file_path):
        """Process a schedule file (PDF or image) and extract table data."""
        try:
            # Add logging to track the process
            self.logger.info(f"[TableOCRProcessor] Processing file: {file_path}")
            
            # Step 1: Preprocess the file
            processed_path, processed_img, original_img = self.preprocessor.preprocess(file_path)
            self.logger.info(f"[TableOCRProcessor] Preprocessed image shape: {processed_img.shape}")
            self.logger.info(f"[TableOCRProcessor] Original image shape: {original_img.shape}")

            # Step 2: Detect table borders and cells
            table_bounds = self.table_detector.detect_table_borders(processed_img)
            self.logger.info(f"[TableOCRProcessor] Table bounds: {table_bounds}")
            
            cells = self.table_detector.detect_cells(processed_img, table_bounds)
            self.logger.info(f"[TableOCRProcessor] Detected {len(cells)} cells")

            if len(cells) == 0:
                self.logger.error("No cells detected in the table")
                raise ValueError("No cells were detected in the table")
            
            # Step 3: Extract text from each cell
            result = {"table_bounds": table_bounds, "cells": []}

            for idx, cell in enumerate(cells):
                try:
                    # Log cell bounds before OCR
                    self.logger.debug(f"[TableOCRProcessor] Processing cell {idx}: {cell}")
                    
                    text = self.ocr_processor.extract_cell_text(original_img, cell)
                    
                    # Log extracted text
                    self.logger.debug(f"[TableOCRProcessor] Cell {idx} text: '{text}'")
                    
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
                    self.logger.error(f"[TableOCRProcessor] Error processing cell {idx}: {str(ocr_error)}")

            # Log summary of extracted text
            texts_found = [c["text"] for c in result["cells"] if c["text"].strip()]
            self.logger.info(f"[TableOCRProcessor] Extracted {len(texts_found)} non-empty texts")
            if texts_found:
                self.logger.info(f"[TableOCRProcessor] Sample texts: {texts_found[:3]}")

            # Step 4: Organize cells into grid structure
            grid, num_rows, num_cols = self.grid_builder.organize_cells_into_grid(result["cells"])
            self.logger.info(f"[TableOCRProcessor] Grid dimensions: {num_rows}x{num_cols}")


            # Step 5: Create comprehensive response object
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

            # Step 6: Create visualization
            self._create_visualization(original_img, table_bounds, result["cells"], 
                                      response["visualization"]["output_file"], num_rows, num_cols)
                
            return response

        except Exception as e:
            self.logger.error(f"Error during processing: {str(e)}")
            raise
        finally:
            # Clean up temporary files
            if 'processed_path' in locals() and os.path.exists(processed_path):
                try:
                    os.remove(processed_path)
                except OSError:
                    pass

    def _create_visualization(self, img, table_bounds, cells, output_file, num_rows, num_cols):
        """Create and save visualization of detected table and cells."""
        vis_img = img.copy()

        # Draw table boundary in red
        cv2.rectangle(
            vis_img,
            (table_bounds["x1"], table_bounds["y1"]),
            (table_bounds["x2"], table_bounds["y2"]),
            (0, 0, 255),
            3,
        )

        # Generate random colors for cells
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(cells), 3)).tolist()

        # Draw cells with random colors
        for cell, color in zip(cells, colors):
            cv2.rectangle(
                vis_img,
                (cell["bounds"]["x1"], cell["bounds"]["y1"]),
                (cell["bounds"]["x2"], cell["bounds"]["y2"]),
                color,
                2,
            )

        # Save visualization
        cv2.imwrite(output_file, vis_img)
        self.logger.info(f"Saved visualization to {output_file}")

        # Don't show matplotlib in API context - check if we're in a notebook/terminal
        try:
            import matplotlib
            if matplotlib.get_backend() != 'Agg':  # Only show if not in headless mode
                plt.figure(figsize=(15, 12))
                plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
                plt.axis("off")
                plt.title(f"Detected Table and Cells ({num_rows}x{num_cols} grid)")
                plt.show()
        except:
            pass 
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Table OCR Processor')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--file', type=str, default='i_a.pdf', help='Input file path')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    log_level = getattr(logging, args.log_level)
    
    # Setup logging once for the entire application
    setup_logging(debug=args.debug, log_level=log_level)
    
    # Create processor
    processor = TableOCRProcessor(debug=args.debug)
    
    try:
        result = processor.process_file(args.file)
        
        if not result["table"]["grid"]:
            print("No cells were detected.")
            exit(1)

        print("\nExtracted Table Data:")
        print(f"File: {result['metadata']['file_name']}")
        print(f"Image size: {result['metadata']['image_size']['width']}x{result['metadata']['image_size']['height']}")
        print(f"Table dimensions: {result['table']['dimensions']['rows']}x{result['table']['dimensions']['columns']}")
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