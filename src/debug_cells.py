"""
Interactive debugging script for table cell detection.
Run this to step through the detection process.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

from src.image_preprocessor import ImagePreprocessor
from src.table_detector import detect_table_borders, detect_cells
from src.cell_ocr import CellOCR

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def interactive_debug(file_path):
    """Interactively debug the table detection process."""
    
    print("="*60)
    print("INTERACTIVE TABLE DETECTION DEBUGGER")
    print("="*60)
    
    # Step 1: Load and preprocess
    print("\n[Step 1] Loading and preprocessing image...")
    preprocessor = ImagePreprocessor()
    processed_path, processed_img, original_img = preprocessor.preprocess(file_path)
    
    print(f"  Original image shape: {original_img.shape}")
    print(f"  Processed image shape: {processed_img.shape}")
    
    # Show original image
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.title("Processed Image")
    plt.savefig("debug_output/step1_preprocessing.png")
    plt.show()
    
    input("Press Enter to continue to Step 2...")
    
    # Step 2: Detect table borders
    print("\n[Step 2] Detecting table borders...")
    table_bounds = detect_table_borders(processed_img)
    print(f"  Table bounds: {table_bounds}")
    
    # Visualize table bounds
    vis_img = original_img.copy()
    cv2.rectangle(
        vis_img,
        (table_bounds["x1"], table_bounds["y1"]),
        (table_bounds["x2"], table_bounds["y2"]),
        (0, 255, 0), 3
    )
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Table Bounds")
    plt.savefig("debug_output/step2_table_bounds.png")
    plt.show()
    
    input("Press Enter to continue to Step 3...")
    
    # Step 3: Detect cells
    print("\n[Step 3] Detecting cells...")
    cells = detect_cells(processed_img, table_bounds)
    print(f"  Detected {len(cells)} cells")
    
    # Visualize all cells
    vis_img = original_img.copy()
    for i, cell in enumerate(cells):
        color = ((i * 37) % 255, (i * 67) % 255, (i * 97) % 255)
        cv2.rectangle(
            vis_img,
            (cell["x1"], cell["y1"]),
            (cell["x2"], cell["y2"]),
            color, 2
        )
        cv2.putText(
            vis_img, str(i),
            (cell["x1"] + 5, cell["y1"] + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )
    
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Cells ({len(cells)} total)")
    plt.savefig("debug_output/step3_cells.png")
    plt.show()
    
    # Step 4: Examine individual cells
    print("\n[Step 4] Examining individual cells...")
    ocr = CellOCR()
    
    for i, cell in enumerate(cells):
        print(f"\n  Cell {i}:")
        print(f"    Position: ({cell['x1']}, {cell['y1']}) to ({cell['x2']}, {cell['y2']})")
        print(f"    Size: {cell['width']}x{cell['height']}")
        
        # Extract and show cell image
        cell_img = original_img[cell["y1"]:cell["y2"], cell["x1"]:cell["x2"]]
        
        # OCR the cell
        text = ocr.extract_cell_text(original_img, cell)
        print(f"    OCR Text: '{text}'")
        
        # Show cell
        plt.figure(figsize=(8, 4))
        plt.imshow(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Cell {i}: '{text[:30]}...' ({cell['width']}x{cell['height']})")
        plt.savefig(f"debug_output/step4_cell_{i}.png")
        
        # Ask if user wants to continue
        response = input(f"    Continue to next cell? (y/n/q to quit): ")
        if response.lower() == 'q':
            break
        elif response.lower() == 'n':
            continue
        
        plt.close()
    
    print("\n" + "="*60)
    print("Debug session complete. Check debug_output/ for saved images.")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    # Create debug output directory
    os.makedirs("debug_output", exist_ok=True)
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "i_a.pdf"
    
    interactive_debug(file_path)