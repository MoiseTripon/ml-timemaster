import datetime
import os
import tempfile

import cv2
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pdf2image
import pytesseract  # type: ignore


def convert_pdf_to_image(pdf_path, dpi=500):
    """
    Converts a PDF file to a high-quality image.
    """
    try:
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        if not images:
            raise ValueError("No pages found in PDF.")

        first_page = images[0]
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            temp_path = tmp.name
            first_page.save(temp_path, "PNG", quality=95)
            print(f"Converted PDF to image: {first_page.size}")
        return temp_path
    except Exception as e:
        raise ValueError(f"Error converting PDF to image: {str(e)}")


def is_blank_image(img):
    """
    Checks if an image is blank (mostly white).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_zero_pixels = cv2.countNonZero(gray)
    total_pixels = gray.shape[0] * gray.shape[1]
    return non_zero_pixels < total_pixels * 0.01  # more than 99% blank


def preprocess_schedule_table(file_path):
    """
    Preprocesses a schedule table from a PDF or image.
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")

    file_type = "pdf" if file_path.lower().endswith(".pdf") else "image"

    if file_type == "pdf":
        print("Converting PDF to image...")
        img_path = convert_pdf_to_image(file_path)
    else:
        img_path = file_path

    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError("Failed to load image.")

        if is_blank_image(img):
            raise ValueError("Error: The input image is blank.")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary)

        # Save preprocessed image
        preprocessed_path = f"preprocessed_{os.path.basename(file_path)}.png"
        cv2.imwrite(preprocessed_path, denoised)

        return preprocessed_path, img
    except Exception as e:
        raise ValueError(f"Error preprocessing file: {str(e)}")
    finally:
        if file_type == "pdf" and os.path.exists(img_path):
            os.remove(img_path)


def remove_duplicate_cells(cells, tolerance=10):
    """
    Remove duplicate cells that overlap significantly.
    Returns a list of unique cells.
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

    print(f"Removed {len(valid_cells) - len(unique_cells)} duplicate cells")
    return unique_cells


def detect_table_borders(img):
    """
    Detects the main table borders in the image.
    Returns the table boundary coordinates.
    """
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 3
    )

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

    print(f"Detected table boundaries: {table_bounds}")
    return table_bounds


def detect_cells(img, table_bounds):
    """
    Detects individual cells within the table.
    Returns a list of cell coordinates.
    """
    # Extract table region
    x, y, w, h = (
        table_bounds["x1"],
        table_bounds["y1"],
        table_bounds["x2"] - table_bounds["x1"],
        table_bounds["y2"] - table_bounds["y1"],
    )
    table_region = img[y : y + h, x : x + w]

    # Convert to grayscale
    gray = cv2.cvtColor(table_region, cv2.COLOR_BGR2GRAY)

    # Apply multiple preprocessing techniques to catch different types of cells
    preprocessed_images = []

    # Method 1: Standard Otsu's thresholding with different parameters
    _, binary1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed_images.append(binary1)

    # Method 2: Multiple adaptive thresholding attempts with different parameters
    binary2 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
    )
    preprocessed_images.append(binary2)

    binary3 = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 4
    )
    preprocessed_images.append(binary3)

    # Method 3: Edge detection with different parameters
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 30, 200)
    preprocessed_images.extend([edges1, edges2])

    # Method 4: Combination of methods
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary4 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    preprocessed_images.append(binary4)

    # Find contours in each preprocessed image
    all_contours = []
    for processed in preprocessed_images:
        contours, _ = cv2.findContours(
            processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        all_contours.extend(contours)

    # Filter and process contours
    cells = []
    min_area = (w * h) * 0.0001  # Minimum area in pixels
    max_area = (w * h) * 0.99  # Maximum area (99% of table area)

    for contour in all_contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x1, y1, width, height = cv2.boundingRect(contour)
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

    # Remove duplicates
    cells = remove_duplicate_cells(cells)

    return cells


def organize_cells_into_grid(cells):
    """
    Organizes cells into a grid structure and calculates rowspan and colspan values.
    Returns a 2D list of cell information and grid dimensions.
    """
    if not cells:
        return [], 0, 0

    # Sort cells by position (top to bottom, left to right)
    sorted_cells = sorted(cells, key=lambda c: (c["bounds"]["y1"], c["bounds"]["x1"]))

    # Find unique row and column positions with tolerance
    tolerance = 5  # 5 pixels tolerance for row/column alignment

    def merge_close_positions(positions, tolerance):
        if not positions:
            return []
        positions = sorted(positions)
        merged = [positions[0]]
        for pos in positions[1:]:
            if pos - merged[-1] > tolerance:
                merged.append(pos)
        return merged

    # Get unique positions and merge those that are very close
    row_positions = merge_close_positions(
        list(set(cell["bounds"]["y1"] for cell in sorted_cells)), tolerance
    )
    col_positions = merge_close_positions(
        list(set(cell["bounds"]["x1"] for cell in sorted_cells)), tolerance
    )

    # Calculate typical column width
    if len(col_positions) > 1:
        col_widths = [
            col_positions[i + 1] - col_positions[i]
            for i in range(len(col_positions) - 1)
        ]
        typical_col_width = sum(col_widths) / len(col_widths)
    else:
        typical_col_width = 100  # default value if only one column

    # Create a grid to store cell information
    temp_grid = []  # Use a temporary grid to store rows before filtering
    occupied_positions = set()  # Track which positions are already taken

    # Process each row
    for row_idx, row_y in enumerate(row_positions):
        current_row = [None] * len(col_positions)  # Initialize row with None values
        has_cells = False  # Track if this row has any cells

        # Process cells that start in this row
        for cell in sorted_cells:
            if abs(cell["bounds"]["y1"] - row_y) <= tolerance:
                has_cells = True
                # Find the starting column index for this cell
                col_idx = 0
                while (
                    col_idx < len(col_positions)
                    and col_positions[col_idx] + tolerance < cell["bounds"]["x1"]
                ):
                    col_idx += 1

                if col_idx >= len(col_positions):  # Safety check
                    continue

                # Calculate rowspan
                rowspan = 1
                cell_bottom = cell["bounds"]["y2"]
                for next_row_y in row_positions[row_idx + 1 :]:
                    if next_row_y - tolerance <= cell_bottom:
                        rowspan += 1
                    else:
                        break

                # Calculate colspan based on width coverage
                cell_right = cell["bounds"]["x2"]
                cell_width = cell_right - cell["bounds"]["x1"]
                colspan = 1

                # Calculate colspan by checking how many columns this cell spans
                remaining_width = cell_width
                next_col = col_idx
                while (
                    next_col + 1 < len(col_positions)
                    and remaining_width > typical_col_width * 0.3
                ):
                    # Get width of current column
                    if next_col + 1 < len(col_positions):
                        col_width = (
                            col_positions[next_col + 1] - col_positions[next_col]
                        )
                    else:
                        col_width = typical_col_width

                    # If we still have enough remaining width, increase colspan
                    if remaining_width > col_width * 0.3:
                        colspan += 1
                        remaining_width -= col_width

                    next_col += 1

                # Check if any of the positions this cell would occupy are already taken
                can_place = True
                for r in range(row_idx, min(row_idx + rowspan, len(row_positions))):
                    for c in range(col_idx, min(col_idx + colspan, len(col_positions))):
                        if (r, c) in occupied_positions:
                            can_place = False
                            break
                    if not can_place:
                        break

                if can_place:
                    # Create cell info
                    cell_info = {
                        "text": cell["text"],
                        "rowspan": rowspan,
                        "colspan": colspan,
                    }

                    # Place the cell and mark its positions as occupied
                    current_row[col_idx] = cell_info
                    for r in range(row_idx, min(row_idx + rowspan, len(row_positions))):
                        for c in range(
                            col_idx, min(col_idx + colspan, len(col_positions))
                        ):
                            occupied_positions.add((r, c))
                            if r == row_idx and c > col_idx:
                                current_row[c] = None  # Mark covered columns as None

        # Only add non-None values at the end of the row if there are no cells with colspan
        if has_cells:
            # Check if there's a cell with colspan in this row
            has_colspan = any(
                cell
                for cell in current_row
                if cell is not None and cell.get("colspan", 1) > 1
            )
            if not has_colspan:
                # Fill in empty cells only if there's no colspan
                for i in range(len(current_row)):
                    if (
                        current_row[i] is None
                        and (row_idx, i) not in occupied_positions
                    ):
                        current_row[i] = {"text": "", "rowspan": 1, "colspan": 1}

        temp_grid.append(current_row)

    # Filter out None values and empty rows
    final_grid = []
    for row in temp_grid:
        # Remove None values from the row
        filtered_row = [cell for cell in row if cell is not None]
        # Only add the row if it contains at least one cell
        if filtered_row:
            final_grid.append(filtered_row)

    # Calculate grid dimensions
    num_rows = len(final_grid)
    num_cols = max((len(row) for row in final_grid), default=0)

    return final_grid, num_rows, num_cols


def process_schedule_file(file_path):
    """
    Processes a schedule file (PDF or image) and extracts table data.
    Returns a comprehensive JSON structure with all table information.
    """
    try:
        # Preprocess the file
        processed_path, original_img = preprocess_schedule_table(file_path)
        print(f"Original image size: {original_img.shape}")

        # Step 1: Detect table borders
        table_bounds = detect_table_borders(original_img)

        # Step 2: Detect individual cells
        cells = detect_cells(original_img, table_bounds)

        # Step 3: Extract text from each cell
        result = {"table_bounds": table_bounds, "cells": []}

        for idx, cell in enumerate(cells):
            try:
                # Extract text from cell
                text = extract_cell_text(original_img, cell)

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
                print(f"Error processing cell {idx}: {str(ocr_error)}")

        # Step 4: Organize cells into grid structure
        grid, num_rows, num_cols = organize_cells_into_grid(result["cells"])

        # Create comprehensive response object
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

        # Create a visualization image
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
        if os.path.exists(processed_path):
            try:
                os.remove(processed_path)
            except OSError:
                pass


def extract_cell_text(img, cell):
    """
    Extracts text from a cell using OCR, attempting different orientations and preprocessing methods.
    Returns an empty string if confidence threshold isn't met.
    """
    try:
        cell_id = f"Cell({cell['x1']},{cell['y1']},{cell['x2']},{cell['y2']})"

        # Extract cell region with padding
        padding = 3  # Increase padding to 3 pixels to avoid cutting off text
        y1 = max(0, cell["y1"] - padding)
        y2 = min(img.shape[0], cell["y2"] + padding)
        x1 = max(0, cell["x1"] - padding)
        x2 = min(img.shape[1], cell["x2"] + padding)
        cell_img = img[y1:y2, x1:x2]

        if (
            cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5
        ):  # Skip very small cells
            return ""

        # Convert to grayscale
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

        # Create different preprocessed versions of the image
        preprocessed_images = []

        # 1. Original grayscale - good baseline for clean text
        preprocessed_images.append(("original", gray))

        # 2. Otsu's thresholding - good for high contrast text
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("otsu", otsu))

        # 3. Adaptive thresholding - good for varying lighting conditions
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images.append(("adaptive", adaptive))

        # 4. Contrast enhancement - helps with low contrast text
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, enhanced_binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        preprocessed_images.append(("enhanced", enhanced_binary))

        # 5. Denoised version - helps with noisy images
        denoised = cv2.fastNlMeansDenoising(gray)
        _, denoised_binary = cv2.threshold(
            denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        preprocessed_images.append(("denoised", denoised_binary))

        # 6. Inverted versions - helps with white text on dark backgrounds
        inverted = cv2.bitwise_not(gray)
        preprocessed_images.append(("inverted", inverted))

        # 7. Sharpened version - helps with blurry text
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        _, sharpened_binary = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        preprocessed_images.append(("sharpened", sharpened_binary))

        # 8. Dilated version - helps with thin or broken text
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(otsu, kernel, iterations=1)
        preprocessed_images.append(("dilated", dilated))

        # Different OCR configurations to try
        configs = [
            "--oem 3 --psm 6",  # Assume uniform block of text - good for paragraphs
            "--oem 3 --psm 7",  # Treat the image as a single text line - good for headers
            "--oem 3 --psm 8",  # Treat the image as a single word - good for short labels
            "--oem 3 --psm 4",  # Assume a single column of text - good for multi-line cells
            "--oem 3 --psm 11",  # Sparse text - find as much text as possible in complex layouts
            "--oem 3 --psm 13",  # Treat the image as a raw line with default OEM - good fallback
        ]

        # Confidence thresholds
        ROTATION_CONFIDENCE_THRESHOLD = (
            70.0  # Only try rotations if confidence is below this
        )
        MINIMUM_CONFIDENCE_THRESHOLD = (
            50.0  # Minimum confidence required to accept text
        )

        best_text = ""
        best_confidence = 0
        best_method = ""
        attempts = 0
        successful_attempts = 0

        # First try all preprocessing methods with original orientation
        for preprocess_name, processed_img in preprocessed_images:
            for config in configs:
                attempts += 1
                try:
                    # Get detailed results including confidence scores
                    results = pytesseract.image_to_data(
                        processed_img,
                        output_type=pytesseract.Output.DICT,
                        config=config,
                    )

                    # Combine all text from this attempt
                    text = " ".join(
                        [word for word in results["text"] if str(word).strip()]
                    )

                    # Calculate average confidence for non-empty words
                    confidences = [
                        conf
                        for conf, word in zip(results["conf"], results["text"])
                        if str(word).strip()
                    ]
                    avg_confidence = (
                        sum(confidences) / len(confidences) if confidences else 0
                    )

                    if text:
                        successful_attempts += 1

                    # Update best text if this attempt has higher confidence
                    if text and avg_confidence > best_confidence:
                        best_text = text
                        best_confidence = avg_confidence
                        best_method = f"{preprocess_name}, angle 0, config {config}"
                except Exception:
                    continue

        # Only try rotations if best confidence is below threshold
        if best_confidence < ROTATION_CONFIDENCE_THRESHOLD:
            # Try rotations with the best preprocessing methods
            # Sort preprocessing methods by their best confidence
            preprocess_confidence = {}
            for preprocess_name, processed_img in preprocessed_images:
                for config in configs:
                    try:
                        results = pytesseract.image_to_data(
                            processed_img,
                            output_type=pytesseract.Output.DICT,
                            config=config,
                        )
                        text = " ".join(
                            [word for word in results["text"] if str(word).strip()]
                        )
                        confidences = [
                            conf
                            for conf, word in zip(results["conf"], results["text"])
                            if str(word).strip()
                        ]
                        avg_confidence = (
                            sum(confidences) / len(confidences) if confidences else 0
                        )

                        if text and (
                            preprocess_name not in preprocess_confidence
                            or avg_confidence > preprocess_confidence[preprocess_name]
                        ):
                            preprocess_confidence[preprocess_name] = avg_confidence
                    except Exception:
                        continue

            # Sort preprocessing methods by confidence
            sorted_preprocess = sorted(
                preprocess_confidence.items(), key=lambda x: x[1], reverse=True
            )

            # Try rotations with the top 3 preprocessing methods
            top_preprocess = (
                [name for name, _ in sorted_preprocess[:3]]
                if sorted_preprocess
                else [p[0] for p in preprocessed_images[:3]]
            )

            for preprocess_name, processed_img in preprocessed_images:
                if preprocess_name not in top_preprocess:
                    continue

                # Try different orientations
                orientations = [
                    (cv2.rotate(processed_img, cv2.ROTATE_90_CLOCKWISE), 90),
                    (cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE), -90),
                ]

                for img_orient, angle in orientations:
                    for config in configs:
                        attempts += 1
                        try:
                            # Get detailed results including confidence scores
                            results = pytesseract.image_to_data(
                                img_orient,
                                output_type=pytesseract.Output.DICT,
                                config=config,
                            )

                            # Combine all text from this attempt
                            text = " ".join(
                                [word for word in results["text"] if str(word).strip()]
                            )

                            # Calculate average confidence for non-empty words
                            confidences = [
                                conf
                                for conf, word in zip(results["conf"], results["text"])
                                if str(word).strip()
                            ]
                            avg_confidence = (
                                sum(confidences) / len(confidences)
                                if confidences
                                else 0
                            )

                            if text:
                                successful_attempts += 1

                            # Update best text if this attempt has higher confidence
                            if text and avg_confidence > best_confidence:
                                best_text = text
                                best_confidence = avg_confidence
                                best_method = (
                                    f"{preprocess_name}, angle {angle}, config {config}"
                                )
                        except Exception:
                            continue

        # If no text was found with confidence-based approach, try a simpler approach
        if not best_text:
            # Try a direct OCR on the original image with different PSM modes
            for config in configs:
                try:
                    text = pytesseract.image_to_string(gray, config=config)
                    if text.strip():
                        # For direct OCR, we can't get confidence scores, so we'll use a lower threshold
                        # We'll estimate confidence at 40% for direct OCR
                        estimated_confidence = 40.0
                        best_text = text
                        best_confidence = estimated_confidence
                        break
                except Exception:
                    continue

        # Check if the best confidence meets the minimum threshold
        if best_confidence < MINIMUM_CONFIDENCE_THRESHOLD:
            # Only print the final result - no text due to low confidence
            print(
                f"{cell_id}: No text (confidence {best_confidence:.1f}% below threshold)"
            )
            return ""

        # Post-process the text
        if best_text:
            # Remove extra whitespace and normalize
            best_text = " ".join(best_text.split())

            # Enhanced I vs 1 disambiguation
            # First, replace vertical bars with I (this is almost always correct)
            best_text = best_text.replace("|", "I")

            # Context-based replacement for distinguishing between I and 1
            words = best_text.split()
            processed_words = []

            for word in words:
                # Check if the word contains only I, 1, or l characters (common confusion)
                if word and all(c in "I1l" for c in word):
                    # If it's a single character, use context to determine if it's I or 1
                    if len(word) == 1:
                        # In most tables, single I is more common than single 1
                        processed_words.append("I")
                    else:
                        # For longer sequences, it's likely a number (e.g., 11, 111)
                        processed_words.append(word.replace("I", "1").replace("l", "1"))
                # Special case for Roman numerals (I, II, III, IV, etc.)
                elif word in [
                    "I",
                    "II",
                    "III",
                    "IV",
                    "V",
                    "VI",
                    "VII",
                    "VIII",
                    "IX",
                    "XI",
                    "XII",
                ]:
                    processed_words.append(word.replace("1", "I").replace("l", "I"))
                # Check for patterns that are likely course/section numbers
                elif any(
                    pattern in word for pattern in ["CS", "MATH", "PHYS", "CHEM", "BIO"]
                ):
                    # In course numbers, 1 is more likely than I
                    processed_words.append(word.replace("I", "1"))
                else:
                    processed_words.append(word)

            best_text = " ".join(processed_words)

            print(f"{cell_id}: '{best_text}' (confidence: {best_confidence:.1f}%)")
        else:
            print(f"{cell_id}: No text detected")

        return best_text.strip()

    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        return ""


# Example usage
if __name__ == "__main__":
    file_path = "i_a.pdf"  # Replace with your PDF or image file
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
