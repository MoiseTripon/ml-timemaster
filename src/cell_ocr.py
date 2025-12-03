"""
OCR module for ML Timemaster.
Contains the CellOCR class for extracting text from table cells.
"""

import logging
import cv2
import numpy as np
import pytesseract  # type: ignore
import os
from src.performance_logger import PerformanceLogger, LogContext, timed_operation

# Create performance-optimized logger
_base_logger = logging.getLogger(__name__)
logger = PerformanceLogger(_base_logger)


class CellOCR:
    """
    Handles OCR operations for extracting text from table cells.
    Uses multiple preprocessing techniques and configurations to maximize accuracy.
    """

    def __init__(
        self,
        rotation_confidence_threshold=70.0,
        minimum_confidence_threshold=50.0,
        high_confidence_threshold=90.0,
        verbose_logging=False,  # New parameter
    ):
        """
        Initialize the CellOCR.

        Args:
            rotation_confidence_threshold (float): Threshold below which rotations are attempted.
            minimum_confidence_threshold (float): Minimum confidence required to accept text.
            high_confidence_threshold (float): Threshold above which text is immediately accepted.
            verbose_logging (bool): Enable verbose logging for debugging
        """
        self.rotation_confidence_threshold = rotation_confidence_threshold
        self.minimum_confidence_threshold = minimum_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.consecutive_low_confidence_threshold = 2
        self.verbose_logging = verbose_logging

        # OCR configurations to try
        self.configs = [
            "--oem 3 --psm 6",  # Assume uniform block of text - good for paragraphs
            "--oem 3 --psm 7",  # Treat the image as a single text line - good for headers
            "--oem 3 --psm 8",  # Treat the image as a single word - good for short labels
            "--oem 3 --psm 4",  # Assume a single column of text - good for multi-line cells
            "--oem 3 --psm 11",  # Sparse text - find as much text as possible in complex layouts
            "--oem 3 --psm 13",  # Treat the image as a raw line with default OEM - good fallback
        ]

    def _preprocess_cell_image(self, gray_img):
        """
        Create different preprocessed versions of a cell image.

        Args:
            gray_img (numpy.ndarray): Grayscale image of the cell.

        Returns:
            list: List of tuples (method_name, preprocessed_image)
        """
        preprocessed_images = []

        # 1. Original grayscale - good baseline for clean text
        preprocessed_images.append(("original", gray_img))

        # 2. Otsu's thresholding - good for high contrast text
        _, otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("otsu", otsu))

        # 3. Adaptive thresholding - good for varying lighting conditions
        adaptive = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images.append(("adaptive", adaptive))

        # Only add more preprocessing methods if verbose logging is enabled
        if self.verbose_logging:
            # 4. Contrast enhancement - helps with low contrast text
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_img)
            _, enhanced_binary = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed_images.append(("enhanced", enhanced_binary))

            # 5. Denoised version - helps with noisy images
            denoised = cv2.fastNlMeansDenoising(gray_img)
            _, denoised_binary = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed_images.append(("denoised", denoised_binary))

            # 6. Inverted versions - helps with white text on dark backgrounds
            inverted = cv2.bitwise_not(gray_img)
            preprocessed_images.append(("inverted", inverted))

            # 7. Sharpened version - helps with blurry text
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(gray_img, -1, kernel)
            _, sharpened_binary = cv2.threshold(
                sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed_images.append(("sharpened", sharpened_binary))

            # 8. Dilated version - helps with thin or broken text
            kernel = np.ones((2, 2), np.uint8)
            dilated = cv2.dilate(otsu, kernel, iterations=1)
            preprocessed_images.append(("dilated", dilated))

        return preprocessed_images

    def _try_ocr_with_config(self, img, config):
        """
        Try OCR with a specific configuration.

        Args:
            img (numpy.ndarray): Image to process.
            config (str): Tesseract configuration string.

        Returns:
            tuple: (text, confidence) or (None, 0) if failed
        """
        try:
            results = pytesseract.image_to_data(
                img, output_type=pytesseract.Output.DICT, config=config
            )

            # Combine all text from this attempt
            text = " ".join([word for word in results["text"] if str(word).strip()])

            # Calculate average confidence for non-empty words
            confidences = [
                conf
                for conf, word in zip(results["conf"], results["text"])
                if str(word).strip()
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return text, avg_confidence
        except Exception:
            return None, 0

    def _post_process_text(self, text):
        """
        Post-process extracted text to fix common OCR errors.

        Args:
            text (str): Raw text from OCR.

        Returns:
            str: Post-processed text.
        """
        if not text:
            return text

        # Remove extra whitespace and normalize
        text = " ".join(text.split())

        # Enhanced I vs 1 disambiguation
        # First, replace vertical bars with I (this is almost always correct)
        text = text.replace("|", "I")

        # Context-based replacement for distinguishing between I and 1
        words = text.split()
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

        return " ".join(processed_words)

    @timed_operation("Cell OCR extraction")
    def extract_cell_text(self, img, cell):
        """
        Extracts text from a cell using OCR.
        """
        cell_id = f"Cell[{cell['x1']},{cell['y1']}-{cell['x2']},{cell['y2']}]"
        
        # Use context manager to control logging level
        with LogContext(__name__, 
                        temp_level=logging.DEBUG if self.verbose_logging else logging.WARNING):
            
            if self.verbose_logging:
                logger.info("="*40)
                logger.info(f"Processing {cell_id}")
                logger.info(f"Cell dimensions: {cell['x2']-cell['x1']}x{cell['y2']-cell['y1']}")
            
            try:
                # Extract cell region with padding
                padding = 3
                y1 = max(0, cell["y1"] - padding)
                y2 = min(img.shape[0], cell["y2"] + padding)
                x1 = max(0, cell["x1"] - padding)
                x2 = min(img.shape[1], cell["x2"] + padding)
                
                if self.verbose_logging:
                    logger.debug(f"Extraction coords (with padding): x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                
                cell_img = img[y1:y2, x1:x2]
                
                if self.verbose_logging:
                    logger.debug(f"Extracted cell image shape: {cell_img.shape}")

                if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                    logger.warning(f"{cell_id}: Cell too small, skipping")
                    return ""

                # Save debug image of the cell only in verbose mode
                if self.verbose_logging and hasattr(self, 'debug_mode') and self.debug_mode:
                    debug_dir = "debug_output/cells"
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(f"{debug_dir}/cell_{cell['x1']}_{cell['y1']}.png", cell_img)

                # Convert to grayscale
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

                # Create preprocessed versions
                preprocessed_images = self._preprocess_cell_image(gray)
                
                if self.verbose_logging:
                    logger.debug(f"Created {len(preprocessed_images)} preprocessed versions")

                best_text = ""
                best_confidence = 0
                best_method = ""

                # Try all preprocessing methods
                for preprocess_name, processed_img in preprocessed_images:
                    # In non-verbose mode, only try first 2 configs for each preprocessing
                    configs_to_try = self.configs if self.verbose_logging else self.configs[:2]
                    
                    for config in configs_to_try:
                        text, avg_confidence = self._try_ocr_with_config(processed_img, config)
                        
                        if text and avg_confidence > best_confidence:
                            best_text = text
                            best_confidence = avg_confidence
                            best_method = f"{preprocess_name}+{config}"
                            
                            if self.verbose_logging:
                                logger.debug(
                                    f"  New best: '{text[:50]}...' "
                                    f"(conf={avg_confidence:.1f}%, method={best_method})"
                                )

                        # Early exit for high confidence
                        if text and avg_confidence >= self.high_confidence_threshold:
                            if self.verbose_logging:
                                logger.info(
                                    f"{cell_id}: HIGH CONFIDENCE ({avg_confidence:.1f}%) "
                                    f"- '{best_text}' [method: {best_method}]"
                                )
                            return self._post_process_text(best_text).strip()

                # Log final result
                if best_confidence >= self.minimum_confidence_threshold:
                    if self.verbose_logging:
                        logger.info(
                            f"{cell_id}: ACCEPTED ({best_confidence:.1f}%) "
                            f"- '{best_text}' [method: {best_method}]"
                        )
                    return self._post_process_text(best_text).strip()
                else:
                    # Only log low confidence results in verbose mode
                    if self.verbose_logging:
                        logger.info(
                            f"{cell_id}: REJECTED (confidence {best_confidence:.1f}% < {self.minimum_confidence_threshold}%) "
                            f"- best attempt: '{best_text}'"
                        )
                    return ""

            except Exception as e:
                logger.error(f"{cell_id}: Error during extraction: {str(e)}", exc_info=self.verbose_logging)
                return ""
            
        # Flush any pending log messages
        if hasattr(logger, 'flush'):
            logger.flush()