"""
OCR module for ML Timemaster.
Contains the CellOCR class for extracting text from table cells.
"""

import logging
import cv2
import numpy as np
import pytesseract
from src.performance_logger import PerformanceLogger, LogContext, timed_operation


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
        verbose_logging=False,
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
        self.verbose_logging = verbose_logging
        
        _base_logger = logging.getLogger(__name__)
        self.logger = PerformanceLogger(_base_logger)

        # OCR configurations to try
        self.configs = [
            "--oem 3 --psm 6",  # Uniform block of text
            "--oem 3 --psm 7",  # Single text line
            "--oem 3 --psm 8",  # Single word
            "--oem 3 --psm 4",  # Single column of text
            "--oem 3 --psm 11",  # Sparse text
            "--oem 3 --psm 13",  # Raw line
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

        # Basic preprocessing methods
        preprocessed_images.append(("original", gray_img))

        _, otsu = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(("otsu", otsu))

        adaptive = cv2.adaptiveThreshold(
            gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        preprocessed_images.append(("adaptive", adaptive))

        # Additional preprocessing in verbose mode
        if self.verbose_logging:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray_img)
            _, enhanced_binary = cv2.threshold(
                enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed_images.append(("enhanced", enhanced_binary))

            denoised = cv2.fastNlMeansDenoising(gray_img)
            _, denoised_binary = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed_images.append(("denoised", denoised_binary))

            inverted = cv2.bitwise_not(gray_img)
            preprocessed_images.append(("inverted", inverted))

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(gray_img, -1, kernel)
            _, sharpened_binary = cv2.threshold(
                sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            preprocessed_images.append(("sharpened", sharpened_binary))

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

            text = " ".join([word for word in results["text"] if str(word).strip()])

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

        text = " ".join(text.split())
        text = text.replace("|", "I")

        words = text.split()
        processed_words = []

        for word in words:
            if word and all(c in "I1l" for c in word):
                if len(word) == 1:
                    processed_words.append("I")
                else:
                    processed_words.append(word.replace("I", "1").replace("l", "1"))
            elif word in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "XI", "XII"]:
                processed_words.append(word.replace("1", "I").replace("l", "I"))
            elif any(pattern in word for pattern in ["CS", "MATH", "PHYS", "CHEM", "BIO"]):
                processed_words.append(word.replace("I", "1"))
            else:
                processed_words.append(word)

        return " ".join(processed_words)

    @timed_operation("Cell OCR extraction")
    def extract_cell_text(self, img, cell):
        """
        Extract text from a cell using OCR.
        
        Args:
            img (numpy.ndarray): Original image
            cell (dict): Cell information with bounds
            
        Returns:
            str: Extracted text
        """
        cell_id = f"Cell[{cell['x1']},{cell['y1']}-{cell['x2']},{cell['y2']}]"
        
        with LogContext(__name__, 
                        temp_level=logging.DEBUG if self.verbose_logging else logging.WARNING):
            
            if self.verbose_logging:
                self.logger.info("="*40)
                self.logger.info(f"Processing {cell_id}")
            
            try:
                # Extract cell region with padding
                padding = 3
                y1 = max(0, cell["y1"] - padding)
                y2 = min(img.shape[0], cell["y2"] + padding)
                x1 = max(0, cell["x1"] - padding)
                x2 = min(img.shape[1], cell["x2"] + padding)
                
                cell_img = img[y1:y2, x1:x2]

                if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                    self.logger.warning(f"{cell_id}: Cell too small, skipping")
                    return ""

                # Convert to grayscale
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

                # Create preprocessed versions
                preprocessed_images = self._preprocess_cell_image(gray)

                best_text = ""
                best_confidence = 0
                best_method = ""

                # Try all preprocessing methods
                for preprocess_name, processed_img in preprocessed_images:
                    configs_to_try = self.configs if self.verbose_logging else self.configs[:2]
                    
                    for config in configs_to_try:
                        text, avg_confidence = self._try_ocr_with_config(processed_img, config)
                        
                        if text and avg_confidence > best_confidence:
                            best_text = text
                            best_confidence = avg_confidence
                            best_method = f"{preprocess_name}+{config}"

                        # Early exit for high confidence
                        if text and avg_confidence >= self.high_confidence_threshold:
                            if self.verbose_logging:
                                self.logger.info(
                                    f"{cell_id}: HIGH CONFIDENCE ({avg_confidence:.1f}%) "
                                    f"- '{best_text}' [method: {best_method}]"
                                )
                            return self._post_process_text(best_text).strip()

                # Return result based on confidence
                if best_confidence >= self.minimum_confidence_threshold:
                    if self.verbose_logging:
                        self.logger.info(
                            f"{cell_id}: ACCEPTED ({best_confidence:.1f}%) - '{best_text}'"
                        )
                    return self._post_process_text(best_text).strip()
                else:
                    if self.verbose_logging:
                        self.logger.info(
                            f"{cell_id}: REJECTED (confidence {best_confidence:.1f}%)"
                        )
                    return ""

            except Exception as e:
                self.logger.error(f"{cell_id}: Error during extraction: {str(e)}", 
                                exc_info=self.verbose_logging)
                return ""
            
        if hasattr(self.logger, 'flush'):
            self.logger.flush()