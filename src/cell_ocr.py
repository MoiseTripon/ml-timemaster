"""
OCR module for ML Timemaster.
Contains the CellOCR class for extracting text from table cells.
"""

import logging
import cv2
import numpy as np
import pytesseract  # type: ignore

logger = logging.getLogger(__name__)


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
    ):
        """
        Initialize the CellOCR.

        Args:
            rotation_confidence_threshold (float): Threshold below which rotations are attempted.
            minimum_confidence_threshold (float): Minimum confidence required to accept text.
            high_confidence_threshold (float): Threshold above which text is immediately accepted.
        """
        self.rotation_confidence_threshold = rotation_confidence_threshold
        self.minimum_confidence_threshold = minimum_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.consecutive_low_confidence_threshold = (
            2  # Number of consecutive low confidence results to consider empty
        )

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

    def extract_cell_text(self, img, cell):
        """
        Extracts text from a cell using OCR, attempting different orientations and preprocessing methods.

        Args:
            img (numpy.ndarray): Full image containing the cell.
            cell (dict): Cell dictionary with 'x1', 'y1', 'x2', 'y2' keys.

        Returns:
            str: Extracted text, or empty string if confidence threshold isn't met.
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

            if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                return ""  # Skip very small cells

            # Convert to grayscale
            gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

            # Create different preprocessed versions of the image
            preprocessed_images = self._preprocess_cell_image(gray)

            best_text = ""
            best_confidence = 0
            attempts = 0
            successful_attempts = 0
            consecutive_low_confidence = 0  # Track consecutive low confidence attempts

            # First try all preprocessing methods with original orientation
            for preprocess_name, processed_img in preprocessed_images:
                for config in self.configs:
                    attempts += 1
                    text, avg_confidence = self._try_ocr_with_config(
                        processed_img, config
                    )

                    if text:
                        successful_attempts += 1

                    # Track consecutive low confidence results
                    if avg_confidence < self.minimum_confidence_threshold:
                        consecutive_low_confidence += 1
                    else:
                        consecutive_low_confidence = 0  # Reset on success

                    # Early exit if confidence is above 90%
                    if text and avg_confidence >= self.high_confidence_threshold:
                        best_text = text
                        best_confidence = avg_confidence
                        logger.info(
                            f"{cell_id}: HIGH CONFIDENCE ({avg_confidence:.1f}%) - '{best_text}'"
                        )
                        return self._post_process_text(best_text).strip()

                    # Update best text if this attempt has higher confidence
                    if text and avg_confidence > best_confidence:
                        best_text = text
                        best_confidence = avg_confidence

                    # Early exit if two consecutive attempts have confidence below 50%
                    if (
                        consecutive_low_confidence
                        >= self.consecutive_low_confidence_threshold
                    ):
                        logger.info(
                            f"{cell_id}: EMPTY (2+ consecutive attempts below 50% confidence)"
                        )
                        return ""

            # Only try rotations if best confidence is below threshold
            if best_confidence < self.rotation_confidence_threshold:
                # Find best preprocessing methods
                preprocess_confidence = {}
                for preprocess_name, processed_img in preprocessed_images:
                    for config in self.configs:
                        text, avg_confidence = self._try_ocr_with_config(
                            processed_img, config
                        )

                        if text and (
                            preprocess_name not in preprocess_confidence
                            or avg_confidence > preprocess_confidence[preprocess_name]
                        ):
                            preprocess_confidence[preprocess_name] = avg_confidence

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
                        (
                            cv2.rotate(processed_img, cv2.ROTATE_90_COUNTERCLOCKWISE),
                            -90,
                        ),
                    ]

                    for img_orient, angle in orientations:
                        for config in self.configs:
                            attempts += 1
                            text, avg_confidence = self._try_ocr_with_config(
                                img_orient, config
                            )

                            if text:
                                successful_attempts += 1

                            # Update best text if this attempt has higher confidence
                            if text and avg_confidence > best_confidence:
                                best_text = text
                                best_confidence = avg_confidence

            # If no text was found with confidence-based approach, try a simpler approach
            if not best_text:
                for config in self.configs:
                    try:
                        text = pytesseract.image_to_string(gray, config=config)
                        if text.strip():
                            # For direct OCR, estimate confidence at 40%
                            estimated_confidence = 40.0
                            best_text = text
                            best_confidence = estimated_confidence
                            break
                    except Exception:
                        continue

            # Check if the best confidence meets the minimum threshold
            if best_confidence < self.minimum_confidence_threshold:
                logger.info(
                    f"{cell_id}: No text (confidence {best_confidence:.1f}% below threshold)"
                )
                return ""

            # Post-process the text
            if best_text:
                best_text = self._post_process_text(best_text)
                logger.info(f"{cell_id}: '{best_text}' (confidence: {best_confidence:.1f}%)")
            else:
                logger.info(f"{cell_id}: No text detected")

            return best_text.strip()

        except Exception as e:
            logger.error(f"Error during extraction: {str(e)}")
            return ""
