"""
OCR module for ML Timemaster.
Contains the CellOCR class for extracting text from table cells.
"""

import logging
import cv2
import numpy as np
import pytesseract
from src.performance_logger import LogContext, timed_operation
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class OCRResult:
    """Holds the result of an OCR attempt."""
    text: str
    confidence: float
    quality_score: float
    psm_mode: int
    preprocessing: str
    rotation: int = 0
    
    def __repr__(self):
        return (f"OCRResult(text='{self.text}', conf={self.confidence:.1f}, "
                f"quality={self.quality_score:.1f}, psm={self.psm_mode}, "
                f"prep='{self.preprocessing}', rot={self.rotation}°)")


class CellOCR:
    """
    Handles OCR operations for extracting text from table cells.
    Uses multiple PSM modes and preprocessing techniques for best results.
    """

    # PSM modes to try, in order of preference for table cells
    PSM_MODES = {
        6: "Single uniform block of text",
        7: "Single text line",
        3: "Fully automatic page segmentation",
        8: "Single word",
        11: "Sparse text",
        13: "Raw line",
        4: "Single column of text",
    }
    
    # Primary PSM modes to try first (faster)
    PRIMARY_PSM_MODES = [6, 7, 3]
    
    # Fallback PSM modes to try if primary modes fail
    FALLBACK_PSM_MODES = [8, 11, 13, 4]

    def __init__(
        self,
        minimum_confidence_threshold=50.0,
        high_confidence_threshold=90.0,
        verbose_logging=False,
        empty_cell_variance_threshold=100.0,
        empty_cell_content_ratio_threshold=0.01,
        rotation_score_boost=1.2,
        min_text_length_for_rotation=3,
        languages="eng+ron",
        osd_confidence_threshold=50.0,
        upscale_factor=2,
        enable_fallback_modes=True,
        fallback_confidence_threshold=30.0,  # Lower threshold for fallback attempts
    ):
        """
        Initialize the CellOCR.

        Args:
            minimum_confidence_threshold (float): Minimum confidence required to accept text.
            high_confidence_threshold (float): Threshold above which text is immediately accepted.
            verbose_logging (bool): Enable verbose logging for debugging.
            empty_cell_variance_threshold (float): Variance threshold below which a cell is considered empty.
            empty_cell_content_ratio_threshold (float): Content pixel ratio below which cell is empty.
            rotation_score_boost (float): Score multiplier for rotated results when rotation is expected.
            min_text_length_for_rotation (int): Minimum text length to consider valid in rotation attempts.
            languages (str): Languages to use for OCR (default: English + Romanian).
            osd_confidence_threshold (float): Minimum confidence for OSD rotation detection.
            upscale_factor (int): Factor to upscale images before OCR (default: 2).
            enable_fallback_modes (bool): Whether to try additional PSM modes if primary modes fail.
            fallback_confidence_threshold (float): Lower confidence threshold for fallback attempts.
        """
        self.minimum_confidence_threshold = minimum_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.verbose_logging = verbose_logging
        self.empty_cell_variance_threshold = empty_cell_variance_threshold
        self.empty_cell_content_ratio_threshold = empty_cell_content_ratio_threshold
        self.rotation_score_boost = rotation_score_boost
        self.min_text_length_for_rotation = min_text_length_for_rotation
        self.languages = languages
        self.osd_confidence_threshold = osd_confidence_threshold
        self.upscale_factor = upscale_factor
        self.enable_fallback_modes = enable_fallback_modes
        self.fallback_confidence_threshold = fallback_confidence_threshold
        
        self.logger = logging.getLogger(__name__)
        
        # Config for OSD
        self.osd_config = "--psm 0"

    def _get_ocr_config(self, psm_mode: int) -> str:
        """Generate OCR config string for a specific PSM mode."""
        return f"--oem 3 --psm {psm_mode} -l {self.languages}"

    def _upscale_image(self, img: np.ndarray, scale: int = None) -> np.ndarray:
        """
        Upscale an image by a given factor for better OCR accuracy.
        
        Args:
            img: Image to upscale.
            scale: Scale factor. Uses self.upscale_factor if None.
            
        Returns:
            Upscaled image.
        """
        if scale is None:
            scale = self.upscale_factor
            
        if scale <= 1:
            return img
        
        new_width = img.shape[1] * scale
        new_height = img.shape[0] * scale
        
        upscaled = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        return upscaled

    def _preprocess_otsu(self, gray_img: np.ndarray) -> np.ndarray:
        """Preprocess using Otsu's thresholding."""
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._ensure_black_text_on_white(binary)

    def _preprocess_adaptive_gaussian(self, gray_img: np.ndarray) -> np.ndarray:
        """Preprocess using adaptive Gaussian thresholding."""
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return self._ensure_black_text_on_white(binary)

    def _preprocess_adaptive_mean(self, gray_img: np.ndarray) -> np.ndarray:
        """Preprocess using adaptive mean thresholding."""
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return self._ensure_black_text_on_white(binary)

    def _preprocess_morphological(self, gray_img: np.ndarray) -> np.ndarray:
        """Preprocess with morphological operations to clean up text."""
        blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = self._ensure_black_text_on_white(binary)
        
        # Morphological opening to remove noise
        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return opened

    def _preprocess_contrast_enhanced(self, gray_img: np.ndarray) -> np.ndarray:
        """Preprocess with CLAHE contrast enhancement."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_img)
        
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._ensure_black_text_on_white(binary)

    def _preprocess_sharpened(self, gray_img: np.ndarray) -> np.ndarray:
        """Preprocess with sharpening before thresholding."""
        # Sharpen the image
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(gray_img, -1, kernel)
        
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._ensure_black_text_on_white(binary)

    def _preprocess_denoise(self, gray_img: np.ndarray) -> np.ndarray:
        """Preprocess with denoising."""
        denoised = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self._ensure_black_text_on_white(binary)

    def _ensure_black_text_on_white(self, binary_img: np.ndarray) -> np.ndarray:
        """Ensure text is black on white background (what Tesseract prefers)."""
        h, w = binary_img.shape
        center_region = binary_img[h//4:3*h//4, w//4:3*w//4]
        white_pixels = np.sum(center_region == 255)
        black_pixels = np.sum(center_region == 0)
        
        if black_pixels > white_pixels:
            return cv2.bitwise_not(binary_img)
        return binary_img

    def _get_preprocessing_methods(self, is_fallback: bool = False) -> List[Tuple[str, callable]]:
        """
        Get list of preprocessing methods to try.
        
        Args:
            is_fallback: If True, include more aggressive preprocessing methods.
            
        Returns:
            List of (name, method) tuples.
        """
        primary_methods = [
            ("otsu", self._preprocess_otsu),
            ("adaptive_gaussian", self._preprocess_adaptive_gaussian),
            ("contrast_enhanced", self._preprocess_contrast_enhanced),
        ]
        
        fallback_methods = [
            ("adaptive_mean", self._preprocess_adaptive_mean),
            ("morphological", self._preprocess_morphological),
            ("sharpened", self._preprocess_sharpened),
            ("denoise", self._preprocess_denoise),
        ]
        
        if is_fallback:
            return primary_methods + fallback_methods
        return primary_methods

    def _is_cell_likely_empty(self, gray_img: np.ndarray, cell_id: str = "") -> bool:
        """
        Check if a cell is likely empty using multiple heuristics.
        
        Args:
            gray_img: Grayscale image of the cell.
            cell_id: Cell identifier for logging.
            
        Returns:
            True if the cell is likely empty, False otherwise.
        """
        # Heuristic 1: Check variance
        variance = np.var(gray_img)
        if variance < self.empty_cell_variance_threshold:
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: Empty (low variance: {variance:.2f})")
            return True
        
        # Heuristic 2: Apply Otsu's threshold and check content ratio
        _, binary = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        kernel = np.ones((3, 3), np.uint8)
        binary_cleaned = cv2.erode(binary, kernel, iterations=1)
        
        content_pixels = np.sum(binary_cleaned == 255)
        total_pixels = binary_cleaned.size
        content_ratio = content_pixels / total_pixels if total_pixels > 0 else 0
        
        if content_ratio < self.empty_cell_content_ratio_threshold:
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: Empty (low content ratio: {content_ratio:.4f})")
            return True
        
        # Heuristic 3: Check edge density
        edges = cv2.Canny(gray_img, 50, 150)
        border = 5
        if edges.shape[0] > 2 * border and edges.shape[1] > 2 * border:
            edges_inner = edges[border:-border, border:-border]
            edge_ratio = np.sum(edges_inner > 0) / edges_inner.size if edges_inner.size > 0 else 0
        else:
            edge_ratio = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        
        if edge_ratio < 0.005:
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: Empty (low edge ratio: {edge_ratio:.4f})")
            return True
        
        # Heuristic 4: Check standard deviation in central region
        h, w = gray_img.shape
        if h > 10 and w > 10:
            central_region = gray_img[h//4:3*h//4, w//4:3*w//4]
            central_std = np.std(central_region)
            if central_std < 5:
                if self.verbose_logging:
                    self.logger.debug(f"{cell_id}: Empty (low central std: {central_std:.2f})")
                return True
        
        return False

    def _detect_orientation_with_osd(self, img: np.ndarray) -> Tuple[int, float]:
        """
        Detect text orientation using Tesseract's OSD.
        
        Args:
            img: Input image (grayscale).
            
        Returns:
            Tuple of (rotation_angle, confidence) or (0, 0) if detection fails.
        """
        try:
            if img.shape[0] < 50 or img.shape[1] < 50:
                return 0, 0
            
            upscaled = self._upscale_image(img)
            
            osd_result = pytesseract.image_to_osd(
                upscaled, config=self.osd_config, output_type=pytesseract.Output.DICT
            )
            
            rotation = int(osd_result.get('rotate', 0))
            confidence = float(osd_result.get('orientation_conf', 0))
            
            if self.verbose_logging:
                self.logger.debug(f"OSD result: rotation={rotation}°, confidence={confidence:.1f}")
            
            if confidence >= self.osd_confidence_threshold:
                if rotation == 90:
                    return 270, confidence
                elif rotation == 180:
                    return 180, confidence
                elif rotation == 270:
                    return 90, confidence
            
            return 0, confidence
            
        except Exception as e:
            if self.verbose_logging:
                self.logger.debug(f"OSD failed: {str(e)}")
            return 0, 0

    def _calculate_text_quality_score(
        self, 
        text: str, 
        confidence: float, 
        is_rotated: bool = False, 
        expects_rotation: bool = False
    ) -> float:
        """
        Calculate a quality score for extracted text.
        
        Args:
            text: Extracted text.
            confidence: OCR confidence.
            is_rotated: Whether this text came from a rotated image.
            expects_rotation: Whether we expect the text to be rotated.
            
        Returns:
            Quality score (0-100+).
        """
        if not text:
            return 0
        
        score = confidence
        text_length = len(text.strip())
        
        # Penalize very short text unless high confidence
        if text_length <= 2 and confidence < 80:
            score *= 0.5
        elif text_length >= 3:
            score *= (1 + min(0.3, text_length / 20))
        
        # Check for meaningful content
        alphanumeric_chars = sum(1 for c in text if c.isalnum())
        total_chars = len(text.replace(" ", ""))
        
        if total_chars > 0:
            alphanumeric_ratio = alphanumeric_chars / total_chars
            
            if alphanumeric_ratio < 0.3:
                score *= 0.6
            elif alphanumeric_ratio > 0.7:
                score *= 1.1
        
        # Pattern boosts
        pattern_boost = 1.0
        
        if re.search(r'\d{1,2}[/-]\d{1,2}', text):  # Dates
            pattern_boost *= 1.2
        if re.search(r'\b[A-Z]{2,4}\s*\d{3,4}\b', text):  # Course codes
            pattern_boost *= 1.3
        if re.search(r'\b\d{1,2}:\d{2}\b', text):  # Times
            pattern_boost *= 1.2
        if re.search(r'\b(MON|TUE|WED|THU|FRI|SAT|SUN|LUN|MAR|MIE|JOI|VIN|SÂM|DUM)\b', text, re.I):
            pattern_boost *= 1.2
        if re.search(r'\b(ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)\b', text, re.I):
            pattern_boost *= 1.15
        
        score *= pattern_boost
        
        # Rotation logic
        if expects_rotation:
            if is_rotated and text_length >= self.min_text_length_for_rotation:
                score *= self.rotation_score_boost
            elif not is_rotated:
                if score < 85:
                    score *= 0.8
        
        # Penalize suspicious patterns
        if re.match(r'^(.)\1+$', text.strip()):
            score *= 0.3
        if re.match(r'^[^\w\s]+$', text):
            score *= 0.2
        
        return score

    def _is_text_valid(self, text: str, confidence: float) -> bool:
        """
        Check if the extracted text is valid (not noise).
        
        Args:
            text: Extracted text.
            confidence: OCR confidence.
            
        Returns:
            True if text appears valid, False otherwise.
        """
        if not text:
            return False
        
        stripped_text = text.strip()
        
        if len(stripped_text) <= 1 and confidence < 80:
            return False
        
        alphanumeric_chars = sum(1 for c in stripped_text if c.isalnum())
        if alphanumeric_chars == 0:
            return False
        
        if len(stripped_text) > 0:
            total_non_space = len(stripped_text.replace(" ", ""))
            if total_non_space > 0:
                alphanumeric_ratio = alphanumeric_chars / total_non_space
                if alphanumeric_ratio < 0.3 and confidence < 70:
                    return False
        
        if re.match(r'^(.)\1{3,}$', stripped_text):
            return False
        
        return True

    def _should_try_rotation(self, cell: dict) -> bool:
        """Determine if the cell text might be rotated based on rowspan."""
        rowspan = cell.get("rowspan", 1)
        if rowspan > 1:
            if self.verbose_logging:
                self.logger.debug(f"Rowspan > 1 detected ({rowspan}), will try rotation")
            return True
        return False

    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate an image by a given angle."""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _run_single_ocr(
        self, 
        img: np.ndarray, 
        psm_mode: int, 
        preprocessing_name: str,
        preprocess_func: callable
    ) -> Tuple[str, float]:
        """
        Run OCR with a specific PSM mode and preprocessing.
        
        Args:
            img: Grayscale image.
            psm_mode: PSM mode to use.
            preprocessing_name: Name of preprocessing method.
            preprocess_func: Preprocessing function to apply.
            
        Returns:
            Tuple of (text, confidence).
        """
        try:
            # Upscale
            upscaled = self._upscale_image(img)
            
            # Preprocess
            preprocessed = preprocess_func(upscaled)
            
            # OCR
            config = self._get_ocr_config(psm_mode)
            results = pytesseract.image_to_data(
                preprocessed, output_type=pytesseract.Output.DICT, config=config
            )
            
            # Extract text and confidence
            text = " ".join([word for word in results["text"] if str(word).strip()])
            
            confidences = [
                conf
                for conf, word in zip(results["conf"], results["text"])
                if str(word).strip() and conf != -1
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return text, avg_confidence
            
        except Exception as e:
            if self.verbose_logging:
                self.logger.debug(f"OCR failed (PSM {psm_mode}, {preprocessing_name}): {str(e)}")
            return "", 0

    def _try_all_ocr_combinations(
        self,
        gray_img: np.ndarray,
        cell_id: str = "",
        expects_rotation: bool = False,
        is_fallback: bool = False,
        rotation: int = 0
    ) -> List[OCRResult]:
        """
        Try multiple OCR configurations and collect all results.
        
        Args:
            gray_img: Grayscale image.
            cell_id: Cell identifier for logging.
            expects_rotation: Whether we expect the text to be rotated.
            is_fallback: Whether to use fallback/aggressive modes.
            rotation: Current rotation angle.
            
        Returns:
            List of OCRResult objects.
        """
        results = []
        
        # Select PSM modes based on fallback status
        psm_modes = self.PRIMARY_PSM_MODES.copy()
        if is_fallback:
            psm_modes.extend(self.FALLBACK_PSM_MODES)
        
        # Get preprocessing methods
        preprocessing_methods = self._get_preprocessing_methods(is_fallback)
        
        for psm_mode in psm_modes:
            for prep_name, prep_func in preprocessing_methods:
                text, confidence = self._run_single_ocr(
                    gray_img, psm_mode, prep_name, prep_func
                )
                
                if text:
                    quality_score = self._calculate_text_quality_score(
                        text, confidence, 
                        is_rotated=(rotation != 0), 
                        expects_rotation=expects_rotation
                    )
                    
                    result = OCRResult(
                        text=text,
                        confidence=confidence,
                        quality_score=quality_score,
                        psm_mode=psm_mode,
                        preprocessing=prep_name,
                        rotation=rotation
                    )
                    results.append(result)
                    
                    if self.verbose_logging:
                        self.logger.debug(f"{cell_id}: {result}")
                    
                    # Early exit if we get a very high quality result
                    if quality_score >= self.high_confidence_threshold and self._is_text_valid(text, confidence):
                        if self.verbose_logging:
                            self.logger.debug(f"{cell_id}: High quality result found, stopping early")
                        return results
        
        return results

    def _try_ocr_comprehensive(
        self,
        gray_img: np.ndarray,
        cell_id: str = "",
        expects_rotation: bool = False
    ) -> Tuple[str, float, int]:
        """
        Comprehensive OCR attempt with multiple strategies.
        
        Args:
            gray_img: Grayscale image.
            cell_id: Cell identifier for logging.
            expects_rotation: Whether we expect the text to be rotated.
            
        Returns:
            Tuple of (best_text, best_quality_score, best_rotation).
        """
        all_results: List[OCRResult] = []
        
        # Detect orientation first
        osd_rotation, osd_confidence = self._detect_orientation_with_osd(gray_img)
        
        # List of rotations to try
        rotations_to_try = [0]  # Always try original
        
        if osd_rotation != 0 and osd_confidence >= self.osd_confidence_threshold:
            rotations_to_try.insert(0, osd_rotation)  # Try OSD suggestion first
        
        if expects_rotation:
            for angle in [90, 270]:
                if angle not in rotations_to_try:
                    rotations_to_try.append(angle)
        
        # Phase 1: Try primary modes with each rotation
        for rotation in rotations_to_try:
            img_to_process = self._rotate_image(gray_img, rotation) if rotation != 0 else gray_img
            
            results = self._try_all_ocr_combinations(
                img_to_process,
                cell_id,
                expects_rotation,
                is_fallback=False,
                rotation=rotation
            )
            all_results.extend(results)
            
            # Check if we found a high-quality result
            valid_results = [r for r in results if self._is_text_valid(r.text, r.confidence)]
            if valid_results:
                best = max(valid_results, key=lambda r: r.quality_score)
                if best.quality_score >= self.high_confidence_threshold:
                    if self.verbose_logging:
                        self.logger.info(f"{cell_id}: Phase 1 found high quality result: {best}")
                    return best.text, best.quality_score, best.rotation
        
        # Phase 2: If no good results, try fallback modes
        if self.enable_fallback_modes:
            current_best = None
            valid_results = [r for r in all_results if self._is_text_valid(r.text, r.confidence)]
            if valid_results:
                current_best = max(valid_results, key=lambda r: r.quality_score)
            
            # Only do fallback if we haven't found anything good enough
            if current_best is None or current_best.quality_score < self.minimum_confidence_threshold:
                if self.verbose_logging:
                    self.logger.debug(f"{cell_id}: Phase 1 insufficient, trying fallback modes")
                
                for rotation in rotations_to_try:
                    img_to_process = self._rotate_image(gray_img, rotation) if rotation != 0 else gray_img
                    
                    results = self._try_all_ocr_combinations(
                        img_to_process,
                        cell_id,
                        expects_rotation,
                        is_fallback=True,
                        rotation=rotation
                    )
                    all_results.extend(results)
        
        # Phase 3: Try with higher upscale if still no good results
        valid_results = [r for r in all_results if self._is_text_valid(r.text, r.confidence)]
        if not valid_results or max(r.quality_score for r in valid_results) < self.minimum_confidence_threshold:
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: Trying with 3x upscale")
            
            # Temporarily increase upscale factor
            original_upscale = self.upscale_factor
            self.upscale_factor = 3
            
            for rotation in rotations_to_try[:2]:  # Only try first 2 rotations
                img_to_process = self._rotate_image(gray_img, rotation) if rotation != 0 else gray_img
                
                # Try just primary modes with higher upscale
                results = self._try_all_ocr_combinations(
                    img_to_process,
                    cell_id,
                    expects_rotation,
                    is_fallback=False,
                    rotation=rotation
                )
                all_results.extend(results)
            
            self.upscale_factor = original_upscale
        
        # Select best result
        valid_results = [r for r in all_results if self._is_text_valid(r.text, r.confidence)]
        
        if not valid_results:
            # If no valid results, try to find anything with some confidence
            if all_results:
                # Lower our standards for edge cases
                semi_valid = [r for r in all_results 
                             if r.text.strip() and r.confidence >= self.fallback_confidence_threshold]
                if semi_valid:
                    best = max(semi_valid, key=lambda r: r.quality_score)
                    if self.verbose_logging:
                        self.logger.debug(f"{cell_id}: Using semi-valid result: {best}")
                    return best.text, best.quality_score, best.rotation
            
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: No valid results found")
            return "", 0, 0
        
        best = max(valid_results, key=lambda r: r.quality_score)
        
        if self.verbose_logging:
            self.logger.info(f"{cell_id}: Best result: {best}")
        
        return best.text, best.quality_score, best.rotation

    def _post_process_text(self, text: str) -> str:
        """
        Post-process extracted text to fix common OCR errors.
        
        Args:
            text: Raw text from OCR.
            
        Returns:
            Post-processed text.
        """
        if not text:
            return text

        text = " ".join(text.split())
        text = text.replace("|", "I")
        
        # Romanian diacritics fixes
        replacements = {
            "ã": "ă", "â": "â", "î": "î", "ş": "ș", "ţ": "ț",
            "Ã": "Ă", "Â": "Â", "Î": "Î", "Ş": "Ș", "Ţ": "Ț",
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)

        words = text.split()
        processed_words = []

        for word in words:
            if word and all(c in "I1l" for c in word):
                if len(word) == 1:
                    processed_words.append("I")
                else:
                    processed_words.append(word.replace("1", "I").replace("l", "I"))
            elif word in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]:
                processed_words.append(word.replace("1", "I").replace("l", "I"))
            elif any(pattern in word.upper() for pattern in ["CS", "MATH", "PHYS", "CHEM", "BIO"]):
                processed_words.append(word.replace("I", "1"))
            else:
                processed_words.append(word)

        return " ".join(processed_words)

    @timed_operation("Cell OCR extraction")
    def extract_cell_text(self, img: np.ndarray, cell: dict) -> str:
        """
        Extract text from a cell using OCR.
        
        Args:
            img: Original image
            cell: Cell information with bounds, text, rowspan, and colspan
            
        Returns:
            Extracted text
        """
        cell_id = f"Cell[{cell['x1']},{cell['y1']}-{cell['x2']},{cell['y2']}]"
        
        with LogContext(__name__, 
                        temp_level=logging.DEBUG if self.verbose_logging else logging.WARNING):
            
            if self.verbose_logging:
                self.logger.info("="*60)
                self.logger.info(f"Processing {cell_id} (rowspan: {cell.get('rowspan', 1)})")
            
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

                # Check if cell is likely empty BEFORE running OCR
                if self._is_cell_likely_empty(gray, cell_id):
                    if self.verbose_logging:
                        self.logger.info(f"{cell_id}: Detected as EMPTY cell")
                    return ""

                # Check if we should expect rotated text
                expects_rotation = self._should_try_rotation(cell)
                
                # Perform comprehensive OCR
                best_text, best_quality_score, best_rotation = self._try_ocr_comprehensive(
                    gray, cell_id, expects_rotation
                )

                # Determine threshold
                quality_threshold = self.minimum_confidence_threshold
                
                if expects_rotation and best_quality_score > quality_threshold * 0.8:
                    quality_threshold *= 0.8

                # Final validation and return
                if best_quality_score >= quality_threshold:
                    if self._is_text_valid(best_text, best_quality_score):
                        if self.verbose_logging:
                            rotation_info = f" (rotation: {best_rotation}°)" if best_rotation else ""
                            self.logger.info(
                                f"{cell_id}: ACCEPTED (score: {best_quality_score:.1f}){rotation_info} "
                                f"- '{best_text}'"
                            )
                        return self._post_process_text(best_text).strip()
                    else:
                        if self.verbose_logging:
                            self.logger.info(f"{cell_id}: REJECTED (invalid text pattern)")
                        return ""
                else:
                    # Last resort: if we have ANY text with reasonable confidence, return it
                    if best_text and best_quality_score >= self.fallback_confidence_threshold:
                        if self.verbose_logging:
                            self.logger.info(
                                f"{cell_id}: ACCEPTED via fallback (score: {best_quality_score:.1f}) "
                                f"- '{best_text}'"
                            )
                        return self._post_process_text(best_text).strip()
                    
                    if self.verbose_logging:
                        self.logger.info(
                            f"{cell_id}: REJECTED (score {best_quality_score:.1f} < {quality_threshold})"
                        )
                    return ""

            except Exception as e:
                self.logger.error(f"{cell_id}: Error during extraction: {str(e)}", 
                                exc_info=self.verbose_logging)
                return ""

    def debug_cell(self, img: np.ndarray, cell: dict, save_path: str = None) -> dict:
        """
        Debug method to analyze a cell and show all OCR attempts.
        
        Args:
            img: Original image
            cell: Cell information
            save_path: Optional path to save debug images
            
        Returns:
            Dictionary with debug information
        """
        cell_id = f"Cell[{cell['x1']},{cell['y1']}-{cell['x2']},{cell['y2']}]"
        
        # Extract cell
        padding = 3
        y1 = max(0, cell["y1"] - padding)
        y2 = min(img.shape[0], cell["y2"] + padding)
        x1 = max(0, cell["x1"] - padding)
        x2 = min(img.shape[1], cell["x2"] + padding)
        
        cell_img = img[y1:y2, x1:x2]
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        
        debug_info = {
            "cell_id": cell_id,
            "size": f"{cell_img.shape[1]}x{cell_img.shape[0]}",
            "variance": float(np.var(gray)),
            "std": float(np.std(gray)),
            "is_empty_detected": self._is_cell_likely_empty(gray),
            "all_results": [],
        }
        
        # Try all combinations
        expects_rotation = self._should_try_rotation(cell)
        
        for rotation in [0, 90, 270]:
            rotated = self._rotate_image(gray, rotation) if rotation != 0 else gray
            upscaled = self._upscale_image(rotated)
            
            for psm_mode in self.PRIMARY_PSM_MODES + self.FALLBACK_PSM_MODES:
                for prep_name, prep_func in self._get_preprocessing_methods(True):
                    text, confidence = self._run_single_ocr(rotated, psm_mode, prep_name, prep_func)
                    
                    debug_info["all_results"].append({
                        "rotation": rotation,
                        "psm": psm_mode,
                        "preprocessing": prep_name,
                        "text": text,
                        "confidence": confidence,
                        "quality_score": self._calculate_text_quality_score(
                            text, confidence, rotation != 0, expects_rotation
                        ),
                        "is_valid": self._is_text_valid(text, confidence) if text else False
                    })
        
        # Sort by quality score
        debug_info["all_results"].sort(key=lambda x: x["quality_score"], reverse=True)
        
        # Save debug images if path provided
        if save_path:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            cv2.imwrite(f"{save_path}/original.png", cell_img)
            cv2.imwrite(f"{save_path}/gray.png", gray)
            
            upscaled = self._upscale_image(gray)
            cv2.imwrite(f"{save_path}/upscaled.png", upscaled)
            
            for prep_name, prep_func in self._get_preprocessing_methods(True):
                processed = prep_func(upscaled)
                cv2.imwrite(f"{save_path}/prep_{prep_name}.png", processed)
        
        return debug_info