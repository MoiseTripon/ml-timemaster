"""
OCR module for ML Timemaster.
Contains the CellOCR class for extracting text from table cells.
Uses PaddleOCR for text recognition, optimized for Apple Silicon.
"""

import logging
import cv2
import numpy as np
from src.performance_logger import LogContext, timed_operation
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import threading
import tempfile
import os
import inspect


@dataclass
class OCRResult:
    """Holds the result of an OCR attempt."""
    text: str
    confidence: float
    quality_score: float
    preprocessing: str
    rotation: int = 0
    
    def __repr__(self):
        return (f"OCRResult(text='{self.text}', conf={self.confidence:.1f}, "
                f"quality={self.quality_score:.1f}, "
                f"prep='{self.preprocessing}', rot={self.rotation}°)")


class CellOCR:
    """
    Handles OCR operations for extracting text from table cells.
    Uses PaddleOCR with multiple preprocessing techniques for best results.
    Optimized for Apple Silicon (M1/M2/M3) Macs.
    """
    
    _ocr_lock = threading.Lock()

    def __init__(
        self,
        minimum_confidence_threshold=50.0,
        high_confidence_threshold=90.0,
        verbose_logging=False,
        empty_cell_variance_threshold=100.0,
        empty_cell_content_ratio_threshold=0.01,
        rotation_score_boost=1.2,
        min_text_length_for_rotation=3,
        languages="en",
        upscale_factor=2,
        enable_fallback_modes=True,
        fallback_confidence_threshold=30.0,
    ):
        """Initialize the CellOCR with PaddleOCR."""
        self.minimum_confidence_threshold = minimum_confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.verbose_logging = verbose_logging
        self.empty_cell_variance_threshold = empty_cell_variance_threshold
        self.empty_cell_content_ratio_threshold = empty_cell_content_ratio_threshold
        self.rotation_score_boost = rotation_score_boost
        self.min_text_length_for_rotation = min_text_length_for_rotation
        self.upscale_factor = upscale_factor
        self.enable_fallback_modes = enable_fallback_modes
        self.fallback_confidence_threshold = fallback_confidence_threshold
        
        self.logger = logging.getLogger(__name__)
        self.languages = self._parse_languages(languages)
        self._ocr = None
        self._valid_ocr_params = None

    def _parse_languages(self, languages: str) -> str:
        """Parse language string to PaddleOCR format."""
        lang_map = {
            "eng": "en",
            "ron": "latin",
            "fra": "fr",
            "deu": "german",
            "spa": "es",
            "ita": "it",
            "eng+ron": "latin",
        }
        
        if languages.lower() in lang_map:
            return lang_map[languages.lower()]
        
        parts = re.split(r'[+,]', languages.lower().strip())
        mapped = [lang_map.get(p.strip(), p.strip()) for p in parts]
        
        if 'latin' in mapped or 'ro' in mapped:
            return 'latin'
        
        return mapped[0] if mapped else 'en'

    def _get_valid_paddleocr_params(self) -> set:
        """Get valid parameters for PaddleOCR constructor."""
        if self._valid_ocr_params is not None:
            return self._valid_ocr_params
        
        try:
            from paddleocr import PaddleOCR
            sig = inspect.signature(PaddleOCR.__init__)
            self._valid_ocr_params = set(sig.parameters.keys()) - {'self'}
            if self.verbose_logging:
                self.logger.info(f"Valid PaddleOCR params: {self._valid_ocr_params}")
        except Exception as e:
            self.logger.warning(f"Could not inspect PaddleOCR params: {e}")
            # Fallback to commonly supported params
            self._valid_ocr_params = {'lang', 'det', 'rec', 'cls'}
        
        return self._valid_ocr_params

    def _filter_ocr_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter parameters to only include valid ones for the installed version."""
        valid_params = self._get_valid_paddleocr_params()
        filtered = {}
        
        for key, value in params.items():
            if key in valid_params:
                filtered[key] = value
            elif self.verbose_logging:
                self.logger.debug(f"Skipping unsupported PaddleOCR param: {key}")
        
        return filtered

    def _get_ocr_instance(self):
        """Get or create a PaddleOCR instance (thread-safe)."""
        with self._ocr_lock:
            if self._ocr is None:
                from paddleocr import PaddleOCR
                
                if self.verbose_logging:
                    self.logger.info(f"Initializing PaddleOCR with language: {self.languages}")
                
                # Define all parameters we'd like to use
                desired_params = {
                    'lang': self.languages,
                    'use_angle_cls': True,
                    'use_gpu': False,
                    'show_log': self.verbose_logging,
                    'det_limit_side_len': 1920,
                    'det_limit_type': 'max',
                    'det_db_thresh': 0.2,
                    'det_db_box_thresh': 0.3,
                    'det_db_unclip_ratio': 1.8,
                    'drop_score': 0.3,
                }
                
                # Filter to only valid params
                valid_params = self._filter_ocr_params(desired_params)
                
                if self.verbose_logging:
                    self.logger.info(f"Using PaddleOCR params: {valid_params}")
                
                try:
                    self._ocr = PaddleOCR(**valid_params)
                except TypeError as e:
                    # If still failing, try minimal params
                    self.logger.warning(f"PaddleOCR init failed with params, trying minimal: {e}")
                    minimal_params = {'lang': self.languages}
                    self._ocr = PaddleOCR(**minimal_params)
                
            return self._ocr

    @property
    def ocr(self):
        """Get the main OCR instance."""
        return self._get_ocr_instance()

    def _get_valid_ocr_call_params(self) -> set:
        """Get valid parameters for the ocr() method call."""
        try:
            sig = inspect.signature(self.ocr.ocr)
            return set(sig.parameters.keys()) - {'self'}
        except Exception:
            return {'img', 'det', 'rec', 'cls'}

    def _upscale_image(self, img: np.ndarray, scale: int = None) -> np.ndarray:
        """Upscale an image by a given factor."""
        if scale is None:
            scale = self.upscale_factor
        if scale <= 1:
            return img
        
        new_width = int(img.shape[1] * scale)
        new_height = int(img.shape[0] * scale)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        """Ensure image is in BGR format."""
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def _ensure_black_text_on_white(self, binary_img: np.ndarray) -> np.ndarray:
        """Ensure text is black on white background."""
        if len(binary_img.shape) == 3:
            gray = cv2.cvtColor(binary_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = binary_img
        
        h, w = gray.shape[:2]
        margin_h = max(1, h // 4)
        margin_w = max(1, w // 4)
        
        center = gray[margin_h:h-margin_h, margin_w:w-margin_w]
        if center.size == 0:
            center = gray
        
        if np.sum(center <= 55) > np.sum(center >= 200):
            return cv2.bitwise_not(gray)
        return gray

    # Preprocessing methods
    def _preprocess_none(self, img: np.ndarray) -> np.ndarray:
        """No preprocessing."""
        return self._ensure_bgr(img)

    def _preprocess_otsu(self, img: np.ndarray) -> np.ndarray:
        """Otsu's thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = self._ensure_black_text_on_white(binary)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _preprocess_adaptive_gaussian(self, img: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        binary = self._ensure_black_text_on_white(binary)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _preprocess_adaptive_mean(self, img: np.ndarray) -> np.ndarray:
        """Adaptive mean thresholding."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        binary = self._ensure_black_text_on_white(binary)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _preprocess_contrast_enhanced(self, img: np.ndarray) -> np.ndarray:
        """CLAHE contrast enhancement."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _preprocess_morphological(self, img: np.ndarray) -> np.ndarray:
        """Morphological operations."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = self._ensure_black_text_on_white(binary)
        kernel = np.ones((2, 2), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)

    def _preprocess_sharpened(self, img: np.ndarray) -> np.ndarray:
        """Sharpening."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = self._ensure_black_text_on_white(binary)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _preprocess_denoise(self, img: np.ndarray) -> np.ndarray:
        """Denoising."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = self._ensure_black_text_on_white(binary)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _get_preprocessing_methods(self, is_fallback: bool = False) -> List[Tuple[str, callable]]:
        """Get list of preprocessing methods."""
        primary = [
            ("none", self._preprocess_none),
            ("contrast_enhanced", self._preprocess_contrast_enhanced),
            ("otsu", self._preprocess_otsu),
        ]
        fallback = [
            ("adaptive_gaussian", self._preprocess_adaptive_gaussian),
            ("adaptive_mean", self._preprocess_adaptive_mean),
            ("morphological", self._preprocess_morphological),
            ("sharpened", self._preprocess_sharpened),
            ("denoise", self._preprocess_denoise),
        ]
        return primary + fallback if is_fallback else primary

    def _is_cell_likely_empty(self, gray_img: np.ndarray, cell_id: str = "") -> bool:
        """Check if a cell is likely empty."""
        if len(gray_img.shape) == 3:
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
        
        variance = np.var(gray_img)
        if variance < self.empty_cell_variance_threshold:
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: Empty (variance: {variance:.2f})")
            return True
        
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        binary_cleaned = cv2.erode(binary, kernel, iterations=1)
        content_ratio = np.sum(binary_cleaned == 255) / binary_cleaned.size
        
        if content_ratio < self.empty_cell_content_ratio_threshold:
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: Empty (content ratio: {content_ratio:.4f})")
            return True
        
        edges = cv2.Canny(gray_img, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        if edge_ratio < 0.005:
            if self.verbose_logging:
                self.logger.debug(f"{cell_id}: Empty (edge ratio: {edge_ratio:.4f})")
            return True
        
        return False

    def _calculate_text_quality_score(
        self, text: str, confidence: float, 
        is_rotated: bool = False, expects_rotation: bool = False
    ) -> float:
        """Calculate quality score (confidence in 0-100)."""
        if not text:
            return 0
        
        score = confidence
        text_length = len(text.strip())
        
        if text_length <= 2 and confidence < 80:
            score *= 0.5
        elif text_length >= 3:
            score *= (1 + min(0.3, text_length / 20))
        
        alphanumeric = sum(1 for c in text if c.isalnum())
        total = len(text.replace(" ", ""))
        
        if total > 0:
            ratio = alphanumeric / total
            if ratio < 0.3:
                score *= 0.6
            elif ratio > 0.7:
                score *= 1.1
        
        # Pattern boosts
        if re.search(r'\d{1,2}[/-]\d{1,2}', text):
            score *= 1.2
        if re.search(r'\b[A-Z]{2,4}\s*\d{3,4}\b', text):
            score *= 1.3
        if re.search(r'\b\d{1,2}:\d{2}\b', text):
            score *= 1.2
        
        if expects_rotation and is_rotated and text_length >= self.min_text_length_for_rotation:
            score *= self.rotation_score_boost
        
        if re.match(r'^(.)\1+$', text.strip()):
            score *= 0.3
        
        return score

    def _is_text_valid(self, text: str, confidence: float) -> bool:
        """Check if text is valid."""
        if not text or not text.strip():
            return False
        
        stripped = text.strip()
        
        if len(stripped) <= 1 and confidence < 80:
            return False
        
        alphanumeric = sum(1 for c in stripped if c.isalnum())
        if alphanumeric == 0:
            return False
        
        if re.match(r'^(.)\1{3,}$', stripped):
            return False
        
        return True

    def _should_try_rotation(self, cell: dict) -> bool:
        """Check if rotation should be tried."""
        return cell.get("rowspan", 1) > 1

    def _rotate_image(self, img: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by angle."""
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _parse_ocr_result(self, result) -> List[Tuple[str, float]]:
        """
        Parse PaddleOCR result into list of (text, confidence) tuples.
        Handles various result formats from different PaddleOCR versions.
        """
        parsed = []
        
        if result is None:
            return parsed
        
        if not isinstance(result, list):
            return parsed
        
        # Handle empty result
        if len(result) == 0:
            return parsed
        
        # PaddleOCR can return different structures:
        # 1. [[line1, line2, ...]] - most common
        # 2. [line1, line2, ...] - sometimes
        # 3. [[[box], (text, conf)], ...] - direct format
        
        def extract_from_line(line):
            """Extract text and confidence from a single line result."""
            if line is None:
                return None
            
            if not isinstance(line, (list, tuple)):
                return None
            
            if len(line) < 2:
                return None
            
            # line is typically [bbox, (text, confidence)]
            text_conf = line[-1]  # Last element should be text/conf
            
            if text_conf is None:
                return None
            
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 2:
                text = str(text_conf[0]).strip()
                try:
                    conf = float(text_conf[1])
                    if text:
                        return (text, conf)
                except (ValueError, TypeError):
                    pass
            
            return None
        
        # Try to parse as [[line1, line2, ...]]
        first_elem = result[0]
        
        if first_elem is None:
            return parsed
        
        if isinstance(first_elem, list) and len(first_elem) > 0:
            # Check if first_elem[0] is a line or another list
            if first_elem[0] is None:
                return parsed
            
            # If first_elem[0] is a list of points (bbox), then first_elem is a line
            if (isinstance(first_elem[0], (list, tuple)) and 
                len(first_elem[0]) > 0 and
                isinstance(first_elem[0][0], (int, float, list, tuple))):
                
                # Check if it's bbox format [[x,y], [x,y], ...] or line format
                if (isinstance(first_elem[0][0], (list, tuple)) or 
                    (isinstance(first_elem[0][0], (int, float)) and len(first_elem) == 2)):
                    # This looks like a single line, result is [line1, line2, ...]
                    for line in result:
                        extracted = extract_from_line(line)
                        if extracted:
                            parsed.append(extracted)
                else:
                    # first_elem is [line1, line2, ...], result is [[lines]]
                    for line in first_elem:
                        extracted = extract_from_line(line)
                        if extracted:
                            parsed.append(extracted)
            else:
                # first_elem contains lines
                for line in first_elem:
                    extracted = extract_from_line(line)
                    if extracted:
                        parsed.append(extracted)
        
        # If nothing parsed, try iterating result directly
        if not parsed:
            for item in result:
                if item is None:
                    continue
                if isinstance(item, list):
                    for line in item:
                        extracted = extract_from_line(line)
                        if extracted:
                            parsed.append(extracted)
        
        return parsed

    def _run_paddleocr(
        self, img: np.ndarray, preprocessing_name: str, preprocess_func: callable
    ) -> Tuple[str, float]:
        """
        Run PaddleOCR with specific preprocessing.
        Returns (text, confidence) where confidence is 0-100.
        """
        try:
            # Apply preprocessing
            preprocessed = preprocess_func(img)
            
            # Ensure correct format
            if preprocessed.dtype != np.uint8:
                preprocessed = preprocessed.astype(np.uint8)
            
            if not preprocessed.flags['C_CONTIGUOUS']:
                preprocessed = np.ascontiguousarray(preprocessed)
            
            if self.verbose_logging:
                self.logger.debug(
                    f"OCR with {preprocessing_name}, shape: {preprocessed.shape}, "
                    f"dtype: {preprocessed.dtype}"
                )
            
            # Try calling OCR with different parameter combinations
            result = None
            
            # Get valid parameters for the ocr() method
            valid_call_params = self._get_valid_ocr_call_params()
            
            # Try with cls parameter if supported
            try:
                if 'cls' in valid_call_params:
                    result = self.ocr.ocr(preprocessed, cls=True)
                else:
                    result = self.ocr.ocr(preprocessed)
            except TypeError:
                # Fallback to no extra params
                result = self.ocr.ocr(preprocessed)
            
            if self.verbose_logging:
                self.logger.debug(f"Raw result type: {type(result)}, result: {result}")
            
            # Parse result
            parsed = self._parse_ocr_result(result)
            
            if self.verbose_logging:
                self.logger.debug(f"Parsed: {parsed}")
            
            if not parsed:
                return "", 0.0
            
            # Combine texts
            texts = [t for t, c in parsed]
            confidences = [c for t, c in parsed]
            
            combined = " ".join(texts)
            
            # Weighted average
            weights = [len(t) for t in texts]
            total_weight = sum(weights)
            if total_weight > 0:
                avg_conf = sum(c * w for c, w in zip(confidences, weights)) / total_weight
            else:
                avg_conf = sum(confidences) / len(confidences)
            
            # Convert to 0-100 scale
            avg_conf_100 = avg_conf * 100
            
            if self.verbose_logging:
                self.logger.debug(f"Result: '{combined}' (conf: {avg_conf_100:.1f})")
            
            return combined, avg_conf_100
            
        except Exception as e:
            if self.verbose_logging:
                self.logger.error(f"OCR failed ({preprocessing_name}): {e}", exc_info=True)
            return "", 0.0

    def _run_paddleocr_with_file(
        self, img: np.ndarray, preprocessing_name: str, preprocess_func: callable
    ) -> Tuple[str, float]:
        """
        Run PaddleOCR by saving to temp file first.
        Fallback method if numpy array doesn't work.
        """
        try:
            preprocessed = preprocess_func(img)
            
            if preprocessed.dtype != np.uint8:
                preprocessed = preprocessed.astype(np.uint8)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            cv2.imwrite(temp_path, preprocessed)
            
            try:
                # Get valid parameters
                valid_call_params = self._get_valid_ocr_call_params()
                
                try:
                    if 'cls' in valid_call_params:
                        result = self.ocr.ocr(temp_path, cls=True)
                    else:
                        result = self.ocr.ocr(temp_path)
                except TypeError:
                    result = self.ocr.ocr(temp_path)
                
                if self.verbose_logging:
                    self.logger.debug(f"File-based result: {result}")
                
                parsed = self._parse_ocr_result(result)
                
                if not parsed:
                    return "", 0.0
                
                texts = [t for t, c in parsed]
                confidences = [c for t, c in parsed]
                combined = " ".join(texts)
                
                weights = [len(t) for t in texts]
                total_weight = sum(weights)
                avg_conf = sum(c * w for c, w in zip(confidences, weights)) / total_weight if total_weight > 0 else 0
                
                return combined, avg_conf * 100
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except Exception as e:
            if self.verbose_logging:
                self.logger.error(f"File-based OCR failed: {e}", exc_info=True)
            return "", 0.0

    def _try_all_ocr_combinations(
        self, img: np.ndarray, cell_id: str = "",
        expects_rotation: bool = False, is_fallback: bool = False, rotation: int = 0
    ) -> List[OCRResult]:
        """Try multiple OCR configurations."""
        results = []
        preprocessing_methods = self._get_preprocessing_methods(is_fallback)
        
        # Upscale
        upscaled = self._upscale_image(img)
        
        for prep_name, prep_func in preprocessing_methods:
            # Try numpy array first
            text, confidence = self._run_paddleocr(upscaled, prep_name, prep_func)
            
            # If no result, try with file
            if not text:
                text, confidence = self._run_paddleocr_with_file(upscaled, prep_name, prep_func)
            
            if text:
                quality = self._calculate_text_quality_score(
                    text, confidence, rotation != 0, expects_rotation
                )
                result = OCRResult(
                    text=text, confidence=confidence,
                    quality_score=quality, preprocessing=prep_name, rotation=rotation
                )
                results.append(result)
                
                if self.verbose_logging:
                    self.logger.debug(f"{cell_id}: {result}")
                
                if quality >= self.high_confidence_threshold and self._is_text_valid(text, confidence):
                    return results
        
        return results

    def _try_ocr_comprehensive(
        self, img: np.ndarray, cell_id: str = "", expects_rotation: bool = False
    ) -> Tuple[str, float, int]:
        """Comprehensive OCR with multiple strategies."""
        all_results: List[OCRResult] = []
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        rotations = [0]
        if expects_rotation:
            rotations.extend([90, 270])
        
        # Phase 1: Primary methods
        for rotation in rotations:
            rotated = self._rotate_image(img, rotation) if rotation else img
            results = self._try_all_ocr_combinations(
                rotated, cell_id, expects_rotation, False, rotation
            )
            all_results.extend(results)
            
            valid = [r for r in results if self._is_text_valid(r.text, r.confidence)]
            if valid:
                best = max(valid, key=lambda r: r.quality_score)
                if best.quality_score >= self.high_confidence_threshold:
                    return best.text, best.quality_score, best.rotation
        
        # Phase 2: Fallback methods
        if self.enable_fallback_modes:
            valid = [r for r in all_results if self._is_text_valid(r.text, r.confidence)]
            current_best = max(valid, key=lambda r: r.quality_score) if valid else None
            
            if not current_best or current_best.quality_score < self.minimum_confidence_threshold:
                for rotation in rotations:
                    rotated = self._rotate_image(img, rotation) if rotation else img
                    results = self._try_all_ocr_combinations(
                        rotated, cell_id, expects_rotation, True, rotation
                    )
                    all_results.extend(results)
        
        # Phase 3: Higher upscale
        valid = [r for r in all_results if self._is_text_valid(r.text, r.confidence)]
        if not valid or max(r.quality_score for r in valid) < self.minimum_confidence_threshold:
            original = self.upscale_factor
            self.upscale_factor = 3
            
            for rotation in rotations[:2]:
                rotated = self._rotate_image(img, rotation) if rotation else img
                results = self._try_all_ocr_combinations(
                    rotated, cell_id, expects_rotation, False, rotation
                )
                all_results.extend(results)
            
            self.upscale_factor = original
        
        # Select best
        valid = [r for r in all_results if self._is_text_valid(r.text, r.confidence)]
        
        if not valid:
            semi = [r for r in all_results if r.text.strip() and r.confidence >= self.fallback_confidence_threshold]
            if semi:
                best = max(semi, key=lambda r: r.quality_score)
                return best.text, best.quality_score, best.rotation
            return "", 0, 0
        
        best = max(valid, key=lambda r: r.quality_score)
        return best.text, best.quality_score, best.rotation

    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text."""
        if not text:
            return text
        
        text = " ".join(text.split())
        text = text.replace("|", "I")
        
        replacements = {
            "ã": "ă", "ş": "ș", "ţ": "ț",
            "Ã": "Ă", "Ş": "Ș", "Ţ": "Ț",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text

    @timed_operation("Cell OCR extraction")
    def extract_cell_text(self, img: np.ndarray, cell: dict) -> str:
        """Extract text from a cell using OCR."""
        cell_id = f"Cell[{cell['x1']},{cell['y1']}-{cell['x2']},{cell['y2']}]"
        
        with LogContext(__name__, temp_level=logging.DEBUG if self.verbose_logging else logging.WARNING):
            if self.verbose_logging:
                self.logger.info("=" * 60)
                self.logger.info(f"Processing {cell_id}")
            
            try:
                padding = 3
                y1 = max(0, cell["y1"] - padding)
                y2 = min(img.shape[0], cell["y2"] + padding)
                x1 = max(0, cell["x1"] - padding)
                x2 = min(img.shape[1], cell["x2"] + padding)
                
                cell_img = img[y1:y2, x1:x2].copy()
                
                if cell_img.size == 0 or cell_img.shape[0] < 5 or cell_img.shape[1] < 5:
                    return ""
                
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
                
                if self._is_cell_likely_empty(gray, cell_id):
                    return ""
                
                expects_rotation = self._should_try_rotation(cell)
                best_text, best_score, best_rotation = self._try_ocr_comprehensive(
                    cell_img, cell_id, expects_rotation
                )
                
                threshold = self.minimum_confidence_threshold
                if expects_rotation and best_score > threshold * 0.8:
                    threshold *= 0.8
                
                if best_score >= threshold and self._is_text_valid(best_text, best_score):
                    return self._post_process_text(best_text).strip()
                
                if best_text and best_score >= self.fallback_confidence_threshold:
                    return self._post_process_text(best_text).strip()
                
                return ""
                
            except Exception as e:
                self.logger.error(f"{cell_id}: {e}", exc_info=self.verbose_logging)
                return ""

    def debug_cell(self, img: np.ndarray, cell: dict, save_path: str = None) -> dict:
        """Debug a cell's OCR attempts."""
        cell_id = f"Cell[{cell['x1']},{cell['y1']}-{cell['x2']},{cell['y2']}]"
        
        padding = 3
        y1 = max(0, cell["y1"] - padding)
        y2 = min(img.shape[0], cell["y2"] + padding)
        x1 = max(0, cell["x1"] - padding)
        x2 = min(img.shape[1], cell["x2"] + padding)
        
        cell_img = img[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY) if len(cell_img.shape) == 3 else cell_img
        
        debug_info = {
            "cell_id": cell_id,
            "size": f"{cell_img.shape[1]}x{cell_img.shape[0]}",
            "variance": float(np.var(gray)),
            "std": float(np.std(gray)),
            "is_empty_detected": self._is_cell_likely_empty(gray),
            "all_results": [],
        }
        
        original_verbose = self.verbose_logging
        self.verbose_logging = True
        
        try:
            if len(cell_img.shape) == 2:
                cell_img = cv2.cvtColor(cell_img, cv2.COLOR_GRAY2BGR)
            
            expects_rotation = self._should_try_rotation(cell)
            
            for rotation in [0, 90, 270]:
                rotated = self._rotate_image(cell_img, rotation) if rotation else cell_img
                upscaled = self._upscale_image(rotated)
                
                for prep_name, prep_func in self._get_preprocessing_methods(True):
                    text1, conf1 = self._run_paddleocr(upscaled, prep_name, prep_func)
                    text2, conf2 = self._run_paddleocr_with_file(upscaled, prep_name, prep_func)
                    
                    text, conf = (text1, conf1) if conf1 >= conf2 else (text2, conf2)
                    method = "numpy" if conf1 >= conf2 else "file"
                    
                    debug_info["all_results"].append({
                        "rotation": rotation,
                        "preprocessing": prep_name,
                        "text": text,
                        "confidence": conf,
                        "quality_score": self._calculate_text_quality_score(text, conf, rotation != 0, expects_rotation),
                        "is_valid": self._is_text_valid(text, conf) if text else False,
                        "method": method,
                    })
        finally:
            self.verbose_logging = original_verbose
        
        debug_info["all_results"].sort(key=lambda x: x["quality_score"], reverse=True)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(f"{save_path}/original.png", cell_img)
            cv2.imwrite(f"{save_path}/gray.png", gray)
            
            upscaled = self._upscale_image(cell_img)
            cv2.imwrite(f"{save_path}/upscaled.png", upscaled)
            
            for prep_name, prep_func in self._get_preprocessing_methods(True):
                processed = prep_func(upscaled)
                cv2.imwrite(f"{save_path}/prep_{prep_name}.png", processed)
        
        return debug_info

    def cleanup(self):
        """Clean up resources."""
        with self._ocr_lock:
            self._ocr = None


# Standalone test function
def test_paddleocr_installation():
    """Test PaddleOCR installation and functionality."""
    import sys
    print(f"Python version: {sys.version}")
    
    try:
        import paddle
        print(f"PaddlePaddle version: {paddle.__version__}")
    except ImportError as e:
        print(f"PaddlePaddle not installed: {e}")
        return False
    
    try:
        from paddleocr import PaddleOCR
        print("PaddleOCR imported successfully")
    except ImportError as e:
        print(f"PaddleOCR not installed: {e}")
        return False
    
    # Check constructor params
    print("\n--- PaddleOCR Constructor Parameters ---")
    try:
        sig = inspect.signature(PaddleOCR.__init__)
        params = list(sig.parameters.keys())
        print(f"Available params: {params}")
    except Exception as e:
        print(f"Could not inspect: {e}")
    
    # Try minimal init
    print("\n--- Testing Initialization ---")
    try:
        ocr = PaddleOCR(lang='en')
        print("Initialization successful!")
    except Exception as e:
        print(f"Initialization failed: {e}")
        return False
    
    # Check ocr method params
    print("\n--- OCR Method Parameters ---")
    try:
        sig = inspect.signature(ocr.ocr)
        params = list(sig.parameters.keys())
        print(f"ocr() params: {params}")
    except Exception as e:
        print(f"Could not inspect ocr method: {e}")
    
    # Test with synthetic image
    print("\n--- Testing OCR ---")
    try:
        # Create test image
        test_img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "Hello World", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        # Test with numpy array
        print("Testing with numpy array...")
        result = ocr.ocr(test_img)
        print(f"Numpy result: {result}")
        
        # Test with file
        print("\nTesting with file...")
        temp_path = "/tmp/paddleocr_test.png"
        cv2.imwrite(temp_path, test_img)
        result_file = ocr.ocr(temp_path)
        print(f"File result: {result_file}")
        
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        return True
        
    except Exception as e:
        print(f"OCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_paddleocr_installation()