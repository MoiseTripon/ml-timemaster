"""
Test suite for CellOCR class.
"""

import pytest
import cv2
import numpy as np
from src.cell_ocr import CellOCR


class TestCellOCR:
    """Test cases for CellOCR class."""
    
    @pytest.fixture
    def ocr(self):
        """Create an OCR instance for testing."""
        return CellOCR(rotation_confidence_threshold=70.0, minimum_confidence_threshold=50.0)
    
    @pytest.fixture
    def sample_image_with_text(self):
        """Create a sample image with text."""
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "HELLO", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        return img
    
    @pytest.fixture
    def blank_cell_image(self):
        """Create a blank image (no text)."""
        return np.ones((50, 50, 3), dtype=np.uint8) * 255
    
    @pytest.fixture
    def small_cell_image(self):
        """Create a very small cell image."""
        return np.ones((3, 3, 3), dtype=np.uint8) * 255
    
    def test_ocr_initialization(self):
        """Test that OCR initializes with correct thresholds."""
        ocr = CellOCR(rotation_confidence_threshold=80.0, minimum_confidence_threshold=60.0, 
                      high_confidence_threshold=95.0)
        assert ocr.rotation_confidence_threshold == 80.0
        assert ocr.minimum_confidence_threshold == 60.0
        assert ocr.high_confidence_threshold == 95.0
    
    def test_ocr_default_initialization(self):
        """Test that OCR uses default thresholds when not specified."""
        ocr = CellOCR()
        assert ocr.rotation_confidence_threshold == 70.0
        assert ocr.minimum_confidence_threshold == 50.0
        assert ocr.high_confidence_threshold == 90.0
    
    def test_ocr_configs_loaded(self, ocr):
        """Test that OCR configurations are properly loaded."""
        assert len(ocr.configs) > 0
        assert all('--oem' in config for config in ocr.configs)
    
    def test_preprocess_cell_image(self, ocr):
        """Test that cell image preprocessing creates multiple versions."""
        gray_img = np.ones((50, 50), dtype=np.uint8) * 128
        preprocessed = ocr._preprocess_cell_image(gray_img)
        
        # Should return a list of tuples
        assert isinstance(preprocessed, list)
        assert len(preprocessed) > 0
        
        # Each item should be a tuple of (name, image)
        for item in preprocessed:
            assert isinstance(item, tuple)
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], np.ndarray)
    
    def test_post_process_text_with_empty_string(self, ocr):
        """Test post-processing with empty string."""
        result = ocr._post_process_text("")
        assert result == ""
    
    def test_post_process_text_with_whitespace(self, ocr):
        """Test post-processing removes extra whitespace."""
        result = ocr._post_process_text("  Hello   World  ")
        assert result == "Hello World"
    
    def test_post_process_text_replaces_vertical_bar(self, ocr):
        """Test that vertical bars are replaced with I."""
        result = ocr._post_process_text("H|LLO")
        assert "|" not in result
        assert "I" in result
    
    def test_extract_cell_text_with_small_cell(self, ocr, sample_image_with_text):
        """Test that very small cells return empty string."""
        small_cell = {'x1': 0, 'y1': 0, 'x2': 2, 'y2': 2}
        result = ocr.extract_cell_text(sample_image_with_text, small_cell)
        assert result == ""
    
    def test_extract_cell_text_with_valid_cell(self, ocr, sample_image_with_text):
        """Test text extraction from a valid cell."""
        cell = {'x1': 10, 'y1': 10, 'x2': 180, 'y2': 90}
        result = ocr.extract_cell_text(sample_image_with_text, cell)
        # Result should be a string (may be empty if OCR fails in test environment)
        assert isinstance(result, str)
    
    def test_extract_cell_text_with_out_of_bounds(self, ocr, sample_image_with_text):
        """Test that out-of-bounds cells are handled gracefully."""
        # Cell partially outside image
        cell = {'x1': -10, 'y1': -10, 'x2': 50, 'y2': 50}
        result = ocr.extract_cell_text(sample_image_with_text, cell)
        assert isinstance(result, str)
    
    def test_extract_cell_text_handles_exceptions(self, ocr):
        """Test that OCR handles exceptions gracefully."""
        # Create invalid image data
        invalid_img = np.array([])
        cell = {'x1': 0, 'y1': 0, 'x2': 10, 'y2': 10}
        result = ocr.extract_cell_text(invalid_img, cell)
        assert result == ""
    
    def test_try_ocr_with_config_returns_tuple(self, ocr):
        """Test that _try_ocr_with_config returns proper tuple."""
        gray_img = np.ones((50, 50), dtype=np.uint8) * 255
        cv2.putText(gray_img, "TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        result = ocr._try_ocr_with_config(gray_img, '--oem 3 --psm 6')
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (str, type(None)))
        assert isinstance(result[1], (int, float))
    
    def test_high_confidence_threshold(self, ocr):
        """Test that high confidence threshold parameter is set correctly."""
        custom_ocr = CellOCR(high_confidence_threshold=85.0)
        assert custom_ocr.high_confidence_threshold == 85.0
    
    def test_consecutive_low_confidence_threshold(self, ocr):
        """Test that consecutive low confidence threshold is set."""
        assert ocr.consecutive_low_confidence_threshold == 2
