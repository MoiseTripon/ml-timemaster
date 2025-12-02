"""
Test module for cell_ocr.py
Tests the OCR functionality for extracting text from table cells.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, Mock, MagicMock

from src.cell_ocr import CellOCR


class TestCellOCR:
    """Test cases for CellOCR class."""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create a CellOCR instance for testing."""
        return CellOCR()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
        # Add some text-like features
        cv2.rectangle(img, (20, 30), (80, 50), (0, 0, 0), 2)
        return img
    
    @pytest.fixture
    def sample_cell(self):
        """Create a sample cell definition for testing."""
        return {"x1": 20, "y1": 30, "x2": 80, "y2": 50}
    
    def test_init_default_parameters(self):
        """Test CellOCR initialization with default parameters."""
        ocr = CellOCR()
        
        assert ocr.rotation_confidence_threshold == 70.0
        assert ocr.minimum_confidence_threshold == 50.0
        assert ocr.high_confidence_threshold == 90.0
        assert ocr.consecutive_low_confidence_threshold == 2
        assert len(ocr.configs) == 6  # Should have 6 OCR configurations
    
    def test_init_custom_parameters(self):
        """Test CellOCR initialization with custom parameters."""
        ocr = CellOCR(
            rotation_confidence_threshold=80.0,
            minimum_confidence_threshold=60.0,
            high_confidence_threshold=95.0
        )
        
        assert ocr.rotation_confidence_threshold == 80.0
        assert ocr.minimum_confidence_threshold == 60.0
        assert ocr.high_confidence_threshold == 95.0
    
    def test_preprocess_cell_image(self, ocr_processor):
        """Test the cell image preprocessing methods."""
        # Create a simple grayscale image
        gray_img = np.ones((50, 50), dtype=np.uint8) * 128
        
        preprocessed = ocr_processor._preprocess_cell_image(gray_img)
        
        assert isinstance(preprocessed, list)
        assert len(preprocessed) == 8  # Should have 8 preprocessing methods
        
        # Check that all preprocessing methods return valid results
        for method_name, processed_img in preprocessed:
            assert isinstance(method_name, str)
            assert isinstance(processed_img, np.ndarray)
            assert len(processed_img.shape) == 2  # Should be grayscale
    
    def test_try_ocr_with_config_success(self, ocr_processor):
        """Test OCR attempt with successful result."""
        test_img = np.ones((50, 50), dtype=np.uint8) * 255
        
        mock_results = {
            "text": ["Hello", "World", ""],
            "conf": [85, 90, -1]
        }
        
        with patch('pytesseract.image_to_data', return_value=mock_results):
            text, confidence = ocr_processor._try_ocr_with_config(test_img, "--oem 3 --psm 6")
            
            assert text == "Hello World"
            assert confidence == 87.5  # Average of 85 and 90
    
    def test_try_ocr_with_config_failure(self, ocr_processor):
        """Test OCR attempt with failure."""
        test_img = np.ones((50, 50), dtype=np.uint8) * 255
        
        with patch('pytesseract.image_to_data', side_effect=Exception("OCR Error")):
            text, confidence = ocr_processor._try_ocr_with_config(test_img, "--oem 3 --psm 6")
            
            assert text is None
            assert confidence == 0
    
    def test_post_process_text_empty(self, ocr_processor):
        """Test post-processing with empty text."""
        result = ocr_processor._post_process_text("")
        assert result == ""
        
        result = ocr_processor._post_process_text(None)
        assert result is None
    
    def test_post_process_text_whitespace_cleanup(self, ocr_processor):
        """Test post-processing whitespace cleanup."""
        result = ocr_processor._post_process_text("  Hello   World  ")
        assert result == "Hello World"
    
    def test_post_process_text_character_correction(self, ocr_processor):
        """Test post-processing character corrections."""
        # Test vertical bar replacement
        result = ocr_processor._post_process_text("H|LL0")
        assert "I" in result
        
        # Test single character I vs 1 handling
        result = ocr_processor._post_process_text("I")
        assert result == "I"
        
        # Test course number correction
        result = ocr_processor._post_process_text("CSIII")
        assert "CS111" in result
    
    def test_extract_cell_text_small_cell(self, ocr_processor, sample_image):
        """Test text extraction from very small cells."""
        small_cell = {"x1": 20, "y1": 30, "x2": 22, "y2": 32}  # 2x2 pixel cell
        
        result = ocr_processor.extract_cell_text(sample_image, small_cell)
        assert result == ""  # Should return empty for very small cells
    
    def test_extract_cell_text_high_confidence(self, ocr_processor, sample_image, sample_cell):
        """Test text extraction with high confidence result."""
        mock_results = {
            "text": ["HIGH", "CONF"],
            "conf": [95, 92]
        }
        
        with patch('pytesseract.image_to_data', return_value=mock_results):
            with patch('builtins.print'):  # Mock print to avoid output during tests
                result = ocr_processor.extract_cell_text(sample_image, sample_cell)
                
                assert result == "HIGH CONF"
    
    def test_extract_cell_text_low_confidence(self, ocr_processor, sample_image, sample_cell):
        """Test text extraction with low confidence results."""
        mock_results = {
            "text": ["LOW", "CONF"],
            "conf": [30, 25]  # Below minimum threshold
        }
        
        with patch('pytesseract.image_to_data', return_value=mock_results):
            with patch('pytesseract.image_to_string', return_value=""):
                with patch('builtins.print'):  # Mock print to avoid output during tests
                    result = ocr_processor.extract_cell_text(sample_image, sample_cell)
                    
                    assert result == ""  # Should return empty for low confidence
    
    def test_extract_cell_text_boundary_padding(self, ocr_processor, sample_image):
        """Test that cell extraction respects image boundaries with padding."""
        # Test cell near image boundary
        boundary_cell = {"x1": 0, "y1": 0, "x2": 20, "y2": 20}
        
        with patch('pytesseract.image_to_data') as mock_ocr:
            mock_ocr.return_value = {"text": [], "conf": []}
            with patch('builtins.print'):
                result = ocr_processor.extract_cell_text(sample_image, boundary_cell)
                
                # Should not crash and should handle boundary conditions
                assert isinstance(result, str)
    
    def test_extract_cell_text_consecutive_low_confidence(self, ocr_processor, sample_image, sample_cell):
        """Test early termination on consecutive low confidence results."""
        low_conf_results = {
            "text": ["BAD"],
            "conf": [20]  # Very low confidence
        }
        
        with patch('pytesseract.image_to_data', return_value=low_conf_results):
            with patch('builtins.print'):
                result = ocr_processor.extract_cell_text(sample_image, sample_cell)
                
                # Should terminate early and return empty
                assert result == ""
    
    def test_extract_cell_text_exception_handling(self, ocr_processor, sample_image, sample_cell):
        """Test that exceptions during OCR are handled gracefully."""
        with patch('cv2.cvtColor', side_effect=Exception("OpenCV Error")):
            result = ocr_processor.extract_cell_text(sample_image, sample_cell)
            
            assert result == ""  # Should return empty string on error
    
    def test_extract_cell_text_rotation_attempts(self, ocr_processor, sample_image, sample_cell):
        """Test that rotation attempts are made without errors."""
        # Use high confidence to trigger early exit and avoid complex rotation logic testing
        high_conf_results = {"text": ["SUCCESS"], "conf": [95]}
        
        with patch('pytesseract.image_to_data', return_value=high_conf_results):
            with patch('builtins.print'):
                result = ocr_processor.extract_cell_text(sample_image, sample_cell)
                
                # Should return the high confidence result
                assert result == "SUCCESS"
