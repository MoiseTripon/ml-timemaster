"""
Test module for image_preprocessor.py
Tests the image preprocessing functionality including PDF conversion and image enhancement.
"""

import pytest
import numpy as np
import cv2
import os
import tempfile
from unittest.mock import patch, Mock, MagicMock

from src.image_preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create an ImagePreprocessor instance for testing."""
        return ImagePreprocessor()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255  # White background
        cv2.rectangle(img, (10, 10), (90, 90), (0, 0, 0), 2)  # Black rectangle
        return img
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create a temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, sample_image)
            yield tmp.name
            # Cleanup
            if os.path.exists(tmp.name):
                os.remove(tmp.name)
    
    def test_init_default_dpi(self):
        """Test ImagePreprocessor initialization with default DPI."""
        preprocessor = ImagePreprocessor()
        assert preprocessor.dpi == 500
    
    def test_init_custom_dpi(self):
        """Test ImagePreprocessor initialization with custom DPI."""
        preprocessor = ImagePreprocessor(dpi=300)
        assert preprocessor.dpi == 300
    
    def test_convert_pdf_to_image_success(self, preprocessor):
        """Test successful PDF to image conversion."""
        mock_image = Mock()
        mock_image.size = (100, 100)
        
        with patch('pdf2image.convert_from_path', return_value=[mock_image]):
            with patch('tempfile.NamedTemporaryFile') as mock_temp:
                mock_temp.return_value.__enter__.return_value.name = "/tmp/test.png"
                
                result = preprocessor.convert_pdf_to_image("test.pdf")
                
                assert result == "/tmp/test.png"
                mock_image.save.assert_called_once()
    
    def test_convert_pdf_to_image_no_pages(self, preprocessor):
        """Test PDF conversion when no pages are found."""
        with patch('pdf2image.convert_from_path', return_value=[]):
            with pytest.raises(ValueError, match="No pages found in PDF"):
                preprocessor.convert_pdf_to_image("empty.pdf")
    
    def test_convert_pdf_to_image_exception(self, preprocessor):
        """Test PDF conversion with exception."""
        with patch('pdf2image.convert_from_path', side_effect=Exception("PDF Error")):
            with pytest.raises(ValueError, match="Error converting PDF to image"):
                preprocessor.convert_pdf_to_image("bad.pdf")
    
    def test_is_blank_image_blank(self, preprocessor):
        """Test blank image detection with a blank image."""
        # Create a mostly white image (>99% white)
        blank_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Add tiny amount of content (less than 1%)
        blank_img[50, 50] = [200, 200, 200]
        
        result = preprocessor.is_blank_image(blank_img)
        assert result is True
    
    def test_is_blank_image_not_blank(self, preprocessor, sample_image):
        """Test blank image detection with a non-blank image."""
        result = preprocessor.is_blank_image(sample_image)
        assert result is False
    
    def test_is_blank_image_edge_case(self, preprocessor):
        """Test blank image detection at the edge case (exactly 1% content)."""
        # Create image with exactly 1% non-white pixels
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        # Add exactly 100 black pixels (1% of 10,000 total pixels)
        img[:10, :10] = [0, 0, 0]  # 100 pixels
        
        result = preprocessor.is_blank_image(img)
        # Should be False as it's at the threshold
        assert result is False
    
    def test_preprocess_pdf_file(self, preprocessor):
        """Test preprocessing of a PDF file."""
        mock_image = Mock()
        mock_image.size = (200, 200)
        
        with patch('os.path.exists', return_value=True):
            with patch.object(preprocessor, 'convert_pdf_to_image', return_value="/tmp/converted.png"):
                with patch('cv2.imread', return_value=np.ones((200, 200, 3), dtype=np.uint8) * 255):
                    with patch.object(preprocessor, 'is_blank_image', return_value=False):
                        with patch('cv2.cvtColor') as mock_cvtColor:
                            with patch('cv2.adaptiveThreshold') as mock_threshold:
                                with patch('cv2.fastNlMeansDenoising') as mock_denoise:
                                    with patch('cv2.threshold', return_value=(127, np.ones((200, 200), dtype=np.uint8) * 255)) as mock_otsu:
                                        with patch('cv2.imwrite') as mock_write:
                                            with patch('os.remove'):
                                                
                                                result_path, processed_img, original_img = preprocessor.preprocess("test.pdf")
                                                
                                                assert isinstance(result_path, str)
                                                assert isinstance(processed_img, np.ndarray)
                                                assert isinstance(original_img, np.ndarray)
                                                assert "preprocessed_test.pdf.png" in result_path
    
    def test_preprocess_image_file(self, preprocessor, temp_image_file):
        """Test preprocessing of an image file."""
        with patch.object(preprocessor, 'is_blank_image', return_value=False):
            with patch('cv2.cvtColor') as mock_cvtColor:
                with patch('cv2.adaptiveThreshold') as mock_threshold:
                    with patch('cv2.fastNlMeansDenoising') as mock_denoise:
                        with patch('cv2.threshold') as mock_otsu:
                            with patch('cv2.imwrite') as mock_write:
                                
                                # Mock the various OpenCV operations
                                mock_cvtColor.return_value = np.ones((100, 100), dtype=np.uint8) * 128
                                mock_threshold.return_value = np.ones((100, 100), dtype=np.uint8) * 255
                                mock_denoise.return_value = np.ones((100, 100), dtype=np.uint8) * 255
                                mock_otsu.return_value = (127, np.ones((100, 100), dtype=np.uint8) * 255)
                                
                                result_path, processed_img, original_img = preprocessor.preprocess(temp_image_file)
                                
                                assert isinstance(result_path, str)
                                assert isinstance(processed_img, np.ndarray)
                                assert isinstance(original_img, np.ndarray)
                                assert os.path.basename(temp_image_file) in result_path
    
    def test_preprocess_file_not_found(self, preprocessor):
        """Test preprocessing with non-existent file."""
        with pytest.raises(ValueError, match="File not found"):
            preprocessor.preprocess("nonexistent.pdf")
    
    def test_preprocess_image_load_failure(self, preprocessor, temp_image_file):
        """Test preprocessing when image loading fails."""
        with patch('cv2.imread', return_value=None):
            with pytest.raises(ValueError, match="Failed to load image"):
                preprocessor.preprocess(temp_image_file)
    
    def test_preprocess_blank_image(self, preprocessor, temp_image_file):
        """Test preprocessing with a blank image."""
        with patch.object(preprocessor, 'is_blank_image', return_value=True):
            with pytest.raises(ValueError, match="The input image is blank"):
                preprocessor.preprocess(temp_image_file)
    
    def test_preprocess_processing_exception(self, preprocessor, temp_image_file):
        """Test preprocessing when processing fails."""
        with patch.object(preprocessor, 'is_blank_image', return_value=False):
            with patch('cv2.cvtColor', side_effect=Exception("Processing error")):
                with pytest.raises(ValueError, match="Error preprocessing file"):
                    preprocessor.preprocess(temp_image_file)
    
    def test_preprocess_pdf_cleanup(self, preprocessor):
        """Test that temporary PDF conversion files are cleaned up."""
        mock_image = Mock()
        mock_image.size = (100, 100)
        
        with patch('os.path.exists', return_value=True):
            with patch.object(preprocessor, 'convert_pdf_to_image', return_value="/tmp/temp_pdf.png"):
                with patch('cv2.imread', return_value=np.ones((100, 100, 3), dtype=np.uint8) * 255):
                    with patch.object(preprocessor, 'is_blank_image', return_value=False):
                        with patch('cv2.cvtColor'), patch('cv2.adaptiveThreshold'), \
                             patch('cv2.fastNlMeansDenoising'), patch('cv2.threshold', return_value=(127, np.ones((100, 100), dtype=np.uint8) * 255)), \
                             patch('cv2.imwrite'):
                            with patch('os.remove') as mock_remove:
                                
                                preprocessor.preprocess("test.pdf")
                                
                                # Should clean up the temporary file
                                mock_remove.assert_called_with("/tmp/temp_pdf.png")
    
    def test_preprocess_output_file_naming(self, preprocessor, temp_image_file):
        """Test that output files are named correctly."""
        with patch.object(preprocessor, 'is_blank_image', return_value=False):
            with patch('cv2.cvtColor'), patch('cv2.adaptiveThreshold'), \
                 patch('cv2.fastNlMeansDenoising'), patch('cv2.threshold', return_value=(127, np.ones((100, 100), dtype=np.uint8) * 255)):
                with patch('cv2.imwrite') as mock_write:
                    
                    result_path, _, _ = preprocessor.preprocess(temp_image_file)
                    
                    expected_filename = f"preprocessed_{os.path.basename(temp_image_file)}.png"
                    assert expected_filename in result_path
                    mock_write.assert_called_once()
