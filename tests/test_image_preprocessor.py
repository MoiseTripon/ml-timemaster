"""
Test suite for ImagePreprocessor class.
"""

import pytest
import cv2
import numpy as np
import os
import tempfile
from src.image_preprocessor import ImagePreprocessor


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance for testing."""
        return ImagePreprocessor(dpi=300)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple test image with some content
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 0), 2)
        cv2.putText(img, "Test", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return img
    
    @pytest.fixture
    def blank_image(self):
        """Create a blank white image for testing."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    def test_preprocessor_initialization(self):
        """Test that preprocessor initializes with correct DPI."""
        preprocessor = ImagePreprocessor(dpi=600)
        assert preprocessor.dpi == 600
    
    def test_preprocessor_default_dpi(self):
        """Test that preprocessor uses default DPI when not specified."""
        preprocessor = ImagePreprocessor()
        assert preprocessor.dpi == 500
    
    def test_is_blank_image_with_blank(self, preprocessor, blank_image):
        """Test that blank images are correctly identified."""
        result = preprocessor.is_blank_image(blank_image)
        assert result is True
    
    def test_is_blank_image_with_content(self, preprocessor, sample_image):
        """Test that images with content are not identified as blank."""
        result = preprocessor.is_blank_image(sample_image)
        assert result is False
    
    def test_is_blank_image_with_mostly_white(self, preprocessor):
        """Test edge case with mostly white image but some content."""
        # Create image that's 98% white
        img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (400, 400), (600, 600), (0, 0, 0), -1)
        result = preprocessor.is_blank_image(img)
        assert result is False
    
    def test_preprocess_nonexistent_file(self, preprocessor):
        """Test that preprocessing raises error for nonexistent files."""
        with pytest.raises(ValueError, match="File not found"):
            preprocessor.preprocess("nonexistent_file.png")
    
    def test_preprocess_image_file(self, preprocessor, sample_image):
        """Test preprocessing of a valid image file."""
        # Save test image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_image)
        
        try:
            preprocessed_path, original_img = preprocessor.preprocess(temp_path)
            
            # Check that preprocessed file was created
            assert os.path.exists(preprocessed_path)
            
            # Check that original image was returned
            assert original_img is not None
            assert original_img.shape == sample_image.shape
            
            # Clean up
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_preprocess_blank_image_raises_error(self, preprocessor, blank_image):
        """Test that preprocessing raises error for blank images."""
        # Save blank image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, blank_image)
        
        try:
            with pytest.raises(ValueError, match="blank"):
                preprocessor.preprocess(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_preprocess_creates_preprocessed_file(self, preprocessor, sample_image):
        """Test that preprocessing creates a preprocessed output file."""
        # Save test image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_image)
        
        try:
            preprocessed_path, _ = preprocessor.preprocess(temp_path)
            
            # Verify preprocessed file exists and has content
            assert os.path.exists(preprocessed_path)
            preprocessed_img = cv2.imread(preprocessed_path)
            assert preprocessed_img is not None
            
            # Clean up
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_preprocess_invalid_image_file(self, preprocessor):
        """Test that preprocessing handles invalid image files."""
        # Create a text file with .png extension
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False, mode='w') as tmp:
            temp_path = tmp.name
            tmp.write("This is not an image")
        
        try:
            with pytest.raises(ValueError):
                preprocessor.preprocess(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_preprocess_includes_binarization(self, preprocessor, sample_image):
        """Test that preprocessing includes binarization step."""
        # Save test image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_image)
        
        try:
            preprocessed_path, _ = preprocessor.preprocess(temp_path)
            
            # Read preprocessed image and check it's binary
            preprocessed_img = cv2.imread(preprocessed_path, cv2.IMREAD_GRAYSCALE)
            assert preprocessed_img is not None
            
            # Binary image should only have values 0 and 255
            unique_values = np.unique(preprocessed_img)
            assert len(unique_values) <= 2
            assert all(val in [0, 255] for val in unique_values)
            
            # Clean up
            if os.path.exists(preprocessed_path):
                os.remove(preprocessed_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
