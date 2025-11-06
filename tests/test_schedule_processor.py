"""
Test suite for ScheduleProcessor class.
"""

import pytest
import cv2
import numpy as np
import os
import tempfile
from src.schedule_processor import ScheduleProcessor


class TestScheduleProcessor:
    """Test cases for ScheduleProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a processor instance for testing."""
        return ScheduleProcessor(
            dpi=300,
            overlap_threshold=0.7,
            min_cell_size=5,
            rotation_confidence_threshold=70.0,
            minimum_confidence_threshold=50.0
        )
    
    @pytest.fixture
    def sample_schedule_image(self):
        """Create a sample schedule table image."""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Draw table border
        cv2.rectangle(img, (50, 50), (550, 350), (0, 0, 0), 3)
        
        # Draw grid lines
        cv2.line(img, (50, 150), (550, 150), (0, 0, 0), 2)
        cv2.line(img, (50, 250), (550, 250), (0, 0, 0), 2)
        cv2.line(img, (250, 50), (250, 350), (0, 0, 0), 2)
        cv2.line(img, (400, 50), (400, 350), (0, 0, 0), 2)
        
        # Add some text
        cv2.putText(img, "Mon", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Tue", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return img
    
    def test_processor_initialization(self):
        """Test that processor initializes with correct parameters."""
        processor = ScheduleProcessor(
            dpi=600,
            overlap_threshold=0.8,
            min_cell_size=10,
            rotation_confidence_threshold=80.0,
            minimum_confidence_threshold=60.0,
            high_confidence_threshold=95.0
        )
        assert processor.preprocessor.dpi == 600
        assert processor.table_analyzer.overlap_threshold == 0.8
        assert processor.table_analyzer.min_cell_size == 10
        assert processor.ocr.rotation_confidence_threshold == 80.0
        assert processor.ocr.minimum_confidence_threshold == 60.0
        assert processor.ocr.high_confidence_threshold == 95.0
    
    def test_processor_default_initialization(self):
        """Test that processor uses default parameters when not specified."""
        processor = ScheduleProcessor()
        assert processor.preprocessor.dpi == 500
        assert processor.table_analyzer.overlap_threshold == 0.7
        assert processor.table_analyzer.min_cell_size == 5
    
    def test_processor_has_required_components(self, processor):
        """Test that processor has all required components."""
        assert hasattr(processor, 'preprocessor')
        assert hasattr(processor, 'table_analyzer')
        assert hasattr(processor, 'ocr')
    
    def test_process_nonexistent_file(self, processor):
        """Test that processing nonexistent file raises error."""
        with pytest.raises(ValueError, match="File not found"):
            processor.process("nonexistent_file.png")
    
    def test_process_valid_image(self, processor, sample_schedule_image):
        """Test processing a valid schedule image."""
        # Save test image to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_schedule_image)
        
        try:
            result = processor.process(temp_path)
            
            # Check result structure
            assert isinstance(result, dict)
            assert 'metadata' in result
            assert 'table' in result
            assert 'visualization' in result
            
            # Check metadata
            assert 'file_name' in result['metadata']
            assert 'file_type' in result['metadata']
            assert 'image_size' in result['metadata']
            assert 'processing_timestamp' in result['metadata']
            
            # Check table data
            assert 'bounds' in result['table']
            assert 'dimensions' in result['table']
            assert 'grid' in result['table']
            assert 'cells' in result['table']
            
            # Clean up visualization file
            if os.path.exists(result['visualization']['output_file']):
                os.remove(result['visualization']['output_file'])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_process_returns_correct_file_type(self, processor, sample_schedule_image):
        """Test that processor correctly identifies file type."""
        # Test with PNG
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_schedule_image)
        
        try:
            result = processor.process(temp_path)
            assert result['metadata']['file_type'] == 'image'
            
            # Clean up
            if os.path.exists(result['visualization']['output_file']):
                os.remove(result['visualization']['output_file'])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_process_creates_visualization(self, processor, sample_schedule_image):
        """Test that processing creates visualization file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_schedule_image)
        
        try:
            result = processor.process(temp_path)
            
            # Check that visualization file was created
            vis_file = result['visualization']['output_file']
            assert os.path.exists(vis_file)
            
            # Verify it's a valid image
            vis_img = cv2.imread(vis_file)
            assert vis_img is not None
            
            # Clean up
            if os.path.exists(vis_file):
                os.remove(vis_file)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_process_cleans_up_temp_files(self, processor, sample_schedule_image):
        """Test that processing cleans up temporary files."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_schedule_image)
        
        try:
            result = processor.process(temp_path)
            
            # The preprocessed file should be cleaned up
            preprocessed_files = [f for f in os.listdir('.') if f.startswith('preprocessed_')]
            
            # Clean up visualization
            if os.path.exists(result['visualization']['output_file']):
                os.remove(result['visualization']['output_file'])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_process_handles_cells_list(self, processor, sample_schedule_image):
        """Test that processor returns cells as a list."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            cv2.imwrite(temp_path, sample_schedule_image)
        
        try:
            result = processor.process(temp_path)
            
            # Cells should be a list
            assert isinstance(result['table']['cells'], list)
            
            # Each cell should have required keys
            for cell in result['table']['cells']:
                assert 'id' in cell
                assert 'bounds' in cell
                assert 'dimensions' in cell
                assert 'text' in cell
            
            # Clean up
            if os.path.exists(result['visualization']['output_file']):
                os.remove(result['visualization']['output_file'])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
