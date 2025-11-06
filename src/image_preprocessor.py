"""
Image preprocessing module for ML Timemaster.
Contains the ImagePreprocessor class for image preprocessing operations.
"""

import cv2
import os
import pdf2image
import tempfile


class ImagePreprocessor:
    """
    Handles image preprocessing operations including PDF conversion,
    blank image detection, and image enhancement.
    """
    
    def __init__(self, dpi=500):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            dpi (int): DPI resolution for PDF to image conversion. Default is 500.
        """
        self.dpi = dpi
    
    def convert_pdf_to_image(self, pdf_path):
        """
        Converts a PDF file to a high-quality image.
        
        Args:
            pdf_path (str): Path to the PDF file.
            
        Returns:
            str: Path to the temporary image file.
            
        Raises:
            ValueError: If the PDF cannot be converted or has no pages.
        """
        try:
            images = pdf2image.convert_from_path(pdf_path, dpi=self.dpi)
            if not images:
                raise ValueError("No pages found in PDF.")
            
            first_page = images[0]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                temp_path = tmp.name
                first_page.save(temp_path, "PNG", quality=95)
                print(f"Converted PDF to image: {first_page.size}")
            return temp_path
        except Exception as e:
            raise ValueError(f"Error converting PDF to image: {str(e)}")
    
    def is_blank_image(self, img):
        """
        Checks if an image is blank (mostly white).
        
        Args:
            img (numpy.ndarray): Input image in BGR format.
            
        Returns:
            bool: True if the image is blank (>99% white), False otherwise.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold to binary - pixels below 250 are considered content
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        non_white_pixels = cv2.countNonZero(binary)
        total_pixels = gray.shape[0] * gray.shape[1]
        return non_white_pixels < total_pixels * 0.01  # more than 99% blank
    
    def preprocess(self, file_path):
        """
        Preprocesses a schedule table from a PDF or image.
        
        Args:
            file_path (str): Path to the PDF or image file.
            
        Returns:
            tuple: (preprocessed_path, original_image)
                - preprocessed_path (str): Path to the preprocessed image file
                - original_image (numpy.ndarray): Original image as numpy array
                
        Raises:
            ValueError: If the file doesn't exist, can't be loaded, or is blank.
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

        file_type = "pdf" if file_path.lower().endswith(".pdf") else "image"
        
        if file_type == "pdf":
            print("Converting PDF to image...")
            img_path = self.convert_pdf_to_image(file_path)
        else:
            img_path = file_path

        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Failed to load image.")
            
            if self.is_blank_image(img):
                raise ValueError("Error: The input image is blank.")
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,
                2
            )
            
            # Denoise
            denoised = cv2.fastNlMeansDenoising(binary)
            
            # Apply binarization after denoising
            _, binarized = cv2.threshold(
                denoised,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Save preprocessed image
            preprocessed_path = f"preprocessed_{os.path.basename(file_path)}.png"
            cv2.imwrite(preprocessed_path, binarized)
            
            return preprocessed_path, img
        except Exception as e:
            raise ValueError(f"Error preprocessing file: {str(e)}")
        finally:
            if file_type == "pdf" and os.path.exists(img_path):
                os.remove(img_path)
