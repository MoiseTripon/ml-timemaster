import sys
print(f"Python version: {sys.version}")

try:
    import paddle
    print(f"PaddlePaddle version: {paddle.__version__}")
except ImportError as e:
    print(f"PaddlePaddle not installed: {e}")

try:
    import paddleocr
    print(f"PaddleOCR version: {paddleocr.__version__}")
except ImportError as e:
    print(f"PaddleOCR not installed: {e}")
except AttributeError:
    print("PaddleOCR installed but version not accessible")

try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"OpenCV not installed: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"NumPy not installed: {e}")

# Check PaddleOCR class signature
print("\n--- PaddleOCR Constructor Parameters ---")
try:
    from paddleocr import PaddleOCR
    import inspect
    sig = inspect.signature(PaddleOCR.__init__)
    print(f"Parameters: {list(sig.parameters.keys())}")
except Exception as e:
    print(f"Could not inspect PaddleOCR: {e}")

# Try to create instance and see what happens
print("\n--- Testing PaddleOCR Initialization ---")
try:
    from paddleocr import PaddleOCR
    # Try minimal initialization
    ocr = PaddleOCR(lang='en')
    print("Basic initialization successful!")
    
    # Check available attributes
    print(f"OCR object type: {type(ocr)}")
    print(f"OCR attributes: {[a for a in dir(ocr) if not a.startswith('_')]}")
except Exception as e:
    print(f"Initialization failed: {e}")

# Test actual OCR
print("\n--- Testing OCR Functionality ---")
try:
    import numpy as np
    import cv2
    from paddleocr import PaddleOCR
    
    ocr = PaddleOCR(lang='en')
    
    # Create test image
    img = np.ones((100, 400, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Hello World", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Save and test with file
    cv2.imwrite("/tmp/test_ocr.png", img)
    
    # Test with file path
    result = ocr.ocr("/tmp/test_ocr.png")
    print(f"File path result: {result}")
    
    # Test with numpy array
    result2 = ocr.ocr(img)
    print(f"Numpy array result: {result2}")
    
except Exception as e:
    print(f"OCR test failed: {e}")
    import traceback
    traceback.print_exc()