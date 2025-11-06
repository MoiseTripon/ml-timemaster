"""
Main entry point for ML Timemaster - OCR Table Detection and Processing.
"""

from src.schedule_processor import ScheduleProcessor


def main():
    """
    Main function to process schedule files.
    """
    file_path = "i_a.pdf"  # Replace with your PDF or image file
    
    try:
        # Create processor with default settings
        processor = ScheduleProcessor(
            dpi=500,
            overlap_threshold=0.7,
            min_cell_size=5,
            rotation_confidence_threshold=70.0,
            minimum_confidence_threshold=50.0,
            high_confidence_threshold=90.0
        )
        
        # Process the file
        result = processor.process(file_path)
        
        # Check if any cells were detected
        if not result['table']['grid']:
            print("No cells were detected.")
            exit(1)
        
        # Display results
        print("\nExtracted Table Data:")
        print(f"File: {result['metadata']['file_name']}")
        print(f"Image size: {result['metadata']['image_size']['width']}x{result['metadata']['image_size']['height']}")
        print(f"Table dimensions: {result['table']['dimensions']['rows']}x{result['table']['dimensions']['columns']}")
        print("\nTable structure:")
        for row in result['table']['grid']:
            print(row)
                
    except ValueError as ve:
        print(f"Error: {str(ve)}")
        if "pdf" in str(ve).lower():
            print("\nEnsure the following for PDFs:")
            print("1. Poppler is installed")
            print("2. The PDF is not corrupted")
            print("3. The file is accessible")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
