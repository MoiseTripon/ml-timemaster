import os
import json
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict

# Import your existing processor
from start import TableOCRProcessor, setup_logging
import logging

# Setup logging with INFO level (less verbose)
setup_logging(debug=False, log_level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Table OCR API", version="1.0.0")

# Create directories for outputs
UPLOAD_DIR = Path("./uploads")
VISUALIZATION_DIR = Path("./visualizations")
UPLOAD_DIR.mkdir(exist_ok=True)
VISUALIZATION_DIR.mkdir(exist_ok=True)

# Configure CORS for NextJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Response model
class ProcessingResponse(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

class CorrectionItem(BaseModel):
    original: str
    corrected: str
    cell_id: Optional[int] = None

class CorrectionsRequest(BaseModel):
    corrections: List[CorrectionItem]

@app.get("/")
async def root():
    return {"message": "Table OCR API is running"}


@app.post("/api/process-file")
async def process_file(file: UploadFile = File(...)):
    """
    Process uploaded file (PNG/PDF) and return extracted table data.
    """
    # Validate file type
    allowed_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not supported. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename
    unique_id = str(uuid.uuid4())
    temp_filename = f"{unique_id}{file_extension}"
    temp_file_path = UPLOAD_DIR / temp_filename
    
    try:
        # Save uploaded file
        file_content = await file.read()
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        
        # Initialize processor WITHOUT api_mode to ensure consistent behavior
        processor = TableOCRProcessor(debug=False, api_mode=False)
        
        # Process the file
        result = processor.process_file(str(temp_file_path))
        
        # Quick validation
        dimensions = result.get('table', {}).get('dimensions', {})
        if dimensions.get('rows', 0) <= 1 and dimensions.get('columns', 0) <= 1:
            logger.warning("Table detection may have failed - single cell detected")
        
        # Move visualization to static directory if it exists
        visualization_filename = None
        if result.get("visualization", {}).get("output_file"):
            old_vis_path = Path(result["visualization"]["output_file"])
            if old_vis_path.exists():
                visualization_filename = f"{unique_id}_visualization.png"
                new_vis_path = VISUALIZATION_DIR / visualization_filename
                shutil.move(str(old_vis_path), str(new_vis_path))
                result["visualization"]["output_file"] = f"/visualizations/{visualization_filename}"
                result["visualization"]["filename"] = visualization_filename  # Add filename for cleanup
        
        return JSONResponse(
            content={
                "success": True,
                "data": result
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )
    finally:
        # Clean up temporary file
        if temp_file_path.exists():
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


@app.delete("/api/cleanup-visualization/{filename}")
async def cleanup_visualization(filename: str):
    """
    Delete a visualization file.
    """
    try:
        # Validate filename to prevent directory traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        vis_path = VISUALIZATION_DIR / filename
        
        if vis_path.exists():
            os.unlink(vis_path)
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Visualization {filename} deleted successfully"
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "success": True,
                    "message": "File not found (may have been already deleted)"
                },
                status_code=200
            )
            
    except Exception as e:
        logger.error(f"Error deleting visualization: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )
        
@app.post("/api/save-corrections")
async def save_corrections(request: CorrectionsRequest):
    """
    Save text corrections to the OCR dictionary.
    """
    try:
        # Use the correct path for the dictionary
        dictionary_path = Path("src/ocr_dictionary.json")
        
        # Check if file exists, if not use the one in root
        if not dictionary_path.exists():
            dictionary_path = Path("ocr_dictionary.json")
        
        if not dictionary_path.exists():
            logger.error(f"Dictionary file not found at {dictionary_path}")
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Dictionary file not found"
                },
                status_code=500
            )
        
        logger.info(f"Loading dictionary from {dictionary_path}")
        
        # Load the dictionary
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            dictionary_data = json.load(f)
        
        # Initialize structures if they don't exist
        if "word_corrections" not in dictionary_data:
            dictionary_data["word_corrections"] = {}
        
        if "phrase_corrections" not in dictionary_data:
            dictionary_data["phrase_corrections"] = {}
        
        # Track what we're adding
        added_corrections = []
        
        # Process each correction
        for correction in request.corrections:
            original = correction.original.strip()
            corrected = correction.corrected.strip()
            
            # Skip if they're the same or empty
            if not original or not corrected or original == corrected:
                continue
            
            # Determine if it's a word or phrase correction
            original_words = original.split()
            corrected_words = corrected.split()
            
            if len(original_words) == 1 and len(corrected_words) == 1:
                # Single word correction
                # Check if we need to add this correction
                existing_corrections = dictionary_data["word_corrections"].get(corrected, [])
                
                # Check if the original is already in the list (case-insensitive)
                already_exists = any(
                    existing.lower() == original.lower() 
                    for existing in existing_corrections
                )
                
                if not already_exists:
                    # Add the correction
                    if corrected not in dictionary_data["word_corrections"]:
                        dictionary_data["word_corrections"][corrected] = []
                    
                    dictionary_data["word_corrections"][corrected].append(original)
                    added_corrections.append({
                        "type": "word",
                        "original": original,
                        "corrected": corrected
                    })
                    logger.info(f"Added word correction: '{original}' -> '{corrected}'")
            else:
                # Multi-word/phrase correction
                existing_corrections = dictionary_data["phrase_corrections"].get(corrected, [])
                
                # Check if already exists
                already_exists = any(
                    existing.lower() == original.lower() 
                    for existing in existing_corrections
                )
                
                if not already_exists:
                    if corrected not in dictionary_data["phrase_corrections"]:
                        dictionary_data["phrase_corrections"][corrected] = []
                    
                    dictionary_data["phrase_corrections"][corrected].append(original)
                    added_corrections.append({
                        "type": "phrase",
                        "original": original,
                        "corrected": corrected
                    })
                    logger.info(f"Added phrase correction: '{original}' -> '{corrected}'")
        
        # Save the updated dictionary if we added anything
        if added_corrections:
            # Sort the corrections for better readability
            for key in dictionary_data["word_corrections"]:
                dictionary_data["word_corrections"][key].sort()
            
            for key in dictionary_data["phrase_corrections"]:
                dictionary_data["phrase_corrections"][key].sort()
            
            # Save to file
            with open(dictionary_path, 'w', encoding='utf-8') as f:
                json.dump(dictionary_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Successfully saved {len(added_corrections)} corrections to {dictionary_path}")
        
        return JSONResponse(
            content={
                "success": True,
                "corrections_added": len(added_corrections),
                "details": added_corrections,
                "message": f"Successfully added {len(added_corrections)} correction(s)"
            },
            status_code=200
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dictionary file: {str(e)}")
        return JSONResponse(
            content={
                "success": False,
                "error": f"Invalid dictionary file format: {str(e)}"
            },
            status_code=500
        )
    except Exception as e:
        logger.error(f"Error saving corrections: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "error": str(e)
            },
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)