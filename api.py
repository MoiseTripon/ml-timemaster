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

# Import your existing processor
from start import TableOCRProcessor, setup_logging
import logging

# Setup logging with DEBUG level for troubleshooting
setup_logging(debug=True, log_level=logging.DEBUG)
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


@app.get("/")
async def root():
    return {"message": "Table OCR API is running"}


@app.post("/api/process-file")
async def process_file(file: UploadFile = File(...)):
    """
    Process uploaded file (PNG/PDF) and return extracted table data.
    """
    logger.info(f"Received file: {file.filename}, size: {file.size}, content_type: {file.content_type}")
    
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
        logger.info(f"Read {len(file_content)} bytes from uploaded file")
        
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file_content)
        
        logger.info(f"Saved file to: {temp_file_path}")
        
        # Verify file was saved correctly
        if not temp_file_path.exists():
            raise Exception("Failed to save uploaded file")
        
        file_size = temp_file_path.stat().st_size
        logger.info(f"Saved file size: {file_size} bytes")
        
        # Initialize processor for each request to ensure clean state
        processor = TableOCRProcessor(debug=True, api_mode=True)
        logger.info("Initialized TableOCRProcessor")
        
        # Process the file
        logger.info(f"Starting file processing: {str(temp_file_path)}")
        result = processor.process_file(str(temp_file_path))
        
        logger.info(f"Processing complete. Found {len(result.get('table', {}).get('cells', []))} cells")
        
        # Log some cell data for debugging
        cells = result.get('table', {}).get('cells', [])
        if cells:
            logger.info(f"First cell text: '{cells[0].get('text', '')}'")
            logger.info(f"Cell count by text length: {sum(1 for c in cells if c.get('text', '').strip())}")
        
        # Log grid data
        grid = result.get('table', {}).get('grid', [])
        logger.info(f"Grid dimensions: {len(grid)} rows")
        if grid:
            logger.info(f"First row: {grid[0]}")
        
        # Move visualization to static directory if it exists
        if result.get("visualization", {}).get("output_file"):
            old_vis_path = Path(result["visualization"]["output_file"])
            if old_vis_path.exists():
                new_vis_filename = f"{unique_id}_visualization.png"
                new_vis_path = VISUALIZATION_DIR / new_vis_filename
                shutil.move(str(old_vis_path), str(new_vis_path))
                result["visualization"]["output_file"] = f"/visualizations/{new_vis_filename}"
                logger.info(f"Moved visualization to: {new_vis_path}")
        
        return JSONResponse(
            content={
                "success": True,
                "data": result
            },
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
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
                logger.info(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as e:
                logger.error(f"Failed to clean up file: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)