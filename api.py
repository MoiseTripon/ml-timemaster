import os
import json
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import your existing processor
from start import TableOCRProcessor, setup_logging
import logging

# Setup logging
setup_logging(debug=False, log_level=logging.INFO)

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
    allow_origins=["http://localhost:3000"],  # NextJS dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for visualizations
app.mount("/visualizations", StaticFiles(directory="visualizations"), name="visualizations")

# Initialize processor once in API mode
processor = TableOCRProcessor(debug=False, api_mode=True)

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
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        result = processor.process_file(str(temp_file_path))
        
        # Move visualization to static directory if it exists
        if result.get("visualization", {}).get("output_file"):
            old_vis_path = Path(result["visualization"]["output_file"])
            if old_vis_path.exists():
                new_vis_filename = f"{unique_id}_visualization.png"
                new_vis_path = VISUALIZATION_DIR / new_vis_filename
                shutil.move(str(old_vis_path), str(new_vis_path))
                
                # Update the result with the new URL
                result["visualization"]["output_file"] = f"/visualizations/{new_vis_filename}"
        
        return JSONResponse(
            content={
                "success": True,
                "data": result
            },
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
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
            except:
                pass


@app.on_event("startup")
async def startup_event():
    """Clean up old files on startup"""
    # Optional: Clean up old visualizations
    for file in VISUALIZATION_DIR.glob("*.png"):
        try:
            # Delete files older than 1 hour
            if (datetime.now() - datetime.fromtimestamp(file.stat().st_mtime)).seconds > 3600:
                file.unlink()
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    # Run without reload when using python api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)