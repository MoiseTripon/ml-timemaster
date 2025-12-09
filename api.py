import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

# Import your existing processor
from start import TableOCRProcessor, setup_logging
import logging

# Setup logging
setup_logging(debug=False, log_level=logging.INFO)

# Initialize FastAPI app
app = FastAPI(title="Table OCR API", version="1.0.0")

# Configure CORS for NextJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # NextJS dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processor once
processor = TableOCRProcessor(debug=False)

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
    
    # Create temporary file
    temp_file = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Process the file
        result = processor.process_file(temp_file_path)
        
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
        if temp_file and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass


@app.get("/api/visualization/{filename}")
async def get_visualization(filename: str):
    """
    Serve visualization images.
    """
    file_path = f"./{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        raise HTTPException(status_code=404, detail="Visualization not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)