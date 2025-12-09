import os
import io
import base64
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import cv2

# Import your existing processor - but we'll modify it
from start import setup_logging
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)

# Store visualization files temporarily
VISUALIZATION_DIR = Path("./temp_visualizations")
VISUALIZATION_DIR.mkdir(exist_ok=True)

# Cleanup old files on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Table OCR API...")
    # Clean old visualization files
    for file in VISUALIZATION_DIR.glob("*.png"):
        try:
            file.unlink()
        except:
            pass
    yield
    # Shutdown
    logger.info("Shutting down Table OCR API...")

# Initialize FastAPI app
app = FastAPI(
    title="Table OCR API",
    version="1.0.0",
    description="API for extracting tables from PDF and image files",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class TableMetadata(BaseModel):
    file_name: str
    file_type: str
    image_size: Dict[str, int]
    processing_timestamp: str

class TableDimensions(BaseModel):
    rows: int
    columns: int

class CellBounds(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Cell(BaseModel):
    id: int
    text: str
    bounds: CellBounds
    dimensions: Dict[str, int]

class TableInfo(BaseModel):
    bounds: Dict[str, int]
    dimensions: TableDimensions
    grid: list
    cells: list

class ProcessingResponse(BaseModel):
    success: bool
    metadata: Optional[TableMetadata] = None
    table: Optional[TableInfo] = None
    visualization_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

# Import and modify the TableOCRProcessor to work better with API
from start import TableOCRProcessor as OriginalProcessor

class APITableOCRProcessor(OriginalProcessor):
    """Modified processor for API usage without matplotlib GUI"""
    
    def _create_visualization(self, img, table_bounds, cells, output_file, num_rows, num_cols):
        """Create and save visualization without displaying it"""
        vis_img = img.copy()

        # Draw table boundary in red
        cv2.rectangle(
            vis_img,
            (table_bounds["x1"], table_bounds["y1"]),
            (table_bounds["x2"], table_bounds["y2"]),
            (0, 0, 255),
            3,
        )

        # Generate random colors for cells
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(cells), 3)).tolist()

        # Draw cells with random colors
        for cell, color in zip(cells, colors):
            cv2.rectangle(
                vis_img,
                (cell["bounds"]["x1"], cell["bounds"]["y1"]),
                (cell["bounds"]["x2"], cell["bounds"]["y2"]),
                color,
                2,
            )

        # Save to temp directory
        output_path = VISUALIZATION_DIR / output_file
        cv2.imwrite(str(output_path), vis_img)
        self.logger.info(f"Saved visualization to {output_path}")
        
        # Return base64 encoded image as alternative
        _, buffer = cv2.imencode('.png', vis_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

# Initialize processor
setup_logging(debug=False, log_level=logging.INFO)
processor = APITableOCRProcessor(debug=False)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Table OCR API is running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "visualization_dir": str(VISUALIZATION_DIR),
        "temp_files": len(list(VISUALIZATION_DIR.glob("*.png")))
    }

def cleanup_visualization(filepath: str):
    """Background task to cleanup visualization files after some time"""
    import time
    time.sleep(300)  # Wait 5 minutes
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up visualization file: {filepath}")
    except Exception as e:
        logger.error(f"Error cleaning up file {filepath}: {e}")

@app.post("/api/process-file", response_model=ProcessingResponse)
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or image file to process")
):
    """
    Process uploaded file (PNG/PDF) and return extracted table data.
    
    Returns:
        ProcessingResponse: JSON response with extracted table data and visualization
    """
    import time
    start_time = time.time()
    
    # Validate file type
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.pdf'}
    file_extension = Path(file.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_extension}' not supported. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Validate file size (e.g., max 10MB)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Reset file pointer
    file.file.seek(0)
    
    temp_file_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_extension,
            dir=VISUALIZATION_DIR
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
        
        # Process the file
        logger.info(f"Processing file: {file.filename}")
        result = processor.process_file(temp_file_path)
        
        # Generate visualization filename
        vis_filename = result["visualization"]["output_file"]
        vis_path = VISUALIZATION_DIR / vis_filename
        
        # Schedule cleanup of visualization file
        if vis_path.exists():
            background_tasks.add_task(cleanup_visualization, str(vis_path))
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return ProcessingResponse(
            success=True,
            metadata=TableMetadata(**result["metadata"]),
            table=TableInfo(**result["table"]),
            visualization_url=f"/api/visualization/{vis_filename}",
            processing_time=round(processing_time, 2)
        )
        
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        return ProcessingResponse(
            success=False,
            error=f"Processing failed: {str(ve)}",
            processing_time=round(time.time() - start_time, 2)
        )
    except Exception as e:
        logger.error(f"Unexpected error processing file: {str(e)}", exc_info=True)
        return ProcessingResponse(
            success=False,
            error=f"An unexpected error occurred: {str(e)}",
            processing_time=round(time.time() - start_time, 2)
        )
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.error(f"Error removing temp file: {e}")

@app.get("/api/visualization/{filename}")
async def get_visualization(filename: str):
    """
    Serve visualization images.
    
    Args:
        filename: Name of the visualization file
        
    Returns:
        FileResponse: The visualization image file
    """
    # Sanitize filename to prevent directory traversal
    safe_filename = Path(filename).name
    file_path = VISUALIZATION_DIR / safe_filename
    
    if file_path.exists() and file_path.is_file():
        return FileResponse(
            path=str(file_path),
            media_type="image/png",
            headers={
                "Cache-Control": "public, max-age=3600"
            }
        )
    else:
        raise HTTPException(status_code=404, detail="Visualization not found")

@app.delete("/api/cleanup")
async def cleanup_old_files():
    """Manual cleanup endpoint for old visualization files"""
    try:
        count = 0
        for file in VISUALIZATION_DIR.glob("*.png"):
            try:
                file.unlink()
                count += 1
            except:
                pass
        return {"message": f"Cleaned up {count} files"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",  # Pass as string to enable reload
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )