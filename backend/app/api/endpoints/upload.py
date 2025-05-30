"""
File upload endpoints for the Deepfake Detection API
"""

import os
import uuid
import aiofiles
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import magic
from loguru import logger

from app.core.config import settings
from app.core.exceptions import UnsupportedFileTypeError, FileProcessingError
from app.utils.file_utils import validate_file_type, get_file_info, create_upload_directory

router = APIRouter()


class UploadResponse(BaseModel):
    """File upload response model"""
    success: bool
    file_id: str
    filename: str
    file_type: str
    file_size: int
    upload_path: str
    timestamp: datetime
    message: str


class BatchUploadResponse(BaseModel):
    """Batch file upload response model"""
    success: bool
    uploaded_files: List[UploadResponse]
    failed_files: List[dict]
    total_files: int
    successful_uploads: int
    failed_uploads: int


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload a single file for deepfake detection
    
    Supports:
    - Videos: .mp4, .avi, .mov, .mkv, .webm
    - Images: .jpg, .jpeg, .png, .bmp, .tiff
    - Audio: .wav, .mp3, .flac, .ogg, .m4a
    """
    
    try:
        # Validate file name
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="File name is required"
            )
        
        # Validate file size
        if not file.size or file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size ({file.size or 0} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)"
            )
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Get file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        # Validate file type
        file_type = validate_file_type(file_extension)
        
        # Create upload directory if it doesn't exist
        upload_dir = create_upload_directory()
        
        # Generate unique filename
        unique_filename = f"{file_id}_{file.filename}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Verify file was saved correctly
        if not os.path.exists(file_path):
            raise FileProcessingError(file.filename, "Failed to save file")
        
        # Get file info
        file_info = get_file_info(file_path)
        
        logger.info(f"File uploaded successfully: {file.filename} -> {file_path}")
        
        return UploadResponse(
            success=True,
            file_id=file_id,
            filename=file.filename,
            file_type=file_type,
            file_size=file_info["size"],
            upload_path=file_path,
            timestamp=datetime.now(),
            message="File uploaded successfully"
        )
        
    except UnsupportedFileTypeError as e:
        logger.error(f"Unsupported file type: {e.file_type}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except FileProcessingError as e:
        logger.error(f"File processing error: {e.message}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during file upload")


@router.post("/upload/batch", response_model=BatchUploadResponse)
async def upload_batch_files(
    files: List[UploadFile] = File(...),
    description: Optional[str] = Form(None)
):
    """
    Upload multiple files for batch deepfake detection
    """
    
    uploaded_files = []
    failed_files = []
    
    for file in files:
        try:
            # Use the single file upload logic
            result = await upload_file(file, description)
            uploaded_files.append(result)
            
        except HTTPException as e:
            failed_files.append({
                "filename": file.filename,
                "error": e.detail,
                "status_code": e.status_code
            })
            logger.error(f"Failed to upload {file.filename}: {e.detail}")
        
        except Exception as e:
            failed_files.append({
                "filename": file.filename,
                "error": str(e),
                "status_code": 500
            })
            logger.error(f"Unexpected error uploading {file.filename}: {str(e)}")
    
    return BatchUploadResponse(
        success=len(failed_files) == 0,
        uploaded_files=uploaded_files,
        failed_files=failed_files,
        total_files=len(files),
        successful_uploads=len(uploaded_files),
        failed_uploads=len(failed_files)
    )


@router.get("/upload/info/{file_id}")
async def get_upload_info(file_id: str):
    """Get information about an uploaded file"""
    
    # Search for file with the given ID
    upload_dir = settings.UPLOAD_DIR
    
    if not os.path.exists(upload_dir):
        raise HTTPException(status_code=404, detail="Upload directory not found")
    
    # Find file with matching ID
    matching_files = [f for f in os.listdir(upload_dir) if f.startswith(file_id)]
    
    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(upload_dir, matching_files[0])
    file_info = get_file_info(file_path)
    
    return {
        "file_id": file_id,
        "filename": matching_files[0],
        "file_path": file_path,
        "file_info": file_info,
        "timestamp": datetime.fromtimestamp(file_info["created_time"])
    }


@router.delete("/upload/{file_id}")
async def delete_uploaded_file(file_id: str):
    """Delete an uploaded file"""
    
    upload_dir = settings.UPLOAD_DIR
    
    if not os.path.exists(upload_dir):
        raise HTTPException(status_code=404, detail="Upload directory not found")
    
    # Find file with matching ID
    matching_files = [f for f in os.listdir(upload_dir) if f.startswith(file_id)]
    
    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(upload_dir, matching_files[0])
    
    try:
        os.remove(file_path)
        logger.info(f"File deleted: {file_path}")
        
        return {
            "success": True,
            "message": f"File {file_id} deleted successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete file")


@router.get("/upload/list")
async def list_uploaded_files():
    """List all uploaded files"""
    
    upload_dir = settings.UPLOAD_DIR
    
    if not os.path.exists(upload_dir):
        return {
            "files": [],
            "total_files": 0,
            "total_size": 0
        }
    
    files_info = []
    total_size = 0
    
    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if os.path.isfile(file_path):
            file_info = get_file_info(file_path)
            
            # Extract file ID from filename
            file_id = filename.split('_')[0] if '_' in filename else filename
            
            files_info.append({
                "file_id": file_id,
                "filename": filename,
                "size": file_info["size"],
                "type": file_info["type"],
                "created_time": datetime.fromtimestamp(file_info["created_time"]),
                "modified_time": datetime.fromtimestamp(file_info["modified_time"])
            })
            
            total_size += file_info["size"]
    
    # Sort by creation time (newest first)
    files_info.sort(key=lambda x: x["created_time"], reverse=True)
    
    return {
        "files": files_info,
        "total_files": len(files_info),
        "total_size": total_size,
        "timestamp": datetime.now()
    }