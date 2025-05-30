"""
Deepfake detection endpoints for the Deepfake Detection API
"""

import os
import time
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from datetime import datetime
from loguru import logger

from app.core.config import settings
from app.core.exceptions import FileProcessingError, InferenceError
from app.models.video_detector import VideoDeepfakeDetector
from app.models.image_detector import ImageDeepfakeDetector
from app.models.audio_detector import AudioDeepfakeDetector
from app.utils.file_utils import get_file_info

router = APIRouter()


class DetectionRequest(BaseModel):
    """Detection request model"""
    file_id: str
    detection_type: Optional[str] = "auto"  # auto, video, image, audio
    confidence_threshold: Optional[float] = 0.5
    detailed_analysis: Optional[bool] = True


class DetectionResult(BaseModel):
    """Detection result model"""
    file_id: str
    filename: str
    file_type: str
    is_deepfake: bool
    confidence_score: float
    detection_type: str
    processing_time: float
    timestamp: datetime
    detailed_results: Optional[Dict[str, Any]] = None


class BatchDetectionRequest(BaseModel):
    """Batch detection request model"""
    file_ids: List[str]
    detection_type: Optional[str] = "auto"
    confidence_threshold: Optional[float] = 0.5
    detailed_analysis: Optional[bool] = True


class BatchDetectionResponse(BaseModel):
    """Batch detection response model"""
    success: bool
    results: List[DetectionResult]
    failed_detections: List[dict]
    total_files: int
    successful_detections: int
    failed_detections_count: int
    total_processing_time: float


# Global model instances (lazy loading)
video_detector = None
image_detector = None
audio_detector = None


def get_video_detector():
    """Get or initialize video detector"""
    global video_detector
    if video_detector is None:
        video_detector = VideoDeepfakeDetector()
    return video_detector


def get_image_detector():
    """Get or initialize image detector"""
    global image_detector
    if image_detector is None:
        image_detector = ImageDeepfakeDetector()
    return image_detector


def get_audio_detector():
    """Get or initialize audio detector"""
    global audio_detector
    if audio_detector is None:
        audio_detector = AudioDeepfakeDetector()
    return audio_detector


def determine_file_type(file_path: str) -> str:
    """Determine the type of file for detection"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in settings.ALLOWED_VIDEO_EXTENSIONS:
        return "video"
    elif file_extension in settings.ALLOWED_IMAGE_EXTENSIONS:
        return "image"
    elif file_extension in settings.ALLOWED_AUDIO_EXTENSIONS:
        return "audio"
    else:
        raise FileProcessingError(file_path, f"Unsupported file type: {file_extension}")


async def perform_detection(file_path: str, detection_type: str, confidence_threshold: float, detailed_analysis: bool) -> Dict[str, Any]:
    """Perform deepfake detection on a file"""
    
    start_time = time.time()
    
    try:
        if detection_type == "video":
            detector = get_video_detector()
            result = await detector.detect(file_path, confidence_threshold, detailed_analysis)
        elif detection_type == "image":
            detector = get_image_detector()
            result = await detector.detect(file_path, confidence_threshold, detailed_analysis)
        elif detection_type == "audio":
            detector = get_audio_detector()
            result = await detector.detect(file_path, confidence_threshold, detailed_analysis)
        else:
            raise InferenceError("unknown", f"Unknown detection type: {detection_type}")
        
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time
        
        return result
        
    except Exception as e:
        logger.error(f"Detection failed for {file_path}: {str(e)}")
        raise InferenceError(detection_type, str(e))


@router.post("/detect", response_model=DetectionResult)
async def detect_deepfake(request: DetectionRequest):
    """
    Detect deepfake in a single uploaded file
    """
    
    try:
        # Find the uploaded file
        upload_dir = settings.UPLOAD_DIR
        matching_files = [f for f in os.listdir(upload_dir) if f.startswith(request.file_id)]
        
        if not matching_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_path = os.path.join(upload_dir, matching_files[0])
        filename = matching_files[0]
        
        # Determine detection type
        detection_type_param = request.detection_type or "auto"
        if detection_type_param == "auto":
            detection_type = determine_file_type(file_path)
        else:
            detection_type = detection_type_param
        
        # Get parameters with defaults
        confidence_threshold = request.confidence_threshold or 0.5
        detailed_analysis = request.detailed_analysis if request.detailed_analysis is not None else True
        
        # Perform detection
        result = await perform_detection(
            file_path, 
            detection_type, 
            confidence_threshold,
            detailed_analysis
        )
        
        logger.info(f"Detection completed for {filename}: {result['is_deepfake']} (confidence: {result['confidence_score']:.3f})")
        
        return DetectionResult(
            file_id=request.file_id,
            filename=filename,
            file_type=detection_type,
            is_deepfake=result["is_deepfake"],
            confidence_score=result["confidence_score"],
            detection_type=detection_type,
            processing_time=result["processing_time"],
            timestamp=datetime.now(),
            detailed_results=result.get("detailed_results") if detailed_analysis else None
        )
        
    except FileProcessingError as e:
        logger.error(f"File processing error: {e.message}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except InferenceError as e:
        logger.error(f"Inference error: {e.message}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error during detection: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during detection")


@router.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_deepfake_batch(request: BatchDetectionRequest):
    """
    Detect deepfake in multiple uploaded files
    """
    
    results = []
    failed_detections = []
    total_start_time = time.time()
    
    for file_id in request.file_ids:
        try:
            detection_request = DetectionRequest(
                file_id=file_id,
                detection_type=request.detection_type,
                confidence_threshold=request.confidence_threshold,
                detailed_analysis=request.detailed_analysis
            )
            
            result = await detect_deepfake(detection_request)
            results.append(result)
            
        except HTTPException as e:
            failed_detections.append({
                "file_id": file_id,
                "error": e.detail,
                "status_code": e.status_code
            })
            logger.error(f"Failed to detect deepfake for {file_id}: {e.detail}")
        
        except Exception as e:
            failed_detections.append({
                "file_id": file_id,
                "error": str(e),
                "status_code": 500
            })
            logger.error(f"Unexpected error detecting deepfake for {file_id}: {str(e)}")
    
    total_processing_time = time.time() - total_start_time
    
    return BatchDetectionResponse(
        success=len(failed_detections) == 0,
        results=results,
        failed_detections=failed_detections,
        total_files=len(request.file_ids),
        successful_detections=len(results),
        failed_detections_count=len(failed_detections),
        total_processing_time=total_processing_time
    )


@router.get("/detect/history")
async def get_detection_history():
    """Get detection history (placeholder for database implementation)"""
    
    # This would typically query a database
    # For now, return a placeholder response
    
    return {
        "message": "Detection history feature will be implemented with database integration",
        "total_detections": 0,
        "recent_detections": [],
        "timestamp": datetime.now()
    }


@router.get("/detect/stats")
async def get_detection_stats():
    """Get detection statistics"""
    
    # This would typically aggregate data from a database
    # For now, return placeholder statistics
    
    return {
        "total_files_processed": 0,
        "deepfakes_detected": 0,
        "authentic_files": 0,
        "average_confidence": 0.0,
        "processing_times": {
            "average": 0.0,
            "min": 0.0,
            "max": 0.0
        },
        "file_types": {
            "video": 0,
            "image": 0,
            "audio": 0
        },
        "timestamp": datetime.now()
    }


@router.post("/detect/live")
async def detect_live_stream():
    """Placeholder for live stream detection"""
    
    return {
        "message": "Live stream detection feature is under development",
        "status": "not_implemented",
        "timestamp": datetime.now()
    }