"""
Health check endpoints for the Deepfake Detection API
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import psutil
# import torch
import time
from datetime import datetime

from app.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    system_info: Dict[str, Any]


class DetailedHealthResponse(BaseModel):
    """Detailed health check response model"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    system_info: Dict[str, Any]
    models_status: Dict[str, bool]
    performance_metrics: Dict[str, Any]


# Track application start time
start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint"""
    
    current_time = time.time()
    uptime = current_time - start_time
    
    # Get basic system info
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        # "gpu_available": torch.cuda.is_available(),
        # "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        "gpu_available": "N/A (torch disabled)",
        "gpu_count": 0
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.VERSION,
        uptime=uptime,
        system_info=system_info
    )


@router.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check():
    """Detailed health check with model status and performance metrics"""
    
    current_time = time.time()
    uptime = current_time - start_time
    
    # Get detailed system info
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "cpu_count": psutil.cpu_count(),
        "memory_total": memory.total,
        "memory_available": memory.available,
        "memory_percent": memory.percent,
        "disk_total": disk.total,
        "disk_free": disk.free,
        "disk_percent": disk.percent,
        # "gpu_available": torch.cuda.is_available(),
        # "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        "gpu_available": "N/A (torch disabled)",
        "gpu_count": 0
    }
    
    # Add GPU info if available
    # if torch.cuda.is_available():
    #     gpu_info = []
    #     for i in range(torch.cuda.device_count()):
    #         gpu_props = torch.cuda.get_device_properties(i)
    #         gpu_memory = torch.cuda.get_device_properties(i).total_memory
    #         gpu_info.append({
    #             "device_id": i,
    #             "name": gpu_props.name,
    #             "total_memory": gpu_memory,
    #             "memory_allocated": torch.cuda.memory_allocated(i),
    #             "memory_reserved": torch.cuda.memory_reserved(i)
    #         })
    #     system_info["gpu_info"] = gpu_info
    system_info["gpu_info"] = "N/A (torch disabled)"
    
    # Check model status (placeholder - will be implemented with actual models)
    models_status = {
        "video_model": True,  # Will check if model files exist and are loadable
        "image_model": True,
        "audio_model": True
    }
    
    # Performance metrics
    performance_metrics = {
        "average_inference_time": 0.0,  # Will be tracked in actual implementation
        "requests_processed": 0,
        "errors_count": 0,
        "cache_hit_rate": 0.0
    }
    
    return DetailedHealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.VERSION,
        uptime=uptime,
        system_info=system_info,
        models_status=models_status,
        performance_metrics=performance_metrics
    )


@router.get("/health/models")
async def models_health_check():
    """Check the health and availability of AI models"""
    
    models_status = {}
    
    # Check if model files exist
    import os
    
    models_to_check = {
        "video_model": settings.VIDEO_MODEL_PATH,
        "image_model": settings.IMAGE_MODEL_PATH,
        "audio_model": settings.AUDIO_MODEL_PATH
    }
    
    for model_name, model_path in models_to_check.items():
        models_status[model_name] = {
            "file_exists": os.path.exists(model_path),
            "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
            "loadable": False,  # Will be checked when models are actually loaded
            "last_used": None,
            "inference_count": 0
        }
    
    return {
        "status": "checked",
        "timestamp": datetime.now(),
        "models": models_status
    }