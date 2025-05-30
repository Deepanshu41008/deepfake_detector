"""
Configuration settings for the Deepfake Detection API
"""

import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings"""
    
    # Environment Settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Deepfake Detection API"
    VERSION: str = "1.0.0"
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 104857600  # 100MB in bytes
    ALLOWED_EXTENSIONS: str = "mp4,avi,mov,mkv,webm,jpg,jpeg,png,bmp,tiff,wav,mp3,flac,ogg,m4a"
    
    # Specific file type extensions
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    ALLOWED_AUDIO_EXTENSIONS: List[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    
    UPLOAD_DIR: str = "uploads"
    
    # Model Settings
    MODELS_DIR: str = "models"
    VIDEO_MODEL_PATH: str = "models/video_model.pth"
    IMAGE_MODEL_PATH: str = "models/image_model.pth"
    AUDIO_MODEL_PATH: str = "models/audio_model.pth"
    ENABLE_GPU: bool = True  # Enable GPU acceleration if available
    
    # Processing Settings
    VIDEO_FRAME_RATE: int = 1
    VIDEO_MAX_FRAMES: int = 30
    IMAGE_SIZE: List[int] = [224, 224]
    AUDIO_SAMPLE_RATE: int = 16000
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./deepfake_detection.db"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://work-1-uzgwgwossxrxhroo.prod-runtime.all-hands.dev",
        "https://work-2-uzgwgwossxrxhroo.prod-runtime.all-hands.dev"
    ]
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v
    
    model_config = {"env_file": ".env", "case_sensitive": True}


# Global settings instance
settings = Settings()