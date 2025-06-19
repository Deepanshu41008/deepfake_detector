"""
Configuration settings for the Deepfake Detection API
"""

import os
from typing import List, Optional
from pathlib import Path # Added
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
    
    # Base directory for backend-relative paths
    # Assuming this config.py is in backend/app/core/
    # So, BACKEND_DIR will point to /app/backend/
    BACKEND_DIR: Path = Path(__file__).resolve().parent.parent.parent

    UPLOAD_DIR: Path = BACKEND_DIR / "uploads"
    
    # Model Settings
    MODELS_DIR: Path = BACKEND_DIR / "models"
    VIDEO_MODEL_PATH: str = "video_model.pth" # Filename only
    IMAGE_MODEL_PATH: str = "image_model.pth" # Filename only
    AUDIO_MODEL_PATH: str = "audio_model.pth" # Filename only
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
    LOG_FILE: Path = BACKEND_DIR / "logs/app.log"
    
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

    @field_validator("UPLOAD_DIR", "MODELS_DIR", "LOG_FILE", mode="before")
    @classmethod
    def ensure_path_strings(cls, v):
        if isinstance(v, Path):
            return str(v)
        return v
    
    model_config = {"env_file": ".env", "case_sensitive": True}


# Global settings instance
settings = Settings()

# Ensure directories exist (this part needs to be run when settings are loaded)
# For now, main.py handles this. If settings were used before main.py's startup,
# this would be a good place.
# settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
# Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)