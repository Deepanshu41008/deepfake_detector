"""
Main FastAPI application for Deepfake Detection System
"""

import os
import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
from loguru import logger

from app.api.endpoints import detection, health, upload
from app.core.config import settings
from app.core.exceptions import setup_exception_handlers

# Configure logging
logger.add(
    backend_dir / "logs/app.log",
    rotation="500 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)

def create_application() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Deepfake Detection API",
        description="AI-powered system for detecting deepfake videos, images, and audio",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Deepfake Detection API",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/api/v1/health"
        }
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
    app.include_router(detection.router, prefix="/api/v1", tags=["detection"])
    
    # Mount static files
    if os.path.exists(backend_dir / "uploads"):
        app.mount("/uploads", StaticFiles(directory=backend_dir / "uploads"), name="uploads")
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize application on startup"""
        logger.info("Starting Deepfake Detection API...")
        
        # Create necessary directories
        os.makedirs(backend_dir / "uploads", exist_ok=True)
        os.makedirs(backend_dir / "logs", exist_ok=True)
        os.makedirs(backend_dir / "models", exist_ok=True)
        
        # Initialize models (lazy loading)
        logger.info("Application startup complete")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown"""
        logger.info("Shutting down Deepfake Detection API...")
    
    return app

app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )