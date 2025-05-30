"""
Custom exceptions and exception handlers for the Deepfake Detection API
"""

from typing import Optional
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
import traceback


class DeepfakeDetectionException(Exception):
    """Base exception for deepfake detection errors"""
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelLoadError(DeepfakeDetectionException):
    """Exception raised when model loading fails"""
    def __init__(self, model_name: str, message: Optional[str] = None):
        self.model_name = model_name
        message = message or f"Failed to load model: {model_name}"
        super().__init__(message, "MODEL_LOAD_ERROR")


class FileProcessingError(DeepfakeDetectionException):
    """Exception raised when file processing fails"""
    def __init__(self, filename: str, message: Optional[str] = None):
        self.filename = filename
        message = message or f"Failed to process file: {filename}"
        super().__init__(message, "FILE_PROCESSING_ERROR")


class UnsupportedFileTypeError(DeepfakeDetectionException):
    """Exception raised when file type is not supported"""
    def __init__(self, file_type: str):
        self.file_type = file_type
        message = f"Unsupported file type: {file_type}"
        super().__init__(message, "UNSUPPORTED_FILE_TYPE")


class InferenceError(DeepfakeDetectionException):
    """Exception raised when model inference fails"""
    def __init__(self, model_name: str, message: Optional[str] = None):
        self.model_name = model_name
        message = message or f"Inference failed for model: {model_name}"
        super().__init__(message, "INFERENCE_ERROR")


def setup_exception_handlers(app: FastAPI):
    """Setup custom exception handlers for the FastAPI app"""
    
    @app.exception_handler(DeepfakeDetectionException)
    async def deepfake_exception_handler(request: Request, exc: DeepfakeDetectionException):
        """Handle custom deepfake detection exceptions"""
        logger.error(f"DeepfakeDetectionException: {exc.message}")
        return JSONResponse(
            status_code=400,
            content={
                "error": True,
                "error_code": exc.error_code,
                "message": exc.message,
                "detail": str(exc)
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        logger.error(f"HTTPException: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "error_code": "HTTP_ERROR",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {str(exc)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "detail": str(exc) if app.debug else "Internal server error"
            }
        )