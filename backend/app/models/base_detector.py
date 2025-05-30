"""
Base class for deepfake detectors
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import time
from loguru import logger

from app.core.config import settings
from app.core.exceptions import ModelLoadError, InferenceError


class BaseDeepfakeDetector(ABC):
    """Base class for all deepfake detectors"""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None):
        self.model_name = model_name
        self.model_path = model_path
        self.model = None
        self.device = self._get_device()
        self.is_loaded = False
        self.inference_count = 0
        self.total_inference_time = 0.0
        
    def _get_device(self) -> torch.device:
        """Get the appropriate device for inference"""
        if settings.ENABLE_GPU and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for inference")
        return device
    
    @abstractmethod
    def _load_model(self) -> nn.Module:
        """Load the specific model architecture"""
        pass
    
    @abstractmethod
    def _preprocess(self, file_path: str) -> torch.Tensor:
        """Preprocess input file for the model"""
        pass
    
    @abstractmethod
    def _postprocess(self, model_output: torch.Tensor, detailed: bool = True) -> Dict[str, Any]:
        """Postprocess model output to get final results"""
        pass
    
    def load_model(self):
        """Load the model if not already loaded"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading {self.model_name} model...")
            self.model = self._load_model()
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"{self.model_name} model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_name} model: {str(e)}")
            raise ModelLoadError(self.model_name, str(e))
    
    async def detect(self, file_path: str, confidence_threshold: float = 0.5, detailed_analysis: bool = True) -> Dict[str, Any]:
        """
        Perform deepfake detection on a file
        
        Args:
            file_path: Path to the input file
            confidence_threshold: Threshold for classification
            detailed_analysis: Whether to return detailed analysis
            
        Returns:
            Dictionary containing detection results
        """
        
        start_time = time.time()
        
        try:
            # Load model if not loaded
            if not self.is_loaded:
                self.load_model()
            
            # Ensure model is loaded
            if self.model is None:
                raise InferenceError(self.model_name, "Model failed to load")
            
            # Preprocess input
            input_tensor = self._preprocess(file_path)
            input_tensor = input_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                model_output = self.model(input_tensor)
            
            # Postprocess results
            results = self._postprocess(model_output, detailed_analysis)
            
            # Apply confidence threshold
            is_deepfake = results["confidence_score"] > confidence_threshold
            results["is_deepfake"] = is_deepfake
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            logger.info(f"{self.model_name} detection completed in {inference_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed for {self.model_name}: {str(e)}")
            raise InferenceError(self.model_name, str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics"""
        avg_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else 0
        
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": str(self.device),
            "inference_count": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": avg_time
        }
    
    def unload_model(self):
        """Unload the model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"{self.model_name} model unloaded")


class DummyModel(nn.Module):
    """Dummy model for testing purposes"""
    
    def __init__(self, input_size: int = 1000, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.fc(x)