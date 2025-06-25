"""
Base class for deepfake detectors
"""

# import torch
# import torch.nn as nn # Keep for type hinting if possible, or remove if causing load issues
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path # Added
import numpy as np # For DummyModel
import time
from loguru import logger

from app.core.config import settings
from app.core.exceptions import ModelLoadError, InferenceError


class BaseDeepfakeDetector(ABC):
    """Base class for all deepfake detectors"""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None):
        self.model_name = model_name
        if model_path:
            self.model_path = Path(settings.MODELS_DIR) / model_path
        else:
            self.model_path = None
        self.model = None
        self.device = self._get_device()
        self.is_loaded = False
        self.inference_count = 0
        self.total_inference_time = 0.0
        
    def _get_device(self): # -> torch.device:
        """Get the appropriate device for inference (dummy version)"""
        if settings.ENABLE_GPU: # and torch.cuda.is_available():
            # device = torch.device("cuda")
            # logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info("GPU check skipped (torch disabled). Reporting 'gpu' if ENABLE_GPU is true.")
            device = "gpu (dummy)"
        else:
            # device = torch.device("cpu")
            logger.info("CPU will be used for inference (torch disabled).")
            device = "cpu (dummy)"
        return device
    
    @abstractmethod
    def _load_model(self): # -> nn.Module:
        """Load the specific model architecture"""
        pass
    
    @abstractmethod
    def _preprocess(self, file_path: str): # -> torch.Tensor:
        """Preprocess input file for the model"""
        pass
    
    @abstractmethod
    def _postprocess(self, model_output, detailed: bool = True) -> Dict[str, Any]: # model_output: torch.Tensor
        """Postprocess model output to get final results"""
        pass
    
    def load_model(self):
        """Load the model if not already loaded"""
        if self.is_loaded:
            return
        
        try:
            logger.info(f"Loading {self.model_name} model...")
            self.model = self._load_model() # This will now be a DummyModel instance
            # self.model.to(self.device) # DummyModel won't have .to()
            # self.model.eval() # DummyModel won't have .eval()
            self.is_loaded = True
            logger.info(f"{self.model_name} model loaded successfully (dummy mode)")
            
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
            # input_tensor = self._preprocess(file_path) # This will raise NotImplementedError in subclasses
            # input_tensor = input_tensor.to(self.device)
            logger.warning(f"Preprocessing for {self.model_name} is disabled. Using dummy output.")
            
            # Perform inference (dummy)
            # with torch.no_grad():
            # model_output = self.model(input_tensor) # self.model is DummyModel
            if self.model and hasattr(self.model, 'forward_dummy'):
                 model_output = self.model.forward_dummy()
            else:
                 model_output = np.random.rand(1, 2) # Dummy output if model is not DummyModel or lacks forward_dummy

            # Postprocess results
            results = self._postprocess(model_output, detailed_analysis) # Subclasses _postprocess are modified
            
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
            
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            logger.info(f"CUDA cache clear skipped (torch disabled). {self.model_name} model unloaded.")


# class DummyModel(nn.Module): # Can't inherit from nn.Module if torch.nn is not imported
class DummyModel:
    """Dummy model for testing purposes (torch-free)"""
    
    def __init__(self, input_size: int = 1000, num_classes: int = 2):
        # super().__init__() # No superclass call
        self.input_size = input_size
        self.num_classes = num_classes
        logger.info(f"DummyModel initialized (torch-free) with input_size={input_size}, num_classes={num_classes}")
        # self.fc = nn.Sequential(
        #     nn.Linear(input_size, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_classes),
        #     nn.Softmax(dim=1)
        # )
    
    def forward_dummy(self): # Renamed from forward, and takes no input for simplicity now
        """Simulates a forward pass, returning random probabilities."""
        # # Flatten input if needed
        # if len(x.shape) > 2:
        #     x = x.view(x.size(0), -1) # This was torch specific
        # return self.fc(x) # This was torch specific

        # Return a random probability distribution for num_classes
        # e.g., for num_classes = 2, something like [0.2, 0.8]
        raw_output = np.random.rand(1, self.num_classes) # Batch size of 1
        # Apply softmax manually if needed, or just ensure sums to 1 (roughly for dummy)
        probabilities = np.exp(raw_output) / np.sum(np.exp(raw_output), axis=1, keepdims=True)
        return probabilities.astype(np.float32)

    def eval(self): # Add dummy eval
      pass

    def to(self, device): # Add dummy to
      pass