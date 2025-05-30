"""
Image deepfake detector using EfficientNet architecture
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Dict, Any
import os
import time
from loguru import logger

from app.models.base_detector import BaseDeepfakeDetector, DummyModel
from app.core.config import settings
from app.core.exceptions import FileProcessingError, InferenceError


class EfficientNetB0(nn.Module):
    """Simplified EfficientNet-B0 for deepfake detection"""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        
        # MBConv blocks (simplified)
        self.blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(32, 64, 3, stride=1, padding=1, groups=32),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            
            # Block 2
            nn.Conv2d(64, 128, 3, stride=2, padding=1, groups=64),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 1),
            nn.BatchNorm2d(128),
            
            # Block 3
            nn.Conv2d(128, 256, 3, stride=2, padding=1, groups=128),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class ImageDeepfakeDetector(BaseDeepfakeDetector):
    """Image deepfake detector using EfficientNet"""
    
    def __init__(self):
        super().__init__("EfficientNet_Image", settings.IMAGE_MODEL_PATH)
        
        # Image processing parameters
        self.image_size = settings.IMAGE_SIZE
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Additional transforms for data augmentation during inference (TTA)
        self.tta_transforms = [
            transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            transforms.Compose([
                transforms.Resize((int(self.image_size[0] * 1.1), int(self.image_size[1] * 1.1))),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]
    
    def _load_model(self) -> nn.Module:
        """Load EfficientNet model"""
        
        # Check if pre-trained model exists
        if os.path.exists(self.model_path):
            try:
                model = EfficientNetB0(num_classes=2)
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Loaded pre-trained model from {self.model_path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
        
        # Use dummy model for demonstration
        logger.warning("Using dummy model for image detection (no pre-trained model found)")
        return DummyModel(input_size=3*224*224, num_classes=2)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load and validate image"""
        
        try:
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Check image size
            if image.size[0] < 32 or image.size[1] < 32:
                raise FileProcessingError(image_path, "Image too small (minimum 32x32)")
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise FileProcessingError(image_path, f"Image loading failed: {str(e)}")
    
    def _preprocess(self, file_path: str) -> torch.Tensor:
        """Preprocess image file"""
        
        # Load image
        image = self._load_image(file_path)
        
        # Transform image
        transformed_image = self.transform(image)
        
        # Add batch dimension
        batch_tensor = transformed_image.unsqueeze(0)
        
        return batch_tensor
    
    def _preprocess_tta(self, file_path: str) -> torch.Tensor:
        """Preprocess image with test-time augmentation"""
        
        # Load image
        image = self._load_image(file_path)
        
        # Apply multiple transforms
        augmented_images = []
        for transform in self.tta_transforms:
            transformed_image = transform(image)
            augmented_images.append(transformed_image)
        
        # Stack into batch
        batch_tensor = torch.stack(augmented_images)
        
        return batch_tensor
    
    def _postprocess(self, model_output: torch.Tensor, detailed: bool = True) -> Dict[str, Any]:
        """Postprocess model output"""
        
        # model_output shape: [batch_size, 2] or [1, 2] for single image
        if len(model_output.shape) == 2 and model_output.shape[0] > 1:
            # Multiple predictions (TTA)
            fake_probs = model_output[:, 0]
            real_probs = model_output[:, 1]
            
            # Average predictions
            avg_fake_prob = torch.mean(fake_probs).item()
            avg_real_prob = torch.mean(real_probs).item()
            
            # Calculate uncertainty (standard deviation)
            uncertainty = torch.std(fake_probs).item()
        else:
            # Single prediction
            fake_prob = model_output[0, 0].item()
            real_prob = model_output[0, 1].item()
            
            avg_fake_prob = fake_prob
            avg_real_prob = real_prob
            uncertainty = 0.0
        
        # Use fake probability as confidence score
        confidence_score = avg_fake_prob
        
        results = {
            "confidence_score": confidence_score,
            "fake_probability": avg_fake_prob,
            "real_probability": avg_real_prob,
            "uncertainty": uncertainty
        }
        
        if detailed:
            detailed_results = {
                "prediction_confidence": confidence_score,
                "class_probabilities": {
                    "fake": avg_fake_prob,
                    "real": avg_real_prob
                },
                "uncertainty_metrics": {
                    "prediction_uncertainty": uncertainty,
                    "confidence_level": "high" if uncertainty < 0.1 else "medium" if uncertainty < 0.2 else "low"
                },
                "model_insights": {
                    "decision_boundary_distance": abs(avg_fake_prob - 0.5),
                    "prediction_strength": max(avg_fake_prob, avg_real_prob),
                    "ambiguity_score": 1.0 - abs(avg_fake_prob - avg_real_prob)
                }
            }
            
            # Add TTA results if available
            if len(model_output.shape) == 2 and model_output.shape[0] > 1:
                tta_results = []
                for i, (fake_p, real_p) in enumerate(zip(model_output[:, 0], model_output[:, 1])):
                    tta_results.append({
                        "augmentation_id": i,
                        "fake_probability": fake_p.item(),
                        "real_probability": real_p.item()
                    })
                detailed_results["tta_analysis"] = tta_results
            
            results["detailed_results"] = detailed_results
        
        return results
    
    async def detect_with_tta(self, file_path: str, confidence_threshold: float = 0.5, detailed_analysis: bool = True) -> Dict[str, Any]:
        """Perform detection with test-time augmentation for better accuracy"""
        
        start_time = time.time()
        
        try:
            # Load model if not loaded
            if not self.is_loaded:
                self.load_model()
            
            # Preprocess with TTA
            input_tensor = self._preprocess_tta(file_path)
            input_tensor = input_tensor.to(self.device)
            
            # Perform inference
            with torch.no_grad():
                model_output = self.model(input_tensor)
            
            # Postprocess results
            results = self._postprocess(model_output, detailed_analysis)
            
            # Apply confidence threshold
            is_deepfake = results["confidence_score"] > confidence_threshold
            results["is_deepfake"] = is_deepfake
            results["used_tta"] = True
            
            # Update statistics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            logger.info(f"{self.model_name} detection with TTA completed in {inference_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"TTA inference failed for {self.model_name}: {str(e)}")
            raise InferenceError(self.model_name, str(e))
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get image metadata"""
        
        try:
            image = Image.open(image_path)
            
            return {
                "width": image.size[0],
                "height": image.size[1],
                "mode": image.mode,
                "format": image.format,
                "resolution": f"{image.size[0]}x{image.size[1]}",
                "aspect_ratio": image.size[0] / image.size[1],
                "megapixels": (image.size[0] * image.size[1]) / 1_000_000
            }
            
        except Exception as e:
            logger.error(f"Error getting image info: {str(e)}")
            return {"error": str(e)}