"""
Audio deepfake detector using Wav2Vec and MFCC-based models
"""

import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from typing import Dict, Any, Tuple
import os
from loguru import logger

from app.models.base_detector import BaseDeepfakeDetector, DummyModel
from app.core.config import settings
from app.core.exceptions import FileProcessingError


class AudioCNN(nn.Module):
    """CNN model for audio deepfake detection using MFCC features"""
    
    def __init__(self, num_classes: int = 2, input_channels: int = 1):
        super().__init__()
        
        # Convolutional layers for MFCC features
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # Calculate the size after conv layers
        # This will be calculated dynamically in forward pass
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = self.fc_layers(x)
        return x


class AudioDeepfakeDetector(BaseDeepfakeDetector):
    """Audio deepfake detector using MFCC features and CNN"""
    
    def __init__(self):
        super().__init__("AudioCNN_MFCC", settings.AUDIO_MODEL_PATH)
        
        # Audio processing parameters
        self.sample_rate = settings.AUDIO_SAMPLE_RATE
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
        self.max_duration = 30  # Maximum audio duration in seconds
        self.segment_duration = 3  # Duration of each segment in seconds
    
    def _load_model(self) -> nn.Module:
        """Load audio CNN model"""
        
        # Check if pre-trained model exists
        if os.path.exists(self.model_path):
            try:
                model = AudioCNN(num_classes=2, input_channels=1)
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Loaded pre-trained model from {self.model_path}")
                return model
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
        
        # Use dummy model for demonstration
        logger.warning("Using dummy model for audio detection (no pre-trained model found)")
        return DummyModel(input_size=13*128, num_classes=2)  # MFCC features flattened
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load and preprocess audio file"""
        
        try:
            # Load audio using librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.max_duration)
            
            if len(audio) == 0:
                raise FileProcessingError(audio_path, "Empty audio file")
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {str(e)}")
            raise FileProcessingError(audio_path, f"Audio loading failed: {str(e)}")
    
    def _extract_mfcc_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features from audio"""
        
        try:
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # Add delta and delta-delta features
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Combine features
            features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting MFCC features: {str(e)}")
            raise FileProcessingError("audio", f"MFCC extraction failed: {str(e)}")
    
    def _segment_audio(self, audio: np.ndarray, sr: int) -> list:
        """Segment audio into fixed-length chunks"""
        
        segment_samples = int(self.segment_duration * sr)
        segments = []
        
        # If audio is shorter than segment duration, pad it
        if len(audio) < segment_samples:
            padded_audio = np.pad(audio, (0, segment_samples - len(audio)), mode='constant')
            segments.append(padded_audio)
        else:
            # Split audio into segments
            for i in range(0, len(audio), segment_samples):
                segment = audio[i:i + segment_samples]
                
                # Pad last segment if necessary
                if len(segment) < segment_samples:
                    segment = np.pad(segment, (0, segment_samples - len(segment)), mode='constant')
                
                segments.append(segment)
        
        return segments
    
    def _preprocess(self, file_path: str) -> torch.Tensor:
        """Preprocess audio file"""
        
        # Load audio
        audio, sr = self._load_audio(file_path)
        
        # Segment audio
        segments = self._segment_audio(audio, sr)
        
        # Extract MFCC features for each segment
        feature_tensors = []
        for segment in segments:
            mfcc_features = self._extract_mfcc_features(segment, sr)
            
            # Convert to tensor and add channel dimension
            feature_tensor = torch.FloatTensor(mfcc_features).unsqueeze(0)  # [1, n_features, time]
            feature_tensors.append(feature_tensor)
        
        # Stack segments into batch
        batch_tensor = torch.stack(feature_tensors)  # [num_segments, 1, n_features, time]
        
        return batch_tensor
    
    def _postprocess(self, model_output: torch.Tensor, detailed: bool = True) -> Dict[str, Any]:
        """Postprocess model output"""
        
        # model_output shape: [num_segments, 2]
        fake_probs = model_output[:, 0]
        real_probs = model_output[:, 1]
        
        # Calculate overall confidence (average across segments)
        avg_fake_prob = torch.mean(fake_probs).item()
        avg_real_prob = torch.mean(real_probs).item()
        
        # Use fake probability as confidence score
        confidence_score = avg_fake_prob
        
        results = {
            "confidence_score": confidence_score,
            "fake_probability": avg_fake_prob,
            "real_probability": avg_real_prob
        }
        
        if detailed:
            # Segment-by-segment analysis
            segment_results = []
            for i, (fake_prob, real_prob) in enumerate(zip(fake_probs, real_probs)):
                segment_results.append({
                    "segment_index": i,
                    "start_time": i * self.segment_duration,
                    "end_time": (i + 1) * self.segment_duration,
                    "fake_probability": fake_prob.item(),
                    "real_probability": real_prob.item(),
                    "is_deepfake": fake_prob.item() > 0.5
                })
            
            # Statistical analysis
            fake_probs_np = fake_probs.detach().cpu().numpy()
            
            detailed_results = {
                "segment_analysis": segment_results,
                "statistics": {
                    "total_segments": len(fake_probs),
                    "deepfake_segments": int(np.sum(fake_probs_np > 0.5)),
                    "authentic_segments": int(np.sum(fake_probs_np <= 0.5)),
                    "min_fake_prob": float(np.min(fake_probs_np)),
                    "max_fake_prob": float(np.max(fake_probs_np)),
                    "std_fake_prob": float(np.std(fake_probs_np)),
                    "consistency_score": 1.0 - float(np.std(fake_probs_np))
                },
                "temporal_analysis": {
                    "most_suspicious_segment": int(np.argmax(fake_probs_np)),
                    "least_suspicious_segment": int(np.argmin(fake_probs_np)),
                    "confidence_trend": fake_probs_np.tolist(),
                    "anomaly_segments": [int(i) for i, prob in enumerate(fake_probs_np) if abs(prob - avg_fake_prob) > 2 * np.std(fake_probs_np)]
                },
                "audio_features": {
                    "segment_duration": self.segment_duration,
                    "sample_rate": self.sample_rate,
                    "mfcc_features": self.n_mfcc,
                    "total_duration": len(fake_probs) * self.segment_duration
                }
            }
            
            results["detailed_results"] = detailed_results
        
        return results
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract additional spectral features for analysis"""
        
        try:
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            
            return {
                "spectral_centroid_mean": float(np.mean(spectral_centroids)),
                "spectral_centroid_std": float(np.std(spectral_centroids)),
                "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
                "spectral_rolloff_std": float(np.std(spectral_rolloff)),
                "zero_crossing_rate_mean": float(np.mean(zcr)),
                "zero_crossing_rate_std": float(np.std(zcr)),
                "chroma_mean": float(np.mean(chroma)),
                "chroma_std": float(np.std(chroma)),
                "tempo": float(tempo)
            }
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {str(e)}")
            return {}
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Get audio metadata"""
        
        try:
            # Load audio info
            audio, sr = self._load_audio(audio_path)
            
            # Basic info
            duration = len(audio) / sr
            
            # Spectral features
            spectral_features = self._extract_spectral_features(audio, sr)
            
            return {
                "duration_seconds": duration,
                "sample_rate": sr,
                "num_samples": len(audio),
                "channels": 1,  # We convert to mono
                "bit_depth": "float32",
                "spectral_features": spectral_features
            }
            
        except Exception as e:
            logger.error(f"Error getting audio info: {str(e)}")
            return {"error": str(e)}