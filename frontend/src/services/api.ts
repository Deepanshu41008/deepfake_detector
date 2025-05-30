import axios from 'axios';

// API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  timeout: 300000, // 5 minutes for large file uploads
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Types
export interface UploadResponse {
  success: boolean;
  file_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  upload_path: string;
  timestamp: string;
  message: string;
}

export interface DetectionResult {
  file_id: string;
  filename: string;
  file_type: string;
  is_deepfake: boolean;
  confidence_score: number;
  detection_type: string;
  processing_time: number;
  timestamp: string;
  detailed_results?: any;
}

export interface BatchDetectionResponse {
  success: boolean;
  results: DetectionResult[];
  failed_detections: any[];
  total_files: number;
  successful_detections: number;
  failed_detections_count: number;
  total_processing_time: number;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  uptime: number;
  system_info: any;
}

// API functions
export const healthAPI = {
  // Get basic health status
  getHealth: () => api.get<HealthResponse>('/health'),
  
  // Get detailed health status
  getDetailedHealth: () => api.get('/health/detailed'),
  
  // Get models health
  getModelsHealth: () => api.get('/health/models'),
};

export const uploadAPI = {
  // Upload single file
  uploadFile: (file: File, description?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (description) {
      formData.append('description', description);
    }
    
    return api.post<UploadResponse>('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / (progressEvent.total || 1)
        );
        // You can use this for progress tracking
        console.log(`Upload progress: ${percentCompleted}%`);
      },
    });
  },
  
  // Upload multiple files
  uploadBatchFiles: (files: File[], description?: string) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });
    if (description) {
      formData.append('description', description);
    }
    
    return api.post('/upload/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },
  
  // Get upload info
  getUploadInfo: (fileId: string) => api.get(`/upload/info/${fileId}`),
  
  // Delete uploaded file
  deleteFile: (fileId: string) => api.delete(`/upload/${fileId}`),
  
  // List uploaded files
  listFiles: () => api.get('/upload/list'),
};

export const detectionAPI = {
  // Detect deepfake in single file
  detectDeepfake: (
    fileId: string,
    detectionType: string = 'auto',
    confidenceThreshold: number = 0.5,
    detailedAnalysis: boolean = true
  ) => {
    return api.post<DetectionResult>('/detect', {
      file_id: fileId,
      detection_type: detectionType,
      confidence_threshold: confidenceThreshold,
      detailed_analysis: detailedAnalysis,
    });
  },
  
  // Batch detection
  detectBatch: (
    fileIds: string[],
    detectionType: string = 'auto',
    confidenceThreshold: number = 0.5,
    detailedAnalysis: boolean = true
  ) => {
    return api.post<BatchDetectionResponse>('/detect/batch', {
      file_ids: fileIds,
      detection_type: detectionType,
      confidence_threshold: confidenceThreshold,
      detailed_analysis: detailedAnalysis,
    });
  },
  
  // Get detection history
  getHistory: () => api.get('/detect/history'),
  
  // Get detection statistics
  getStats: () => api.get('/detect/stats'),
  
  // Live stream detection (placeholder)
  detectLiveStream: () => api.post('/detect/live'),
};

// Utility functions
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const getFileTypeIcon = (fileType: string): string => {
  switch (fileType.toLowerCase()) {
    case 'video':
      return '🎥';
    case 'image':
      return '🖼️';
    case 'audio':
      return '🎵';
    default:
      return '📄';
  }
};

export const getConfidenceColor = (confidence: number): string => {
  if (confidence < 0.3) return 'text-success-600';
  if (confidence < 0.7) return 'text-warning-600';
  return 'text-danger-600';
};

export const getConfidenceLabel = (confidence: number): string => {
  if (confidence < 0.3) return 'Likely Authentic';
  if (confidence < 0.7) return 'Uncertain';
  return 'Likely Deepfake';
};

// Simplified API for direct file upload and detection
export interface SimpleDetectionResult {
  filename: string;
  file_type: string;
  is_deepfake: boolean;
  confidence: number;
  processing_time: number;
  file_size: number;
  details?: any;
}

export const detectDeepfake = async (files: File[]): Promise<SimpleDetectionResult[]> => {
  const formData = new FormData();
  
  files.forEach((file) => {
    formData.append('files', file);
  });

  const response = await api.post<SimpleDetectionResult[]>('/detect/batch', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  // Save results to localStorage for history
  const results = response.data;
  const historyItems = results.map(result => ({
    id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
    filename: result.filename,
    file_type: result.file_type,
    is_deepfake: result.is_deepfake,
    confidence: result.confidence,
    processing_time: result.processing_time,
    timestamp: new Date().toISOString(),
    file_size: result.file_size
  }));

  const existingHistory = JSON.parse(localStorage.getItem('deepfake_detection_history') || '[]');
  const updatedHistory = [...historyItems, ...existingHistory];
  
  // Keep only the last 100 items
  const limitedHistory = updatedHistory.slice(0, 100);
  localStorage.setItem('deepfake_detection_history', JSON.stringify(limitedHistory));

  return results;
};

export default api;