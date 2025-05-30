import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import FileUpload from '../components/FileUpload';
import { detectDeepfake, SimpleDetectionResult } from '../services/api';

const Upload: React.FC = () => {
  const navigate = useNavigate();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [notification, setNotification] = useState<{ type: 'success' | 'error'; message: string } | null>(null);

  const handleFilesSelected = async (files: File[]) => {
    if (files.length === 0) return;

    setIsAnalyzing(true);
    setNotification(null);

    try {
      const results = await detectDeepfake(files);
      
      // Navigate to results page with the detection results
      navigate('/results', { 
        state: { 
          results: results.map(result => ({
            filename: result.filename,
            is_deepfake: result.is_deepfake,
            confidence: result.confidence,
            processing_time: result.processing_time,
            file_type: result.file_type,
            file_size: result.file_size,
            details: result.details
          }))
        } 
      });
    } catch (error: any) {
      console.error('Detection failed:', error);
      setNotification({
        type: 'error',
        message: error.response?.data?.detail || 'Failed to analyze files. Please try again.'
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleUploadError = (error: string) => {
    setNotification({
      type: 'error',
      message: error
    });
    
    // Clear notification after 5 seconds
    setTimeout(() => setNotification(null), 5000);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Upload Media for Analysis
          </h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload your video, image, or audio files to detect potential deepfakes and AI-generated content. 
            Our advanced AI models will analyze your media and provide detailed authenticity reports.
          </p>
        </div>

        {/* Notification */}
        {notification && (
          <div className={`mb-6 p-4 rounded-md ${
            notification.type === 'success' 
              ? 'bg-green-50 border border-green-200 text-green-800' 
              : 'bg-red-50 border border-red-200 text-red-800'
          }`}>
            <div className="flex">
              <div className="flex-shrink-0">
                {notification.type === 'success' ? (
                  <svg className="h-5 w-5 text-green-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                )}
              </div>
              <div className="ml-3">
                <p className="text-sm font-medium">{notification.message}</p>
              </div>
            </div>
          </div>
        )}

        {/* File Upload Component */}
        <div className="bg-white rounded-lg shadow-lg p-8">
          <FileUpload
            onFilesSelected={handleFilesSelected}
            onError={handleUploadError}
            isProcessing={isAnalyzing}
            processingText="Analyzing files..."
          />
        </div>

        {/* Information Cards */}
        <div className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center mb-4">
              <div className="flex-shrink-0">
                <svg className="h-8 w-8 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="ml-3 text-lg font-medium text-gray-900">Video Analysis</h3>
            </div>
            <p className="text-gray-600">
              Advanced frame-by-frame analysis using XceptionNet to detect manipulated video content with high accuracy.
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center mb-4">
              <div className="flex-shrink-0">
                <svg className="h-8 w-8 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="ml-3 text-lg font-medium text-gray-900">Image Detection</h3>
            </div>
            <p className="text-gray-600">
              EfficientNet-based analysis with Test Time Augmentation for robust deepfake image identification.
            </p>
          </div>

          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center mb-4">
              <div className="flex-shrink-0">
                <svg className="h-8 w-8 text-indigo-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
                </svg>
              </div>
              <h3 className="ml-3 text-lg font-medium text-gray-900">Audio Authentication</h3>
            </div>
            <p className="text-gray-600">
              MFCC feature extraction and CNN models to identify synthetic speech and audio manipulations.
            </p>
          </div>
        </div>

        {/* Supported Formats */}
        <div className="mt-8 bg-gray-100 rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Supported File Formats</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Video</h4>
              <p className="text-sm text-gray-600">MP4, AVI, MOV, MKV, WebM</p>
            </div>
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Image</h4>
              <p className="text-sm text-gray-600">JPG, JPEG, PNG, BMP, TIFF</p>
            </div>
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Audio</h4>
              <p className="text-sm text-gray-600">WAV, MP3, FLAC, OGG, M4A</p>
            </div>
          </div>
          <div className="mt-4 text-sm text-gray-500">
            <p><strong>Maximum file size:</strong> 100MB per file</p>
            <p><strong>Batch processing:</strong> Up to 10 files at once</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Upload;