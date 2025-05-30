import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeftIcon, CheckCircleIcon, XCircleIcon } from '@heroicons/react/24/outline';
import ConfidenceMeter from '../components/ConfidenceMeter';

interface DetectionResult {
  filename: string;
  is_deepfake: boolean;
  confidence: number;
  processing_time: number;
  file_type: string;
  file_size: number;
  details?: {
    model_used?: string;
    frames_analyzed?: number;
    audio_features?: string[];
  };
}

const Results: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const results = location.state?.results as DetectionResult[] || [];

  const handleBackToUpload = () => {
    navigate('/upload');
  };

  const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getResultIcon = (isDeepfake: boolean) => {
    return isDeepfake ? (
      <XCircleIcon className="h-8 w-8 text-red-500" />
    ) : (
      <CheckCircleIcon className="h-8 w-8 text-green-500" />
    );
  };

  const getResultText = (isDeepfake: boolean) => {
    return isDeepfake ? 'Deepfake Detected' : 'Authentic Media';
  };

  const getResultColor = (isDeepfake: boolean) => {
    return isDeepfake ? 'text-red-600' : 'text-green-600';
  };

  if (results.length === 0) {
    return (
      <div className="min-h-screen bg-gray-50 py-12">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900 mb-8">No Results Found</h1>
            <p className="text-gray-600 mb-8">No detection results to display.</p>
            <button
              onClick={handleBackToUpload}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              <ArrowLeftIcon className="h-4 w-4 mr-2" />
              Back to Upload
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <button
            onClick={handleBackToUpload}
            className="inline-flex items-center text-indigo-600 hover:text-indigo-500 mb-4"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-2" />
            Back to Upload
          </button>
          <h1 className="text-3xl font-bold text-gray-900">Detection Results</h1>
          <p className="text-gray-600 mt-2">
            Analysis complete for {results.length} file{results.length > 1 ? 's' : ''}
          </p>
        </div>

        <div className="space-y-6">
          {results.map((result, index) => (
            <div key={index} className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  {getResultIcon(result.is_deepfake)}
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {result.filename}
                    </h3>
                    <p className={`text-sm font-medium ${getResultColor(result.is_deepfake)}`}>
                      {getResultText(result.is_deepfake)}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">Processing Time</p>
                  <p className="text-lg font-semibold text-gray-900">
                    {result.processing_time.toFixed(2)}s
                  </p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-sm font-medium text-gray-900 mb-3">Confidence Score</h4>
                  <ConfidenceMeter confidence={result.confidence} />
                </div>

                <div className="space-y-3">
                  <div>
                    <h4 className="text-sm font-medium text-gray-900 mb-2">File Information</h4>
                    <div className="space-y-1 text-sm text-gray-600">
                      <p><span className="font-medium">Type:</span> {result.file_type.toUpperCase()}</p>
                      <p><span className="font-medium">Size:</span> {formatFileSize(result.file_size)}</p>
                      {result.details?.model_used && (
                        <p><span className="font-medium">Model:</span> {result.details.model_used}</p>
                      )}
                      {result.details?.frames_analyzed && (
                        <p><span className="font-medium">Frames Analyzed:</span> {result.details.frames_analyzed}</p>
                      )}
                    </div>
                  </div>

                  {result.details?.audio_features && (
                    <div>
                      <h4 className="text-sm font-medium text-gray-900 mb-2">Audio Features</h4>
                      <div className="flex flex-wrap gap-1">
                        {result.details.audio_features.map((feature, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                          >
                            {feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {result.is_deepfake && (
                <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
                  <p className="text-sm text-red-800">
                    <strong>Warning:</strong> This media appears to be artificially generated or manipulated. 
                    Please verify the source and consider the implications before sharing or acting on this content.
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="mt-8 text-center">
          <button
            onClick={handleBackToUpload}
            className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Analyze More Files
          </button>
        </div>
      </div>
    </div>
  );
};

export default Results;