import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, XMarkIcon, DocumentIcon } from '@heroicons/react/24/outline';

interface FileUploadProps {
  onFilesSelected?: (files: File[]) => void;
  onError?: (error: string) => void;
  maxFiles?: number;
  maxFileSize?: number; // in bytes
  isProcessing?: boolean;
  processingText?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFilesSelected,
  onError,
  maxFiles = 10,
  maxFileSize = 100 * 1024 * 1024, // 100MB
  isProcessing = false,
  processingText = 'Processing...'
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);

  const acceptedFileTypes = {
    'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
    'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
    'audio/*': ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
  };

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxFileSize) {
      return `File "${file.name}" is too large. Maximum size is ${formatFileSize(maxFileSize)}.`;
    }

    // Check file type
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase();
    const allowedExtensions = Object.values(acceptedFileTypes).flat();
    
    if (!allowedExtensions.includes(fileExtension)) {
      return `File type "${fileExtension}" is not supported.`;
    }

    return null;
  };

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    if (rejectedFiles.length > 0) {
      const error = rejectedFiles[0].errors[0]?.message || 'File rejected';
      onError?.(error);
      return;
    }

    // Validate files
    const validFiles: File[] = [];
    for (const file of acceptedFiles) {
      const error = validateFile(file);
      if (error) {
        onError?.(error);
        return;
      }
      validFiles.push(file);
    }

    // Check total file count
    const totalFiles = selectedFiles.length + validFiles.length;
    if (totalFiles > maxFiles) {
      onError?.(` Maximum ${maxFiles} files allowed. You selected ${totalFiles} files.`);
      return;
    }

    setSelectedFiles(prev => [...prev, ...validFiles]);
  }, [selectedFiles, maxFiles, maxFileSize, onError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: acceptedFileTypes,
    maxFiles,
    disabled: isProcessing
  });

  const removeFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const handleAnalyze = () => {
    if (selectedFiles.length === 0) {
      onError?.('Please select at least one file to analyze.');
      return;
    }
    onFilesSelected?.(selectedFiles);
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (file: File) => {
    if (file.type.startsWith('video/')) return '🎥';
    if (file.type.startsWith('image/')) return '🖼️';
    if (file.type.startsWith('audio/')) return '🎵';
    return '📄';
  };

  return (
    <div className="w-full">
      {/* Drop Zone */}
      <div
        {...getRootProps()}
        className={`
          relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive 
            ? 'border-indigo-500 bg-indigo-50' 
            : 'border-gray-300 hover:border-gray-400'
          }
          ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}
        `}
      >
        <input {...getInputProps()} />
        
        {isProcessing ? (
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600 mb-4"></div>
            <p className="text-lg font-medium text-gray-900">{processingText}</p>
            <p className="text-sm text-gray-500 mt-2">Please wait while we analyze your files...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <CloudArrowUpIcon className="h-12 w-12 text-gray-400 mb-4" />
            <p className="text-lg font-medium text-gray-900 mb-2">
              {isDragActive ? 'Drop files here' : 'Drag & drop files here'}
            </p>
            <p className="text-sm text-gray-500 mb-4">
              or click to browse files
            </p>
            <div className="text-xs text-gray-400">
              <p>Supported: Video (MP4, AVI, MOV, MKV, WebM)</p>
              <p>Images (JPG, PNG, BMP, TIFF) • Audio (WAV, MP3, FLAC, OGG, M4A)</p>
              <p>Max file size: {formatFileSize(maxFileSize)} • Max files: {maxFiles}</p>
            </div>
          </div>
        )}
      </div>

      {/* Selected Files */}
      {selectedFiles.length > 0 && (
        <div className="mt-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">
            Selected Files ({selectedFiles.length})
          </h3>
          <div className="space-y-3">
            {selectedFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{getFileIcon(file)}</span>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">
                      {formatFileSize(file.size)} • {file.type}
                    </p>
                  </div>
                </div>
                {!isProcessing && (
                  <button
                    onClick={() => removeFile(index)}
                    className="text-gray-400 hover:text-red-500 transition-colors"
                  >
                    <XMarkIcon className="h-5 w-5" />
                  </button>
                )}
              </div>
            ))}
          </div>

          {/* Analyze Button */}
          <div className="mt-6 flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={isProcessing || selectedFiles.length === 0}
              className={`
                px-8 py-3 rounded-md font-medium text-white transition-colors
                ${isProcessing || selectedFiles.length === 0
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
                }
              `}
            >
              {isProcessing ? processingText : `Analyze ${selectedFiles.length} File${selectedFiles.length > 1 ? 's' : ''}`}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;