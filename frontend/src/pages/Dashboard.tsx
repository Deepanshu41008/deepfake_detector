import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { healthAPI, detectionAPI, uploadAPI } from '../services/api';
import ConfidenceMeter from '../components/ConfidenceMeter';

interface SystemStats {
  totalFiles: number;
  deepfakesDetected: number;
  authenticFiles: number;
  averageConfidence: number;
  systemHealth: string;
  uptime: number;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<SystemStats>({
    totalFiles: 0,
    deepfakesDetected: 0,
    authenticFiles: 0,
    averageConfidence: 0,
    systemHealth: 'unknown',
    uptime: 0,
  });
  const [recentFiles, setRecentFiles] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Load system health
      const healthResponse = await healthAPI.getHealth();
      
      // Load detection stats
      const statsResponse = await detectionAPI.getStats();
      
      // Load recent files
      const filesResponse = await uploadAPI.listFiles();

      setStats({
        totalFiles: statsResponse.data.total_files_processed || 0,
        deepfakesDetected: statsResponse.data.deepfakes_detected || 0,
        authenticFiles: statsResponse.data.authentic_files || 0,
        averageConfidence: statsResponse.data.average_confidence || 0,
        systemHealth: healthResponse.data.status,
        uptime: healthResponse.data.uptime,
      });

      // Get the 5 most recent files
      setRecentFiles(filesResponse.data.files.slice(0, 5));

    } catch (err: any) {
      setError(err.response?.data?.message || 'Failed to load dashboard data');
    } finally {
      setIsLoading(false);
    }
  };

  const formatUptime = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  const getFileTypeIcon = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    if (['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext || '')) return '🎥';
    if (['jpg', 'jpeg', 'png', 'bmp', 'tiff'].includes(ext || '')) return '🖼️';
    if (['wav', 'mp3', 'flac', 'ogg', 'm4a'].includes(ext || '')) return '🎵';
    return '📄';
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="loading-spinner" />
        <span className="ml-2 text-gray-600">Loading dashboard...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6 text-center">
        <div className="text-danger-600 text-lg mb-4">⚠️ Error Loading Dashboard</div>
        <p className="text-gray-600 mb-4">{error}</p>
        <button
          onClick={loadDashboardData}
          className="btn-primary"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          🔍 Deepfake Detection Dashboard
        </h1>
        <p className="text-lg text-gray-600 max-w-2xl mx-auto">
          AI-powered system for detecting deepfake videos, images, and audio.
          Upload your media files to verify their authenticity.
        </p>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link
          to="/upload"
          className="card p-6 hover:shadow-lg transition-shadow cursor-pointer group"
        >
          <div className="text-center">
            <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">📤</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Files</h3>
            <p className="text-gray-600 text-sm">
              Upload videos, images, or audio files for deepfake detection
            </p>
          </div>
        </Link>

        <Link
          to="/history"
          className="card p-6 hover:shadow-lg transition-shadow cursor-pointer group"
        >
          <div className="text-center">
            <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">📋</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">View History</h3>
            <p className="text-gray-600 text-sm">
              Browse your previous detection results and analysis
            </p>
          </div>
        </Link>

        <Link
          to="/about"
          className="card p-6 hover:shadow-lg transition-shadow cursor-pointer group"
        >
          <div className="text-center">
            <div className="text-4xl mb-3 group-hover:scale-110 transition-transform">ℹ️</div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Learn More</h3>
            <p className="text-gray-600 text-sm">
              Understand how deepfake detection works and best practices
            </p>
          </div>
        </Link>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Files</p>
              <p className="text-2xl font-bold text-gray-900">{stats.totalFiles}</p>
            </div>
            <div className="text-3xl">📊</div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Deepfakes Detected</p>
              <p className="text-2xl font-bold text-danger-600">{stats.deepfakesDetected}</p>
            </div>
            <div className="text-3xl">⚠️</div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Authentic Files</p>
              <p className="text-2xl font-bold text-success-600">{stats.authenticFiles}</p>
            </div>
            <div className="text-3xl">✅</div>
          </div>
        </div>

        <div className="card p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">System Status</p>
              <p className={`text-sm font-medium ${
                stats.systemHealth === 'healthy' ? 'text-success-600' : 'text-danger-600'
              }`}>
                {stats.systemHealth === 'healthy' ? '🟢 Online' : '🔴 Issues'}
              </p>
              <p className="text-xs text-gray-500">Uptime: {formatUptime(stats.uptime)}</p>
            </div>
            <div className="text-3xl">🖥️</div>
          </div>
        </div>
      </div>

      {/* Average Confidence */}
      {stats.averageConfidence > 0 && (
        <div className="card p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Average Detection Confidence
          </h3>
          <ConfidenceMeter
            confidence={stats.averageConfidence}
            size="lg"
            label="Overall System Confidence"
          />
        </div>
      )}

      {/* Recent Files */}
      <div className="card p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Recent Files</h3>
          <Link to="/history" className="text-primary-600 hover:text-primary-700 text-sm">
            View All →
          </Link>
        </div>

        {recentFiles.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <div className="text-4xl mb-2">📁</div>
            <p>No files uploaded yet</p>
            <Link to="/upload" className="text-primary-600 hover:text-primary-700 text-sm">
              Upload your first file →
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {recentFiles.map((file, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <span className="text-lg">{getFileTypeIcon(file.filename)}</span>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.filename}</p>
                    <p className="text-xs text-gray-500">
                      {new Date(file.created_time).toLocaleDateString()} • {(file.size / 1024 / 1024).toFixed(1)} MB
                    </p>
                  </div>
                </div>
                <Link
                  to={`/results/${file.file_id}`}
                  className="text-primary-600 hover:text-primary-700 text-sm"
                >
                  Analyze →
                </Link>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Features Overview */}
      <div className="card p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Detection Capabilities</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl mb-2">🎥</div>
            <h4 className="font-medium text-gray-900">Video Detection</h4>
            <p className="text-sm text-gray-600 mt-1">
              Analyze video frames using XceptionNet for face manipulation detection
            </p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl mb-2">🖼️</div>
            <h4 className="font-medium text-gray-900">Image Detection</h4>
            <p className="text-sm text-gray-600 mt-1">
              Detect AI-generated images using EfficientNet architecture
            </p>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl mb-2">🎵</div>
            <h4 className="font-medium text-gray-900">Audio Detection</h4>
            <p className="text-sm text-gray-600 mt-1">
              Identify synthetic speech using MFCC features and CNN models
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;