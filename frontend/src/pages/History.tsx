import React, { useState, useEffect } from 'react';
import { 
  ClockIcon, 
  DocumentTextIcon, 
  TrashIcon,
  EyeIcon,
  FunnelIcon,
  MagnifyingGlassIcon
} from '@heroicons/react/24/outline';
import ConfidenceMeter from '../components/ConfidenceMeter';

interface HistoryItem {
  id: string;
  filename: string;
  file_type: string;
  is_deepfake: boolean;
  confidence: number;
  processing_time: number;
  timestamp: string;
  file_size: number;
}

const History: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [filteredHistory, setFilteredHistory] = useState<HistoryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'authentic' | 'deepfake'>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'confidence' | 'filename'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    // Load history from localStorage
    const savedHistory = localStorage.getItem('deepfake_detection_history');
    if (savedHistory) {
      const parsedHistory = JSON.parse(savedHistory);
      setHistory(parsedHistory);
      setFilteredHistory(parsedHistory);
    }
  }, []);

  useEffect(() => {
    // Apply filters and search
    let filtered = history.filter(item => {
      const matchesSearch = item.filename.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFilter = 
        filterType === 'all' || 
        (filterType === 'authentic' && !item.is_deepfake) ||
        (filterType === 'deepfake' && item.is_deepfake);
      
      return matchesSearch && matchesFilter;
    });

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;
      
      switch (sortBy) {
        case 'timestamp':
          aValue = new Date(a.timestamp);
          bValue = new Date(b.timestamp);
          break;
        case 'confidence':
          aValue = a.confidence;
          bValue = b.confidence;
          break;
        case 'filename':
          aValue = a.filename.toLowerCase();
          bValue = b.filename.toLowerCase();
          break;
        default:
          aValue = a.timestamp;
          bValue = b.timestamp;
      }

      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

    setFilteredHistory(filtered);
  }, [history, searchTerm, filterType, sortBy, sortOrder]);

  const clearHistory = () => {
    if (window.confirm('Are you sure you want to clear all history? This action cannot be undone.')) {
      localStorage.removeItem('deepfake_detection_history');
      setHistory([]);
      setFilteredHistory([]);
    }
  };

  const deleteItem = (id: string) => {
    const updatedHistory = history.filter(item => item.id !== id);
    setHistory(updatedHistory);
    localStorage.setItem('deepfake_detection_history', JSON.stringify(updatedHistory));
  };

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatFileSize = (bytes: number): string => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getResultBadge = (isDeepfake: boolean) => {
    return isDeepfake ? (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
        Deepfake
      </span>
    ) : (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
        Authentic
      </span>
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Detection History</h1>
          <p className="text-gray-600 mt-2">
            View and manage your previous deepfake detection results
          </p>
        </div>

        {/* Controls */}
        <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Search */}
            <div className="relative">
              <MagnifyingGlassIcon className="h-5 w-5 absolute left-3 top-3 text-gray-400" />
              <input
                type="text"
                placeholder="Search files..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              />
            </div>

            {/* Filter */}
            <div className="relative">
              <FunnelIcon className="h-5 w-5 absolute left-3 top-3 text-gray-400" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as any)}
                className="pl-10 w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
              >
                <option value="all">All Results</option>
                <option value="authentic">Authentic Only</option>
                <option value="deepfake">Deepfake Only</option>
              </select>
            </div>

            {/* Sort By */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            >
              <option value="timestamp">Sort by Date</option>
              <option value="confidence">Sort by Confidence</option>
              <option value="filename">Sort by Filename</option>
            </select>

            {/* Sort Order */}
            <select
              value={sortOrder}
              onChange={(e) => setSortOrder(e.target.value as any)}
              className="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500"
            >
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </div>

          {history.length > 0 && (
            <div className="mt-4 flex justify-end">
              <button
                onClick={clearHistory}
                className="inline-flex items-center px-3 py-2 border border-red-300 shadow-sm text-sm leading-4 font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
              >
                <TrashIcon className="h-4 w-4 mr-2" />
                Clear All History
              </button>
            </div>
          )}
        </div>

        {/* Results */}
        {filteredHistory.length === 0 ? (
          <div className="text-center py-12">
            <DocumentTextIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-medium text-gray-900">No history found</h3>
            <p className="mt-1 text-sm text-gray-500">
              {history.length === 0 
                ? "You haven't analyzed any files yet." 
                : "No files match your current filters."}
            </p>
          </div>
        ) : (
          <div className="bg-white shadow-sm rounded-lg overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      File
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Result
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Confidence
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Processing Time
                    </th>
                    <th className="relative px-6 py-3">
                      <span className="sr-only">Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {filteredHistory.map((item) => (
                    <tr key={item.id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div>
                          <div className="text-sm font-medium text-gray-900">
                            {item.filename}
                          </div>
                          <div className="text-sm text-gray-500">
                            {item.file_type.toUpperCase()} • {formatFileSize(item.file_size)}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {getResultBadge(item.is_deepfake)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="w-32">
                          <ConfidenceMeter confidence={item.confidence} size="sm" />
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        <div className="flex items-center">
                          <ClockIcon className="h-4 w-4 mr-1" />
                          {formatDate(item.timestamp)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {item.processing_time.toFixed(2)}s
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button
                          onClick={() => deleteItem(item.id)}
                          className="text-red-600 hover:text-red-900"
                        >
                          <TrashIcon className="h-4 w-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Summary Stats */}
        {history.length > 0 && (
          <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-900">{history.length}</div>
                <div className="text-sm text-gray-500">Total Analyses</div>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {history.filter(item => !item.is_deepfake).length}
                </div>
                <div className="text-sm text-gray-500">Authentic Files</div>
              </div>
            </div>
            <div className="bg-white rounded-lg shadow-sm p-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {history.filter(item => item.is_deepfake).length}
                </div>
                <div className="text-sm text-gray-500">Deepfakes Detected</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default History;