import React from 'react';

interface ConfidenceMeterProps {
  confidence: number; // 0 to 1
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
  showLabel?: boolean;
}

const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({
  confidence,
  label,
  size = 'md',
  showPercentage = true,
  showLabel = true,
}) => {
  // Clamp confidence between 0 and 1
  const clampedConfidence = Math.max(0, Math.min(1, confidence));
  const percentage = Math.round(clampedConfidence * 100);

  // Determine color based on confidence level
  const getColor = (conf: number) => {
    if (conf < 0.3) return 'success'; // Likely authentic (green)
    if (conf < 0.7) return 'warning'; // Uncertain (yellow)
    return 'danger'; // Likely deepfake (red)
  };

  const color = getColor(clampedConfidence);

  // Size configurations
  const sizeConfig = {
    sm: {
      height: 'h-2',
      indicator: 'w-3 h-3',
      text: 'text-xs',
      spacing: 'space-y-1',
    },
    md: {
      height: 'h-3',
      indicator: 'w-4 h-4',
      text: 'text-sm',
      spacing: 'space-y-2',
    },
    lg: {
      height: 'h-4',
      indicator: 'w-5 h-5',
      text: 'text-base',
      spacing: 'space-y-3',
    },
  };

  const config = sizeConfig[size];

  // Color classes
  const colorClasses = {
    success: {
      bg: 'bg-success-500',
      text: 'text-success-700',
      border: 'border-success-600',
    },
    warning: {
      bg: 'bg-warning-500',
      text: 'text-warning-700',
      border: 'border-warning-600',
    },
    danger: {
      bg: 'bg-danger-500',
      text: 'text-danger-700',
      border: 'border-danger-600',
    },
  };

  const colorClass = colorClasses[color];

  // Get confidence label
  const getConfidenceLabel = (conf: number) => {
    if (conf < 0.3) return 'Likely Authentic';
    if (conf < 0.7) return 'Uncertain';
    return 'Likely Deepfake';
  };

  return (
    <div className={`w-full ${config.spacing}`}>
      {/* Header */}
      {(showLabel || showPercentage) && (
        <div className="flex justify-between items-center">
          {showLabel && (
            <span className={`font-medium ${config.text} ${colorClass.text}`}>
              {label || getConfidenceLabel(clampedConfidence)}
            </span>
          )}
          {showPercentage && (
            <span className={`font-mono ${config.text} ${colorClass.text}`}>
              {percentage}%
            </span>
          )}
        </div>
      )}

      {/* Confidence meter */}
      <div className="relative">
        {/* Background gradient */}
        <div
          className={`w-full ${config.height} rounded-full overflow-hidden`}
          style={{
            background: 'linear-gradient(90deg, #22c55e 0%, #fbbf24 50%, #ef4444 100%)',
          }}
        />

        {/* Indicator */}
        <div
          className={`absolute top-1/2 transform -translate-y-1/2 -translate-x-1/2 ${config.indicator} ${colorClass.bg} border-2 ${colorClass.border} rounded-full shadow-sm`}
          style={{
            left: `${percentage}%`,
          }}
        />
      </div>

      {/* Scale labels */}
      <div className="flex justify-between text-xs text-gray-500 mt-1">
        <span>Authentic</span>
        <span>Uncertain</span>
        <span>Deepfake</span>
      </div>
    </div>
  );
};

export default ConfidenceMeter;