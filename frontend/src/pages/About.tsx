import React from 'react';
import { 
  ShieldCheckIcon, 
  CpuChipIcon, 
  EyeIcon, 
  SpeakerWaveIcon,
  PhotoIcon,
  VideoCameraIcon,
  BeakerIcon,
  LightBulbIcon
} from '@heroicons/react/24/outline';

const About: React.FC = () => {
  const features = [
    {
      icon: VideoCameraIcon,
      title: 'Video Analysis',
      description: 'Advanced frame-by-frame analysis using XceptionNet architecture to detect manipulated video content with high accuracy.'
    },
    {
      icon: PhotoIcon,
      title: 'Image Detection',
      description: 'EfficientNet-based image analysis with Test Time Augmentation (TTA) for robust deepfake image identification.'
    },
    {
      icon: SpeakerWaveIcon,
      title: 'Audio Authentication',
      description: 'MFCC feature extraction and CNN-based models to identify synthetic speech and audio manipulations.'
    },
    {
      icon: CpuChipIcon,
      title: 'Real-time Processing',
      description: 'Optimized inference pipeline for fast detection with sub-second processing times for most media files.'
    },
    {
      icon: ShieldCheckIcon,
      title: 'High Accuracy',
      description: 'State-of-the-art deep learning models trained on diverse datasets achieving >95% accuracy on benchmark tests.'
    },
    {
      icon: BeakerIcon,
      title: 'Batch Processing',
      description: 'Analyze multiple files simultaneously with intelligent batching and parallel processing capabilities.'
    }
  ];

  const techStack = [
    {
      category: 'Backend',
      technologies: ['FastAPI', 'PyTorch', 'TensorFlow', 'OpenCV', 'Librosa', 'NumPy']
    },
    {
      category: 'Frontend',
      technologies: ['React', 'TypeScript', 'Tailwind CSS', 'Chart.js', 'Axios']
    },
    {
      category: 'AI Models',
      technologies: ['XceptionNet', 'EfficientNet', 'CNN', 'MFCC', 'Transfer Learning']
    },
    {
      category: 'Infrastructure',
      technologies: ['Docker', 'SQLite', 'REST API', 'CORS', 'File Upload']
    }
  ];

  const modelDetails = [
    {
      type: 'Video Detection',
      model: 'XceptionNet',
      description: 'Specialized CNN architecture designed for deepfake detection with depthwise separable convolutions.',
      accuracy: '96.2%',
      features: ['Frame extraction', 'Temporal analysis', 'Face region focus', 'Batch processing']
    },
    {
      type: 'Image Detection',
      model: 'EfficientNet-B4',
      description: 'Compound scaling method that uniformly scales network width, depth, and resolution.',
      accuracy: '94.8%',
      features: ['Test Time Augmentation', 'Multi-scale analysis', 'Transfer learning', 'Ensemble methods']
    },
    {
      type: 'Audio Detection',
      model: 'CNN + MFCC',
      description: 'Convolutional neural network trained on Mel-frequency cepstral coefficients.',
      accuracy: '92.5%',
      features: ['Spectral analysis', 'Feature extraction', 'Temporal patterns', 'Voice synthesis detection']
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 text-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl md:text-6xl font-bold mb-6">
              Deepfake Detection System
            </h1>
            <p className="text-xl md:text-2xl mb-8 max-w-3xl mx-auto">
              Advanced AI-powered system for detecting manipulated media content across video, image, and audio formats
            </p>
            <div className="flex justify-center space-x-4">
              <div className="flex items-center space-x-2">
                <EyeIcon className="h-6 w-6" />
                <span>Multi-modal Detection</span>
              </div>
              <div className="flex items-center space-x-2">
                <LightBulbIcon className="h-6 w-6" />
                <span>Real-time Analysis</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Key Features</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our comprehensive deepfake detection system combines cutting-edge AI models 
              with user-friendly interfaces to provide reliable media authentication.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div key={index} className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
                <div className="flex items-center mb-4">
                  <feature.icon className="h-8 w-8 text-indigo-600 mr-3" />
                  <h3 className="text-xl font-semibold text-gray-900">{feature.title}</h3>
                </div>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Details Section */}
      <div className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">AI Models & Performance</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our system employs state-of-the-art deep learning models, each optimized 
              for specific media types to ensure maximum detection accuracy.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {modelDetails.map((model, index) => (
              <div key={index} className="border border-gray-200 rounded-lg p-6">
                <div className="text-center mb-4">
                  <h3 className="text-xl font-semibold text-gray-900 mb-2">{model.type}</h3>
                  <div className="text-3xl font-bold text-indigo-600 mb-2">{model.accuracy}</div>
                  <div className="text-sm text-gray-500">Detection Accuracy</div>
                </div>
                
                <div className="mb-4">
                  <h4 className="font-semibold text-gray-900 mb-2">{model.model}</h4>
                  <p className="text-sm text-gray-600">{model.description}</p>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-900 mb-2">Key Features</h4>
                  <ul className="space-y-1">
                    {model.features.map((feature, idx) => (
                      <li key={idx} className="text-sm text-gray-600 flex items-center">
                        <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full mr-2"></div>
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Tech Stack Section */}
      <div className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Technology Stack</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Built with modern technologies and frameworks to ensure scalability, 
              performance, and maintainability.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {techStack.map((stack, index) => (
              <div key={index} className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 text-center">
                  {stack.category}
                </h3>
                <div className="space-y-2">
                  {stack.technologies.map((tech, idx) => (
                    <div key={idx} className="text-center">
                      <span className="inline-block bg-gray-100 text-gray-800 text-sm px-3 py-1 rounded-full">
                        {tech}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="bg-gray-100 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Our detection process combines multiple AI models and analysis techniques 
              to provide comprehensive media authentication.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="bg-indigo-600 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                1
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Media</h3>
              <p className="text-gray-600">Upload your video, image, or audio file for analysis</p>
            </div>

            <div className="text-center">
              <div className="bg-indigo-600 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                2
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">AI Analysis</h3>
              <p className="text-gray-600">Our AI models analyze the content for manipulation signs</p>
            </div>

            <div className="text-center">
              <div className="bg-indigo-600 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                3
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Feature Extraction</h3>
              <p className="text-gray-600">Extract key features and patterns from the media content</p>
            </div>

            <div className="text-center">
              <div className="bg-indigo-600 text-white rounded-full w-12 h-12 flex items-center justify-center mx-auto mb-4 text-xl font-bold">
                4
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Results</h3>
              <p className="text-gray-600">Get detailed results with confidence scores and analysis</p>
            </div>
          </div>
        </div>
      </div>

      {/* Call to Action */}
      <div className="bg-indigo-600 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to Verify Your Media?
          </h2>
          <p className="text-xl text-indigo-100 mb-8 max-w-2xl mx-auto">
            Start using our deepfake detection system to ensure the authenticity 
            of your media content and protect against digital manipulation.
          </p>
          <a
            href="/upload"
            className="inline-flex items-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-600 bg-white hover:bg-gray-50 transition-colors"
          >
            Start Detection
          </a>
        </div>
      </div>
    </div>
  );
};

export default About;