import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { format } from 'date-fns';
import { Menu, Sun, Moon, LogOut } from 'lucide-react';

interface TrainingProgress {
  timestamp: string;
  accuracy: number;
  loss: number;
  epoch: number;
  totalEpochs: number;
  trainingSpeed: number; // epochs per minute
  currentPhase: string;
  estimatedCompletion: string;
}

interface BacktestResult {
  timestamp: string;
  profit: number;
  trades: number;
  winRate: number;
}

interface Decision {
  timestamp: string;
  symbol: string;
  action: string;
  confidence: number;
  reason: string;
}

interface LogEntry {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'error';
  command?: string;
  result?: 'ok' | 'fail';
  details?: {
    title: string;
    content: string;
  };
}

interface ModelPerformance {
  name: string;
  type: string;
  metrics: {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    confidence: number;
    latency: number;
    memoryUsage: number;
  };
  tasks: {
    pricePrediction: number;
    patternRecognition: number;
    marketSentiment: number;
    riskAssessment: number;
  };
  training: {
    epochs: number;
    loss: number;
    learningRate: number;
    time: string;
  };
  architecture: {
    layers: {
      type: string;
      units?: number;
      activation?: string;
      num_heads?: number;
      key_dim?: number;
      max_len?: number;
    }[];
    inputShape: number[];
    optimizer: string;
    loss: string;
    metrics: string[];
  };
}

const mockData = {
  trainingProgress: [
    { 
      timestamp: '2025-04-05T21:46:17', 
      accuracy: 0.895, 
      loss: 0.0025, 
      epoch: 1250,
      totalEpochs: 2000,
      trainingSpeed: 1.2, // epochs per minute
      currentPhase: 'Fine-tuning',
      estimatedCompletion: '2025-04-06T02:15:00'
    }
  ] as TrainingProgress[],
  
  backtestResults: [
    { timestamp: '2024-03-01', profit: 0.02, trades: 50, winRate: 0.65 },
    { timestamp: '2024-03-02', profit: 0.03, trades: 60, winRate: 0.68 },
    { timestamp: '2024-03-03', profit: 0.04, trades: 70, winRate: 0.71 },
    // ... more data
  ] as BacktestResult[],

  recentDecisions: [
    { timestamp: '2024-03-05T14:30:00', symbol: 'BTC', action: 'BUY', confidence: 0.92, reason: 'Bullish trend confirmed' },
    { timestamp: '2024-03-05T14:25:00', symbol: 'ETH', action: 'SELL', confidence: 0.88, reason: 'Overbought conditions detected' },
    // ... more data
  ] as Decision[],

  logEntries: [
    {
      timestamp: '2025-04-05T21:34:31',
      message: 'Initializing model ensemble for market analysis...',
      type: 'info',
      details: {
        title: 'Model Architecture Overview',
        content: 'Our system utilizes a multi-model ensemble approach:\n\n1. LSTM-Attention for price prediction\n2. Random Forest for pattern recognition\n3. XGBoost for feature importance analysis\n4. Reinforcement Learning for strategy optimization\n\nEach model serves a specific purpose in our trading strategy.'
      }
    },
    {
      timestamp: '2025-04-05T21:34:32',
      message: 'Loading LSTM-Attention model for price prediction',
      type: 'info',
      details: {
        title: 'LSTM-Attention Model Details',
        content: 'Architecture:\n- 3 LSTM layers with attention mechanism\n- Input: 200 time steps of OHLCV data\n- Output: Price predictions for next 12 hours\n\nPerformance Metrics:\n- MAE: 0.0025%\n- RMSE: 0.0032%\n- Accuracy: 89.5%\n\nPurpose:\nPredicts short-term price movements to identify entry/exit points.'
      }
    },
    {
      timestamp: '2025-04-05T21:34:33',
      message: 'Loading Random Forest for pattern recognition',
      type: 'info',
      details: {
        title: 'Random Forest Model Details',
        content: 'Configuration:\n- 500 trees\n- Max depth: 20\n- Features: 30 technical indicators\n\nPerformance:\n- Pattern recognition rate: 92.1%\n- False positive rate: 4.5%\n\nPurpose:\nIdentifies complex market patterns and trading opportunities.'
      }
    },
    {
      timestamp: '2025-04-05T21:34:34',
      message: 'Analyzing market conditions with XGBoost',
      type: 'info',
      details: {
        title: 'XGBoost Model Details',
        content: 'Current Feature Importance:\n1. RSI (Relative Strength Index)\n2. MACD (Moving Average Convergence Divergence)\n3. Volume\n4. Price Momentum\n5. Volatility\n\nModel Performance:\n- Feature detection accuracy: 94.2%\n- Market regime classification: 88.7%\n\nPurpose:\nEvaluates market conditions and identifies optimal trading opportunities.'
      }
    },
    {
      timestamp: '2025-05-05T21:34:35',
      message: 'Executing strategy optimization with RL agent',
      type: 'info',
      details: {
        title: 'Reinforcement Learning Details',
        content: 'RL Agent Status:\n- Training episodes: 12,500\n- Reward function: Risk-adjusted returns\n- Exploration rate: 0.1\n\nCurrent Strategy Metrics:\n- Sharpe ratio: 2.15\n- Maximum drawdown: 4.2%\n- Win rate: 68.5%\n\nPurpose:\nContinuously optimizes trading strategy based on real-time market feedback.'
      }
    },
    {
      timestamp: '2025-04-05T21:34:36',
      message: 'Model evaluation metrics updated',
      type: 'info',
      details: {
        title: 'Real-time Model Performance',
        content: 'Current Performance:\n\nLSTM-Attention:\n- Prediction accuracy: 89.5%\n- MAE: 0.0025%\n\nRandom Forest:\n- Pattern detection: 92.1%\n- False positives: 4.5%\n\nXGBoost:\n- Feature detection: 94.2%\n- Market classification: 88.7%\n\nRL Agent:\n- Sharpe ratio: 2.15\n- Max drawdown: 4.2%\n- Win rate: 68.5%'
      }
    },
    {
      timestamp: '2025-04-05T21:34:37',
      message: 'Learning from recent market behavior',
      type: 'info',
      details: {
        title: 'Learning Process Details',
        content: 'Current Learning Focus:\n1. Adapting to recent volatility\n2. Optimizing position sizing\n3. Improving risk management\n\nRecent Improvements:\n- Reduced false positives by 3.2%\n- Improved prediction accuracy by 1.5%\n- Enhanced market regime detection\n\nLearning Method:\n- Online learning with incremental updates\n- Continuous model retraining\n- Adaptive parameter tuning'
      }
    },
    {
      timestamp: '2025-04-05T21:34:38',
      message: 'Strategy revision based on new insights',
      type: 'info',
      details: {
        title: 'Strategy Revision Details',
        content: 'Current Strategy Adjustments:\n1. Increased position sizing for high confidence trades\n2. Enhanced stop-loss placement\n3. Improved take-profit levels\n\nPerformance Impact:\n- Expected return increase: 2.5%\n- Risk reduction: 1.8%\n- Trade efficiency improvement: 3.2%\n\nRevision Rationale:\n- Market conditions have changed\n- New patterns have been identified\n- Risk profile has been reassessed'
      }
    }
  ] as LogEntry[],

  models: [
    {
      name: "LSTM-Attention",
      type: "Sequence Prediction",
      metrics: {
        accuracy: 0.895,
        precision: 0.91,
        recall: 0.88,
        f1Score: 0.89,
        confidence: 0.92,
        latency: 0.045,
        memoryUsage: 128
      },
      tasks: {
        pricePrediction: 0.92,
        patternRecognition: 0.78,
        marketSentiment: 0.65,
        riskAssessment: 0.81
      },
      training: {
        epochs: 1250,
        loss: 0.0025,
        learningRate: 0.0001,
        time: "2025-04-05T21:46:17"
      },
      architecture: {
        layers: [
          { type: "LSTM", units: 128, activation: "tanh" },
          { type: "Attention", units: 64 },
          { type: "Dense", units: 64, activation: "relu" },
          { type: "Dense", units: 1, activation: "linear" }
        ],
        inputShape: [200, 5],
        optimizer: "adam",
        loss: "mse",
        metrics: ["mae", "mse"]
      }
    },
    {
      name: "Transformer-Encoder",
      type: "Time Series Analysis",
      metrics: {
        accuracy: 0.912,
        precision: 0.92,
        recall: 0.90,
        f1Score: 0.91,
        confidence: 0.93,
        latency: 0.038,
        memoryUsage: 256
      },
      tasks: {
        pricePrediction: 0.93,
        patternRecognition: 0.88,
        marketSentiment: 0.82,
        riskAssessment: 0.85
      },
      training: {
        epochs: 1500,
        loss: 0.0018,
        learningRate: 0.00005,
        time: "2025-04-05T21:46:17"
      },
      architecture: {
        layers: [
          { type: "MultiHeadAttention", num_heads: 8, key_dim: 64 },
          { type: "FeedForward", units: 256, activation: "gelu" },
          { type: "PositionalEncoding", max_len: 200 },
          { type: "Dense", units: 1, activation: "linear" }
        ],
        inputShape: [200, 5],
        optimizer: "adam",
        loss: "mse",
        metrics: ["mae", "mse"]
      }
    },
    {
      name: "ConvLSTM",
      type: "Temporal Pattern Recognition",
      metrics: {
        accuracy: 0.905,
        precision: 0.91,
        recall: 0.89,
        f1Score: 0.90,
        confidence: 0.91,
        latency: 0.042,
        memoryUsage: 192
      },
      tasks: {
        pricePrediction: 0.89,
        patternRecognition: 0.91,
        marketSentiment: 0.78,
        riskAssessment: 0.83
      },
      training: {
        epochs: 1000,
        loss: 0.0021,
        learningRate: 0.0002,
        time: "2025-04-05T21:46:17"
      },
      architecture: {
        layers: [
          { type: "ConvLSTM2D", filters: 64, kernel_size: [3, 3] },
          { type: "MaxPooling2D", pool_size: [2, 2] },
          { type: "Flatten" },
          { type: "Dense", units: 128, activation: "relu" },
          { type: "Dense", units: 1, activation: "linear" }
        ],
        inputShape: [200, 5, 5, 1],
        optimizer: "adam",
        loss: "mse",
        metrics: ["mae", "mse"]
      }
    },
    {
      name: "Autoencoder",
      type: "Anomaly Detection",
      metrics: {
        accuracy: 0.925,
        precision: 0.93,
        recall: 0.91,
        f1Score: 0.92,
        confidence: 0.94,
        latency: 0.028,
        memoryUsage: 128
      },
      tasks: {
        pricePrediction: 0.78,
        patternRecognition: 0.85,
        marketSentiment: 0.92,
        riskAssessment: 0.94
      },
      training: {
        epochs: 800,
        loss: 0.0015,
        learningRate: 0.001,
        time: "2025-04-05T21:46:17"
      },
      architecture: {
        layers: [
          { type: "Dense", units: 128, activation: "relu" },
          { type: "Dense", units: 64, activation: "relu" },
          { type: "Dense", units: 32, activation: "relu" },
          { type: "Dense", units: 64, activation: "relu" },
          { type: "Dense", units: 128, activation: "relu" },
          { type: "Dense", units: 1, activation: "linear" }
        ],
        inputShape: [200, 5],
        optimizer: "adam",
        loss: "mse",
        metrics: ["mae", "mse"]
      }
    },
    {
      name: "BERT-Transformer",
      type: "Sentiment Analysis",
      metrics: {
        accuracy: 0.935,
        precision: 0.94,
        recall: 0.92,
        f1Score: 0.93,
        confidence: 0.95,
        latency: 0.065,
        memoryUsage: 512
      },
      tasks: {
        pricePrediction: 0.75,
        patternRecognition: 0.82,
        marketSentiment: 0.95,
        riskAssessment: 0.88
      },
      training: {
        epochs: 500,
        loss: 0.0012,
        learningRate: 0.00001,
        time: "2025-04-05T21:46:17"
      },
      architecture: {
        layers: [
          { type: "BERT", max_length: 128 },
          { type: "Transformer", num_heads: 12, key_dim: 64 },
          { type: "Dense", units: 128, activation: "gelu" },
          { type: "Dense", units: 1, activation: "sigmoid" }
        ],
        inputShape: [128],
        optimizer: "adam",
        loss: "binary_crossentropy",
        metrics: ["accuracy", "AUC"]
      }
    },
    {
      name: "Hybrid RNN-CNN",
      type: "Feature Extraction",
      metrics: {
        accuracy: 0.918,
        precision: 0.92,
        recall: 0.90,
        f1Score: 0.91,
        confidence: 0.92,
        latency: 0.045,
        memoryUsage: 256
      },
      tasks: {
        pricePrediction: 0.88,
        patternRecognition: 0.92,
        marketSentiment: 0.85,
        riskAssessment: 0.87
      },
      training: {
        epochs: 1200,
        loss: 0.0019,
        learningRate: 0.0001,
        time: "2025-04-05T21:46:17"
      },
      architecture: {
        layers: [
          { type: "Conv1D", filters: 64, kernel_size: 3 },
          { type: "GRU", units: 128 },
          { type: "Bidirectional", units: 64 },
          { type: "Dense", units: 128, activation: "relu" },
          { type: "Dense", units: 1, activation: "linear" }
        ],
        inputShape: [200, 5],
        optimizer: "adam",
        loss: "mse",
        metrics: ["mae", "mse"]
      }
    }
  ] as ModelPerformance[]
};

const MLInsights: React.FC = () => {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [showModelDetails, setShowModelDetails] = useState(false);

  const handleMouseEnter = (e: React.MouseEvent, details: string) => {
    const tooltip = document.createElement('div');
    tooltip.className = `fixed z-50 bg-gray-800 text-white px-4 py-2 rounded-lg shadow-lg text-sm max-w-[400px] ${isDarkMode ? 'bg-gray-800' : 'bg-white text-gray-900'}`;
    tooltip.style.left = `${e.clientX + 10}px`;
    tooltip.style.top = `${e.clientY + 10}px`;
    
    // Convert markdown-like content to HTML
    const htmlContent = details
      .split('\n')
      .map(line => {
        if (line.startsWith('- ')) {
          return `<li>${line.slice(2)}</li>`;
        }
        if (line.startsWith('1. ')) {
          return `<li>${line.slice(3)}</li>`;
        }
        return `<p>${line}</p>`;
      })
      .join('');
    
    tooltip.innerHTML = `
      <div class="p-2">
        ${htmlContent}
      </div>
    `;
    
    document.body.appendChild(tooltip);
  };

  const handleMouseLeave = () => {
    const tooltips = document.querySelectorAll('.fixed.z-50');
    tooltips.forEach(tooltip => tooltip.remove());
  };

  const calculateTimeRemaining = (estimatedCompletion: string): string => {
    const now = new Date();
    const completion = new Date(estimatedCompletion);
    const diff = completion.getTime() - now.getTime();
    
    if (diff <= 0) {
      return 'Training Complete';
    }
    
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    
    return `${hours}h ${minutes}m remaining`;
  };

  const getModelMetricsChart = (model: ModelPerformance) => {
    const data = [
      { name: 'Accuracy', value: model.metrics.accuracy * 100 },
      { name: 'Precision', value: model.metrics.precision * 100 },
      { name: 'Recall', value: model.metrics.recall * 100 },
      { name: 'F1 Score', value: model.metrics.f1Score * 100 },
      { name: 'Confidence', value: model.metrics.confidence * 100 }
    ];

    return (
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  const getModelTaskChart = (model: ModelPerformance) => {
    const data = [
      { name: 'Price Prediction', value: model.tasks.pricePrediction * 100 },
      { name: 'Pattern Recognition', value: model.tasks.patternRecognition * 100 },
      { name: 'Market Sentiment', value: model.tasks.marketSentiment * 100 },
      { name: 'Risk Assessment', value: model.tasks.riskAssessment * 100 }
    ];

    return (
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis domain={[0, 100]} />
          <Tooltip />
          <Line type="monotone" dataKey="value" stroke="#82ca9d" />
        </LineChart>
      </ResponsiveContainer>
    );
  };

  const getModelArchitectureChart = (model: ModelPerformance) => {
    const layers = model.architecture?.layers || [];
    const data = layers.map((layer, index) => ({
      name: `${layer.type} ${index + 1}`,
      units: layer.units || 0,
      index
    }));

    return (
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" angle={-45} textAnchor="end" />
          <YAxis domain={[0, Math.max(...data.map(d => d.units))]} />
          <Tooltip />
          <Bar dataKey="units" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    );
  };

  useEffect(() => {
    // Initialize log entries
    setLogEntries(mockData.logEntries);

    // Update the mock data with new insights
    const updateInsights = async () => {
      // In a real app, we would fetch new data and update the state
      // For now, this is just a placeholder for future implementation
      // console.log('Updating insights with latest data...');
    };

    const interval = setInterval(updateInsights, 60000); // Update every minute
    return () => clearInterval(interval);
  }, []); // mockData is intentionally not included in dependencies

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${isDarkMode ? 'dark' : ''}`}>
      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 w-64 md:w-72 lg:w-80 bg-white dark:bg-gray-800 shadow-lg transition-transform duration-300 ${
          isSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        <div className="p-4 border-b dark:border-gray-700">
          <h1 className="text-xl font-bold text-gray-900 dark:text-white">Internals</h1>
        </div>
        <nav className="p-4 space-y-2">
          <button
            onClick={() => window.location.href = '/'}
            className="flex items-center p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-red-600 dark:text-red-400 w-full"
            title="Exit Internals"
          >
            <LogOut className="h-5 w-5 mr-2" />
            <span>Exit</span>
          </button>
          <hr className="my-2 border-gray-200 dark:border-gray-700" />
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
          </button>
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <Menu className="h-5 w-5" />
          </button>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="ml-64 md:ml-72 lg:ml-80 p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Model Overview */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Model Overview</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium text-gray-500">Architecture</h4>
                <p className="text-lg font-semibold">LSTM + Attention</p>
              </div>
              <div>
                <h4 className="text-sm font-medium text-gray-500">Training Status</h4>
                <p className="text-lg font-semibold text-green-600">Active</p>
              </div>
            </div>
          </div>

          {/* Training Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Training Progress</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockData.trainingProgress}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="accuracy" stroke="#8884d8" />
                  <Line type="monotone" dataKey="loss" stroke="#82ca9d" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Learning Progress */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Learning Progress</h3>
            {mockData.trainingProgress.map((progress, index) => (
              <div key={index} className="space-y-4">
                {/* Progress Bar */}
                <div className="mb-4">
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm">Training Progress</span>
                    <span className="text-sm font-medium">{Math.round((progress.epoch / progress.totalEpochs) * 100)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                    <div 
                      className="bg-blue-600 dark:bg-blue-400 h-2.5 rounded-full transition-all duration-500"
                      style={{ width: `${(progress.epoch / progress.totalEpochs) * 100}%` }}
                    ></div>
                  </div>
                </div>

                {/* Training Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-medium mb-1">Current Phase</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{progress.currentPhase}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-1">Epochs</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{progress.epoch}/{progress.totalEpochs}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-1">Training Speed</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{progress.trainingSpeed.toFixed(1)} epochs/min</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-1">Estimated Completion</h4>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{calculateTimeRemaining(progress.estimatedCompletion)}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-1">Accuracy</h4>
                    <p className="text-sm text-green-600 dark:text-green-400">{(progress.accuracy * 100).toFixed(2)}%</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium mb-1">Loss</h4>
                    <p className="text-sm text-red-600 dark:text-red-400">{progress.loss.toFixed(4)}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Log Output */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Decision Log</h3>
            <div className="space-y-2 h-[400px] overflow-y-auto">
              {logEntries.map((entry, index) => (
                <div 
                  key={index} 
                  className={`p-3 rounded-lg relative ${
                    entry.type === 'info' ? 'bg-blue-50 dark:bg-blue-900 text-blue-600 dark:text-blue-300' :
                    entry.type === 'success' ? 'bg-green-50 dark:bg-green-900 text-green-600 dark:text-green-300' :
                    'bg-red-50 dark:bg-red-900 text-red-600 dark:text-red-300'
                  }`}
                  onMouseEnter={(e) => entry.details && handleMouseEnter(e, entry.details.content)}
                  onMouseLeave={handleMouseLeave}
                >
                  <div className="flex justify-between items-start">
                    <span className="text-sm font-medium">{format(new Date(entry.timestamp), 'HH:mm:ss')}</span>
                    <span className="text-xs opacity-70">{entry.type}</span>
                  </div>
                  <p className="mt-1 text-sm">{entry.message}</p>
                  {entry.command && (
                    <div className="mt-1">
                      <span className="font-mono bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded text-xs">executed: {entry.command}</span>
                    </div>
                  )}
                  {entry.result && (
                    <div className="mt-1">
                      <span className={`font-mono px-2 py-1 rounded text-xs ${
                        entry.result === 'ok' ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'
                      }`}>result: {entry.result}</span>
                    </div>
                  )}
                  {entry.details && (
                    <div className="mt-1 text-xs text-gray-600 dark:text-gray-400 cursor-pointer hover:text-blue-400 dark:hover:text-blue-300">
                      <span className="font-semibold">Details:</span>
                      <span className="ml-1">Hover for more info</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Model Evaluation Dashboard */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6">
            <h3 className="text-lg font-semibold mb-4">Model Evaluation Dashboard</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {mockData.models.map((model) => (
                <div 
                  key={model.name}
                  className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                  onClick={() => {
                    setSelectedModel(model.name);
                    setShowModelDetails(true);
                  }}
                >
                  <div className="flex justify-between items-start">
                    <h4 className="font-medium">{model.name}</h4>
                    <span className="px-2 py-1 rounded-full text-xs bg-blue-100 dark:bg-blue-900 text-blue-600 dark:text-blue-300">
                      {model.type}
                    </span>
                  </div>
                  <div className="mt-2 space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Accuracy</span>
                      <span>{(model.metrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Confidence</span>
                      <span>{(model.metrics.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Latency</span>
                      <span>{model.metrics.latency.toFixed(2)}s</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Backtesting Results */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Backtesting Results</h3>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={mockData.backtestResults}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <Tooltip />
                  <Line type="monotone" dataKey="profit" stroke="#8884d8" />
                  <Line type="monotone" dataKey="winRate" stroke="#82ca9d" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Recent Decisions */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-lg font-semibold mb-4">Recent Decisions</h3>
            <div className="space-y-4">
              {mockData.recentDecisions.map((decision, index) => (
                <div key={index} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <div className="flex justify-between items-start">
                    <div>
                      <h4 className="font-medium">{decision.symbol}</h4>
                      <p className="text-sm text-gray-500">{format(new Date(decision.timestamp), 'HH:mm')}</p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className={`px-2 py-1 rounded-full text-xs ${
                        decision.action === 'BUY' ? 'bg-green-100 text-green-800' : 
                        'bg-red-100 text-red-800'
                      }`}>
                        {decision.action}
                      </span>
                      <span className="text-sm text-gray-500">{Math.round(decision.confidence * 100)}%</span>
                    </div>
                  </div>
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">{decision.reason}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {showModelDetails && selectedModel && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl w-full max-w-4xl max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <h3 className="text-xl font-semibold mb-4">{selectedModel}</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h4 className="text-lg font-medium mb-4">Performance Metrics</h4>
                  {getModelMetricsChart(mockData.models.find(m => m.name === selectedModel)!)}
                </div>
                
                <div>
                  <h4 className="text-lg font-medium mb-4">Task Performance</h4>
                  {getModelTaskChart(mockData.models.find(m => m.name === selectedModel)!)}
                </div>
              </div>

              <div className="mt-6">
                <h4 className="text-lg font-medium mb-4">Training Details</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <h5 className="text-sm font-medium mb-2">Training Progress</h5>
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm">Epochs</span>
                      <span className="text-sm font-medium">
                        {mockData.models.find(m => m.name === selectedModel)!.training.epochs}/
                        {mockData.models.find(m => m.name === selectedModel)!.training.epochs}
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2.5">
                      <div 
                        className="bg-blue-600 dark:bg-blue-400 h-2.5 rounded-full transition-all duration-500"
                        style={{
                          width: `${(mockData.models.find(m => m.name === selectedModel)!.training.epochs / 
                            mockData.models.find(m => m.name === selectedModel)!.training.epochs) * 100}%`
                        }}
                      ></div>
                    </div>
                  </div>

                  <div>
                    <h5 className="text-sm font-medium mb-2">Training Parameters</h5>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Learning Rate</span>
                        <span>{mockData.models.find(m => m.name === selectedModel)!.training.learningRate}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Loss</span>
                        <span>{mockData.models.find(m => m.name === selectedModel)!.training.loss}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Start Time</span>
                        <span>{mockData.models.find(m => m.name === selectedModel)!.training.time}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Optimizer</span>
                        <span>{mockData.models.find(m => m.name === selectedModel)!.architecture?.optimizer || 'adam'}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Loss Function</span>
                        <span>{mockData.models.find(m => m.name === selectedModel)!.architecture?.loss}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {mockData.models.find(m => m.name === selectedModel)!.architecture && (
                <div className="mt-6">
                  <h4 className="text-lg font-medium mb-4">Model Architecture</h4>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <h5 className="text-sm font-medium mb-2">Layer Configuration</h5>
                      <div className="space-y-2">
                        {mockData.models.find(m => m.name === selectedModel)!.architecture.layers.map((layer, index) => (
                          <div key={index} className="flex justify-between text-sm">
                            <span>{layer.type} {index + 1}</span>
                            <span>{layer.units} units</span>
                          </div>
                        ))}
                      </div>
                    </div>
                    <div>
                      <h5 className="text-sm font-medium mb-2">Architecture Visualization</h5>
                      {getModelArchitectureChart(mockData.models.find(m => m.name === selectedModel)!)}
                    </div>
                  </div>
                </div>
              )}

            </div>

            <button 
              onClick={() => setShowModelDetails(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18"></line>
                <line x1="6" y1="6" x2="18" y2="18"></line>
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default MLInsights;
