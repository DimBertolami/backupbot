import React from 'react';

interface BotPerformanceCardProps {
  onRefresh?: () => void;
  loading?: boolean;
}

const BotPerformanceCard: React.FC<BotPerformanceCardProps> = ({ 
  onRefresh = () => {},
  loading = false
}) => {
  return (
    <div className="theme-card rounded-xl p-6 mt-8">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
          Bot Performance
        </h2>
        <div className="flex items-center space-x-2">
          <div className="text-sm text-gray-600 dark:text-gray-300">
            Last update: {new Date().toLocaleTimeString()}
          </div>
          <button 
            onClick={onRefresh}
            disabled={loading}
            className="px-3 py-1 text-sm rounded bg-blue-500 text-white hover:bg-blue-600 transition-colors"
          >
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Confidence Score Chart */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Confidence Score</h3>
          <div className="h-64">
            <div className="relative h-full">
              <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 opacity-20 rounded-lg"></div>
              <div className="relative h-full p-4">
                <div className="h-full flex items-center justify-center">
                  <div className="text-4xl font-bold text-blue-500 dark:text-blue-300">82%</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Overall Profit Chart */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Overall Profit</h3>
          <div className="h-64">
            <div className="relative h-full">
              <div className="absolute inset-0 bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 opacity-20 rounded-lg"></div>
              <div className="relative h-full p-4">
                <div className="h-full flex items-center justify-center">
                  <div className="text-4xl font-bold text-green-500 dark:text-green-300">+12.8%</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Learning Curve Chart */}
        <div className="space-y-4 col-span-2">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Learning Curve</h3>
          <div className="h-64">
            <div className="relative h-full">
              <div className="absolute inset-0 bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 opacity-20 rounded-lg"></div>
              <div className="relative h-full p-4">
                <div className="h-full flex items-center justify-center">
                  <div className="text-4xl font-bold text-orange-500 dark:text-orange-300">87%</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Performance Metrics */}
        <div className="space-y-4 col-span-2">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Recent Performance</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-green-50 dark:bg-green-900/10 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-300">Win Rate</div>
              <div className="text-2xl font-bold text-green-600 dark:text-green-300">85%</div>
            </div>
            <div className="p-4 bg-blue-50 dark:bg-blue-900/10 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-300">Average Profit</div>
              <div className="text-2xl font-bold text-blue-600 dark:text-blue-300">2.3%</div>
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-900/10 rounded-lg">
              <div className="text-sm text-gray-600 dark:text-gray-300">Drawdown</div>
              <div className="text-2xl font-bold text-purple-600 dark:text-purple-300">-0.8%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BotPerformanceCard;
