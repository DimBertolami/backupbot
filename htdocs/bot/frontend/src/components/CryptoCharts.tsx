import React, { useState, useEffect } from 'react';
import { ChartData } from '../types/ChartData';
import { ChevronLeft, ChevronRight, Eye, EyeOff } from 'lucide-react';

interface CryptoChartsProps {
  refreshInterval?: number; // in milliseconds
}

const CryptoCharts: React.FC<CryptoChartsProps> = ({ refreshInterval = 0 }) => {
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [activeSlide, setActiveSlide] = useState<number>(0);
  const [isHidden, setIsHidden] = useState<boolean>(false);
  
  // Chart types for carousel
  const chartTypes = ['price_3d', 'candlestick', 'indicators_combined', 'signals_3d', 'signals'];
  const chartTitles = {
    'price_3d': '3D Price Surface',
    'candlestick': 'Price Chart',
    'indicators_combined': 'Price & All Indicators',
    'signals_3d': '3D Trading Signals',
    'signals': 'Trading Signals'
  };

  // Function to fetch chart data
  const fetchChartData = async () => {
    try {
      setLoading(true);
      const response = await fetch('/charts/chart_data.json');
      
      if (!response.ok) {
        throw new Error(`Error fetching chart data: ${response.status}`);
      }
      
      const data = await response.json();
      setChartData(data);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      console.error('Failed to load chart data:', err);
      setError('Failed to load cryptocurrency charts');
    } finally {
      setLoading(false);
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchChartData();
    
    // Setup refresh interval if specified
    if (refreshInterval > 0) {
      const intervalId = setInterval(fetchChartData, refreshInterval);
      return () => clearInterval(intervalId);
    }
  }, [refreshInterval]);

  if (loading && !chartData) {
    return <div className="loading">Loading cryptocurrency charts...</div>;
  }

  if (error && !chartData) {
    return <div className="error">{error}</div>;
  }

  if (!chartData) {
    return <div className="no-data">No chart data available</div>;
  }

  // Navigation functions for carousel
  const nextSlide = () => {
    setActiveSlide((prev) => 
      prev === chartTypes.length - 1 ? 0 : prev + 1
    );
  };

  const prevSlide = () => {
    setActiveSlide((prev) => 
      prev === 0 ? chartTypes.length - 1 : prev - 1
    );
  };

  const toggleVisibility = () => {
    setIsHidden(!isHidden);
  };

  // If charts are hidden, just show the controls
  if (isHidden) {
    return (
      <div className="crypto-charts hidden-charts">
        <div className="charts-controls">
          <h2>Cryptocurrency Analysis</h2>
          <button 
            className="visibility-toggle" 
            onClick={toggleVisibility}
            aria-label="Show charts"
          >
            <Eye size={20} />
            <span>Show Charts</span>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="crypto-charts">
      <div className="charts-header">
        <h2>Cryptocurrency Analysis</h2>
        <div className="charts-actions">
          {lastUpdate && (
            <p className="update-time">
              Last updated: {lastUpdate.toLocaleString()}
            </p>
          )}
          <button 
            className="visibility-toggle" 
            onClick={toggleVisibility}
            aria-label="Hide charts"
          >
            <EyeOff size={20} />
            <span>Hide Charts</span>
          </button>
        </div>
      </div>
      
      <div className="chart-carousel">
        {/* Carousel Navigation */}
        <button 
          className="carousel-nav carousel-prev" 
          onClick={prevSlide}
          aria-label="Previous chart"
        >
          <ChevronLeft />
        </button>

        {/* Chart Display */}
        <div className="carousel-container">
          {chartTypes.map((type, index) => {
            const isActive = index === activeSlide;
            
            // Handle different chart types and their data sources
            let dataType = type;
            let hasHtmlChart = false;
            let htmlSrc = '';
            
            // Map chart types to data structure keys
            if (type === 'indicators_combined') {
              dataType = 'indicators';
              hasHtmlChart = !!chartData?.chartPaths?.indicators_html;
              htmlSrc = chartData?.chartPaths?.indicators_html || '';
            } else if (type === 'price_3d') {
              hasHtmlChart = !!chartData?.chartPaths?.price_3d_html;
              htmlSrc = chartData?.chartPaths?.price_3d_html || '';
            } else if (type === 'signals_3d') {
              hasHtmlChart = !!chartData?.chartPaths?.signals_3d_html;
              htmlSrc = chartData?.chartPaths?.signals_3d_html || '';
            }
            
            const hasChart = chartData?.chartPaths?.[dataType as keyof typeof chartData.chartPaths];
            const title = chartTitles[type as keyof typeof chartTitles];
            
            return (
              <div 
                key={type} 
                className={`carousel-slide ${isActive ? 'active' : ''}`}
                style={{ display: isActive ? 'block' : 'none' }}
              >
                {hasHtmlChart ? (
                  // Render the HTML chart in an iframe
                  <div className={`chart-container chart-${type}`}>
                    <h3>{title}</h3>
                    <iframe
                      src={htmlSrc}
                      title={title}
                      className="chart-iframe"
                      style={{ width: '100%', height: '600px', border: 'none' }}
                      sandbox="allow-same-origin allow-scripts"
                    />
                  </div>
                ) : hasChart ? (
                  <div className={`chart-container chart-${type}`}>
                    <h3>{title}</h3>
                    <img 
                      src={chartData.chartPaths[dataType as keyof typeof chartData.chartPaths]} 
                      alt={title} 
                      className="chart-image"
                    />
                  </div>
                ) : (
                  <div className="no-chart">
                    <p>No {title.toLowerCase()} available</p>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Carousel Navigation */}
        <button 
          className="carousel-nav carousel-next" 
          onClick={nextSlide}
          aria-label="Next chart"
        >
          <ChevronRight />
        </button>
      </div>

      {/* Enhanced Navigation Controls */}
      <div className="carousel-controls">
        <div className="carousel-dots">
          {chartTypes.map((_, index) => (
            <button
              key={index}
              className={`carousel-dot ${index === activeSlide ? 'active' : ''}`}
              onClick={() => setActiveSlide(index)}
              aria-label={`Go to chart ${index + 1}`}
            />
          ))}
        </div>
        
        <div className="carousel-actions">
          <button 
            className="scroll-button prev-button" 
            onClick={prevSlide}
            aria-label="Scroll left"
          >
            Prev
          </button>
          
          <button 
            className="scroll-button next-button" 
            onClick={nextSlide}
            aria-label="Scroll right"
          >
            Next
          </button>
        </div>
      </div>
      
      <button 
        className="refresh-button" 
        onClick={fetchChartData}
        disabled={loading}
      >
        {loading ? 'Refreshing...' : 'Refresh Charts'}
      </button>
    </div>
  );
};

export default CryptoCharts;
