"""
React TypeScript Integration Module for Cryptocurrency Charting

This module provides functions to generate cryptocurrency charts and export them
in a format that can be easily consumed by TypeScript React applications.
"""

import os
import sys
import json
import shutil
import base64
import numpy as np
import pandas as pd
from datetime import datetime
import traceback

# Import our plotting utilities
from crypto_plots import (
    plot_candlestick,
    plot_technical_indicators,
    plot_trading_signals,
    generate_all_charts
)

# Get data from the cryptocurrency bot
try:
    from fetchall import fe_preprocess
except ImportError:
    print("Warning: Could not import from fetchall, using sample data instead")
    fe_preprocess = None

# Default paths for React projects
REACT_PROJECTS = {
    'main': '/opt/lampp/htdocs/bot/frontend',
    'new': '/opt/lampp/htdocs/bot/frontend2',
    'legacy': '/opt/lampp/htdocs/bot/oldfront'
}

def generate_charts_for_react(data=None, predictions=None, target_dir=None, 
                             project_name='main', return_base64=False):
    """
    Generate cryptocurrency charts for React and export them to the specified project.
    
    Args:
        data (pd.DataFrame): Cryptocurrency data with price and indicators
        predictions (np.array): Trading signal predictions
        target_dir (str): Target directory within the React project
        project_name (str): Name of the React project ('main', 'new', or 'legacy')
        return_base64 (bool): Whether to return base64 encoded images
        
    Returns:
        dict: Chart data configuration for React
    """
    # Validate project name
    if project_name not in REACT_PROJECTS:
        print(f"Unknown project: {project_name}. Using 'main' instead.")
        project_name = 'main'
    
    # Get project directory
    react_project_dir = REACT_PROJECTS[project_name]
    
    # Set default target directory if not provided
    if target_dir is None:
        target_dir = 'public/charts'
    
    # Create full path to chart directory
    chart_dir = os.path.join(react_project_dir, target_dir)
    
    # Create directory if it doesn't exist
    os.makedirs(chart_dir, exist_ok=True)
    
    # If no data provided, try to get it
    if data is None:
        if fe_preprocess:
            try:
                print("Fetching cryptocurrency data...")
                data = fe_preprocess(exch="binance")
                if data is None:
                    raise ValueError("Failed to get data from fe_preprocess")
            except Exception as e:
                print(f"Error fetching data: {e}")
                data = create_sample_data()
        else:
            data = create_sample_data()
    
    # Generate charts and save them
    print(f"Generating charts for React project: {project_name}")
    print(f"Saving charts to: {chart_dir}")
    
    result = {
        'timestamp': datetime.now().isoformat(),
        'chartPaths': {},
        'base64Data': {}
    }
    
    # Generate candlestick chart
    try:
        candlestick_path = os.path.join(chart_dir, 'candlestick.png')
        candlestick_b64 = plot_candlestick(
            data, 
            title='Cryptocurrency Price',
            save_path=candlestick_path,
            return_base64=return_base64
        )
        
        if os.path.exists(candlestick_path):
            result['chartPaths']['candlestick'] = f"{target_dir}/candlestick.png"
            
        if return_base64 and candlestick_b64:
            result['base64Data']['candlestick'] = candlestick_b64
    except Exception as e:
        print(f"Error generating candlestick chart: {e}")
        traceback.print_exc()
    
    # Generate technical indicators chart
    try:
        indicators_path = os.path.join(chart_dir, 'indicators.png')
        indicators_b64 = plot_technical_indicators(
            data,
            title='Technical Indicators',
            save_path=indicators_path,
            return_base64=return_base64
        )
        
        if os.path.exists(indicators_path):
            result['chartPaths']['indicators'] = f"{target_dir}/indicators.png"
            
        if return_base64 and indicators_b64:
            result['base64Data']['indicators'] = indicators_b64
    except Exception as e:
        print(f"Error generating technical indicators chart: {e}")
        traceback.print_exc()
    
    # Generate trading signals chart
    if predictions is not None:
        try:
            signals_path = os.path.join(chart_dir, 'signals.png')
            signals_b64 = plot_trading_signals(
                data,
                model_predictions=predictions,
                title='Trading Signals',
                save_path=signals_path,
                return_base64=return_base64
            )
            
            if os.path.exists(signals_path):
                result['chartPaths']['signals'] = f"{target_dir}/signals.png"
                
            if return_base64 and signals_b64:
                result['base64Data']['signals'] = signals_b64
        except Exception as e:
            print(f"Error generating trading signals chart: {e}")
            traceback.print_exc()
    
    # Create TypeScript type definition file
    ts_types = """
export interface ChartData {
    timestamp: string;
    chartPaths: {
        candlestick?: string;
        indicators?: string;
        signals?: string;
    };
}
"""
    
    # Save TypeScript definition
    types_dir = os.path.join(react_project_dir, 'src/types')
    os.makedirs(types_dir, exist_ok=True)
    with open(os.path.join(types_dir, 'ChartData.ts'), 'w') as f:
        f.write(ts_types)
    
    # Save JSON configuration for React
    json_path = os.path.join(chart_dir, 'chart_data.json')
    with open(json_path, 'w') as f:
        # Don't include base64 data in JSON file (too large)
        json.dump(
            {
                'timestamp': result['timestamp'],
                'chartPaths': result['chartPaths']
            }, 
            f, 
            indent=2
        )
    
    # Create an example React component file
    react_component = """import React, { useState, useEffect } from 'react';
import { ChartData } from '../types/ChartData';

interface CryptoChartsProps {
  refreshInterval?: number; // in milliseconds
}

const CryptoCharts: React.FC<CryptoChartsProps> = ({ refreshInterval = 0 }) => {
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

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

  return (
    <div className="crypto-charts">
      <h2>Cryptocurrency Analysis</h2>
      {lastUpdate && (
        <p className="update-time">
          Last updated: {lastUpdate.toLocaleString()}
        </p>
      )}
      
      <div className="chart-grid">
        {chartData.chartPaths.candlestick && (
          <div className="chart-container">
            <h3>Price Chart</h3>
            <img 
              src={chartData.chartPaths.candlestick} 
              alt="Cryptocurrency price chart" 
              className="chart-image"
            />
          </div>
        )}

        {chartData.chartPaths.indicators && (
          <div className="chart-container">
            <h3>Technical Indicators</h3>
            <img 
              src={chartData.chartPaths.indicators} 
              alt="Technical indicators" 
              className="chart-image"
            />
          </div>
        )}

        {chartData.chartPaths.signals && (
          <div className="chart-container">
            <h3>Trading Signals</h3>
            <img 
              src={chartData.chartPaths.signals} 
              alt="Trading signals" 
              className="chart-image"
            />
          </div>
        )}
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
"""
    
    # Save component file
    components_dir = os.path.join(react_project_dir, 'src/components')
    os.makedirs(components_dir, exist_ok=True)
    with open(os.path.join(components_dir, 'CryptoCharts.tsx'), 'w') as f:
        f.write(react_component)
    
    print(f"Successfully exported charts for React project: {project_name}")
    print(f"TypeScript type definition saved to: {types_dir}/ChartData.ts")
    print(f"Example React component saved to: {components_dir}/CryptoCharts.tsx")
    print(f"Chart data JSON saved to: {json_path}")
    
    return result

def create_sample_data():
    """Create sample data when real data is not available"""
    print("Generating sample data for demonstration...")
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate random OHLC data with trends
    base = 50000 + np.cumsum(np.random.normal(0, 500, 100))
    # Ensure volatility is always positive
    daily_volatility = np.abs(np.random.normal(loc=800, scale=300, size=100))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': base,
        'close': base + np.random.normal(0, 100, 100),
        'high': base + daily_volatility,
        'low': base - daily_volatility * 0.5,  # Make sure low is not too low
        'volume': np.abs(np.random.normal(loc=1000000, scale=500000, size=100))
    })
    
    # Ensure high is the highest and low is the lowest each day
    for i in range(len(data)):
        values = [data.loc[i, 'open'], data.loc[i, 'close']]
        data.loc[i, 'high'] = max(values) + abs(np.random.normal(0, 200))
        data.loc[i, 'low'] = min(values) - abs(np.random.normal(0, 200))
    
    # Add some indicators for demonstration
    data['SMA'] = data['close'].rolling(window=14).mean()
    data['EMA'] = data['close'].ewm(span=14).mean()
    data['RSI'] = 50 + np.random.normal(0, 15, 100)  # Simulated RSI
    data['MACD'] = data['EMA'] - data['SMA']  # Simulated MACD
    
    # Simulate Bollinger Bands
    std = data['close'].rolling(window=20).std()
    data['MiddleBand'] = data['close'].rolling(window=20).mean()
    data['UpperBand'] = data['MiddleBand'] + (std * 2)
    data['LowerBand'] = data['MiddleBand'] - (std * 2)
    
    return data

def main():
    """Main function to deploy charts to React project"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Generate cryptocurrency charts for React')
    parser.add_argument('--project', '-p', choices=['main', 'new', 'legacy'], default='main',
                        help='Target React project (default: main)')
    parser.add_argument('--dir', '-d', default='public/charts',
                        help='Target directory within React project (default: public/charts)')
    parser.add_argument('--sample', '-s', action='store_true',
                        help='Use sample data instead of fetching real data')
    
    args = parser.parse_args()
    
    # Get data
    data = None
    if args.sample:
        data = create_sample_data()
    else:
        # Try to get real data
        try:
            if fe_preprocess:
                data = fe_preprocess(exch="binance")
            else:
                print("Module fetchall not available, using sample data")
                data = create_sample_data()
        except Exception as e:
            print(f"Error fetching real data: {e}")
            data = create_sample_data()
    
    # Import prediction function from tradebot
    from tradebot import train_model
    
    # Train model and get signals
    _, predictions = train_model(data)
    
    # Generate charts for the specified React project
    generate_charts_for_react(
        data=data,
        predictions=full_predictions,
        target_dir=args.dir,
        project_name=args.project
    )

if __name__ == "__main__":
    main()
