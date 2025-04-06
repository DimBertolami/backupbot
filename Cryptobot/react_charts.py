"""
React-friendly Crypto Charting Module

This adapter module connects your existing cryptocurrency trading bot
with modern visualization capabilities that work well with React frontends.
"""

import os
import numpy as np
import pandas as pd
import json
import base64
import matplotlib
# Use Agg backend to avoid display issues
matplotlib.use('Agg')

# Import the main plotting module
from crypto_plots import (
    plot_candlestick,
    plot_technical_indicators,
    plot_trading_signals,
    generate_all_charts
)

# Function to create base64 encoded plots for React
def create_charts_for_react(data, predictions=None, save_dir='public/charts'):
    """
    Create cryptocurrency charts and return them in a React-friendly format.
    
    Args:
        data (pd.DataFrame): Cryptocurrency data with price and indicators
        predictions (np.array): Trading signal predictions (optional)
        save_dir (str): Directory to save chart images within the React public folder
        
    Returns:
        dict: Chart data in a format that can be used in React components
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate all charts with base64 encoding
    chart_data = generate_all_charts(
        data, 
        predictions=predictions,
        save_dir=save_dir,
        return_base64=True
    )
    
    # Convert to a React-friendly format
    react_chart_data = {
        'timestamp': datetime.now().isoformat(),
        'chartPaths': {
            'candlestick': os.path.join(save_dir, 'candlestick.png'),
            'indicators': os.path.join(save_dir, 'indicators.png'),
            'signals': os.path.join(save_dir, 'signals.png') if predictions is not None else None
        },
        'base64Data': chart_data
    }
    
    # Save the chart data as JSON for React to consume
    json_path = os.path.join(save_dir, 'chart_data.json')
    with open(json_path, 'w') as f:
        # Convert only the paths to JSON (base64 data could be too large)
        json.dump(
            {
                'timestamp': react_chart_data['timestamp'],
                'chartPaths': react_chart_data['chartPaths']
            }, 
            f, 
            indent=2
        )
    
    return react_chart_data

# Function to adapt to the existing plot_exchange_data in fetchall.py
def plot_exchange_data_react(model=None, data=None, exchange_name='binance', 
                            color='black', features=None, predictions=None):
    """
    A drop-in replacement for the plot_exchange_data function that works with React.
    
    Args:
        model: Trained model (for feature importance)
        data: DataFrame with price and indicator data
        exchange_name: Name of exchange for labeling
        color: Color for price line
        features: List of feature names
        predictions: Model predictions
        
    Returns:
        dict: Chart data in a format that can be used in React
    """
    if data is None:
        print("Error: No data provided for plotting")
        return None
    
    # Create a React app charts directory if it doesn't exist
    react_chart_dir = os.path.join(os.getcwd(), 'react_charts')
    os.makedirs(react_chart_dir, exist_ok=True)
    
    # Generate the React-friendly charts
    charts = create_charts_for_react(
        data,
        predictions=predictions,
        save_dir=react_chart_dir
    )
    
    print(f"Charts generated successfully. Available at {react_chart_dir}")
    print(f"Use these chart paths in your React components:")
    for chart_type, path in charts['chartPaths'].items():
        if path:
            print(f"- {chart_type}: {path}")
    
    # Create an HTML example file to demonstrate usage
    html_example = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Crypto Chart Example</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .chart-container {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
            h1, h2 {{ color: #333; }}
        </style>
    </head>
    <body>
        <h1>Cryptocurrency Charts - {exchange_name}</h1>
        
        <div class="chart-container">
            <h2>Candlestick Chart</h2>
            <img src="data:image/png;base64,{charts['base64Data'].get('candlestick', '')}" alt="Candlestick Chart">
        </div>
        
        <div class="chart-container">
            <h2>Technical Indicators</h2>
            <img src="data:image/png;base64,{charts['base64Data'].get('indicators', '')}" alt="Technical Indicators">
        </div>
    """
    
    if 'signals' in charts['base64Data'] and charts['base64Data']['signals']:
        html_example += f"""
        <div class="chart-container">
            <h2>Trading Signals</h2>
            <img src="data:image/png;base64,{charts['base64Data'].get('signals', '')}" alt="Trading Signals">
        </div>
        """
    
    html_example += """
    </body>
    </html>
    """
    
    # Save the HTML example
    html_path = os.path.join(react_chart_dir, 'chart_example.html')
    with open(html_path, 'w') as f:
        f.write(html_example)
    
    print(f"HTML example created at: {html_path}")
    
    return charts

# React component example snippet (for reference)
REACT_COMPONENT_EXAMPLE = """
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const CryptoCharts = () => {
    const [chartData, setChartData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        // Fetch the chart data JSON from your server
        axios.get('/charts/chart_data.json')
            .then(response => {
                setChartData(response.data);
                setLoading(false);
            })
            .catch(err => {
                setError('Failed to load chart data');
                setLoading(false);
                console.error(err);
            });
    }, []);

    if (loading) return <div>Loading cryptocurrency charts...</div>;
    if (error) return <div>{error}</div>;
    if (!chartData) return <div>No chart data available</div>;

    return (
        <div className="crypto-charts">
            <h2>Cryptocurrency Analysis</h2>
            <p>Last updated: {new Date(chartData.timestamp).toLocaleString()}</p>
            
            <div className="chart-container">
                <h3>Price Chart</h3>
                <img 
                    src={chartData.chartPaths.candlestick} 
                    alt="Cryptocurrency price chart" 
                    className="chart-image"
                />
            </div>

            <div className="chart-container">
                <h3>Technical Indicators</h3>
                <img 
                    src={chartData.chartPaths.indicators} 
                    alt="Technical indicators" 
                    className="chart-image"
                />
            </div>

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
    );
};

export default CryptoCharts;
"""

# Example usage
if __name__ == "__main__":
    from datetime import datetime
    
    try:
        # Try to import and use your existing code
        from fetchall import fe_preprocess, make_decision
        
        print("Fetching cryptocurrency data...")
        data = fe_preprocess(exch="binance")
        
        if data is not None:
            # Generate some example predictions (1 for buy, -1 for sell)
            # In real use, these would come from your models
            np.random.seed(42)  # For reproducible example
            predictions = np.random.choice([1, -1], size=min(30, len(data)))
            
            # Pad with NaN to match data length
            full_predictions = np.full(len(data), np.nan)
            full_predictions[-len(predictions):] = predictions
            
            # Create the React-friendly charts
            plot_exchange_data_react(
                data=data,
                exchange_name="Binance BTC/USDT",
                predictions=full_predictions
            )
        else:
            print("Failed to fetch data for plotting")
    except ImportError:
        print("Could not import from fetchall.py - generating example data instead")
        
        # Generate example data if fetchall.py can't be imported
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.normal(loc=50000, scale=1000, size=100),
            'high': np.random.normal(loc=51000, scale=1000, size=100),
            'low': np.random.normal(loc=49000, scale=1000, size=100),
            'close': np.random.normal(loc=50500, scale=1000, size=100),
            'volume': np.random.normal(loc=1000000, scale=200000, size=100),
            'SMA': np.random.normal(loc=50200, scale=800, size=100),
            'EMA': np.random.normal(loc=50300, scale=800, size=100),
            'RSI': np.random.normal(loc=50, scale=10, size=100),
            'MACD': np.random.normal(loc=0, scale=100, size=100)
        })
        
        # Example predictions
        predictions = np.random.choice([1, -1], size=100)
        
        # Create the React-friendly charts with example data
        plot_exchange_data_react(
            data=data,
            exchange_name="Example Data",
            predictions=predictions
        )
    
    print("\nHow to use in React:")
    print("1. Copy the chart files to your React app's public directory")
    print("2. Use the React component example to display the charts")
    print("3. The component can be modified to fit your app's design")
    print("\nReact component example saved in the variable REACT_COMPONENT_EXAMPLE")
