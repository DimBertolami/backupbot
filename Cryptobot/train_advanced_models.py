#!/usr/bin/env python3
"""
Advanced Training Pipeline for Cryptocurrency Trading Bot

This script integrates the advanced deep learning models, performance tracking,
and self-improvement system with the existing trading bot infrastructure.

Usage:
    python train_advanced_models.py [--iterations N] [--data-source source]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json

# Import our custom modules
from model_trainer import AdvancedModelTrainer
from performance_tracker import ModelPerformanceTracker, TradingStrategyOptimizer
from advanced_dl_models import (
    build_transformer_model,
    build_inception_time_model,
    build_temporal_fusion_transformer
)

# Import existing modules
try:
    from fetchall import fe_preprocess
    from crypto_data_processing import fetch_historical_data
    from model_evaluation import evaluate_trading_performance
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some functionality may be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("advanced_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("advanced_training")


def fetch_large_dataset(source="binance", timeframe="1h", days=180, pairs=None):
    """
    Fetch a large dataset for training advanced models.
    
    Args:
        source (str): Data source (binance, coingecko, etc.)
        timeframe (str): Data timeframe
        days (int): Number of days of historical data
        pairs (list): List of trading pairs
        
    Returns:
        pd.DataFrame: Combined dataset
    """
    if pairs is None:
        pairs = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
    
    logger.info(f"Fetching {days} days of data for {len(pairs)} pairs from {source}")
    
    combined_data = None
    
    try:
        if source == "binance" and fe_preprocess:
            # Use existing preprocessing function
            combined_data = fe_preprocess(exch=source)
            logger.info(f"Fetched data using fe_preprocess: {combined_data.shape if combined_data is not None else 'None'}")
        
        # If the above doesn't work or returns no data, try direct fetch
        if combined_data is None or combined_data.empty:
            logger.info(f"Falling back to direct data fetching")
            all_data = []
            
            for pair in pairs:
                try:
                    pair_data = fetch_historical_data(
                        exchange=source,
                        symbol=pair,
                        timeframe=timeframe,
                        limit=days * 24  # Assuming hourly data
                    )
                    
                    if pair_data is not None and not pair_data.empty:
                        # Add symbol as a feature
                        pair_data['symbol'] = pair
                        all_data.append(pair_data)
                        logger.info(f"Fetched {len(pair_data)} records for {pair}")
                except Exception as e:
                    logger.error(f"Error fetching data for {pair}: {e}")
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                logger.info(f"Combined data shape: {combined_data.shape}")
            else:
                logger.warning("No data fetched from any source")
    
    except Exception as e:
        logger.error(f"Error in data fetching: {e}")
    
    # If no real data could be fetched, create synthetic data for demonstration
    if combined_data is None or combined_data.empty:
        logger.warning("Using synthetic data for demonstration")
        combined_data = create_synthetic_data(days=days)
    
    return combined_data


def create_synthetic_data(days=180):
    """
    Create synthetic data for demonstration purposes.
    
    Args:
        days (int): Number of days of data to generate
        
    Returns:
        pd.DataFrame: Synthetic dataset
    """
    logger.info(f"Generating {days} days of synthetic data")
    
    # Generate date range
    hours = days * 24
    dates = pd.date_range(start='2023-01-01', periods=hours, freq='H')
    
    # Generate price data with realistic patterns
    base_price = 50000
    
    # Add trend component
    trend = np.cumsum(np.random.normal(0, 50, hours))
    
    # Add cyclical component (daily and weekly patterns)
    daily_cycle = 1000 * np.sin(np.linspace(0, 2*np.pi*days, hours))
    weekly_cycle = 2000 * np.sin(np.linspace(0, 2*np.pi*days/7, hours))
    
    # Add random noise
    noise = np.random.normal(0, 500, hours)
    
    # Combine components
    close_prices = base_price + trend + daily_cycle + weekly_cycle + noise
    
    # Generate OHLC data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices - np.random.normal(0, 100, hours),
        'close': close_prices,
        'high': close_prices + np.abs(np.random.normal(0, 200, hours)),
        'low': close_prices - np.abs(np.random.normal(0, 200, hours)),
        'volume': np.abs(np.random.normal(1000000, 500000, hours))
    })
    
    # Ensure high is highest and low is lowest
    for i in range(len(data)):
        values = [data.loc[i, 'open'], data.loc[i, 'close']]
        data.loc[i, 'high'] = max(data.loc[i, 'high'], max(values))
        data.loc[i, 'low'] = min(data.loc[i, 'low'], min(values))
    
    # Add technical indicators
    # Simple Moving Averages
    data['sma_10'] = data['close'].rolling(window=10).mean()
    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_100'] = data['close'].rolling(window=100).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    data['ema_10'] = data['close'].ewm(span=10).mean()
    data['ema_20'] = data['close'].ewm(span=20).mean()
    data['ema_50'] = data['close'].ewm(span=50).mean()
    
    # MACD
    data['macd'] = data['ema_12'] = data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean()
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # RSI (simplified)
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['bb_middle'] = data['sma_20']
    data['bb_stddev'] = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_stddev']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_stddev']
    
    # Create target variable (next hour price direction)
    data['return'] = data['close'].pct_change()
    data['next_return'] = data['return'].shift(-1)
    data['target'] = (data['next_return'] > 0).astype(int)
    
    # Drop NaN values
    data = data.dropna().reset_index(drop=True)
    
    # Add symbol column
    data['symbol'] = 'BTC/USDT'
    
    logger.info(f"Generated synthetic data with shape: {data.shape}")
    return data


def preprocess_data(data):
    """
    Preprocess data for advanced model training.
    
    Args:
        data (pd.DataFrame): Raw data
        
    Returns:
        tuple: (processed_data, feature_list)
    """
    logger.info("Preprocessing data for model training")
    
    # Copy data to avoid modifying original
    df = data.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Add more features if not already present
    
    # Price-based features
    if 'close' in df.columns and 'open' in df.columns:
        # Percentage change
        df['pct_change'] = df['close'].pct_change()
        
        # Price range
        df['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Volume-based features
    if 'volume' in df.columns:
        # Volume change
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages
        df['volume_sma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_10']
    
    # Indicator-based features (if not already calculated)
    
    # RSI
    if 'rsi' not in df.columns and 'close' in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    if 'macd' not in df.columns and 'close' in df.columns:
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    if 'bb_upper' not in df.columns and 'close' in df.columns:
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['bb_stddev'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + 2 * df['bb_stddev']
        df['bb_lower'] = df['sma_20'] - 2 * df['bb_stddev']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # ATR (Average True Range)
    if 'atr' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        df['atr_ratio'] = df['atr'] / df['close']
    
    # Target variable (if not present)
    if 'target' not in df.columns and 'close' in df.columns:
        df['return'] = df['close'].pct_change()
        df['next_return'] = df['return'].shift(-1)
        df['target'] = (df['next_return'] > 0).astype(int)
    
    # Define features to use (adjust based on available columns)
    base_features = [
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    technical_features = [
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 
        'atr', 'atr_ratio'
    ]
    
    derived_features = [
        'pct_change', 'price_range', 'gap',
        'volume_change', 'volume_ratio'
    ]
    
    # Filter to only include columns that exist
    features = base_features + [f for f in technical_features + derived_features if f in df.columns]
    
    # Drop NaN values
    df = df.dropna().reset_index(drop=True)
    
    logger.info(f"Preprocessing complete. Data shape: {df.shape}, Features: {len(features)}")
    return df, features


def run_advanced_training(data_source="binance", iterations=3, timeframe="1h", days=180):
    """
    Run the advanced training pipeline with multiple iterations for self-improvement.
    
    Args:
        data_source (str): Source to fetch data from
        iterations (int): Number of training/improvement iterations
        timeframe (str): Data timeframe
        days (int): Number of days of historical data
        
    Returns:
        tuple: (best_model, performance_metrics, trainer)
    """
    # Create directories for results
    os.makedirs("advanced_models", exist_ok=True)
    os.makedirs("performance_db", exist_ok=True)
    os.makedirs("training_data", exist_ok=True)
    
    # Fetch large dataset
    data = fetch_large_dataset(source=data_source, timeframe=timeframe, days=days)
    
    # Preprocess data
    processed_data, features = preprocess_data(data)
    
    # Save processed data for later use
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = f"training_data/processed_data_{timestamp}.csv"
    processed_data.to_csv(data_file, index=False)
    logger.info(f"Saved processed data to {data_file}")
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(
        base_dir='training_data',
        performance_db_path='performance_db',
        model_save_dir='advanced_models'
    )
    
    # Run multiple training iterations
    best_model = None
    best_metrics = None
    
    logger.info(f"Starting training pipeline with {iterations} iterations")
    
    for i in range(iterations):
        logger.info(f"Starting iteration {i+1}/{iterations}")
        
        # Run a training cycle
        model, metrics, params = trainer.run_training_cycle(
            processed_data, features, 'target'
        )
        
        # Update best model if this is the first iteration or if better than previous best
        if best_model is None or (
            metrics and 
            'profit_percentage' in metrics.get(list(metrics.keys())[0], {}) and
            metrics[list(metrics.keys())[0]]['profit_percentage'] > 
            best_metrics[list(best_metrics.keys())[0]]['profit_percentage']
        ):
            best_model = model
            best_metrics = metrics
        
        logger.info(f"Completed iteration {i+1}")
    
    # Save final performance summary
    summary_file = trainer.performance_tracker.save_performance_summary()
    
    # Generate performance graphs
    best_model_id = None
    if best_metrics:
        best_model_type = list(best_metrics.keys())[0]
        best_model_id = f"{best_model_type}_iter{trainer.current_iteration-1}"
        
        # Plot performance history
        performance_fig = trainer.performance_tracker.plot_performance_history(
            model_id=best_model_id, metric='profit_percentage'
        )
        
        if performance_fig:
            plot_file = f"training_data/performance_plot_{timestamp}.png"
            performance_fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved performance plot to {plot_file}")
    
    logger.info("Advanced training pipeline completed successfully")
    return best_model, best_metrics, trainer


def export_model_for_production(model, model_type, trainer):
    """
    Export the trained model for production use.
    
    Args:
        model: Trained model
        model_type (str): Type of model
        trainer (AdvancedModelTrainer): Trainer instance
        
    Returns:
        str: Path to exported model
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = "production_models"
    os.makedirs(export_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(export_dir, f"{model_type}_{timestamp}.h5")
    model.save(model_path)
    
    # Save parameters
    params_path = os.path.join(export_dir, f"{model_type}_{timestamp}_params.json")
    with open(params_path, 'w') as f:
        json.dump(trainer.current_params, f, indent=2)
    
    # Save model info
    info_path = os.path.join(export_dir, f"{model_type}_{timestamp}_info.json")
    with open(info_path, 'w') as f:
        json.dump({
            'model_type': model_type,
            'timestamp': timestamp,
            'features': trainer.features if hasattr(trainer, 'features') else [],
            'parameters': trainer.current_params
        }, f, indent=2, default=str)
    
    logger.info(f"Exported model for production use to {model_path}")
    return model_path


def export_for_react_frontend(trainer, best_model_id=None):
    """
    Export training results for the React frontend.
    
    Args:
        trainer (AdvancedModelTrainer): Trainer instance
        best_model_id (str, optional): ID of the best model
        
    Returns:
        str: Path to exported JSON
    """
    # Get comparison of all models
    comparison = trainer.performance_tracker.compare_models(top_n=10)
    
    if comparison.empty:
        logger.warning("No models available for comparison")
        return None
    
    # Convert to JSON-serializable format
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'models': comparison.to_dict(orient='records'),
        'best_model_id': best_model_id,
        'improvement_history': trainer.improvement_history
    }
    
    # Save to file
    os.makedirs("frontend_data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = f"frontend_data/training_results_{timestamp}.json"
    
    with open(export_path, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    logger.info(f"Exported training results for React frontend to {export_path}")
    
    # Copy to React frontend directory if available
    react_dir = "/opt/lampp/htdocs/bot/frontend/public/model_data"
    if os.path.exists("/opt/lampp/htdocs/bot/frontend"):
        os.makedirs(react_dir, exist_ok=True)
        react_path = f"{react_dir}/training_results.json"
        
        # Copy file
        import shutil
        shutil.copy(export_path, react_path)
        logger.info(f"Copied results to React frontend: {react_path}")
    
    return export_path


def main():
    """Main function to run the advanced training pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Advanced Training Pipeline for Crypto Trading Bot')
    parser.add_argument('--iterations', type=int, default=3, help='Number of training iterations')
    parser.add_argument('--data-source', type=str, default="binance", help='Data source')
    parser.add_argument('--timeframe', type=str, default="1h", help='Timeframe')
    parser.add_argument('--days', type=int, default=180, help='Number of days of data')
    args = parser.parse_args()
    
    logger.info(f"Starting advanced training pipeline with arguments: {args}")
    
    try:
        # Run training pipeline
        best_model, best_metrics, trainer = run_advanced_training(
            data_source=args.data_source,
            iterations=args.iterations,
            timeframe=args.timeframe,
            days=args.days
        )
        
        # Get best model ID
        best_model_id = None
        if best_metrics:
            best_model_type = list(best_metrics.keys())[0]
            best_model_id = f"{best_model_type}_iter{trainer.current_iteration-1}"
        
        # Export model for production
        if best_model:
            export_model_for_production(best_model, best_model_type, trainer)
        
        # Export results for React frontend
        export_for_react_frontend(trainer, best_model_id)
        
        logger.info("Advanced training pipeline completed successfully")
        print("\n========== TRAINING COMPLETE ==========")
        print(f"Best model: {best_model_id}")
        
        if best_metrics:
            best_model_metrics = best_metrics[list(best_metrics.keys())[0]]
            print("\nPerformance Metrics:")
            for key, value in best_model_metrics.items():
                print(f"  {key}: {value}")
        
        print("\nCheck the 'advanced_models' directory for saved models")
        print("Check the 'frontend_data' directory for React-compatible results")
        print("========================================\n")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
