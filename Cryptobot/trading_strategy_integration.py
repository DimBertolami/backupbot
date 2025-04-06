"""
Trading Strategy Integration Module

This module integrates the advanced ML models with the existing trading strategy.
It provides a unified interface for model selection, prediction, and execution.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
import datetime
import logging
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import joblib

# Import existing components
from fetchall import (
    fe_preprocess, 
    fetch_binance_data, 
    calculate_indicators,
    plot_exchange_data
)

# Import advanced components
try:
    from model_trainer import AdvancedModelTrainer
    from performance_tracker import ModelPerformanceTracker
    from advanced_dl_models import (
        build_transformer_model, 
        build_inception_time_model,
        build_temporal_fusion_transformer
    )
except ImportError as e:
    print(f"Warning: Could not import advanced models: {e}")
    print("Some functionality may be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_strategy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_strategy")


class IntegratedTradingStrategy:
    """
    Integrated trading strategy that combines traditional and ML approaches.
    """
    
    def __init__(self, models_dir="advanced_models"):
        """
        Initialize the integrated trading strategy.
        
        Args:
            models_dir (str): Directory containing trained models
        """
        self.models_dir = models_dir
        self.models = {}
        self.active_model = None
        self.active_model_id = None
        self.performance_tracker = None
        self.features = None
        self.scaler = None
        self.sequence_length = 60  # Default
        
        # Load performance tracker if available
        try:
            self.performance_tracker = ModelPerformanceTracker(db_path='performance_db')
            logger.info("Loaded performance tracker")
        except Exception as e:
            logger.warning(f"Could not load performance tracker: {e}")
        
        # Try to load best model
        self.load_best_model()
        logger.info("Integrated trading strategy initialized")
    
    def load_best_model(self):
        """
        Load the best performing model from the models directory.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        # Check for best model pointer
        best_model_path = os.path.join(self.models_dir, "best_model.json")
        
        if os.path.exists(best_model_path):
            try:
                with open(best_model_path, 'r') as f:
                    best_model_info = json.load(f)
                
                model_path = best_model_info.get('path')
                model_type = best_model_info.get('best_model_type')
                
                if model_path and os.path.exists(model_path):
                    logger.info(f"Loading best model: {model_path}")
                    self.active_model = tf.keras.models.load_model(model_path)
                    self.active_model_id = f"{model_type}_{best_model_info.get('iteration', 0)}"
                    
                    # Load parameters
                    model_dir = os.path.dirname(model_path)
                    params_path = os.path.join(model_dir, "parameters.json")
                    
                    if os.path.exists(params_path):
                        with open(params_path, 'r') as f:
                            params = json.load(f)
                        self.sequence_length = params.get('sequence_length', self.sequence_length)
                    
                    logger.info(f"Successfully loaded best model: {self.active_model_id}")
                    return True
                
            except Exception as e:
                logger.error(f"Error loading best model: {e}")
        
        # If we get here, try to find any model in production_models
        try:
            prod_dir = "production_models"
            if os.path.exists(prod_dir):
                model_files = [f for f in os.listdir(prod_dir) if f.endswith('.h5')]
                
                if model_files:
                    # Get most recent model
                    model_files.sort(reverse=True)
                    model_path = os.path.join(prod_dir, model_files[0])
                    
                    logger.info(f"Loading production model: {model_path}")
                    self.active_model = tf.keras.models.load_model(model_path)
                    self.active_model_id = os.path.splitext(model_files[0])[0]
                    
                    # Get parameters if available
                    params_file = self.active_model_id + "_params.json"
                    params_path = os.path.join(prod_dir, params_file)
                    
                    if os.path.exists(params_path):
                        with open(params_path, 'r') as f:
                            params = json.load(f)
                        self.sequence_length = params.get('sequence_length', self.sequence_length)
                    
                    logger.info(f"Successfully loaded production model: {self.active_model_id}")
                    return True
        
        except Exception as e:
            logger.error(f"Error loading production model: {e}")
        
        logger.warning("No models found. Running with traditional strategy only.")
        return False
    
    def prepare_data_for_prediction(self, data):
        """
        Prepare data for model prediction.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            tuple: (X, original_data)
        """
        # Ensure we have all required features
        required_features = [
            'open', 'high', 'low', 'close', 'volume',
            'SMA', 'EMA', 'RSI', 'MACD',
            'UpperBand', 'MiddleBand', 'LowerBand'
        ]
        
        # Check if data contains the required features
        missing_features = [f for f in required_features if f not in data.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Try to calculate indicators
            try:
                data = calculate_indicators(data)
                logger.info("Calculated indicators for missing features")
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
        
        # Select only available features from the required list
        available_features = [f for f in required_features if f in data.columns]
        
        # Scale features
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            data_scaled = self.scaler.fit_transform(data[available_features])
        else:
            data_scaled = self.scaler.transform(data[available_features])
        
        # Create sequences
        X = []
        if len(data) >= self.sequence_length:
            for i in range(len(data) - self.sequence_length + 1):
                X.append(data_scaled[i:i+self.sequence_length])
            X = np.array(X)
        else:
            logger.warning(f"Not enough data for sequence length {self.sequence_length}")
            # Pad sequences if necessary
            padding = np.zeros((self.sequence_length - len(data), len(available_features)))
            padded_data = np.vstack([padding, data_scaled])
            X = np.array([padded_data])
        
        self.features = available_features
        return X, data
    
    def predict(self, data=None):
        """
        Generate predictions using the active model.
        
        Args:
            data (pd.DataFrame, optional): Input data. If None, fetch new data.
            
        Returns:
            tuple: (pd.DataFrame with predictions, raw_predictions)
        """
        # Get data if not provided
        if data is None:
            try:
                data = fe_preprocess(exch="binance")
                logger.info("Fetched new data for prediction")
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                return None, None
        
        # Make predictions with traditional strategy if no ML model
        if self.active_model is None:
            logger.info("No active ML model. Using traditional strategy.")
            return self.traditional_strategy(data), None
        
        # Prepare data for ML prediction
        X, data = self.prepare_data_for_prediction(data)
        
        if X is None or len(X) == 0:
            logger.error("Failed to prepare data for prediction")
            return None, None
        
        # Generate predictions
        try:
            raw_predictions = self.active_model.predict(X)
            
            # Convert to trading signals
            signals = np.where(raw_predictions > 0.5, 1, -1)
            
            # Add predictions to data
            result = data.copy()
            
            # Handle case where prediction length doesn't match data length
            if len(signals) == len(result):
                result['ml_signal'] = signals
            elif len(signals) < len(result):
                # Pad with NaN at the beginning
                padding = np.full(len(result) - len(signals), np.nan)
                result['ml_signal'] = np.concatenate([padding, signals.flatten()])
            else:
                # Take most recent predictions
                result['ml_signal'] = signals[-len(result):].flatten()
            
            # Combine with traditional signals if available
            if 'target' in result.columns:
                # Weighted combination
                result['combined_signal'] = 0.7 * result['ml_signal'] + 0.3 * result['target']
                result['final_signal'] = np.where(result['combined_signal'] > 0, 1, -1)
            else:
                result['final_signal'] = result['ml_signal']
            
            logger.info(f"Generated predictions using model {self.active_model_id}")
            return result, raw_predictions
        
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return None, None
    
    def traditional_strategy(self, data):
        """
        Apply traditional trading strategy based on technical indicators.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with trading signals
        """
        result = data.copy()
        
        # Make sure we have technical indicators
        if not all(f in result.columns for f in ['RSI', 'MACD', 'UpperBand', 'LowerBand']):
            try:
                result = calculate_indicators(result)
            except Exception as e:
                logger.error(f"Error calculating indicators: {e}")
                return result
        
        # Generate trading signals
        signals = []
        
        for i in range(len(result)):
            # Default to hold
            signal = 0
            
            try:
                # RSI strategy
                if result.iloc[i]['RSI'] < 30:
                    signal += 1  # Oversold - buy signal
                elif result.iloc[i]['RSI'] > 70:
                    signal -= 1  # Overbought - sell signal
                
                # MACD strategy
                if 'MACD_signal' in result.columns:
                    if result.iloc[i]['MACD'] > result.iloc[i]['MACD_signal']:
                        signal += 1  # Bullish - buy signal
                    elif result.iloc[i]['MACD'] < result.iloc[i]['MACD_signal']:
                        signal -= 1  # Bearish - sell signal
                
                # Bollinger Bands strategy
                if result.iloc[i]['close'] < result.iloc[i]['LowerBand']:
                    signal += 1  # Price below lower band - buy signal
                elif result.iloc[i]['close'] > result.iloc[i]['UpperBand']:
                    signal -= 1  # Price above upper band - sell signal
            
            except Exception as e:
                logger.error(f"Error calculating traditional signal at index {i}: {e}")
            
            # Final signal: buy if sum > 0, sell if sum < 0, hold if sum = 0
            signals.append(1 if signal > 0 else (-1 if signal < 0 else 0))
        
        result['trad_signal'] = signals
        result['final_signal'] = signals  # Without ML, traditional is final
        
        logger.info("Generated predictions using traditional strategy")
        return result
    
    def execute_trades(self, predictions, backtesting=False, initial_balance=10000):
        """
        Execute trades based on predictions.
        
        Args:
            predictions (pd.DataFrame): Data with predictions
            backtesting (bool): Whether to run in backtesting mode
            initial_balance (float): Initial balance for backtesting
            
        Returns:
            pd.DataFrame: Trade results
        """
        if predictions is None or len(predictions) == 0:
            logger.error("No predictions available for trade execution")
            return None
        
        # Make sure we have final signals
        if 'final_signal' not in predictions.columns:
            logger.error("No trading signals in predictions")
            return None
        
        trades = []
        balance = initial_balance
        position = 0
        entry_price = 0
        
        for i in range(1, len(predictions)):
            timestamp = predictions.iloc[i]['timestamp']
            current_price = predictions.iloc[i]['close']
            signal = predictions.iloc[i]['final_signal']
            
            # Skip if signal is NaN
            if pd.isna(signal):
                continue
            
            # Execute trades based on signals
            if signal == 1 and position == 0:  # Buy signal
                if backtesting:
                    # In backtesting, we can buy immediately
                    position = balance / current_price
                    entry_price = current_price
                    balance = 0
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'buy',
                        'price': current_price,
                        'amount': position,
                        'balance': balance,
                        'portfolio_value': position * current_price
                    })
                else:
                    # In real trading, we would place an order
                    logger.info(f"BUY signal at {timestamp}: price={current_price}")
                    # Placeholder for actual trading API call
            
            elif signal == -1 and position > 0:  # Sell signal
                if backtesting:
                    # In backtesting, we can sell immediately
                    balance = position * current_price
                    pnl = position * (current_price - entry_price)
                    
                    trades.append({
                        'timestamp': timestamp,
                        'action': 'sell',
                        'price': current_price,
                        'amount': position,
                        'pnl': pnl,
                        'pnl_pct': (current_price / entry_price - 1) * 100,
                        'balance': balance,
                        'portfolio_value': balance
                    })
                    
                    position = 0
                    entry_price = 0
                else:
                    # In real trading, we would place an order
                    logger.info(f"SELL signal at {timestamp}: price={current_price}")
                    # Placeholder for actual trading API call
        
        # Create trades DataFrame
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Calculate cumulative metrics
            if backtesting and not trades_df.empty:
                trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
                
                # Calculate final portfolio value
                final_value = trades_df.iloc[-1]['balance']
                if position > 0:
                    final_value = position * predictions.iloc[-1]['close']
                
                logger.info(f"Backtesting results: Final value: {final_value}, Return: {(final_value/initial_balance-1)*100:.2f}%")
            
            return trades_df
        
        return pd.DataFrame()
    
    def plot_trading_results(self, data, trades=None, show_indicators=True):
        """
        Plot trading results with indicators and trade signals.
        
        Args:
            data (pd.DataFrame): Price data with signals
            trades (pd.DataFrame, optional): Trade execution data
            show_indicators (bool): Whether to show technical indicators
            
        Returns:
            matplotlib.figure.Figure: Plot figure
        """
        try:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot price
            ax.plot(data['timestamp'], data['close'], label='Price', color='black')
            
            # Plot indicators if requested
            if show_indicators:
                if 'SMA' in data.columns:
                    ax.plot(data['timestamp'], data['SMA'], label='SMA', color='blue', alpha=0.7)
                
                if 'EMA' in data.columns:
                    ax.plot(data['timestamp'], data['EMA'], label='EMA', color='orange', alpha=0.7)
                
                if all(col in data.columns for col in ['UpperBand', 'MiddleBand', 'LowerBand']):
                    ax.plot(data['timestamp'], data['UpperBand'], color='green', alpha=0.3, linestyle='--')
                    ax.plot(data['timestamp'], data['MiddleBand'], color='green', alpha=0.3, linestyle='-')
                    ax.plot(data['timestamp'], data['LowerBand'], color='green', alpha=0.3, linestyle='--')
            
            # Plot signals
            if 'final_signal' in data.columns:
                # Buy signals
                buy_signals = data[data['final_signal'] == 1]
                ax.scatter(buy_signals['timestamp'], buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
                
                # Sell signals
                sell_signals = data[data['final_signal'] == -1]
                ax.scatter(sell_signals['timestamp'], sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
            
            # Plot executed trades
            if trades is not None and not trades.empty:
                buy_trades = trades[trades['action'] == 'buy']
                sell_trades = trades[trades['action'] == 'sell']
                
                ax.scatter(buy_trades['timestamp'], buy_trades['price'], marker='o', color='lime', s=150, label='Buy Executed')
                ax.scatter(sell_trades['timestamp'], sell_trades['price'], marker='o', color='darkred', s=150, label='Sell Executed')
            
            # Style the plot
            ax.set_title(f"Trading Strategy Results: {self.active_model_id or 'Traditional Strategy'}", fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            plt.tight_layout()
            
            # Add performance metrics if available
            if trades is not None and not trades.empty and 'pnl' in trades.columns:
                total_trades = len(trades)
                win_trades = len(trades[trades['pnl'] > 0])
                loss_trades = len(trades[trades['pnl'] < 0])
                win_rate = win_trades / total_trades if total_trades > 0 else 0
                
                total_pnl = trades['pnl'].sum()
                avg_pnl = trades['pnl'].mean()
                
                text = (
                    f"Total Trades: {total_trades}\n"
                    f"Win Rate: {win_rate:.2%}\n"
                    f"Total P&L: ${total_pnl:.2f}\n"
                    f"Avg P&L/Trade: ${avg_pnl:.2f}"
                )
                
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting trading results: {e}")
            return None
    
    def run(self, data=None, backtesting=False, initial_balance=10000):
        """
        Run the integrated trading strategy.
        
        Args:
            data (pd.DataFrame, optional): Input data. If None, fetch new data.
            backtesting (bool): Whether to run in backtesting mode
            initial_balance (float): Initial balance for backtesting
            
        Returns:
            tuple: (predictions, trades, plot)
        """
        # Get predictions
        predictions, _ = self.predict(data)
        
        if predictions is None:
            logger.error("Failed to generate predictions")
            return None, None, None
        
        # Execute trades
        trades = self.execute_trades(predictions, backtesting, initial_balance)
        
        # Generate plot
        plot = self.plot_trading_results(predictions, trades)
        
        # Save results for React frontend
        self.save_results_for_frontend(predictions, trades, plot)
        
        return predictions, trades, plot
    
    def save_results_for_frontend(self, predictions, trades, plot):
        """
        Save trading results for the React frontend.
        
        Args:
            predictions (pd.DataFrame): Prediction results
            trades (pd.DataFrame): Trade execution results
            plot (matplotlib.figure.Figure): Results plot
        """
        if predictions is None:
            return
        
        # Create directory if needed
        react_dir = "/opt/lampp/htdocs/bot/frontend/public/trading_data"
        os.makedirs(react_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save predictions
            if predictions is not None:
                # Convert to simpler format for JSON
                pred_json = {
                    'timestamp': timestamp,
                    'model_id': self.active_model_id or "traditional",
                    'data': predictions.tail(100).to_dict(orient='records')
                }
                
                with open(f"{react_dir}/latest_predictions.json", 'w') as f:
                    json.dump(pred_json, f, default=str, indent=2)
            
            # Save trades
            if trades is not None and not trades.empty:
                trades_json = {
                    'timestamp': timestamp,
                    'model_id': self.active_model_id or "traditional",
                    'trades': trades.to_dict(orient='records')
                }
                
                with open(f"{react_dir}/latest_trades.json", 'w') as f:
                    json.dump(trades_json, f, default=str, indent=2)
            
            # Save plot
            if plot is not None:
                plot.savefig(f"{react_dir}/latest_trading_plot.png", dpi=300, bbox_inches='tight')
            
            logger.info(f"Saved trading results for React frontend")
        
        except Exception as e:
            logger.error(f"Error saving results for frontend: {e}")


# Example usage
if __name__ == "__main__":
    # Create the integrated strategy
    strategy = IntegratedTradingStrategy()
    
    # Run backtesting
    try:
        # Get data
        data = fe_preprocess(exch="binance")
        
        if data is not None:
            # Run strategy
            predictions, trades, plot = strategy.run(data, backtesting=True)
            
            if plot is not None:
                plot.savefig("trading_results.png", dpi=300, bbox_inches='tight')
                print(f"Results saved to trading_results.png")
            
            if trades is not None:
                print(f"Generated {len(trades)} trades")
                if not trades.empty and 'pnl' in trades.columns:
                    total_pnl = trades['pnl'].sum()
                    print(f"Total P&L: ${total_pnl:.2f}")
        else:
            print("Failed to load data for backtesting")
    
    except Exception as e:
        print(f"Error running backtesting: {e}")
