"""
Performance Tracking System for Cryptocurrency Trading Models

This module implements tools for tracking, logging, and analyzing the performance
of machine learning models used in cryptocurrency trading, with capabilities for
self-improvement and strategy evolution.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import datetime
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tabulate import tabulate
from pymongo import MongoClient
import logging
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_performance.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("performance_tracker")

# Suppress warnings
warnings.filterwarnings('ignore')


class ModelPerformanceTracker:
    """
    Track and analyze the performance of trading models over time.
    """
    
    def __init__(self, db_path='performance_db', use_mongodb=False, mongodb_uri=None):
        """
        Initialize the performance tracker.
        
        Args:
            db_path (str): Path to store performance data
            use_mongodb (bool): Whether to use MongoDB for storage
            mongodb_uri (str): MongoDB connection URI
        """
        self.db_path = db_path
        self.use_mongodb = use_mongodb
        
        # Create storage directory if needed
        if not use_mongodb and not os.path.exists(db_path):
            os.makedirs(db_path)
        
        # Connect to MongoDB if requested
        if use_mongodb:
            if mongodb_uri is None:
                mongodb_uri = "mongodb://localhost:27017/"
            self.client = MongoClient(mongodb_uri)
            self.db = self.client.trading_performance
            self.models_collection = self.db.models
            self.trades_collection = self.db.trades
            self.metrics_collection = self.db.metrics
            logger.info(f"Connected to MongoDB at {mongodb_uri}")
        
        # Track current performance metrics
        self.current_metrics = {}
        self.historical_metrics = []
        
        logger.info(f"Performance tracker initialized with storage at {db_path}")
    
    def record_model_performance(self, model_id, model_type, trading_pair, timeframe,
                               metrics, predictions=None, trades=None, parameters=None):
        """
        Record performance metrics for a model.
        
        Args:
            model_id (str): Unique identifier for the model
            model_type (str): Type of model (e.g., 'lstm', 'transformer')
            trading_pair (str): Trading pair (e.g., 'BTC-USD')
            timeframe (str): Timeframe used (e.g., '1h', '1d')
            metrics (dict): Performance metrics
            predictions (pd.DataFrame, optional): Prediction results
            trades (pd.DataFrame, optional): Trading results
            parameters (dict, optional): Model parameters
            
        Returns:
            str: ID of the recorded performance entry
        """
        timestamp = datetime.datetime.now()
        
        # Create performance record
        performance_data = {
            "model_id": model_id,
            "model_type": model_type,
            "trading_pair": trading_pair,
            "timeframe": timeframe,
            "timestamp": timestamp,
            "metrics": metrics,
            "parameters": parameters if parameters else {}
        }
        
        # Store in MongoDB or local filesystem
        if self.use_mongodb:
            # Store performance data
            result = self.metrics_collection.insert_one(performance_data)
            performance_id = str(result.inserted_id)
            
            # Store predictions if provided
            if predictions is not None and not predictions.empty:
                pred_data = predictions.to_dict(orient='records')
                for p in pred_data:
                    p['model_id'] = model_id
                    p['performance_id'] = performance_id
                self.db.predictions.insert_many(pred_data)
            
            # Store trades if provided
            if trades is not None and not trades.empty:
                trades_data = trades.to_dict(orient='records')
                for t in trades_data:
                    t['model_id'] = model_id
                    t['performance_id'] = performance_id
                self.trades_collection.insert_many(trades_data)
        else:
            # Store in local filesystem
            performance_id = f"{model_id}_{timestamp.strftime('%Y%m%d%H%M%S')}"
            performance_file = os.path.join(self.db_path, f"{performance_id}.json")
            
            # Save metrics and parameters
            with open(performance_file, 'w') as f:
                json.dump(performance_data, f, default=str, indent=2)
            
            # Save predictions if provided
            if predictions is not None and not predictions.empty:
                pred_file = os.path.join(self.db_path, f"{performance_id}_predictions.csv")
                predictions.to_csv(pred_file, index=False)
            
            # Save trades if provided
            if trades is not None and not trades.empty:
                trades_file = os.path.join(self.db_path, f"{performance_id}_trades.csv")
                trades.to_csv(trades_file, index=False)
        
        # Update current metrics
        self.current_metrics = {
            "model_id": model_id,
            "timestamp": timestamp,
            **metrics
        }
        
        # Add to historical metrics
        self.historical_metrics.append({
            "model_id": model_id,
            "timestamp": timestamp,
            **metrics
        })
        
        logger.info(f"Recorded performance for model {model_id}: {metrics}")
        return performance_id
    
    def calculate_trading_metrics(self, trades_df, initial_balance=10000):
        """
        Calculate comprehensive trading metrics from a dataframe of trades.
        
        Args:
            trades_df (pd.DataFrame): DataFrame with trades
            initial_balance (float): Initial account balance
            
        Returns:
            dict: Trading metrics
        """
        if trades_df is None or trades_df.empty:
            logger.warning("No trades provided for metrics calculation")
            return {}
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'action', 'price', 'amount', 'profit']
        missing_cols = [col for col in required_cols if col not in trades_df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in trades DataFrame: {missing_cols}")
            return {}
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] < 0])
        
        if total_trades == 0:
            win_rate = 0
        else:
            win_rate = winning_trades / total_trades
        
        # Calculate returns
        total_profit = trades_df['profit'].sum()
        profit_percentage = (total_profit / initial_balance) * 100
        
        # Calculate drawdown
        balance = initial_balance
        balance_history = [balance]
        
        for profit in trades_df['profit']:
            balance += profit
            balance_history.append(balance)
        
        balance_series = pd.Series(balance_history)
        running_max = balance_series.cummax()
        drawdown = (balance_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Calculate Sharpe ratio (simplified)
        if len(trades_df) > 1:
            returns = trades_df['profit'] / initial_balance
            sharpe_ratio = np.sqrt(365) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0
        
        # Average profit per trade
        avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0
        
        # Profit factor
        total_gains = trades_df[trades_df['profit'] > 0]['profit'].sum()
        total_losses = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        profit_factor = total_gains / total_losses if total_losses != 0 else float('inf')
        
        metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "profit_percentage": profit_percentage,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_profit_per_trade": avg_profit_per_trade,
            "profit_factor": profit_factor,
            "final_balance": balance
        }
        
        return metrics
    
    def compare_models(self, model_ids=None, top_n=5, metric='profit_percentage'):
        """
        Compare performance of multiple models.
        
        Args:
            model_ids (list, optional): List of model IDs to compare
            top_n (int): Number of top models to return
            metric (str): Metric to sort by
            
        Returns:
            pd.DataFrame: Comparison of model performance
        """
        # Get all performance records
        if self.use_mongodb:
            if model_ids:
                cursor = self.metrics_collection.find({"model_id": {"$in": model_ids}})
            else:
                cursor = self.metrics_collection.find()
            performance_records = list(cursor)
        else:
            performance_records = []
            for file in os.listdir(self.db_path):
                if file.endswith('.json') and not file.endswith('_config.json'):
                    with open(os.path.join(self.db_path, file), 'r') as f:
                        record = json.load(f)
                        if model_ids is None or record.get('model_id') in model_ids:
                            performance_records.append(record)
        
        if not performance_records:
            logger.warning("No performance records found for comparison")
            return pd.DataFrame()
        
        # Extract metrics for comparison
        comparison_data = []
        for record in performance_records:
            metrics = record.get('metrics', {})
            
            row = {
                "model_id": record.get('model_id'),
                "model_type": record.get('model_type'),
                "trading_pair": record.get('trading_pair'),
                "timeframe": record.get('timeframe'),
                "timestamp": record.get('timestamp')
            }
            
            # Add all metrics
            for m_key, m_value in metrics.items():
                row[m_key] = m_value
            
            comparison_data.append(row)
        
        # Convert to DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by the specified metric and get top N
        if metric in comparison_df.columns:
            comparison_df = comparison_df.sort_values(by=metric, ascending=False).head(top_n)
        
        return comparison_df
    
    def plot_performance_history(self, model_id=None, metric='profit_percentage'):
        """
        Plot historical performance of a model or all models.
        
        Args:
            model_id (str, optional): Model ID to plot
            metric (str): Metric to plot
            
        Returns:
            matplotlib.figure.Figure: Performance plot
        """
        # Get performance data
        if model_id:
            if self.use_mongodb:
                records = list(self.metrics_collection.find({"model_id": model_id}))
            else:
                records = []
                for file in os.listdir(self.db_path):
                    if file.endswith('.json') and model_id in file:
                        with open(os.path.join(self.db_path, file), 'r') as f:
                            records.append(json.load(f))
        else:
            # Use historical metrics
            records = self.historical_metrics
        
        if not records:
            logger.warning(f"No historical data found for model_id={model_id}")
            return None
        
        # Extract data for plotting
        plot_data = []
        for record in records:
            timestamp = record.get('timestamp')
            if isinstance(timestamp, str):
                timestamp = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            metrics = record.get('metrics', {})
            if not metrics and metric in record:
                # If metrics are in the root of the record
                value = record[metric]
            else:
                value = metrics.get(metric)
            
            if value is not None:
                plot_data.append({
                    'timestamp': timestamp,
                    'model_id': record.get('model_id'),
                    'value': value
                })
        
        if not plot_data:
            logger.warning(f"No data for metric '{metric}' in records")
            return None
        
        # Convert to DataFrame and sort by timestamp
        plot_df = pd.DataFrame(plot_data)
        plot_df = plot_df.sort_values('timestamp')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if model_id:
            # Single model
            ax.plot(plot_df['timestamp'], plot_df['value'], marker='o', linestyle='-')
            ax.set_title(f"{metric} History for Model {model_id}")
        else:
            # Multiple models
            for model, group in plot_df.groupby('model_id'):
                ax.plot(group['timestamp'], group['value'], marker='o', linestyle='-', label=model)
            ax.legend(title="Model ID")
            ax.set_title(f"{metric} History Across Models")
        
        ax.set_xlabel("Date")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def save_performance_summary(self, file_path=None):
        """
        Save a summary of performance metrics to a file.
        
        Args:
            file_path (str, optional): Path to save summary
            
        Returns:
            str: Path to saved summary file
        """
        # Generate summary from historical data
        if not self.historical_metrics:
            logger.warning("No historical metrics available for summary")
            return None
        
        # Convert to DataFrame
        summary_df = pd.DataFrame(self.historical_metrics)
        
        # Generate default file path if not provided
        if file_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_path = os.path.join(self.db_path, f"performance_summary_{timestamp}.csv")
        
        # Save summary
        summary_df.to_csv(file_path, index=False)
        logger.info(f"Performance summary saved to {file_path}")
        
        return file_path
    
    def get_best_model(self, metric='profit_percentage'):
        """
        Get the best performing model based on a specific metric.
        
        Args:
            metric (str): Metric to use for ranking
            
        Returns:
            dict: Best model info
        """
        comparison = self.compare_models(top_n=1, metric=metric)
        
        if comparison.empty:
            logger.warning("No models found for comparison")
            return None
        
        return comparison.iloc[0].to_dict()


class TradingStrategyOptimizer:
    """
    Optimize trading strategies based on historical performance.
    """
    
    def __init__(self, performance_tracker):
        """
        Initialize the strategy optimizer.
        
        Args:
            performance_tracker (ModelPerformanceTracker): Performance tracking system
        """
        self.performance_tracker = performance_tracker
        self.improvement_history = []
        logger.info("Trading strategy optimizer initialized")
    
    def analyze_model_weaknesses(self, model_id, trades_df):
        """
        Analyze model weaknesses based on trading history.
        
        Args:
            model_id (str): Model ID
            trades_df (pd.DataFrame): DataFrame with trades
            
        Returns:
            dict: Weakness analysis
        """
        if trades_df is None or trades_df.empty:
            logger.warning("No trades provided for weakness analysis")
            return {}
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'action', 'price', 'profit', 'market_condition']
        missing_cols = [col for col in required_cols if col not in trades_df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns for detailed analysis: {missing_cols}")
            # Add minimal market condition if missing
            if 'market_condition' in missing_cols and 'profit' in trades_df.columns:
                # Try to infer basic market condition
                trades_df['market_trend'] = np.where(trades_df['profit'] > 0, 'up', 'down')
        
        # Analyze performance across different market conditions
        if 'market_condition' in trades_df.columns:
            condition_performance = trades_df.groupby('market_condition').agg({
                'profit': ['sum', 'mean', 'count'],
                'action': 'count'
            })
            
            # Find worst performing conditions
            if not condition_performance.empty:
                worst_condition = condition_performance['profit']['sum'].idxmin()
                worst_condition_data = {
                    'condition': worst_condition,
                    'total_profit': condition_performance['profit']['sum'][worst_condition],
                    'avg_profit': condition_performance['profit']['mean'][worst_condition],
                    'trade_count': condition_performance['action']['count'][worst_condition]
                }
            else:
                worst_condition_data = {}
        else:
            condition_performance = pd.DataFrame()
            worst_condition_data = {}
        
        # Analyze timing of losing trades
        losing_trades = trades_df[trades_df['profit'] < 0]
        if not losing_trades.empty and 'timestamp' in losing_trades.columns:
            # Convert timestamps if they're strings
            if isinstance(losing_trades['timestamp'].iloc[0], str):
                losing_trades['timestamp'] = pd.to_datetime(losing_trades['timestamp'])
            
            # Group by hour of day
            losing_trades['hour'] = losing_trades['timestamp'].dt.hour
            losing_by_hour = losing_trades.groupby('hour').agg({
                'profit': ['sum', 'count']
            })
            
            worst_hour = losing_by_hour['profit']['sum'].idxmin()
            time_weakness = {
                'worst_hour': worst_hour,
                'loss_at_worst_hour': losing_by_hour['profit']['sum'][worst_hour],
                'trades_at_worst_hour': losing_by_hour['profit']['count'][worst_hour]
            }
        else:
            time_weakness = {}
        
        # Analyze consecutive losses (drawdowns)
        trades_df['win'] = trades_df['profit'] > 0
        streaks = (trades_df['win'] != trades_df['win'].shift()).cumsum()
        losing_streaks = trades_df[~trades_df['win']].groupby(streaks).size()
        
        max_consecutive_losses = losing_streaks.max() if not losing_streaks.empty else 0
        
        # Compile weakness analysis
        weakness_analysis = {
            'model_id': model_id,
            'timestamp': datetime.datetime.now(),
            'worst_market_condition': worst_condition_data,
            'time_weakness': time_weakness,
            'max_consecutive_losses': max_consecutive_losses,
            'total_losing_trades': len(losing_trades),
            'avg_loss_per_losing_trade': losing_trades['profit'].mean() if not losing_trades.empty else 0
        }
        
        return weakness_analysis
    
    def suggest_improvements(self, model_id, weakness_analysis, model_params=None):
        """
        Suggest improvements based on weakness analysis.
        
        Args:
            model_id (str): Model ID
            weakness_analysis (dict): Weakness analysis
            model_params (dict, optional): Current model parameters
            
        Returns:
            dict: Suggested improvements
        """
        suggestions = {
            'model_id': model_id,
            'timestamp': datetime.datetime.now(),
            'parameter_adjustments': {},
            'strategy_changes': [],
            'data_improvements': []
        }
        
        # Suggest parameter adjustments based on weaknesses
        if model_params:
            # Adjust risk parameters based on consecutive losses
            max_losses = weakness_analysis.get('max_consecutive_losses', 0)
            if max_losses > 5:
                suggestions['parameter_adjustments']['risk_level'] = 'lower'
                suggestions['parameter_adjustments']['position_size'] = 'reduce'
                suggestions['strategy_changes'].append(
                    "Implement more aggressive stop-loss due to frequent consecutive losses"
                )
        
        # Suggest data improvements
        worst_condition = weakness_analysis.get('worst_market_condition', {}).get('condition')
        if worst_condition:
            suggestions['data_improvements'].append(
                f"Add more training data for '{worst_condition}' market conditions"
            )
            
            if worst_condition == 'volatile':
                suggestions['strategy_changes'].append(
                    "Consider adding volatility indicators like ATR or Bollinger Band Width"
                )
            elif worst_condition == 'trending':
                suggestions['strategy_changes'].append(
                    "Consider adding trend strength indicators like ADX"
                )
            elif worst_condition == 'sideways':
                suggestions['strategy_changes'].append(
                    "Consider adding range-bound indicators like RSI or Stochastic oscillators"
                )
        
        # Suggest timing adjustments
        time_weakness = weakness_analysis.get('time_weakness', {})
        if time_weakness and 'worst_hour' in time_weakness:
            suggestions['strategy_changes'].append(
                f"Consider avoiding trades around hour {time_weakness['worst_hour']} or adjusting strategy for this time period"
            )
        
        # Log suggestions
        logger.info(f"Generated improvement suggestions for model {model_id}")
        
        # Add to improvement history
        self.improvement_history.append({
            'model_id': model_id,
            'timestamp': datetime.datetime.now(),
            'weakness_analysis': weakness_analysis,
            'suggestions': suggestions
        })
        
        return suggestions
    
    def implement_suggestions(self, suggestions, model_params):
        """
        Implement suggested improvements automatically.
        
        Args:
            suggestions (dict): Improvement suggestions
            model_params (dict): Current model parameters
            
        Returns:
            dict: Updated model parameters
        """
        updated_params = model_params.copy()
        
        # Implement parameter adjustments
        param_adjustments = suggestions.get('parameter_adjustments', {})
        
        # Risk level adjustment
        if 'risk_level' in param_adjustments:
            if param_adjustments['risk_level'] == 'lower':
                # Reduce risk by adjusting relevant parameters
                if 'stop_loss_pct' in updated_params:
                    updated_params['stop_loss_pct'] *= 0.8  # Tighter stop loss
                
                if 'take_profit_pct' in updated_params:
                    updated_params['take_profit_pct'] *= 1.2  # Higher take profit
        
        # Position size adjustment
        if 'position_size' in param_adjustments:
            if param_adjustments['position_size'] == 'reduce':
                if 'position_size_pct' in updated_params:
                    updated_params['position_size_pct'] *= 0.8  # Smaller position size
        
        # Implement feature engineering changes
        strategy_changes = suggestions.get('strategy_changes', [])
        
        for change in strategy_changes:
            if "volatility indicators" in change.lower():
                updated_params['use_atr'] = True
                updated_params['use_bbands'] = True
            
            elif "trend strength indicators" in change.lower():
                updated_params['use_adx'] = True
                updated_params['use_macd'] = True
            
            elif "range-bound indicators" in change.lower():
                updated_params['use_rsi'] = True
                updated_params['use_stoch'] = True
        
        logger.info(f"Implemented suggested improvements: {updated_params}")
        return updated_params


# Example usage
if __name__ == "__main__":
    # Create tracker
    tracker = ModelPerformanceTracker(db_path='performance_db')
    
    # Sample metrics
    sample_metrics = {
        "accuracy": 0.65,
        "precision": 0.70,
        "recall": 0.62,
        "f1_score": 0.66,
        "profit_percentage": 8.5,
        "max_drawdown": -12.3,
        "sharpe_ratio": 1.2
    }
    
    # Record performance
    tracker.record_model_performance(
        model_id="lstm_model_001",
        model_type="lstm",
        trading_pair="BTC-USD",
        timeframe="1h",
        metrics=sample_metrics
    )
    
    # Create optimizer
    optimizer = TradingStrategyOptimizer(tracker)
    
    # Sample trades for analysis
    sample_trades = pd.DataFrame({
        "timestamp": pd.date_range(start="2023-01-01", periods=10, freq="H"),
        "action": ["buy", "sell"] * 5,
        "price": [40000, 41000, 39000, 42000, 43000, 40000, 41000, 39000, 38000, 40000],
        "amount": [0.1] * 10,
        "profit": [100, -200, 300, 150, -100, 200, -150, -300, 400, 100],
        "market_condition": ["trending", "volatile", "sideways", "trending", "volatile", 
                           "sideways", "trending", "volatile", "sideways", "trending"]
    })
    
    # Analyze weaknesses
    weakness_analysis = optimizer.analyze_model_weaknesses("lstm_model_001", sample_trades)
    
    # Get suggestions
    current_params = {
        "stop_loss_pct": 0.05,
        "take_profit_pct": 0.1,
        "position_size_pct": 0.2,
        "use_rsi": False,
        "use_macd": True,
        "use_bbands": False,
        "use_atr": False
    }
    
    suggestions = optimizer.suggest_improvements("lstm_model_001", weakness_analysis, current_params)
    
    # Implement suggestions
    updated_params = optimizer.implement_suggestions(suggestions, current_params)
    
    print("Original parameters:", current_params)
    print("\nSuggested improvements:", suggestions)
    print("\nUpdated parameters:", updated_params)
