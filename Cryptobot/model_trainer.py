"""
Advanced Model Training System for Cryptocurrency Trading

This module integrates advanced deep learning models with performance tracking
to create a self-improving trading system.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime
import joblib
import json
import logging

# Import our custom modules
from advanced_dl_models import (
    build_transformer_model,
    build_inception_time_model, 
    build_temporal_fusion_transformer,
    build_ensemble_model
)
from performance_tracker import ModelPerformanceTracker, TradingStrategyOptimizer
from deep_learning_models import DeepLearningTrader, prepare_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_trainer")


class AdvancedModelTrainer:
    """
    Trains, evaluates, and optimizes advanced deep learning models for trading.
    Implements a self-improvement loop based on performance metrics.
    """
    
    def __init__(self, base_dir='training_data', 
                performance_db_path='performance_db',
                model_save_dir='advanced_models'):
        """
        Initialize the advanced model trainer.
        
        Args:
            base_dir (str): Directory for training data
            performance_db_path (str): Path for performance database
            model_save_dir (str): Directory to save trained models
        """
        self.base_dir = base_dir
        self.model_save_dir = model_save_dir
        
        # Create directories if they don't exist
        for dir_path in [base_dir, model_save_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        
        # Initialize performance tracker and optimizer
        self.performance_tracker = ModelPerformanceTracker(db_path=performance_db_path)
        self.strategy_optimizer = TradingStrategyOptimizer(self.performance_tracker)
        
        # Track training iterations for self-improvement
        self.current_iteration = 0
        self.improvement_history = []
        
        # Default model parameters
        self.default_model_params = {
            'sequence_length': 60,
            'batch_size': 32,
            'epochs': 50,
            'patience': 10,
            'learning_rate': 0.001,
            'dropout_rate': 0.2,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.1,
            'position_size_pct': 0.2,
            'use_rsi': True,
            'use_macd': True,
            'use_bbands': True,
            'use_atr': True
        }
        
        # Current parameters (will be updated during self-improvement)
        self.current_params = self.default_model_params.copy()
        logger.info("Advanced Model Trainer initialized")
    
    def prepare_training_data(self, data, features, target_col='target', 
                             test_size=0.2, val_size=0.2):
        """
        Prepare data for model training with proper sequence generation.
        
        Args:
            data (pd.DataFrame): Preprocessed data with features and target
            features (list): List of feature columns to use
            target_col (str): Name of the target column
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scalers)
        """
        logger.info(f"Preparing data with {len(features)} features")
        
        # Store feature list for later use
        self.features = features
        
        # Ensure data is sorted by time
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Split data into training and testing sets
        train_data, test_data = train_test_split(
            data, test_size=test_size, shuffle=False
        )
        
        # Further split training data into train and validation
        train_data, val_data = train_test_split(
            train_data, test_size=val_size, shuffle=False
        )
        
        # Scale features
        feature_scaler = MinMaxScaler()
        
        # Fit scaler on training data only
        train_data[features] = feature_scaler.fit_transform(train_data[features])
        val_data[features] = feature_scaler.transform(val_data[features])
        test_data[features] = feature_scaler.transform(test_data[features])
        
        # Get sequence length from parameters
        seq_length = self.current_params['sequence_length']
        
        # Generate sequences
        X_train, y_train = prepare_sequences(train_data, features, seq_length, target_col)
        X_val, y_val = prepare_sequences(val_data, features, seq_length, target_col)
        X_test, y_test = prepare_sequences(test_data, features, seq_length, target_col)
        
        logger.info(f"Data preparation complete: X_train shape: {X_train.shape}")
        
        # Store scalers for later use
        scalers = {
            'feature_scaler': feature_scaler
        }
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scalers
    
    def train_models(self, X_train, y_train, X_val, y_val, model_types=None):
        """
        Train multiple advanced deep learning models.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
            model_types (list, optional): List of model types to train
            
        Returns:
            dict: Trained models
        """
        if model_types is None:
            model_types = ['transformer', 'inception', 'tft']
        
        logger.info(f"Training {len(model_types)} model types: {model_types}")
        
        input_shape = X_train.shape[1:]
        models = {}
        histories = {}
        
        # Set up callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.current_params['patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
        ]
        
        # Train each model type
        for model_type in model_types:
            logger.info(f"Training {model_type} model")
            
            # Build the appropriate model
            if model_type == 'transformer':
                model = build_transformer_model(
                    input_shape=input_shape,
                    dropout_rate=self.current_params['dropout_rate']
                )
            elif model_type == 'inception':
                model = build_inception_time_model(
                    input_shape=input_shape,
                    dropout_rate=self.current_params['dropout_rate']
                )
            elif model_type == 'tft':
                model = build_temporal_fusion_transformer(
                    input_shape=input_shape,
                    dropout_rate=self.current_params['dropout_rate']
                )
            else:
                logger.warning(f"Unknown model type: {model_type}")
                continue
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.current_params['epochs'],
                batch_size=self.current_params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Store model and history
            models[model_type] = model
            histories[model_type] = history
        
        # After training individual models, build an ensemble
        if len(models) > 1:
            logger.info("Training ensemble model")
            
            # Get predictions from each model
            base_preds = []
            base_models = []
            
            for model_type, model in models.items():
                base_models.append(model)
            
            # Build and train ensemble
            ensemble_model = build_ensemble_model(
                models=base_models,
                input_shape=input_shape
            )
            
            # Fine-tune ensemble (with a few epochs)
            ensemble_history = ensemble_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=10,  # Fewer epochs for fine-tuning
                batch_size=self.current_params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Add ensemble to models
            models['ensemble'] = ensemble_model
            histories['ensemble'] = ensemble_history
        
        logger.info(f"Model training complete: {list(models.keys())}")
        return models, histories
    
    def evaluate_models(self, models, X_test, y_test, data_test=None):
        """
        Evaluate trained models on test data.
        
        Args:
            models (dict): Dictionary of trained models
            X_test (np.array): Test features
            y_test (np.array): Test targets
            data_test (pd.DataFrame, optional): Original test data
            
        Returns:
            dict: Evaluation results
        """
        results = {}
        best_model = None
        best_performance = -float('inf')
        
        logger.info(f"Evaluating {len(models)} models")
        
        for model_type, model in models.items():
            # Get predictions
            y_pred_prob = model.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate basic metrics
            accuracy = np.mean(y_pred == y_test)
            precision = np.sum((y_pred == 1) & (y_test == 1)) / (np.sum(y_pred == 1) + 1e-10)
            recall = np.sum((y_pred == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            
            # Create trades from predictions for backtesting
            trades = None
            trading_metrics = {}
            
            if data_test is not None:
                # Generate trades based on predictions
                try:
                    # Use the last part of test data (aligned with predictions)
                    backtest_data = data_test.iloc[-len(y_pred):]
                    
                    # Simple trading simulation
                    initial_cash = 10000
                    positions = 0
                    cash = initial_cash
                    trades_list = []
                    
                    for i in range(len(y_pred) - 1):
                        signal = y_pred[i][0] if len(y_pred[i].shape) > 0 else y_pred[i]
                        price = backtest_data.iloc[i]['close']
                        next_price = backtest_data.iloc[i+1]['close']
                        timestamp = backtest_data.iloc[i]['timestamp']
                        
                        # Determine market condition based on price changes
                        pct_change = (price / backtest_data.iloc[max(0, i-5):i+1]['close'].mean() - 1) * 100
                        if abs(pct_change) < 1:
                            market_condition = 'sideways'
                        elif pct_change > 0:
                            market_condition = 'trending'
                        else:
                            market_condition = 'volatile'
                        
                        if signal == 1 and positions == 0:  # Buy signal
                            position_size = cash * self.current_params['position_size_pct']
                            positions = position_size / price
                            cash -= position_size
                            
                            trades_list.append({
                                'timestamp': timestamp,
                                'action': 'buy',
                                'price': price,
                                'amount': positions,
                                'profit': 0,
                                'cash': cash,
                                'market_condition': market_condition
                            })
                        
                        elif signal == 0 and positions > 0:  # Sell signal
                            profit = positions * (next_price - price)
                            cash += positions * next_price
                            
                            trades_list.append({
                                'timestamp': timestamp,
                                'action': 'sell',
                                'price': next_price,
                                'amount': positions,
                                'profit': profit,
                                'cash': cash,
                                'market_condition': market_condition
                            })
                            
                            positions = 0
                    
                    # Create trades DataFrame
                    trades = pd.DataFrame(trades_list)
                    
                    # Calculate trading performance metrics
                    if not trades.empty:
                        trading_metrics = self.performance_tracker.calculate_trading_metrics(
                            trades, initial_balance=initial_cash
                        )
                
                except Exception as e:
                    logger.error(f"Error in trade simulation: {e}")
            
            # Combine all metrics
            model_results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                **trading_metrics
            }
            
            results[model_type] = model_results
            
            # Track best model
            if 'profit_percentage' in trading_metrics:
                performance = trading_metrics['profit_percentage']
            else:
                performance = f1  # Use F1 if trading metrics not available
            
            if performance > best_performance:
                best_model = model_type
                best_performance = performance
            
            # Record performance in tracker
            model_id = f"{model_type}_iter{self.current_iteration}"
            self.performance_tracker.record_model_performance(
                model_id=model_id,
                model_type=model_type,
                trading_pair="BTC-USD",  # Default
                timeframe="1h",  # Default
                metrics=model_results,
                trades=trades,
                parameters=self.current_params
            )
            
            logger.info(f"Model {model_type} evaluation: {model_results}")
        
        return results, best_model
    
    def self_improve(self, best_model_type, models, evaluation_results, trades=None):
        """
        Implement self-improvement based on model performance.
        
        Args:
            best_model_type (str): Type of the best performing model
            models (dict): Trained models
            evaluation_results (dict): Evaluation metrics
            trades (pd.DataFrame): Trading results
            
        Returns:
            dict: Updated parameters
        """
        logger.info(f"Starting self-improvement cycle for iteration {self.current_iteration}")
        
        # Analyze model weaknesses
        model_id = f"{best_model_type}_iter{self.current_iteration}"
        
        if trades is not None and not trades.empty:
            weakness_analysis = self.strategy_optimizer.analyze_model_weaknesses(
                model_id, trades
            )
            
            # Get improvement suggestions
            suggestions = self.strategy_optimizer.suggest_improvements(
                model_id, weakness_analysis, self.current_params
            )
            
            # Implement suggestions
            updated_params = self.strategy_optimizer.implement_suggestions(
                suggestions, self.current_params
            )
            
            # Track improvement
            self.improvement_history.append({
                'iteration': self.current_iteration,
                'model_id': model_id,
                'evaluation': evaluation_results.get(best_model_type, {}),
                'weakness_analysis': weakness_analysis,
                'suggestions': suggestions,
                'old_params': self.current_params.copy(),
                'new_params': updated_params
            })
            
            # Update current parameters
            self.current_params = updated_params
            
            logger.info(f"Self-improvement complete. Updated parameters: {updated_params}")
        else:
            logger.warning("No trade data available for self-improvement")
        
        # Increment iteration counter
        self.current_iteration += 1
        
        return self.current_params
    
    def save_models(self, models, evaluation_results, best_model_type):
        """
        Save trained models and evaluation results.
        
        Args:
            models (dict): Trained models
            evaluation_results (dict): Evaluation metrics
            best_model_type (str): Type of the best performing model
            
        Returns:
            dict: Paths to saved models
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_paths = {}
        
        for model_type, model in models.items():
            # Create model directory
            model_dir = os.path.join(
                self.model_save_dir, 
                f"{model_type}_iter{self.current_iteration}_{timestamp}"
            )
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, "model.h5")
            model.save(model_path)
            
            # Save evaluation results
            eval_path = os.path.join(model_dir, "evaluation.json")
            with open(eval_path, 'w') as f:
                json.dump(
                    evaluation_results.get(model_type, {}), 
                    f, 
                    indent=2, 
                    default=str
                )
            
            # Save parameters
            params_path = os.path.join(model_dir, "parameters.json")
            with open(params_path, 'w') as f:
                json.dump(self.current_params, f, indent=2)
            
            save_paths[model_type] = {
                'model': model_path,
                'evaluation': eval_path,
                'parameters': params_path
            }
            
            logger.info(f"Saved {model_type} model to {model_path}")
        
        # Mark best model
        best_model_path = os.path.join(self.model_save_dir, "best_model.json")
        with open(best_model_path, 'w') as f:
            json.dump({
                'best_model_type': best_model_type,
                'iteration': self.current_iteration - 1,  # Because we already incremented in self_improve
                'timestamp': timestamp,
                'path': save_paths.get(best_model_type, {}).get('model', None),
                'evaluation': evaluation_results.get(best_model_type, {})
            }, f, indent=2, default=str)
        
        logger.info(f"Marked {best_model_type} as best model")
        return save_paths
    
    def run_training_cycle(self, data, features, target_col='target'):
        """
        Run a complete training cycle with self-improvement.
        
        Args:
            data (pd.DataFrame): Preprocessed data with features and target
            features (list): List of feature columns to use
            target_col (str): Name of the target column
            
        Returns:
            tuple: (best_model, evaluation_results, updated_parameters)
        """
        logger.info(f"Starting training cycle {self.current_iteration}")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, scalers = self.prepare_training_data(
            data, features, target_col
        )
        
        # Train models
        models, histories = self.train_models(X_train, y_train, X_val, y_val)
        
        # Evaluate models
        evaluation_results, best_model_type = self.evaluate_models(
            models, X_test, y_test, data_test=data
        )
        
        # Self-improve based on results
        # Assuming we can get the trades for the best model from the evaluation process
        best_model_id = f"{best_model_type}_iter{self.current_iteration}"
        trades = None  # In a real implementation, you'd retrieve this from evaluation_results
        
        updated_params = self.self_improve(
            best_model_type, models, evaluation_results, trades
        )
        
        # Save models
        save_paths = self.save_models(models, evaluation_results, best_model_type)
        
        logger.info(f"Training cycle {self.current_iteration-1} complete")
        return models.get(best_model_type), evaluation_results, updated_params


# Example usage
if __name__ == "__main__":
    from fetchall import fe_preprocess
    
    # Get data
    try:
        # Use your actual data fetching function
        data = fe_preprocess(exch="binance")
        
        if data is None or data.empty:
            raise ValueError("No data returned from fe_preprocess")
    except Exception as e:
        # Create sample data for demonstration
        print(f"Error fetching real data: {e}")
        print("Creating sample data for demonstration...")
        
        # Generate sample data
        import numpy as np
        from datetime import datetime, timedelta
        
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
        
        base_price = 50000 + np.cumsum(np.random.normal(0, 100, 1000))
        data = pd.DataFrame({
            'timestamp': dates,
            'open': base_price,
            'high': base_price + np.random.normal(0, 200, 1000),
            'low': base_price - np.random.normal(0, 200, 1000),
            'close': base_price + np.random.normal(0, 100, 1000),
            'volume': np.abs(np.random.normal(1000000, 500000, 1000))
        })
        
        # Ensure high is highest and low is lowest
        for i in range(len(data)):
            values = [data.loc[i, 'open'], data.loc[i, 'close']]
            data.loc[i, 'high'] = max(data.loc[i, 'high'], max(values))
            data.loc[i, 'low'] = min(data.loc[i, 'low'], min(values))
        
        # Add technical indicators
        data['sma_10'] = data['close'].rolling(window=10).mean()
        data['sma_30'] = data['close'].rolling(window=30).mean()
        data['rsi'] = 50 + np.random.normal(0, 10, 1000)  # Placeholder
        data['macd'] = data['sma_10'] - data['sma_30']  # Simplified MACD
        data['macd_signal'] = data['macd'].rolling(window=9).mean()
        
        # Add target (simplified for demonstration)
        data['return'] = data['close'].pct_change()
        data['target'] = (data['return'].shift(-1) > 0).astype(int)
        
        # Drop NaN values
        data = data.dropna().reset_index(drop=True)
    
    # Define features
    features = [
        'open', 'high', 'low', 'close', 'volume',
        'sma_10', 'sma_30', 'rsi', 'macd', 'macd_signal'
    ]
    
    # Create trainer
    trainer = AdvancedModelTrainer()
    
    # Run a training cycle
    best_model, results, updated_params = trainer.run_training_cycle(
        data, features, 'target'
    )
    
    print("Training complete!")
    print(f"Best model evaluation: {results}")
    print(f"Updated parameters: {updated_params}")
    
    # Export results to performance summary
    trainer.performance_tracker.save_performance_summary()
