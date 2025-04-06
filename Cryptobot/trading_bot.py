#!/usr/bin/env python3
"""
Advanced Cryptocurrency Trading Bot

This script integrates all the enhanced components (data fetching, preprocessing,
model training, and backtesting) into a complete trading system with deep learning
capabilities.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from crypto_data_processing import (
    fetch_crypto_data, calculate_advanced_indicators, 
    preprocess_crypto_data, prepare_trading_data
)
from deep_learning_models import DeepLearningTrader
from model_evaluation import (
    evaluate_classification_model, evaluate_deep_learning_model,
    evaluate_trading_performance, compare_models, backtest_ensemble
)

# Import scikit-learn models for comparison
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup directory for models and visualizations
MODEL_DIR = os.path.join(os.getcwd(), 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    """Main function to run the trading bot"""
    print("========== Cryptocurrency Trading Bot ==========")
    print("Starting data collection and model training pipeline...")
    
    # Step 1: Fetch and prepare data from different sources
    print("\n1. Fetching and preprocessing data...")
    
    # Try different data sources (CoinGecko is most reliable for free access)
    data_sources = [
        ('coingecko', 'bitcoin'),
        ('yahoofinance', 'BTC-USD'),
        # Uncomment to use Binance (requires API setup)
        # ('binance', 'BTCUSDT')
    ]
    
    # Dictionary to store processed data from each source
    processed_data = {}
    
    for source, symbol in data_sources:
        print(f"\nFetching data from {source} for {symbol}...")
        data, features = prepare_trading_data(
            source=source,
            symbol=symbol,
            lookback_days=730,
            feature_set='advanced'
        )
        
        if data is not None and not data.empty:
            print(f"Successfully processed data from {source}")
            processed_data[source] = (data, features)
        else:
            print(f"Failed to get usable data from {source}")
    
    # Use the first successful data source for model training
    if not processed_data:
        print("Error: Failed to fetch usable data from any source")
        return
    
    source = list(processed_data.keys())[0]
    data, features = processed_data[source]
    print(f"Using data from {source} for model training")
    
    # Step 2: Prepare data for machine learning
    print("\n2. Preparing data for machine learning...")
    
    # Split data for training and testing
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # Define features and target for traditional ML models
    X_train = train_data[features]
    y_train = train_data['target']
    X_test = test_data[features]
    y_test = test_data['target']
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Step 3: Train traditional machine learning models
    print("\n3. Training traditional machine learning models...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
    }
    
    # Train and evaluate each model
    model_results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        print(f"Evaluating {name}...")
        metrics = evaluate_classification_model(
            model, X_test, y_test, model_name=name
        )
        
        if metrics:
            model_results[name] = metrics
    
    # Step 4: Train deep learning models
    print("\n4. Training deep learning models...")
    
    # Sequence length for time series models
    sequence_length = 60
    
    # Prepare sequence data for deep learning models
    dl_trainer = DeepLearningTrader(
        model_type='lstm',
        sequence_length=sequence_length,
        batch_size=32,
        epochs=50,
        save_dir=MODEL_DIR
    )
    
    X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq = dl_trainer.prepare_data(
        data, features, target_col='target', test_size=0.2, val_size=0.2
    )
    
    # Train LSTM model
    print("\nTraining LSTM model...")
    lstm_model = dl_trainer.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    
    # Save the model
    dl_trainer.save_model(model_name="lstm_crypto_model")
    
    # Plot training history
    dl_trainer.plot_training_history()
    
    # Evaluate LSTM model
    print("\nEvaluating LSTM model...")
    lstm_metrics = dl_trainer.evaluate(X_test_seq, y_test_seq)
    
    # Evaluate with our custom deep learning evaluator
    lstm_dl_metrics = evaluate_deep_learning_model(
        lstm_model, X_test_seq, y_test_seq,
        sequence_length=sequence_length,
        features=features,
        actual_prices=test_data['close'],
        model_name="LSTM Model",
        save_dir=MODEL_DIR
    )
    
    # Step 5: Perform backtesting
    print("\n5. Backtesting trading strategies...")
    
    # Backtest deep learning model
    backtest_results, backtest_metrics = dl_trainer.backtest(
        test_data, features, initial_cash=10000, commission=0.001, plot=True
    )
    
    # Step 6: Compare all models
    print("\n6. Comparing all models...")
    
    # Compare traditional ML models
    ml_models = [models[name] for name in models.keys()]
    ml_model_names = list(models.keys())
    
    compare_models(
        ml_models, X_test, y_test,
        actual_prices=test_data['close'],
        model_names=ml_model_names
    )
    
    # Summary of results
    print("\n========== Trading Bot Results Summary ==========")
    print(f"Data Source: {source} ({len(data)} data points)")
    print(f"Features used: {len(features)} technical indicators")
    print("\nTraditional Machine Learning Model Performance:")
    
    for name, result in model_results.items():
        print(f"  - {name}: Accuracy = {result['accuracy']:.4f}, F1 = {result['f1']:.4f}")
    
    print("\nDeep Learning Model Performance:")
    print(f"  - LSTM: Accuracy = {lstm_metrics['accuracy']:.4f}")
    
    if 'trading_metrics' in lstm_dl_metrics and lstm_dl_metrics['trading_metrics']:
        tm = lstm_dl_metrics['trading_metrics']
        print(f"\nLSTM Trading Performance:")
        print(f"  - Annual Return: {tm['strategy_annual_return']:.2%}")
        print(f"  - Sharpe Ratio: {tm['strategy_sharpe']:.2f}")
        print(f"  - Max Drawdown: {tm['max_strategy_drawdown']:.2%}")
        print(f"  - Win Rate: {tm['win_rate']:.2%}")
    
    print("\nTrading bot setup completed successfully!")
    print("==============================================")
    
    return {
        'data': data,
        'features': features,
        'models': models,
        'dl_model': lstm_model,
        'backtest_results': backtest_results
    }

if __name__ == "__main__":
    main()
