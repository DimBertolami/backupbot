"""
Advanced Cryptocurrency Data Processing Module

This module provides enhanced data fetching, preprocessing, and feature engineering
for cryptocurrency trading data from multiple sources.
"""

import os
import numpy as np
import pandas as pd
import requests
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime, timedelta
import yfinance as yf
import json
import traceback

# Fetch historical data from CoinGecko through their free API
def fetch_coingecko_data(coin_id='bitcoin', vs_currency='usd', days=365):
    """
    Fetch historical data from CoinGecko API - free and developer-friendly.
    
    Args:
        coin_id (str): ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')
        vs_currency (str): Currency to compare against (e.g., 'usd', 'eur')
        days (int): Number of days of data to retrieve
        
    Returns:
        pd.DataFrame: DataFrame containing price and market data with timestamp index
    """
    try:
        # CoinGecko API endpoint for historical market data
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': vs_currency,
            'days': days,
            'interval': 'daily'
        }
        
        print(f"Fetching {days} days of {coin_id} data from CoinGecko...")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"Error: CoinGecko API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
        data = response.json()
        
        # Process the data into a pandas DataFrame
        prices = data.get('prices', [])
        market_caps = data.get('market_caps', [])
        total_volumes = data.get('total_volumes', [])
        
        if not prices:
            print("No price data returned from CoinGecko")
            return None
            
        # Create DataFrame with all data points
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Add market cap and volume if available
        if market_caps and len(market_caps) == len(prices):
            df['market_cap'] = [item[1] for item in market_caps]
        
        if total_volumes and len(total_volumes) == len(prices):
            df['volume'] = [item[1] for item in total_volumes]
        
        # Create OHLC-like data (CoinGecko only provides closing prices)
        df['open'] = df['close'].shift(1)
        df['high'] = df['close'] * 1.01  # Approximate
        df['low'] = df['close'] * 0.99   # Approximate
        
        # Fill first row's open with its close
        if not df.empty:
            df.loc[0, 'open'] = df.loc[0, 'close']
        
        print(f"Successfully retrieved {len(df)} days of {coin_id} data")
        return df
        
    except Exception as e:
        print(f"Error fetching CoinGecko data: {e}")
        traceback.print_exc()
        return None

# Fetch historical data from Alpha Vantage (they offer free crypto data as well)
def fetch_alphavantage_data(symbol='BTC', market='USD', interval='daily', outputsize='full'):
    """
    Fetch historical cryptocurrency data from Alpha Vantage API.
    
    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        market (str): Market to compare against (e.g., 'USD', 'EUR')
        interval (str): Time interval ('daily', 'weekly', 'monthly')
        outputsize (str): 'compact' (last 100 data points) or 'full' (all available data)
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with timestamp index
    """
    try:
        # You can get a free API key from https://www.alphavantage.co/support/#api-key
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        base_url = 'https://www.alphavantage.co/query'
        
        params = {
            'function': 'DIGITAL_CURRENCY_DAILY',
            'symbol': symbol,
            'market': market,
            'apikey': api_key
        }
        
        if interval == 'weekly':
            params['function'] = 'DIGITAL_CURRENCY_WEEKLY'
        elif interval == 'monthly':
            params['function'] = 'DIGITAL_CURRENCY_MONTHLY'
        
        print(f"Fetching {symbol}/{market} {interval} data from Alpha Vantage...")
        response = requests.get(base_url, params=params)
        
        if response.status_code != 200:
            print(f"Error: Alpha Vantage API returned status code {response.status_code}")
            return None
            
        data = response.json()
        
        # Check for error messages
        if 'Error Message' in data:
            print(f"Alpha Vantage API Error: {data['Error Message']}")
            return None
            
        # Extract time series data
        if 'Time Series (Digital Currency Daily)' in data:
            time_series = data['Time Series (Digital Currency Daily)']
        elif 'Time Series (Digital Currency Weekly)' in data:
            time_series = data['Time Series (Digital Currency Weekly)']
        elif 'Time Series (Digital Currency Monthly)' in data:
            time_series = data['Time Series (Digital Currency Monthly)']
        else:
            print("No time series data found in Alpha Vantage response")
            return None
        
        # Convert to DataFrame
        df_list = []
        for date, values in time_series.items():
            row = {
                'timestamp': date,
                'open': float(values[f'1a. open ({market})']),
                'high': float(values[f'2a. high ({market})']),
                'low': float(values[f'3a. low ({market})']),
                'close': float(values[f'4a. close ({market})']),
                'volume': float(values[f'5. volume']),
                'market_cap': float(values[f'6. market cap ({market})']) if f'6. market cap ({market})' in values else None
            }
            df_list.append(row)
        
        if not df_list:
            print("No data points extracted from Alpha Vantage response")
            return None
            
        df = pd.DataFrame(df_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Successfully retrieved {len(df)} data points from Alpha Vantage")
        return df
        
    except Exception as e:
        print(f"Error fetching Alpha Vantage data: {e}")
        traceback.print_exc()
        return None

# Function to get historical cryptocurrency data from various APIs
def fetch_crypto_data(source='coingecko', **kwargs):
    """
    Unified function to fetch cryptocurrency data from various sources.
    
    Args:
        source (str): Data source ('coingecko', 'alphavantage', 'yahoofinance', 'binance')
        **kwargs: Additional parameters specific to each data source
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data
    """
    if source == 'coingecko':
        return fetch_coingecko_data(**kwargs)
    elif source == 'alphavantage':
        return fetch_alphavantage_data(**kwargs)
    elif source == 'yahoofinance':
        # Default values for Yahoo Finance
        symbol = kwargs.get('symbol', 'BTC-USD')
        interval = kwargs.get('interval', '1d')
        period = kwargs.get('period', '2y')
        
        # Call the external function that should be imported
        try:
            from fetchall import fetch_yfinance_data
            return fetch_yfinance_data(symbol=symbol, interval=interval, period=period)
        except ImportError:
            print("fetch_yfinance_data function not found, using internal implementation")
            try:
                data = yf.download(tickers=symbol, interval=interval, period=period)
                if data.empty:
                    return None
                data = data.reset_index()
                data.columns = [col.lower() for col in data.columns]
                data = data.rename(columns={'date': 'timestamp'})
                return data
            except Exception as e:
                print(f"Error fetching Yahoo Finance data: {e}")
                return None
    elif source == 'binance':
        # For Binance, rely on the external function
        try:
            from fetchall import fetch_binance_data
            return fetch_binance_data(**kwargs)
        except ImportError:
            print("fetch_binance_data function not found")
            return None
    else:
        print(f"Unknown data source: {source}")
        return None

# Advanced indicator calculation with error handling
def calculate_advanced_indicators(df):
    """
    Calculate a comprehensive set of technical indicators for cryptocurrency data.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    try:
        if df is None or df.empty:
            print("No data provided for indicator calculation")
            return None
            
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Ensure required columns exist
        required_cols = ['close']
        for col in required_cols:
            if col not in data.columns:
                print(f"Missing required column: {col}")
                return None
        
        # Ensure numeric type for calculations
        for col in ['close', 'open', 'high', 'low']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Basic moving averages
        data['SMA14'] = data['close'].rolling(window=14).mean()  # Simple Moving Average
        data['EMA14'] = data['close'].ewm(span=14, adjust=False).mean()  # Exponential Moving Average
        data['EMA'] = data['close'].ewm(span=14).mean()  # Exponential Moving Average (alternate)
        
        # Basic momentum indicators
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)  # Relative Strength Index
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )  # MACD
        
        # Volatility indicators
        if 'high' in data.columns and 'low' in data.columns:
            data['UpperBand'], data['MiddleBand'], data['LowerBand'] = talib.BBANDS(
                data['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )  # Bollinger Bands
            
            data['ATR'] = talib.ATR(
                data['high'], data['low'], data['close'], timeperiod=14
            )  # Average True Range
        
        # Volume-based indicators
        if 'volume' in data.columns:
            data['OBV'] = talib.OBV(data['close'], data['volume'])  # On-Balance Volume
            
            # Calculate Chaikin Money Flow
            try:
                mfm = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
                mfm = mfm.replace([np.inf, -np.inf], np.nan)
                mfv = mfm * data['volume']
                data['CMF'] = mfv.rolling(20).sum() / data['volume'].rolling(20).sum()
            except:
                print("Could not calculate Chaikin Money Flow")
        
        # Trend indicators
        if 'high' in data.columns and 'low' in data.columns:
            data['ADX'] = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)  # ADX
            
        # Additional indicators
        data['WILLR'] = talib.WILLR(data['high'], data['low'], data['close'], timeperiod=14)  # Williams %R
        
        # Add lookback close prices (can be useful for ML models)
        for i in [3, 7, 14, 30]:
            if len(data) > i:
                col_name = f'close_lag_{i}'
                data[col_name] = data['close'].shift(i)
        
        # Add price change features
        data['price_change_1d'] = data['close'].pct_change(1)
        data['price_change_7d'] = data['close'].pct_change(7)
        
        # Handle NaN values
        # Dropping NaN is usually done after selecting the features to use
        return data
        
    except Exception as e:
        print(f"Error calculating indicators: {e}")
        traceback.print_exc()
        return None

# Advanced preprocessing with improved NaN handling
def preprocess_crypto_data(data, features=None, scale_method='minmax'):
    """
    Preprocess cryptocurrency data for machine learning models.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLCV data and indicators
        features (list): Specific features to use (if None, will use common indicators)
        scale_method (str): Scaling method ('minmax' or 'standard')
        
    Returns:
        tuple: (DataFrame with preprocessed data, list of features used)
    """
    try:
        if data is None or data.empty:
            print("No data provided for preprocessing")
            return None, None
            
        # Calculate indicators if they don't exist
        if 'RSI' not in data.columns:
            data = calculate_advanced_indicators(data)
            
        if data is None:
            return None, None
        
        # Define default features if not provided
        if features is None:
            features = [
                'SMA14', 'EMA14', 'RSI', 'MACD', 'MACD_signal', 
                'UpperBand', 'MiddleBand', 'LowerBand'
            ]
            
            # Add advanced features if available
            for feature in ['ATR', 'OBV', 'ADX', 'WILLR', 'CMF']:
                if feature in data.columns:
                    features.append(feature)
        
        # Filter to available features
        available_features = [f for f in features if f in data.columns]
        if len(available_features) < len(features):
            print(f"Some requested features are not available. Using: {available_features}")
            features = available_features
            
        if not features:
            print("No valid features available for preprocessing")
            return None, None
        
        # Handle NaN values in indicators
        for feature in features:
            nan_count = data[feature].isna().sum()
            if nan_count > 0:
                print(f"Handling {nan_count} NaN values in {feature}")
                data[feature] = data[feature].fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        if scale_method == 'minmax':
            scaler = MinMaxScaler()
        else:  # standard scaling
            scaler = StandardScaler()
            
        # Apply scaling
        data[features] = scaler.fit_transform(data[features])
        
        # Create target variable (future price movement)
        data['target'] = data['close'].shift(-1) > data['close']
        data['target'] = data['target'].astype(int)
        data['target'] = data['target'].apply(lambda x: 1 if x == 1 else -1)  # 1 for Buy, -1 for Sell
        
        # Drop rows with NaN values after all processing
        initial_rows = len(data)
        data = data.dropna(subset=features + ['target']).reset_index(drop=True)
        print(f"Removed {initial_rows - len(data)} rows with NaN values")
        
        return data, features
        
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        traceback.print_exc()
        return None, None

# Complete pipeline for fetching, processing, and preparing data
def prepare_trading_data(source='coingecko', symbol='bitcoin', lookback_days=730, feature_set='all'):
    """
    Complete pipeline to fetch and prepare cryptocurrency data for trading models.
    
    Args:
        source (str): Data source ('coingecko', 'alphavantage', 'yahoofinance', 'binance')
        symbol (str): Symbol or coin ID to fetch
        lookback_days (int): Number of days of historical data to fetch
        feature_set (str): 'basic', 'advanced', or 'all' for feature selection
        
    Returns:
        tuple: (DataFrame with processed data, list of features used)
    """
    print(f"Preparing trading data from {source} for {symbol}...")
    
    # Configure source-specific parameters
    if source == 'coingecko':
        data = fetch_coingecko_data(coin_id=symbol, days=lookback_days)
    elif source == 'alphavantage':
        data = fetch_alphavantage_data(symbol=symbol)
    elif source == 'yahoofinance':
        data = fetch_crypto_data(source='yahoofinance', symbol=f"{symbol}-USD", period=f"{lookback_days}d")
    elif source == 'binance':
        data = fetch_crypto_data(source='binance', symbol=f"{symbol}USDT", lookback=f"{lookback_days} days ago UTC")
    else:
        print(f"Unknown source: {source}")
        return None, None
    
    if data is None:
        print(f"Failed to fetch data from {source}")
        return None, None
    
    # Calculate all indicators
    data = calculate_advanced_indicators(data)
    
    # Select features based on feature_set
    if feature_set == 'basic':
        features = ['SMA14', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']
    elif feature_set == 'advanced':
        features = [
            'SMA14', 'EMA14', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'UpperBand', 'MiddleBand', 'LowerBand', 'ATR', 'ADX', 'WILLR'
        ]
    else:  # 'all'
        features = None  # Use all available indicators
    
    # Preprocess the data
    processed_data, used_features = preprocess_crypto_data(data, features=features)
    
    return processed_data, used_features

# If this file is run directly, demonstrate the functionality
if __name__ == "__main__":
    # Example usage
    data, features = prepare_trading_data(source='coingecko', symbol='bitcoin', lookback_days=365)
    
    if data is not None:
        print(f"Successfully prepared data with {len(data)} rows and the following features:")
        print(features)
        print("\nDataset shape:", data.shape)
        print("\nSample data:")
        print(data.head())
    else:
        print("Failed to prepare data.")
