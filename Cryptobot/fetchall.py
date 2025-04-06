'''	Author: Dimi Bertolami										   date: 16-03-2025
        ----------------------										   ----------------
1.0)    This bot came to life after realising that I suck at guessing which cryptocurrency is going to make me profit
1.1) 	install required packages
1.2) 	fetch historical price data and technical indicators.
2)   	Feature Engineering: Create features based on historical data and technical indicators (e.g., RSI, moving averages).
3)   	preprocess the datait for machine learning (model training for example normalize, generate technical indicators).
4)   	Train machine learning mode  (LSTM, Decision Trees, or RL agent).
5)   	Evaluate the models on a validation dataset or new data using metrics such as accuracy, precision, recall (for classification models), or profitability (for RL).
6)   	Use the model's predictions to implement a Buy/Hold/Sell strategy.

Explanation of Dependencies:
	numpy → Array operations
	pandas → Data manipulation
	matplotlib → Static plotting
	seaborn → Enhanced visualizations
	plotly → Interactive charts
	scikit-learn → ML utilities (RandomForest, train_test_split)
	xgboost → XGBoost model
	tensorflow → Deep learning models (LSTM, CNN)
'''

import os
import random
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import seaborn as sns
from ta.utils import dropna
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import requests, talib, json
import plotly.graph_objects as go
import tensorflow as tf
from ta import add_all_ta_features
from ta.volatility import BollingerBands
from python_bitvavo_api.bitvavo import Bitvavo
from binance.client import Client as BinanceClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, TimeDistributed, RepeatVector, Bidirectional
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Try to import dotenv for secure environment variable loading
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed. Using os.environ directly.")
    # You can install it with: pip install python-dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TensorFlow oneDNN warning
warnings.filterwarnings('ignore')  # Suppress minor warnings

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device NOT found')
else:
    print('Found GPU at: {}'.format(device_name))

# API Keys - loaded securely from environment variables
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BITVAVO_API_KEY = os.getenv("BITVAVO_API_KEY")
BITVAVO_API_SECRET = os.getenv("BITVAVO_API_SECRET")
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

# Check if required API keys are present
if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    print("Warning: Binance API keys not found in environment variables. Set these in a .env file for production use.")
    
if not BITVAVO_API_KEY or not BITVAVO_API_SECRET:
    print("Warning: Bitvavo API keys not found in environment variables. Set these in a .env file for production use.")
    
if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    print("Warning: Alpaca API keys not found in environment variables. Set these in a .env file for production use.")

# Split Data
def split_data(df, features, target):
    X = df[features]
    y = df[target].replace({-1: 2})  # Map SELL (-1) to 2 to preserve the signal
                                     #    y = df[target].replace(-1, 0)  # Convert -1 to 0 for binary classification
    y = df[target]
    return train_test_split(X, y, test_size=0.2, shuffle=False)

'''
lr_model = train_model(LinearRegression, X_train, y_train)
LR_model = train_model(LogisticRegression, X_train, y_train)
KNC_model = train_model(KNeighborsClassifier, X_train, y_train)
DTC_model = train_model(DecisionTreeClassifier, X_train, y_train)
DTR_model = train_model(DecisionTreeRegressor, X_train, y_train)
RFR_model = train_model(RandomForestRegressor,  X_train, y_train)

'''
def train_model(model_class, X_train, y_train):
    """
    Train a machine learning model with proper instance handling.
    
    Args:
        model_class: Class of the model to be trained (not an instance)
        X_train: Training features
        y_train: Target values
        
    Returns:
        Trained model instance
    """
    try:
        # Get model name for better error handling
        if hasattr(model_class, '__name__'):
            model_name = model_class.__name__
        else:
            model_name = str(model_class)
        
        print(f"Training {model_name} model...")
            
        # Create model instance based on its type
        if isinstance(model_class, str):
            # Handle case where model is passed as a string name
            if model_class == "LinearRegression":
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            else:
                raise ValueError(f"Unknown model string: {model_class}")
        elif hasattr(model_class, 'n_estimators'):
            # Models that use n_estimators parameter (forests, etc.)
            model = model_class(n_estimators=100, random_state=42)
        else:
            # Default case for most other models
            model = model_class()
            
        # Train the model
        model.fit(X_train, y_train)
        print(f"{model_name} trained successfully")
        return model
    except Exception as e:
        print(f"Error training {model_name if 'model_name' in locals() else 'model'}: {e}")
        return None

# Train Random Forest
def train_Random_Forest(X_train, y_train, n_estimators=100):
#    feature_names = [f"feature {i}" for i in range(X.shape[1])]
#    forest = RandomForestClassifier(random_state=0)
#    forest.fit(X_train, y_train)

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    return model

# Train XGBoost
#def train_xgboost(X_train, y_train):
#    model = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, eval_metric='logloss')
#    model.fit(X_train, y_train)
#    return model

def train_xgboost(X_train, y_train):
    y_train = y_train.replace({-1: 0, 1: 1, 0: 2})  # Ensure labels start at 0
    model = xgb.XGBClassifier(objective='multi:softmax', num_class=2, eval_metric='mlogloss', n_estimators=100, max_depth=3, max_leaves=5)
    model.fit(X_train, y_train)
    return model

# Train LSTM
def train_LSTM(X_train, y_train, lookback=730, units=50, epochs=100):
    timesteps=400           # dimensionality of the input sequence
    features=3            # dimensionality of each input representation in the sequence
    LSTMoutputDimension = 2 # dimensionality of the LSTM outputs (Hidden & Cell states)

    input = Input(shape=(timesteps, features))
    output= LSTM(LSTMoutputDimension)(input)
    model_LSTM = Model(inputs=input, outputs=output)
    W = model_LSTM.layers[1].get_weights()[0]
    U = model_LSTM.layers[1].get_weights()[1]
    b = model_LSTM.layers[1].get_weights()[2]
    print("Shapes of Matrices and Vecors:")
    print("Input [batch_size, timesteps, feature] ", input.shape)
    print("Input feature/dimension (x in formulations)", input.shape[2])
    print("Number of Hidden States/LSTM units (cells)/dimensionality of the output space (h in formulations)", LSTMoutputDimension)
    print("W", W.shape)
    print("U", U.shape)
    print("b", b.shape)
    model_LSTM.summary()
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(units),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    return model

# Train CNN
def train_CNN(X_train, y_train, filters=64, kernel_size=2, epochs=100):
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=(X_train.shape[1], 1)),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    return model

def make_decision(models, X_test):
    """
    Make trading decisions based on ensemble predictions.
    
    Args:
        models: List of trained model instances
        X_test: Test features
        
    Returns:
        numpy.ndarray: Final trading decisions
    """
    if not models:
        print("Error: No models provided for decision making")
        return np.array([])
        
    try:
        predictions = []
        
        for i, model in enumerate(models):
            try:
                print(f"Getting predictions from model {i+1}/{len(models)}")
                # Handle different model types appropriately
                pred = model.predict(X_test)
                
                # Convert predictions to numpy array if needed
                if isinstance(pred, list):
                    pred = np.array(pred)
                    
                # Flatten if multi-dimensional
                if len(pred.shape) > 1:
                    pred = pred.flatten()
                    
                # Print prediction stats for debugging
                print(f"Model {i+1} prediction shape: {pred.shape}, min: {pred.min()}, max: {pred.max()}")                
                predictions.append(pred)
                
            except Exception as e:
                print(f"Error getting predictions from model {i+1}: {e}")
                # Skip this model's predictions
                continue
            
        if not predictions:
            print("Error: No valid predictions generated from any model")
            return np.array([])
            
        # Stack predictions and compute ensemble average
        predictions = np.array(predictions)
        print(f"Ensemble predictions shape: {predictions.shape}")
        
        # Compute final decision (mean of all model predictions)
        final_decision = np.round(predictions.mean(axis=0))  # Ensemble averaging
        
        return final_decision
        
    except Exception as e:
        print(f"Error in making decision: {e}")
        return np.array([])

def apply_risk_management(predictions, stop_loss=0.02, take_profit=0.05):
    decisions = []
    for pred in predictions:
        pred = int(round(pred))  # Ensure integer output (no floating-point issues)
        if pred == 1:
            decisions.append("BUY")
        elif pred == -1:
            decisions.append("SELL")
        else:
            decisions.append("HOLD")
    return decisions

# Visualization Functions
def plot_signals(df, predictions):
    df['decision'] = predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='price'))
    fig.add_trace(go.Scatter(x=df[df['decision'] == 1].index, y=df[df['decision'] == 1]['close'], mode='markers', marker=dict(color='green', size=8), name='BUY'))
    fig.add_trace(go.Scatter(x=df[df['decision'] == -1].index, y=df[df['decision'] == -1]['close'], mode='markers', marker=dict(color='red', size=8), name='SELL'))
    fig.update_layout(title='Trading Signals', xaxis_title='time', yaxis_title='price')
    fig.show()

def plot_feature_importance(model, features):
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        plt.figure(figsize=(10,5))
        sns.barplot(x=importance, y=features)
        plt.title('Feature Importance')
        plt.show()
    else:
        print(f"model {model}: has no feature called feature_importances_")

def plot_cumulative_returns(df, predictions):
    print(f"under construction..")
#    df['Strategy Returns'] = df['close'] * predictions
#    df['Cumulative Returns'] = (1 + df['Strategy Returns']).cumprod()
#    plt.figure(figsize=(10,5))
#    plt.plot(df['Cumulative Returns'], label='Strategy')
#    plt.title('Cumulative Returns')
#    plt.legend()
#    plt.show()

# Calculate technical indicators
def calculate_indicators(df):
    df['SMA14'] = df['close'].rolling(window=14).mean()  								# Simple Moving Average
    df['EMA14'] = df['close'].ewm(span=14, adjust=False).mean()  							# Exponential Moving Average
    df['EMA'] = df['close'].ewm(span=14).mean()                       							# , adjust=False # Exponential Moving Average (14-period) technical indicator
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)  									# Relative Strength Index
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)  		# MACD
    df['UpperBand'], df['MiddleBand'], df['LowerBand'] = talib.BBANDS(df['close'], timeperiod=20)  			# Bollinger Bands
    df = df.dropna()  													# Drop NaN values
    return df

# replace NaN with zero in the data
def nz(value, default=0):
    if np.isnan(value):
        return default
    return value

# fetch historical data from Binance, returns a dataframe
def fetch_binance_data(symbol='BTC-USDT', interval='1h', lookback='730 days ago UTC'):
    """
    Fetch historical data from Binance API with error handling.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
        interval (str): Time interval between candles (e.g., '1h', '1d')
        lookback (str): How far back to fetch data (e.g., '730 days ago UTC')
        
    Returns:
        pd.DataFrame: DataFrame containing price and volume data with timestamp index, or None if failed
    """
    try:
        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            print("Warning: Binance API keys are not set properly.")
            return None
            
        binance_client = BinanceClient(BINANCE_API_KEY, BINANCE_API_SECRET)
        
        try:
            klines = binance_client.get_historical_klines(symbol, BinanceClient.KLINE_INTERVAL_1DAY, lookback)
            
            if not klines:
                print(f"Warning: No data returned from Binance for {symbol}")
                return None
        except Exception as api_error:
            print(f"Error fetching data from Binance API: {api_error}")
            return None
            
        data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
                                            'SMA', 'EMA', 'RSI', 'target'])
        data['close'] = pd.to_numeric(data['close'], errors='coerce')
        data['close'] = data['close'].astype(float)
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        return data
    except Exception as e:
        print(f"Error fetching Binance data: {e}")
        return None

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
        return None

# Fetch historical data from yahoo finance, returning a dataframe
def fetch_yfinance_data(symbol='BTCUSD', interval='1h', period="730"):
    """
    Fetch historical data from Yahoo Finance API with error handling.
    
    Args:
        symbol (str): Trading pair or stock symbol (e.g., 'BTCUSD')
        interval (str): Time interval between data points (e.g., '1h', '1d')
        period (str): Period to fetch data for (e.g., '730' days)
        
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data with timestamp index
    """
    try:
        # Download data from Yahoo Finance
        data = yf.download(tickers=symbol, interval=interval, period=period)
        
        if data.empty:
            print(f"Warning: No data returned from Yahoo Finance for {symbol}")
            return None
            
        # Format the data for consistency with other sources
        data.columns = data.columns.get_level_values(0)
        data = data.reset_index()  # Ensure 'Date' is a normal column
        data = data.rename(columns={'Date': 'timestamp'})
        data.columns = [col.lower() for col in data.columns]
        
        # Convert data types
        numeric_cols = ['close', 'high', 'low', 'open', 'volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
        # Ensure close column is properly formatted
        if 'close' in data.columns:
            data['close'] = data['close'].astype(float)
            
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        return data
    except Exception as e:
        print(f"Error fetching Yahoo Finance data: {e}")
        return None

def fe_preprocess(exch="binance"):

# if exch=='binance':
    binance_data = fetch_binance_data()
    coingecko_data = fetch_coingecko_data()
    #coinmarketcap_data = coinmarketcap_data()

    # Merge datasets
    if binance_data is not None and coingecko_data is not None:
        binance_data = binance_data.merge(coingecko_data, on='timestamp')
    elif binance_data is None and coingecko_data is not None:
        binance_data = coingecko_data
    elif binance_data is not None and coingecko_data is None:
        pass  # Keep binance_data as is
    else:
        print("Warning: No valid data available from either source")
        return None
    # Handle cases where coinmarketcap_data may not be defined
    try:
        if 'coinmarketcap_data' in locals() and coinmarketcap_data is not None and binance_data is not None:
            binance_data = binance_data.merge(coinmarketcap_data, on='timestamp')
        elif binance_data is not None:
            print("Warning: Using Binance data only - CoinMarketCap not available")
        else:
            print("Error: No valid Binance data available")
            return None
    except NameError:
        if binance_data is not None:
            print("Warning: Using Binance data only - CoinMarketCap not configured")
        else:
            print("Error: No valid Binance data available")
            return None
    binance_data['target'] = binance_data['close'].shift(-1) > binance_data['close']
    binance_data['target'] = binance_data['target'].astype(int)									# force dataframe's target as type int
    binance_data['target'] = binance_data['target'].apply(lambda x: 1 if x == 1 else -1)					# Target variable (Buy=1, Hold=0, Sell=-1)
    binance_data['close'].fillna(0) 		                                                                       	 	# Fill NaN values with the last valid observation
    features = ['SMA', 'EMA14', 'EMA', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand'] 					# technical indicators
    scaler = MinMaxScaler()													# start the scaler
    #binance_data[features] = scaler.fit_transform(binance_data[features])
    binance_data = calculate_indicators(binance_data)
    		# apply the technical indicators to the scaler

    return binance_data



def plot_exchange_data(model=None, data=None, exchange_name='binance', color='black', features=None, predictions=None):
    """
    Plot exchange data with technical indicators and predictions.
    
    Args:
        model: Trained model to extract feature importance (can be None)
        data: DataFrame containing price and indicator data
        exchange_name: Name of exchange for labeling
        color: Color for price line
        features: List of feature names for feature importance
        predictions: Model predictions to overlay on price chart
    """
    if data is None:
        print("Error: No data provided for plotting")
        return
        
    try:
        # Create plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        if predictions is not None:
            print(f"Plotting with {len(predictions)} predictions")
            decisions = predictions
        else:
            decisions = None
            print("No predictions provided for plotting")
        
        # Plot price data
        ax1.plot(data['timestamp'], data['close'], label=f'{exchange_name} BTC', color='black')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        
        # Create twin axis for indicators
        ax2 = ax1.twinx()
        
        # Define indicators to plot based on exchange
        indicator_map = {
            'binance': {
                'SMA': ('SMA', 'pink', 'dashed'),
                'EMA14': ('EMA14', 'yellow', 'dotted'),
                'MACD': ('MACD', 'orange', 'dashed'),
                'RSI': ('RSI', 'aquamarine', 'dashdot'),
                'UpperBand': ('UpperBand', 'fuchsia', (0, (5, 2))),
                'MiddleBand': ('MiddleBand', 'darkgoldenrod', (0, (5, 10))),
                'LowerBand': ('LowerBand', 'gold', (0, (10, 5)))
            },
            'other': {
                'SMA14': ('SMA14', 'pink', 'dashed'),
                'EMA14': ('EMA14', 'yellow', 'dotted'),
                'MACD': ('MACD', 'orange', 'dashed'),
                'RSI': ('RSI', 'aquamarine', 'dashdot'),
                'UpperBand': ('UpperBand', 'fuchsia', (0, (5, 2))),
                'MiddleBand': ('MiddleBand', 'darkgoldenrod', (0, (5, 10))),
                'LowerBand': ('LowerBand', 'gold', (0, (10, 5)))
            }
        }
        
        # Select indicators based on exchange type
        indicators = indicator_map['binance'] if exchange_name == 'binance' else indicator_map['other']
        
        # Plot each indicator if it exists in the data
        for name, (col, color, style) in indicators.items():
            if col in data.columns:
                ax2.plot(data['timestamp'], data[col], label=name, linestyle=style, color=color)
            else:
                print(f"Warning: Indicator '{col}' not found in data columns")
                
        ax2.set_ylabel('Indicators')
        ax2.legend(loc='upper right')
        
        # Set title and display
        plt.title(f"Historical Crypto Data from {exchange_name}")
        plt.tight_layout()
        plt.show()
        
        # Plot additional visualizations
        try:
            if data is not None:
                # Plot buy/sell signals
                plot_signals(data, predictions=decisions)
            
            # Plot feature importance if model and features provided
            if model is not None and features is not None:
                plot_feature_importance(model, features)
            elif model is not None and not hasattr(model, "feature_importances_"):
                print("Model does not have feature_importances_ attribute, skipping feature importance plot")
            elif features is None:
                print("No features provided for feature importance plot")
            
            # Plot cumulative returns
            if data is not None:
                plot_cumulative_returns(data, predictions=decisions)
        except Exception as e:
            print(f"Error in additional plotting: {e}")
            
    except Exception as e:
        print(f"Error plotting exchange data: {e}")

def main():
    """Main function that orchestrates the cryptocurrency trading bot workflow."""
    # Fetch and preprocess data
    binance_data = fe_preprocess(exch='binance')
    binfeatures = ['SMA', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']
    X_train, X_test, y_train, y_test = split_data(binance_data, binfeatures, 'target')
    feature_names = binfeatures
    
    # Train various models
    print("Training Random Forest model...")
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_train, y_train)
    rf_model = train_model(RandomForestClassifier, X_train, y_train)
    
    print("Training LSTM model...")
    lstm_model = train_LSTM(X_train, y_train)
    
    print("Training CNN model...")
    cnn_model = train_CNN(X_train, y_train)
    
    print("Training traditional ML models...")
    lr_model = train_model(LinearRegression, X_train, y_train)
    LR_model = train_model(LogisticRegression, X_train, y_train)
    KNC_model = train_model(KNeighborsClassifier, X_train, y_train)
    DTC_model = train_model(DecisionTreeClassifier, X_train, y_train)
    DTR_model = train_model(DecisionTreeRegressor, X_train, y_train)
    RFR_model = train_model(RandomForestRegressor,  X_train, y_train)
    
    # Model evaluation and decision making
    # Currently only using RandomForest for decisions, but we have an ensemble available
    models = [rf_model]
    binmodels = [rf_model, lstm_model, cnn_model, lr_model, LR_model, KNC_model, DTC_model, DTR_model, RFR_model]
    
    # Make predictions
    print("Making trading decisions...")
    bindecisions = make_decision(models, X_test)  # Changed from X_train to X_test for proper evaluation
    binfinal_trades = apply_risk_management(bindecisions)
    print("Final trade decisions: ", binfinal_trades)
    
    # Evaluate model performance
    predictions = rf_model.predict(X_test)
    print("Model accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
   
    return binance_data, binmodels, bindecisions

if __name__ == "__main__":
    print("Starting cryptocurrency trading bot...")
    binance_data, binmodels, bindecisions = main()



yf_data = fe_preprocess(exch='yahoofinance')
yffeatures = ['SMA14', 'EMA14', 'RSI', 'MACD', 'UpperBand', 'MiddleBand', 'LowerBand']
X_train, X_test, y_train, y_test = split_data(yf_data, yffeatures, 'target')
feature_names = yffeatures
rf_model = train_model(RandomForestClassifier, X_train, y_train)
lstm_model = train_LSTM(X_train, y_train)
cnn_model = train_CNN(X_train, y_train)
lr_model = train_model(LinearRegression, X_train, y_train)
LR_model = train_model(LogisticRegression, X_train, y_train)
KNC_model = train_model(KNeighborsClassifier, X_train, y_train)
DTC_model = train_model(DecisionTreeClassifier, X_train, y_train)
DTR_model = train_model(DecisionTreeRegressor, X_train, y_train)
RFR_model = train_model(RandomForestRegressor,  X_train, y_train)
models=[rf_model]
yfmodels=[rf_model, lstm_model, cnn_model, lr_model, LR_model, KNC_model, DTC_model, DTR_model, RFR_model]
yfdecisions = make_decision(models, X_train)
yffinal_trades = apply_risk_management(yfdecisions)
print("final trade decisions: ", yffinal_trades)

# Evaluate model performance
yfmodels = [rf_model, lstm_model, cnn_model, lr_model, LR_model, KNC_model, DTC_model, DTR_model, RFR_model]
yfdecisions = make_decision(yfmodels, X_train)
yffinal_trades = apply_risk_management(yfdecisions)
print("final trade decisions: ", yffinal_trades)

# Evaluate model performance
yfmodels = [rf_model, lstm_model, cnn_model, lr_model, LR_model, KNC_model, DTC_model, DTR_model, RFR_model]
yfdecisions = make_decision(yfmodels, X_train)
yffinal_trades = apply_risk_management(yfdecisions)
print("final trade decisions: ", yffinal_trades)

# Evaluate model performance
yfeval = evaluate_classification_model(yfmodels, X_test, y_test, model_name="yfmodels")
print("\nEvaluation Results:\n", yfeval)

# Visualize model performance
visualize_model(yfmodels, X_test, y_test, model_name="yfmodels")

# Save model
save_model(yfmodels, "yfmodels")

# Load model
yfmodels = load_model("yfmodels")

# Make predictions
yfdecisions = make_decision(yfmodels, X_test)
yffinal_trades = apply_risk_management(yfdecisions)
print("final trade decisions: ", yffinal_trades)

