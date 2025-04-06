#!/usr/bin/env python3
import os
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor

# Setup logging
logging.basicConfig(
    filename="crypto_trading.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)

SYMBOL = "BTC-USD"
INTERVAL = "5m"  # Example: "1m", "5m", "1h"
LIMIT = 1000
THRESHOLD = 0.01  # 1% threshold for buy/sell decision

# Fetch historical data
def fetch_yfinance_data(symbol, interval, period="30d"):
    print(f"Fetching data for {symbol} with interval {interval}...")
    try:
        df = yf.download(tickers=symbol, interval=interval, period=period)
        df.reset_index(inplace=True)
        print(f"Successfully fetched {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Train a model and generate predictions
def train_model(df):
    X = df[["Open", "High", "Low", "Close", "Volume"]]
    y = df["Close"].shift(-1).ffill().values.ravel()  # Flatten to 1D

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Generate predictions for entire dataset
    predictions = model.predict(X)
    
    # Convert predictions to trading signals (1 for buy, -1 for sell, 0 for hold)
    signals = np.zeros(len(predictions))
    signals[predictions > df["Close"] * (1 + THRESHOLD)] = 1   # Buy signals
    signals[predictions < df["Close"] * (1 - THRESHOLD)] = -1  # Sell signals
    
    return model, signals

# Make a decision
def make_decision(latest_price, predicted_price, threshold):
    print(f"Latest Price: {latest_price}, Predicted Price: {predicted_price}")
    if predicted_price > latest_price * (1 + threshold):
        return "BUY"
    elif predicted_price < latest_price * (1 - threshold):
        return "SELL"
    else:
        return "HOLD"

# Execute a trade
def execute_trade(decision, symbol, amount):
    if decision == "BUY":
        # Example API call to execute a buy order
        print(f"Executing BUY order for {amount} of {symbol}")
        # Here you would add the API call to execute the trade
    elif decision == "SELL":
        # Example API call to execute a sell order
        print(f"Executing SELL order for {amount} of {symbol}")
        # Here you would add the API call to execute the trade
    else:
        print("No trade executed. Decision was HOLD.")

from flask import Flask, jsonify  # Import Flask for API
app = Flask(__name__)  # Initialize Flask app

@app.route('/data', methods=['GET'])  # Create an API endpoint
def get_data():
    # Fetch data from Yahoo Finance
    yahoo_data = fetch_yfinance_data(SYMBOL, INTERVAL, period="30d")
    
    # Fetch data from Binance
    binance_data = fetch_binance_data(symbol='BTCUSDT', interval='1h', lookback='30 days ago UTC')
    
    # Fetch data from Bitvavo
    bitvavo_data = fetch_bitvavo_data(symbol='BTC-EUR', interval='1h', start_date="2023-03-18", end_date="2025-03-18")
    
    # Structure the response
    response = {
        "yahoo": yahoo_data.to_dict(orient='records'),
        "binance": binance_data.to_dict(orient='records'),
        "bitvavo": bitvavo_data.to_dict(orient='records')
    }
    
    return jsonify(response)

# Live trading simulation
def live_trading():  
    # Fetch data from various sources and display it
    df = fetch_yfinance_data(SYMBOL, INTERVAL, period="30d")
    if df is None or df.empty:
        print("No data fetched, exiting.")
        return

    model, signals = train_model(df)
    print("Model trained and signals generated.")

    # Get the latest signal
    latest_signal = signals[-1]
    decision = "BUY" if latest_signal == 1 else "SELL" if latest_signal == -1 else "HOLD"
    execute_trade(decision, SYMBOL, 0.001)

# Run the script
if __name__ == "__main__":
    live_trading()
