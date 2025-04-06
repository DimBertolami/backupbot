#!/usr/bin/env python3
"""
Trading Signals Update Script

This script simulates and updates trading signals for cryptocurrencies in the wallet.
It generates buy/sell/hold signals every 5 minutes based on ML model predictions.
"""

import os
import json
import time
import random
import datetime
import numpy as np
import argparse
import re
from pathlib import Path
import signal
import logging

# Set up logging
logging.basicConfig(
    filename='trading_signals.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, stopping gracefully...")
    raise KeyboardInterrupt

# Set up signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Import our trading database module
try:
    from trading_db import save_trading_signals, generate_bot_thoughts
except ImportError:
    print("Warning: trading_db module not found. Database features will be disabled.")
    # Create dummy functions if the module is not available
    def save_trading_signals(*args, **kwargs): pass
    def generate_bot_thoughts(*args, **kwargs): return []

# Cryptocurrencies in the wallet
WALLET_CRYPTOS = [
    {"symbol": "BTC/USDT", "base_price": 52000, "volatility": 0.015},
    {"symbol": "ETH/USDT", "base_price": 3100, "volatility": 0.02},
    {"symbol": "SOL/USDT", "base_price": 149, "volatility": 0.03},
    {"symbol": "ADA/USDT", "base_price": 0.52, "volatility": 0.025},
    {"symbol": "BNB/USDT", "base_price": 580, "volatility": 0.018}
]

# Signal generation models (simulated)
MODELS = [
    "transformer_v1.2", 
    "inceptiontime_v1.0", 
    "lstm_gru_ensemble", 
    "temporal_fusion_transformer_v1"
]

# Output directory for frontend
FRONTEND_DIR = Path("/opt/lampp/htdocs/bot/frontend/public/trading_data")
CONFIG_FILE = FRONTEND_DIR / "trading_config.json"

def generate_price(base_price, volatility, trend_bias=0):
    """Generate a realistic price based on volatility and trend"""
    change = np.random.normal(trend_bias, volatility) 
    return base_price * (1 + change)

def generate_trading_signal(price_history, current_price, prev_signal=None):
    """Generate a trading signal based on price movement patterns"""
    if len(price_history) < 3:
        signal = random.choice(["BUY", "HOLD", "SELL"])
        confidence = random.uniform(0.5, 0.7)
        return signal, confidence
    
    # Calculate recent trends
    short_trend = (current_price / price_history[-1]) - 1
    medium_trend = (current_price / price_history[-3]) - 1
    
    # Basic momentum strategy
    if short_trend > 0.005 and medium_trend > 0.01:
        signal = "BUY"  # Strong uptrend
        confidence = random.uniform(0.75, 0.95)
    elif short_trend < -0.005 and medium_trend < -0.01:
        signal = "SELL"  # Strong downtrend
        confidence = random.uniform(0.75, 0.95)
    else:
        # Sideways movement - favor holding or previous signal
        if prev_signal and random.random() > 0.3:
            signal = prev_signal
            confidence = random.uniform(0.55, 0.75)
        else:
            signal = random.choices(["BUY", "HOLD", "SELL"], weights=[0.3, 0.4, 0.3])[0]
            confidence = random.uniform(0.55, 0.75)
    
    return signal, confidence

def update_trading_signals(price_history=None, prev_signals=None):
    """Update trading signals for all cryptocurrencies in the wallet"""
    if price_history is None:
        price_history = {crypto["symbol"]: [] for crypto in WALLET_CRYPTOS}
    
    if prev_signals is None:
        prev_signals = {crypto["symbol"]: None for crypto in WALLET_CRYPTOS}
    
    now = datetime.datetime.now()
    signals = []
    
    # Determine the current market trend (slightly biased)
    market_trend = np.random.normal(0, 0.005)
    
    for crypto in WALLET_CRYPTOS:
        symbol = crypto["symbol"]
        base_price = crypto["base_price"]
        volatility = crypto["volatility"]
        
        # Apply some market correlation with individual noise
        current_price = generate_price(base_price, volatility, market_trend)
        
        # Store price history (keep last 24 points = 2 hours)
        if symbol in price_history:
            price_history[symbol].append(current_price)
            if len(price_history[symbol]) > 24:
                price_history[symbol] = price_history[symbol][-24:]
        else:
            price_history[symbol] = [current_price]
        
        # Calculate 24h price change (or simulate if history is shorter)
        if len(price_history[symbol]) >= 24:
            price_change_24h = (current_price / price_history[symbol][0] - 1) * 100
        else:
            price_change_24h = np.random.normal(market_trend * 100 * 24, 3)
        
        # Generate signal and confidence
        signal, confidence = generate_trading_signal(
            price_history[symbol], 
            current_price,
            prev_signals[symbol]
        )
        
        # Update previous signal
        prev_signals[symbol] = signal
        
        signals.append({
            "symbol": symbol,
            "currentPrice": round(current_price, 8 if current_price < 1 else 2),
            "signal": signal,
            "confidence": confidence,
            "timestamp": now.isoformat(),
            "priceChange24h": round(price_change_24h, 2)
        })
    
    # Select a model
    model_id = random.choice(MODELS)
    
    # Generate bot thoughts based on current signals
    bot_thoughts = generate_bot_thoughts(signals, model_id)
    
    # Create the full data object
    data = {
        "timestamp": now.isoformat(),
        "model_id": model_id,
        "signals": signals,
        "autoRefresh": True,
        "thoughts": bot_thoughts
    }
    
    # Save to file for frontend
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    with open(FRONTEND_DIR / "live_trading_status.json", "w") as f:
        json.dump(data, f, indent=2)
    
    # Save data to SQLite database
    try:
        save_trading_signals(signals, model_id)
    except Exception as e:
        logging.error(f"Error saving to database: {str(e)}")
        print(f"Error saving to database: {str(e)}")
    
    print(f"[{now.strftime('%H:%M:%S')}] Updated trading signals:")
    for signal in signals:
        print(f"  {signal['symbol']}: {signal['signal']} @ ${signal['currentPrice']} " 
              f"({signal['priceChange24h']:+.2f}%) - {signal['confidence']*100:.1f}% confidence")
    
    # Also print a random thought
    if bot_thoughts:
        thought = random.choice(bot_thoughts)
        print(f"\n🤖 Bot thought: {thought}")
    
    return price_history, prev_signals

def parse_interval(interval_str):
    """Parse interval string (like '5m', '1h') into seconds"""
    if not interval_str:
        return 5 * 60  # Default to 5 minutes
        
    match = re.match(r'^(\d+)([mhdw])$', interval_str.lower())
    if not match:
        return 5 * 60  # Default to 5 minutes if format is invalid
        
    value, unit = match.groups()
    value = int(value)
    
    if unit == 'm':  # minutes
        return value * 60
    elif unit == 'h':  # hours
        return value * 60 * 60
    elif unit == 'd':  # days
        return value * 24 * 60 * 60
    elif unit == 'w':  # weeks
        return value * 7 * 24 * 60 * 60
    
    return 5 * 60  # Default fallback

def get_config():
    """Read configuration from the config file"""
    if not CONFIG_FILE.exists():
        # Default config
        return {
            "update_interval": "5m",
            "auto_refresh": True,
            "last_modified": datetime.datetime.now().isoformat()
        }
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error reading config file: {str(e)}")
        print(f"Error reading config file: {str(e)}")
        # Return default config if file read fails
        return {
            "update_interval": "5m",
            "auto_refresh": True,
            "last_modified": datetime.datetime.now().isoformat()
        }

def update_config(config):
    """Write updated configuration to the config file"""
    config["last_modified"] = datetime.datetime.now().isoformat()
    
    try:
        os.makedirs(FRONTEND_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error writing config file: {str(e)}")
        print(f"Error writing config file: {str(e)}")
        return False

def run_update_loop(initial_interval=None):
    """Run the update loop continuously, checking for config changes"""
    price_history = None
    prev_signals = None
    last_config_check = 0
    config_check_interval = 30  # Check config every 30 seconds
    
    # Get initial config or use provided interval
    config = get_config()
    if initial_interval:
        config["update_interval"] = f"{initial_interval}m"
        update_config(config)
    
    interval_seconds = parse_interval(config["update_interval"])
    
    print(f"Starting trading signal updates every {config['update_interval']}...")
    logging.info(f"Starting trading signal updates with interval: {config['update_interval']}")
    
    last_update = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time to update signals
            if current_time - last_update >= interval_seconds:
                try:
                    price_history, prev_signals = update_trading_signals(price_history, prev_signals)
                    logging.info("Successfully updated trading signals")
                    print(f"Next update in {config['update_interval']}...")
                    last_update = current_time
                except Exception as e:
                    logging.error(f"Error updating trading signals: {str(e)}")
                    print(f"Error updating trading signals: {str(e)}")
                    # Wait a bit before retrying
                    time.sleep(60)
            
            # Check for config changes periodically
            if current_time - last_config_check >= config_check_interval:
                try:
                    new_config = get_config()
                    if new_config["update_interval"] != config["update_interval"]:
                        print(f"Interval changed: {config['update_interval']} → {new_config['update_interval']}")
                        logging.info(f"Interval changed: {config['update_interval']} → {new_config['update_interval']}")
                        config = new_config
                        interval_seconds = parse_interval(config["update_interval"])
                    last_config_check = current_time
                except Exception as e:
                    logging.error(f"Error checking config: {str(e)}")
            
            # Short sleep to prevent CPU hogging
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.error(f"Error during sleep: {str(e)}")
                
    except KeyboardInterrupt:
        logging.info("Update loop stopped by user.")
        print("Update loop stopped by user.")
    except Exception as e:
        logging.error(f"Unexpected error in update loop: {str(e)}")
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and update trading signals")
    parser.add_argument("--interval", type=int, default=None, 
                        help="Update interval in minutes (default: read from config)")
    parser.add_argument("--once", action="store_true", 
                        help="Run once and exit (don't loop)")
    parser.add_argument("--set-interval", type=str, 
                        help="Set a new interval (format: 1m, 5m, 15m, 1h, etc.)")
    
    args = parser.parse_args()
    
    # If just setting a new interval without running
    if args.set_interval and not args.once:
        config = get_config()
        config["update_interval"] = args.set_interval
        if update_config(config):
            print(f"Successfully set interval to {args.set_interval}")
        else:
            print("Failed to update interval")
        exit(0)
    
    if args.once:
        update_trading_signals()
    else:
        run_update_loop(args.interval)
