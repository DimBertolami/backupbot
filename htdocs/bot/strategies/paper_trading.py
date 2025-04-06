import os
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *
import threading
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_trading")

class PaperTradingStrategy:
    """
    Paper trading strategy for Binance that can be easily switched to live trading.
    This class simulates trading with virtual funds while using real market data.
    
    Features:
    - Paper trading with virtual funds
    - Live trading with real funds (when configured)
    - Auto-execution of suggested trades based on confidence levels
    - Configurable refresh intervals
    - Performance tracking and metrics
    """
    def __init__(self, config_file='trading_config.json'):
        """
        Initialize the paper trading strategy with configuration.
        
        Args:
            config_file (str): Path to the JSON configuration file
        """
        self.config_file = config_file
        self.config = {}  # Initialize config as empty dict before loading
        self.trade_history = []  # Initialize trade_history before loading state
        
        # Default values for essential parameters
        self.mode = 'paper'
        self.symbols = ['BTCUSDT']
        self.base_currency = 'USDT'
        self.initial_balance = 10000
        self.risk_percentage = 2
        self.strategy_params = {
            'short_window': 50,
            'long_window': 200,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30
        }
        self.intervals = ['1h']
        self.stop_loss_pct = 5
        self.take_profit_pct = 10
        
        # Auto-execution of suggested trades
        self.auto_execute_suggested_trades = False
        self.min_confidence_threshold = 0.75  # Only execute trades with confidence above this threshold
        self.suggested_trades = []  # List to store suggested trades from the dashboard
        self.last_suggested_trade_check = 0  # Timestamp of last check for suggested trades
        self.suggested_trade_refresh_interval = 60  # Default to 1 minute
        
        # Load configuration
        self.load_config()
        
        # Initialize account state
        self.reset_account()
        
        # Initialize runtime variables
        self.is_running = False
        self.thread = None
        self.orders = []
        
        # Keep API key state at startup
        self._last_api_key = self.config.get('api_key', '')
        self._last_api_secret = self.config.get('api_secret', '')
        
        # Load any existing state if available
        self.load_state()
        
        # Try to recover API keys from backup if they are missing or empty in config
        self.recover_api_keys_if_needed()
        
        # Initialize Binance client with potentially recovered API keys
        self.client = Client(
            self.config.get('api_key', ''), 
            self.config.get('api_secret', '')
        )
        
        # Validate the API keys
        self.validate_api_keys()
        
        logger.info(f"Paper Trading Strategy initialized with {self.base_currency} balance: {self.balance}")

    def load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
                
            # Extract essential configuration parameters
            self.mode = self.config.get('mode', 'paper')  # 'paper' or 'live'
            self.symbols = self.config.get('symbols', ['BTCUSDT'])
            self.base_currency = self.config.get('base_currency', 'USDT')
            self.initial_balance = self.config.get('initial_balance', 10000)
            self.risk_percentage = self.config.get('risk_percentage', 2)
            self.strategy_params = self.config.get('strategy_params', {
                'short_window': 50,
                'long_window': 200,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            })
            self.intervals = self.config.get('intervals', ['1h'])
            self.stop_loss_pct = self.config.get('stop_loss_pct', 5)
            self.take_profit_pct = self.config.get('take_profit_pct', 10)
            
            # Load auto-execution settings
            self.auto_execute_suggested_trades = self.config.get('auto_execute_suggested_trades', False)
            self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.75)
            self.suggested_trade_refresh_interval = self.config.get('suggested_trade_refresh_interval', 60)
            
            logger.info(f"Configuration loaded from {self.config_file}")
            if self.auto_execute_suggested_trades:
                logger.info(f"Auto-execution of suggested trades is ENABLED with min confidence: {self.min_confidence_threshold}")
            else:
                logger.info("Auto-execution of suggested trades is DISABLED")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            # Set defaults if config loading fails
            self.mode = 'paper'
            self.symbols = ['BTCUSDT']
            self.base_currency = 'USDT'
            self.initial_balance = 10000
            self.risk_percentage = 2
            self.strategy_params = {
                'short_window': 50,
                'long_window': 200,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            }
            self.intervals = ['1h']
            self.stop_loss_pct = 5
            self.take_profit_pct = 10

    def save_config(self):
        """Save current configuration to JSON file."""
        try:
            # Create the directory if it doesn't exist
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            # Create a backup of the existing config file if it exists
            if os.path.exists(self.config_file):
                backup_file = f"{self.config_file}.bak"
                try:
                    with open(self.config_file, 'r') as src:
                        with open(backup_file, 'w') as dst:
                            dst.write(src.read())
                except Exception as backup_err:
                    logger.warning(f"Could not create config backup: {backup_err}")
                    
            # Save the updated config
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            logger.info(f"Configuration saved to {self.config_file}")
            
            # Also update the client if API keys have changed
            self.update_client()
            
            # Save a backup of the API keys for redundancy
            self.backup_api_keys()
        except Exception as e:
            logger.error(f"Error saving config: {e}")

    def update_client(self):
        """Update the Binance client if API keys have changed."""
        try:
            current_api_key = self.config.get('api_key', '')
            current_api_secret = self.config.get('api_secret', '')
            
            # Only update client if keys have changed
            if (not hasattr(self, 'client') or 
                current_api_key != self._last_api_key or 
                current_api_secret != self._last_api_secret):
                
                # If we have valid keys, update the client
                if current_api_key and current_api_secret:
                    logger.info("Updating Binance client with new API credentials")
                    self.client = Client(current_api_key, current_api_secret)
                    self._last_api_key = current_api_key
                    self._last_api_secret = current_api_secret
                    return True
            return False
        except Exception as e:
            logger.error(f"Error updating client: {e}")
            return False
    
    def backup_api_keys(self):
        """Save backup copies of API keys for redundancy."""
        try:
            api_key = self.config.get('api_key', '')
            api_secret = self.config.get('api_secret', '')
            
            # Only backup if we have valid keys
            if not api_key or not api_secret:
                return False
                
            # Save to state file
            self.save_state()
            
            # Create dedicated backup file
            backup_dir = os.path.dirname(self.config_file)
            backup_file = os.path.join(backup_dir, 'api_keys_backup.json')
            
            backup_data = {
                "api_key": api_key,
                "api_secret": api_secret,
                "timestamp": datetime.now().isoformat(),
                "source": self.config_file
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=4)
                
            logger.info(f"API keys backed up to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Error backing up API keys: {e}")
            return False
    
    def recover_api_keys_if_needed(self):
        """Try to recover API keys from backup files if they are missing in config."""
        # Only attempt recovery if API keys are missing or empty
        if self.config.get('api_key') and self.config.get('api_secret'):
            logger.info("API keys already present in config, no recovery needed")
            return True
            
        logger.info("API keys missing or empty, attempting recovery...")
        
        # Try to recover from backup file first
        backup_dir = os.path.dirname(self.config_file)
        backup_file = os.path.join(backup_dir, 'api_keys_backup.json')
        
        if os.path.exists(backup_file):
            try:
                with open(backup_file, 'r') as f:
                    backup = json.load(f)
                    
                api_key = backup.get('api_key', '')
                api_secret = backup.get('api_secret', '')
                    
                if api_key and api_secret:
                    logger.info("Recovered API keys from backup file")
                    self.config['api_key'] = api_key
                    self.config['api_secret'] = api_secret
                    self.save_config()
                    return True
            except Exception as e:
                logger.error(f"Error recovering API keys from backup: {e}")
                
        # Next try state file
        state_file = 'paper_trading_state.json'
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    
                api_keys = state.get('api_keys', {})
                api_key = api_keys.get('key', '')
                api_secret = api_keys.get('secret', '')
                
                if api_key and api_secret:
                    logger.info("Recovered API keys from state file")
                    self.config['api_key'] = api_key
                    self.config['api_secret'] = api_secret
                    self.save_config()
                    return True
            except Exception as e:
                logger.error(f"Error recovering API keys from state file: {e}")
        
        # If we're still here, try looking for api keys in other locations
        possible_locations = [
            '/opt/lampp/htdocs/bot/frontend/trading_data/api_keys.json',
            '/opt/lampp/htdocs/bot/frontend/public/api_keys.json',
            '/opt/lampp/htdocs/bot/api_keys.json'
        ]
        
        for location in possible_locations:
            if os.path.exists(location):
                try:
                    with open(location, 'r') as f:
                        keys = json.load(f)
                        
                    api_key = keys.get('api_key', '')
                    api_secret = keys.get('api_secret', '')
                    
                    if api_key and api_secret:
                        logger.info(f"Recovered API keys from {location}")
                        self.config['api_key'] = api_key
                        self.config['api_secret'] = api_secret
                        self.save_config()
                        return True
                except Exception as e:
                    logger.error(f"Error reading API keys from {location}: {e}")
        
        logger.warning("Could not recover API keys from any backup location")
        return False
        
    def validate_api_keys(self):
        """Validate that the current API keys are working."""
        api_key = self.config.get('api_key', '')
        api_secret = self.config.get('api_secret', '')
        
        if not api_key or not api_secret:
            logger.warning("No API keys configured, some features may be unavailable")
            return False
            
        try:
            # Test API connection with a simple request
            server_time = self.client.get_server_time()
            if server_time:
                logger.info("API keys validated successfully")
                return True
        except BinanceAPIException as e:
            logger.error(f"API key validation failed: {e}")
            if e.code == -2015:  # Invalid API key
                # Don't clear the keys automatically, just report the issue
                logger.warning("API keys are invalid but keeping them in config")
            return False
        except Exception as e:
            logger.error(f"Error validating API keys: {e}")
            return False
    
    def reset_account(self):
        """Reset account to initial state."""
        self.balance = self.initial_balance
        self.holdings = {symbol: 0 for symbol in self.symbols}
        self.open_orders = []
        self.equity_history = [{'timestamp': datetime.now().isoformat(), 'equity': self.balance}]
        self.last_prices = {}
        self.save_state()

    def load_state(self):
        """Load trading state from disk if available."""
        # Ensure we have default values for all state attributes
        if not hasattr(self, 'balance'):
            self.balance = self.initial_balance
        if not hasattr(self, 'holdings'):
            self.holdings = {symbol: 0 for symbol in self.symbols}
        if not hasattr(self, 'open_orders'):
            self.open_orders = []
        if not hasattr(self, 'equity_history'):
            self.equity_history = []
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        if not hasattr(self, 'last_prices'):
            self.last_prices = {}
            
        # Get state file path - use the same directory as config file
        state_dir = os.path.dirname(self.config_file)
        state_file = os.path.join(state_dir, 'paper_trading_state.json')
        
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.balance = state.get('balance', self.initial_balance)
                self.holdings = state.get('holdings', {symbol: 0 for symbol in self.symbols})
                self.open_orders = state.get('open_orders', [])
                self.equity_history = state.get('equity_history', [])
                self.trade_history = state.get('trade_history', [])
                self.last_prices = state.get('last_prices', {})
                
                # Restore API keys from state if they exist
                api_keys = state.get('api_keys', {})
                if api_keys and api_keys.get('key') and api_keys.get('secret'):
                    # Only update config if keys from state are not empty
                    if not self.config.get('api_key') or not self.config.get('api_secret'):
                        logger.info("Restoring API keys from saved state")
                        self.config['api_key'] = api_keys.get('key')
                        self.config['api_secret'] = api_keys.get('secret')
                        self.save_config()  # Make sure they're saved to config file too
                
                # Restore trading mode if available
                if 'mode' in state:
                    self.mode = state.get('mode')
                
                logger.info(f"Trading state loaded from {state_file}")
            else:
                logger.info("No saved state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def save_state(self):
        """Save current trading state to disk."""
        # Ensure we have default values for all state attributes
        if not hasattr(self, 'balance'):
            self.balance = self.initial_balance
        if not hasattr(self, 'holdings'):
            self.holdings = {symbol: 0 for symbol in self.symbols}
        if not hasattr(self, 'open_orders'):
            self.open_orders = []
        if not hasattr(self, 'equity_history'):
            self.equity_history = []
        if not hasattr(self, 'trade_history'):
            self.trade_history = []
        if not hasattr(self, 'last_prices'):
            self.last_prices = {}
        
        # Create directory for state file if it doesn't exist
        state_dir = os.path.dirname(self.config_file)
        os.makedirs(state_dir, exist_ok=True)
        
        state_file = os.path.join(state_dir, 'paper_trading_state.json')
        
        try:
            # Include API keys in the state file as well for redundancy
            state = {
                'balance': self.balance,
                'holdings': self.holdings,
                'open_orders': self.open_orders,
                'equity_history': self.equity_history,
                'trade_history': self.trade_history,
                'last_prices': self.last_prices,
                'api_keys': {
                    'key': self.config.get('api_key', ''),
                    'secret': self.config.get('api_secret', '')
                },
                'mode': self.mode,
                'is_running': getattr(self, 'is_running', False),
                'last_updated': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
                
            logger.info(f"Trading state saved to {state_file}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")

    def fetch_market_data(self, symbol, interval, limit=500):
        """
        Fetch market data from Binance.
        
        Args:
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT')
            interval (str): Candlestick interval (e.g., '1h', '4h', '1d')
            limit (int): Number of candlesticks to fetch
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            candles = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Set index to timestamp
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            self.calculate_indicators(df)
            
            # Update last price
            self.last_prices[symbol] = float(df['close'].iloc[-1])
            
            logger.debug(f"Fetched {len(df)} candlesticks for {symbol} at {interval} interval")
            return df
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df):
        """
        Calculate technical indicators for trading signals.
        
        Args:
            df (pandas.DataFrame): OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with added indicators
        """
        # Calculate moving averages
        df['SMA_short'] = df['close'].rolling(window=self.strategy_params['short_window']).mean()
        df['SMA_long'] = df['close'].rolling(window=self.strategy_params['long_window']).mean()
        
        # Calculate EMA
        df['EMA_short'] = df['close'].ewm(span=self.strategy_params['short_window'], adjust=False).mean()
        df['EMA_long'] = df['close'].ewm(span=self.strategy_params['long_window'], adjust=False).mean()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.strategy_params['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.strategy_params['rsi_period']).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df

    def generate_signals(self, df):
        """
        Generate trading signals based on calculated indicators.
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            
        Returns:
            int: 1 for buy, -1 for sell, 0 for hold
        """
        # Get the latest data
        latest = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else latest
        
        # Moving Average Crossover Signal
        ma_signal = 0
        if latest['SMA_short'] > latest['SMA_long'] and previous['SMA_short'] <= previous['SMA_long']:
            ma_signal = 1  # Buy signal
        elif latest['SMA_short'] < latest['SMA_long'] and previous['SMA_short'] >= previous['SMA_long']:
            ma_signal = -1  # Sell signal
            
        # RSI Signal
        rsi_signal = 0
        if latest['RSI'] < self.strategy_params['rsi_oversold']:
            rsi_signal = 1  # Buy signal
        elif latest['RSI'] > self.strategy_params['rsi_overbought']:
            rsi_signal = -1  # Sell signal
            
        # Combine signals (you can adjust the weighting or logic)
        # For now, we'll prioritize MA crossover but consider RSI too
        if ma_signal != 0:
            return ma_signal
        return rsi_signal

    def paper_trade(self, symbol, signal, current_price, is_suggested=False, confidence=0.0):
        """
        Execute a paper trade based on the given signal.
        
        Args:
            symbol (str): Trading pair symbol
            signal (int): Trading signal (1=buy, -1=sell, 0=hold)
            current_price (float): Current market price
            is_suggested (bool): Whether this trade is from suggested trades
            confidence (float): Confidence level for suggested trades
            
        Returns:
            dict: Trade details if executed, None otherwise
        """
        if signal == 0:
            return None
            
        timestamp = datetime.now().isoformat()
        
        if signal > 0:  # Buy signal
            # Check if we have enough balance
            if self.balance <= 0:
                logger.info(f"Insufficient balance ({self.balance} {self.base_currency}) to buy {symbol}")
                return None
                
            # Calculate position size based on risk percentage
            risk_amount = self.balance * (self.risk_percentage / 100)
            quantity = risk_amount / current_price
            
            # Round down to appropriate decimal places
            quantity = self.round_quantity(symbol, quantity)
            
            if quantity <= 0:
                logger.info(f"Calculated quantity too small to execute buy for {symbol}")
                return None
                
            # Calculate the actual cost
            cost = quantity * current_price
            
            # Update balance and holdings
            self.balance -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
            
            # Record the trade
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'side': 'BUY',
                'quantity': quantity,
                'price': current_price,
                'value': cost,
                'balance_after': self.balance,
                'holdings_after': self.holdings[symbol],
                'type': 'paper' if self.mode == 'paper' else 'live'
            }
            
            self.trade_history.append(trade)
            logger.info(f"BUY {quantity} {symbol} at {current_price} = {cost} {self.base_currency}")
            
            return trade
            
        elif signal < 0:  # Sell signal
            # Check if we have any holdings to sell
            current_holdings = self.holdings.get(symbol, 0)
            if current_holdings <= 0:
                logger.info(f"No holdings of {symbol} to sell")
                return None
                
            # Calculate the value
            value = current_holdings * current_price
            
            # Update balance and holdings
            self.balance += value
            self.holdings[symbol] = 0
            
            # Record the trade
            trade = {
                'timestamp': timestamp,
                'symbol': symbol,
                'side': 'SELL',
                'quantity': current_holdings,
                'price': current_price,
                'value': value,
                'balance_after': self.balance,
                'holdings_after': 0,
                'type': 'paper' if self.mode == 'paper' else 'live'
            }
            
            self.trade_history.append(trade)
            logger.info(f"SELL {current_holdings} {symbol} at {current_price} = {value} {self.base_currency}")
            
            return trade
            
        return None

    def live_trade(self, symbol, signal, current_price, is_suggested=False, confidence=0.0):
        """
        Execute a live trade on Binance based on the given signal.
        
        Args:
            symbol (str): Trading pair symbol
            signal (int): Trading signal (1=buy, -1=sell, 0=hold)
            current_price (float): Current market price
            is_suggested (bool): Whether this trade is from suggested trades
            confidence (float): Confidence level for suggested trades
            
        Returns:
            dict: Trade details if executed, None otherwise
        """
        if signal == 0 or self.mode != 'live':
            return None
            
        try:
            timestamp = datetime.now().isoformat()
            
            if signal > 0:  # Buy signal
                # Get account balance
                account = self.client.get_account()
                base_asset_balance = next((float(balance['free']) for balance in account['balances'] 
                                         if balance['asset'] == self.base_currency), 0)
                
                if base_asset_balance <= 0:
                    logger.info(f"Insufficient balance ({base_asset_balance} {self.base_currency}) to buy {symbol}")
                    return None
                
                # Calculate position size based on risk percentage
                risk_amount = base_asset_balance * (self.risk_percentage / 100)
                quantity = risk_amount / current_price
                
                # Round down to appropriate decimal places
                quantity = self.round_quantity(symbol, quantity)
                
                if quantity <= 0:
                    logger.info(f"Calculated quantity too small to execute buy for {symbol}")
                    return None
                
                # Place a market buy order
                order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                
                # Record the trade
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': quantity,
                    'price': current_price,  # Estimated price
                    'value': quantity * current_price,  # Estimated value
                    'order_id': order['orderId'],
                    'type': 'live'
                }
                
                self.trade_history.append(trade)
                logger.info(f"LIVE BUY {quantity} {symbol} at ~{current_price}")
                
                return trade
                
            elif signal < 0:  # Sell signal
                # Get current holdings
                account = self.client.get_account()
                asset = symbol.replace(self.base_currency, '')
                asset_balance = next((float(balance['free']) for balance in account['balances'] 
                                    if balance['asset'] == asset), 0)
                
                if asset_balance <= 0:
                    logger.info(f"No holdings of {asset} to sell")
                    return None
                
                # Place a market sell order
                order = self.client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=asset_balance
                )
                
                # Record the trade
                trade = {
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'side': 'SELL',
                    'quantity': asset_balance,
                    'price': current_price,  # Estimated price
                    'value': asset_balance * current_price,  # Estimated value
                    'order_id': order['orderId'],
                    'type': 'live'
                }
                
                self.trade_history.append(trade)
                logger.info(f"LIVE SELL {asset_balance} {symbol} at ~{current_price}")
                
                return trade
                
        except BinanceAPIException as e:
            logger.error(f"Binance API error during live trading: {e}")
        except Exception as e:
            logger.error(f"Error executing live trade: {e}")
            
        return None

    def round_quantity(self, symbol, quantity):
        """
        Round quantity to appropriate decimal places based on exchange rules.
        
        Args:
            symbol (str): Trading pair symbol
            quantity (float): Quantity to round
            
        Returns:
            float: Rounded quantity
        """
        try:
            # Get exchange info for the symbol
            exchange_info = self.client.get_exchange_info()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if symbol_info:
                lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
                
                if lot_size_filter:
                    step_size = float(lot_size_filter['stepSize'])
                    precision = len(str(step_size).rstrip('0').split('.')[1]) if '.' in str(step_size) else 0
                    return np.floor(quantity * 10**precision) / 10**precision
            
            # Default to 5 decimal places if we can't determine
            return np.floor(quantity * 10**5) / 10**5
            
        except Exception as e:
            logger.warning(f"Error determining quantity precision: {e}")
            # Default to 5 decimal places
            return np.floor(quantity * 10**5) / 10**5

    def calculate_portfolio_value(self):
        """
        Calculate the current portfolio value.
        
        Returns:
            float: Total portfolio value (balance + holdings value)
        """
        # Start with cash balance
        total_value = self.balance
        
        # Add value of all holdings
        for symbol, quantity in self.holdings.items():
            if quantity > 0:
                # Get latest price
                price = self.last_prices.get(symbol)
                if price:
                    total_value += quantity * price
                else:
                    # Try to fetch the current price
                    try:
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                        self.last_prices[symbol] = price
                        total_value += quantity * price
                    except Exception as e:
                        logger.warning(f"Error fetching price for {symbol}: {e}")
        
        return total_value

    def update_equity_history(self):
        """Update the equity history with the current portfolio value."""
        current_value = self.calculate_portfolio_value()
        self.equity_history.append({
            'timestamp': datetime.now().isoformat(),
            'equity': current_value
        })
        
        # Keep only the last 10000 points to prevent the file from growing too large
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-10000:]
            
        logger.info(f"Portfolio value: {current_value} {self.base_currency}")

    def get_performance_metrics(self):
        """
        Calculate and return performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        if not self.equity_history or len(self.equity_history) < 2:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_loss': 0,
                'return_pct': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
        # Extract equity values
        equity_values = [entry['equity'] for entry in self.equity_history]
        initial_equity = equity_values[0]
        current_equity = equity_values[-1]
        
        # Calculate basic metrics
        profit_loss = current_equity - initial_equity
        return_pct = (profit_loss / initial_equity) * 100 if initial_equity > 0 else 0
        
        # Calculate win rate
        if not self.trade_history:
            win_rate = 0
        else:
            # Consider a trade successful if selling at a higher price than buying
            buy_prices = {}
            successful_trades = 0
            total_completed_trades = 0
            
            for trade in self.trade_history:
                symbol = trade['symbol']
                if trade['side'] == 'BUY':
                    buy_prices[symbol] = trade['price']
                elif trade['side'] == 'SELL' and symbol in buy_prices:
                    total_completed_trades += 1
                    if trade['price'] > buy_prices[symbol]:
                        successful_trades += 1
                    buy_prices.pop(symbol)
            
            win_rate = (successful_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0
        
        # Calculate drawdown
        max_equity = max(equity_values)
        max_drawdown = ((max_equity - min(equity_values)) / max_equity) * 100 if max_equity > 0 else 0
        
        # Calculate Sharpe ratio (if we have enough data)
        if len(equity_values) > 30:
            equity_series = pd.Series(equity_values)
            returns = equity_series.pct_change().dropna()
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': len(self.trade_history),
            'win_rate': win_rate,
            'profit_loss': profit_loss,
            'return_pct': return_pct,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def export_results(self, filename=None):
        """
        Export trading results to JSON file.
        
        Args:
            filename (str): Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_results_{timestamp}.json"
            
        results = {
            'config': self.config,
            'performance': self.get_performance_metrics(),
            'equity_history': self.equity_history,
            'trade_history': self.trade_history,
            'final_balance': self.balance,
            'final_holdings': self.holdings,
            'exported_at': datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info(f"Trading results exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

    def run_trading_cycle(self):
        """Run a single trading cycle across all symbols and intervals."""
        for symbol in self.symbols:
            for interval in self.intervals:
                try:
                    # Fetch latest market data
                    df = self.fetch_market_data(symbol, interval)
                    
                    if df.empty:
                        logger.warning(f"No data available for {symbol} at {interval} interval")
                        continue
                    
                    # Generate trading signals
                    signal = self.generate_signals(df)
                    
                    # Get current price
                    current_price = float(df['close'].iloc[-1])
                    
                    # Execute trade based on mode
                    if self.mode == 'paper':
                        self.paper_trade(symbol, signal, current_price)
                    elif self.mode == 'live':
                        self.live_trade(symbol, signal, current_price)
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle for {symbol} at {interval}: {e}")
        
        # Update portfolio value and save state
        self.update_equity_history()
        self.save_state()

    def fetch_suggested_trades(self):
        """
        Fetch suggested trades from the trading dashboard.
        
        Returns:
            list: List of suggested trades with symbol, signal, and confidence
        """
        try:
            # Try to fetch suggested trades from the trading status file
            suggested_trades = []
            
            # Check multiple possible locations for the trading status file
            possible_paths = [
                '/opt/lampp/htdocs/bot/frontend/trading_data/live_trading_status.json',
                '/opt/lampp/htdocs/bot/trading_data/live_trading_status.json',
                '/home/dim/git/Cryptobot/trading_data/live_trading_status.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        data = json.load(f)
                        if 'signals' in data and isinstance(data['signals'], list):
                            for signal in data['signals']:
                                if 'symbol' in signal and 'signal' in signal and 'confidence' in signal:
                                    # Convert symbol format if needed (e.g., BTC/USDT to BTCUSDT)
                                    symbol = signal['symbol'].replace('/', '')
                                    
                                    # Only include BUY or SELL signals (not HOLD)
                                    if signal['signal'] in ['BUY', 'SELL']:
                                        suggested_trades.append({
                                            'symbol': symbol,
                                            'signal': signal['signal'],
                                            'confidence': signal['confidence'],
                                            'price': signal.get('currentPrice', 0)
                                        })
                            break  # Stop after finding valid data
            
            if suggested_trades:
                logger.info(f"Found {len(suggested_trades)} suggested trades")
            return suggested_trades
            
        except Exception as e:
            logger.error(f"Error fetching suggested trades: {e}")
            return []
    
    def check_and_execute_suggested_trades(self):
        """
        Check for suggested trades and execute them if they meet the confidence threshold.
        """
        # Only check for suggested trades at the configured interval
        current_time = time.time()
        if current_time - self.last_suggested_trade_check < self.suggested_trade_refresh_interval:
            return
        
        self.last_suggested_trade_check = current_time
        logger.info("Checking for suggested trades...")
        
        # Fetch suggested trades
        suggested_trades = self.fetch_suggested_trades()
        
        # Execute trades that meet the confidence threshold
        for trade in suggested_trades:
            try:
                symbol = trade['symbol']
                signal_str = trade['signal']
                confidence = trade['confidence']
                price = trade.get('price', 0)
                
                # Skip trades below the confidence threshold
                if confidence < self.min_confidence_threshold:
                    logger.info(f"Skipping {signal_str} for {symbol} - confidence {confidence:.2f} below threshold {self.min_confidence_threshold}")
                    continue
                
                # Convert signal string to integer signal
                signal = 1 if signal_str == 'BUY' else -1 if signal_str == 'SELL' else 0
                
                # Skip HOLD signals
                if signal == 0:
                    continue
                
                # Get current price if not provided
                if price <= 0:
                    try:
                        ticker = self.client.get_symbol_ticker(symbol=symbol)
                        price = float(ticker['price'])
                    except Exception as e:
                        logger.error(f"Error getting price for {symbol}: {e}")
                        continue
                
                logger.info(f"Auto-executing suggested {signal_str} for {symbol} with confidence {confidence:.2f} at price {price}")
                
                # Execute trade based on mode
                if self.mode == 'paper':
                    self.paper_trade(symbol, signal, price, is_suggested=True, confidence=confidence)
                elif self.mode == 'live':
                    self.live_trade(symbol, signal, price, is_suggested=True, confidence=confidence)
                
            except Exception as e:
                logger.error(f"Error executing suggested trade: {e}")
    
    def start(self, interval_seconds=60):
        """
        Start the trading strategy.
        
        Args:
            interval_seconds (int): Seconds between trading cycles
        """
        if self.is_running:
            logger.warning("Trading already running")
            return
            
        self.is_running = True
        logger.info(f"Starting {'paper' if self.mode == 'paper' else 'live'} trading with {len(self.symbols)} symbols")
        
        # Update the suggested trade refresh interval based on the trading interval
        self.suggested_trade_refresh_interval = interval_seconds
        logger.info(f"Setting suggested trade refresh interval to {self.suggested_trade_refresh_interval} seconds")
        
        def trading_loop():
            while self.is_running:
                try:
                    self.run_trading_cycle()
                    # Sleep until next cycle
                    time.sleep(interval_seconds)
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(10)  # Shorter sleep on error
        
        # Start trading in a separate thread
        self.thread = threading.Thread(target=trading_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop the trading strategy."""
        if not self.is_running:
            logger.warning("Trading already stopped")
            return
            
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=10)
        
        logger.info("Trading stopped")
        self.save_state()
        self.export_results()
        
    def switch_mode(self, new_mode):
        """
        Switch between paper and live trading modes.
        
        Args:
            new_mode (str): 'paper' or 'live'
        """
        if new_mode not in ['paper', 'live']:
            logger.error(f"Invalid mode: {new_mode}")
            return
            
        # Check for API keys if switching to live mode
        if new_mode == 'live' and (not self.config.get('api_key') or not self.config.get('api_secret')):
            logger.error("Cannot switch to live mode: API keys not configured")
            return False
            
        was_running = self.is_running
        
        # Stop trading if running
        if was_running:
            self.stop()
            
        # Switch mode
        self.mode = new_mode
        self.config['mode'] = new_mode
        self.save_config()
        self.save_state()  # Ensure we save the state immediately after mode change
        
        logger.info(f"Switched to {new_mode} trading mode")
        
        # Restart if it was running
        if was_running:
            self.start()
            
        return True


if __name__ == "__main__":
    # Load paper trading strategy from config file
    strategy = PaperTradingStrategy(config_file='/opt/lampp/htdocs/bot/frontend/trading_data/trading_config.json')
    
    # Enable auto-execution of suggested trades
    strategy.auto_execute_suggested_trades = True
    strategy.min_confidence_threshold = 0.75  # Only execute trades with 75% or higher confidence
    
    # Save the updated configuration
    strategy.save_config()
    
    # Start paper trading
    strategy.start(interval_seconds=300)  # Check every 5 minutes
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop trading on keyboard interrupt
        strategy.stop()
        logger.info("Trading stopped by user")
