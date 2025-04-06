#!/usr/bin/env python3
"""
Paper Trading CLI for Binance
This script provides a command-line interface to start, stop, and manage paper trading.
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from strategies.paper_trading import PaperTradingStrategy

def get_status(strategy):
    """Get and print strategy status."""
    status = "Running" if strategy.is_running else "Stopped"
    mode = strategy.mode
    balance = f"{strategy.balance:.2f} {strategy.base_currency}"
    
    portfolio_value = strategy.calculate_portfolio_value()
    holdings_value = portfolio_value - strategy.balance
    metrics = strategy.get_performance_metrics()
    
    print(f"\n--- Paper Trading Status ---")
    print(f"Status: {status}")
    print(f"Mode: {mode.upper()}")
    print(f"Balance: {balance}")
    print(f"Holdings Value: {holdings_value:.2f} {strategy.base_currency}")
    print(f"Total Portfolio: {portfolio_value:.2f} {strategy.base_currency}")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit/Loss: {metrics['profit_loss']:.2f} {strategy.base_currency} ({metrics['return_pct']:.2f}%)")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Display auto-execution settings
    auto_execute = "ENABLED" if hasattr(strategy, 'auto_execute_suggested_trades') and strategy.auto_execute_suggested_trades else "DISABLED"
    print(f"\nAuto-Execute Suggested Trades: {auto_execute}")
    if hasattr(strategy, 'auto_execute_suggested_trades') and strategy.auto_execute_suggested_trades:
        print(f"Min Confidence Threshold: {strategy.min_confidence_threshold:.2f}")
        print(f"Refresh Interval: {strategy.suggested_trade_refresh_interval} seconds")
    
    print("\nSymbols:")
    for symbol in strategy.symbols:
        print(f"  - {symbol}: {strategy.holdings.get(symbol, 0):.8f} (Last: {strategy.last_prices.get(symbol, 'N/A')})")
    
    print("\nRecent trades:")
    for trade in strategy.trade_history[-5:]:
        print(f"  {trade['timestamp']} | {trade['side']} {trade['quantity']} {trade['symbol']} @ {trade['price']}")
    
    print("")

def save_api_keys(config_file, api_key, api_secret):
    """Save API keys to config file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        config['api_key'] = api_key
        config['api_secret'] = api_secret
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"API keys saved to {config_file}")
    except Exception as e:
        print(f"Error saving API keys: {e}")

def main():
    parser = argparse.ArgumentParser(description='Paper Trading CLI for Binance')
    
    # Define commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start paper trading')
    start_parser.add_argument('--interval', type=int, default=300, help='Trading cycle interval in seconds')
    
    # Stop command
    subparsers.add_parser('stop', help='Stop paper trading')
    
    # Status command
    subparsers.add_parser('status', help='Get trading status')
    
    # Switch mode command
    switch_parser = subparsers.add_parser('switch', help='Switch between paper and live trading')
    switch_parser.add_argument('mode', choices=['paper', 'live'], help='Trading mode')
    
    # Reset command
    subparsers.add_parser('reset', help='Reset paper trading account to initial state')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export trading results')
    export_parser.add_argument('--filename', type=str, help='Output filename')
    
    # API keys command
    api_parser = subparsers.add_parser('api', help='Set Binance API keys')
    api_parser.add_argument('key', help='Binance API key')
    api_parser.add_argument('secret', help='Binance API secret')
    
    # Auto-execute command
    auto_execute_parser = subparsers.add_parser('auto-execute', help='Enable/disable auto-execution of suggested trades')
    auto_execute_parser.add_argument('--enabled', type=str, choices=['true', 'false'], required=True, 
                                    help='Enable or disable auto-execution of suggested trades')
    auto_execute_parser.add_argument('--confidence', type=float, default=0.75, 
                                    help='Minimum confidence threshold for auto-execution (0.0-1.0)')
    auto_execute_parser.add_argument('--interval', type=int, default=60, 
                                    help='Refresh interval for checking suggested trades in seconds')
    
    # Execute trade command
    execute_trade_parser = subparsers.add_parser('execute-trade', help='Execute a single trade')
    execute_trade_parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTCUSDT)')
    execute_trade_parser.add_argument('--side', type=str, choices=['BUY', 'SELL'], required=True, help='Trade side')
    execute_trade_parser.add_argument('--price', type=float, help='Trade price (optional, will use current market price if not provided)')
    execute_trade_parser.add_argument('--quantity', type=float, help='Trade quantity (optional)')
    execute_trade_parser.add_argument('--confidence', type=float, default=0.75, help='Trade confidence score')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             'frontend/trading_data/trading_config.json')
    
    # Initialize strategy
    strategy = PaperTradingStrategy(config_file=config_file)
    
    # Process commands
    if args.command == 'start':
        if strategy.is_running:
            print("Trading is already running")
        else:
            print(f"Starting {strategy.mode} trading with {len(strategy.symbols)} symbols")
            print(f"Trading cycle interval: {args.interval} seconds")
            strategy.start(interval_seconds=args.interval)
            get_status(strategy)
    
    elif args.command == 'stop':
        if not strategy.is_running:
            print("Trading is already stopped")
        else:
            print("Stopping trading...")
            strategy.stop()
            print("Trading stopped")
    
    elif args.command == 'status':
        get_status(strategy)
    
    elif args.command == 'switch':
        if args.mode == strategy.mode:
            print(f"Already in {args.mode} mode")
        else:
            # Check if API keys are set for live mode
            if args.mode == 'live' and (not strategy.config.get('api_key') or not strategy.config.get('api_secret')):
                print("API keys not set. Please set API keys first using the 'api' command.")
            else:
                print(f"Switching to {args.mode} mode...")
                strategy.switch_mode(args.mode)
                get_status(strategy)
    
    elif args.command == 'reset':
        confirmation = input("Are you sure you want to reset the paper trading account? (y/n): ")
        if confirmation.lower() == 'y':
            print("Resetting account...")
            was_running = strategy.is_running
            if was_running:
                strategy.stop()
            strategy.reset_account()
            if was_running:
                strategy.start()
            print("Account reset to initial state")
            get_status(strategy)
        else:
            print("Reset cancelled")
    
    elif args.command == 'export':
        filename = args.filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_results_{timestamp}.json"
        strategy.export_results(filename)
        print(f"Results exported to {filename}")
    
    elif args.command == 'api':
        save_api_keys(config_file, args.key, args.secret)
        print("To use these keys for live trading, switch to live mode with: python paper_trading_cli.py switch live")
    
    elif args.command == 'auto-execute':
        enabled = args.enabled.lower() == 'true'
        confidence = max(0.0, min(1.0, args.confidence))  # Ensure confidence is between 0 and 1
        interval = max(1, args.interval)  # Ensure interval is at least 1 second
        
        # Update strategy settings
        strategy.auto_execute_suggested_trades = enabled
        strategy.min_confidence_threshold = confidence
        strategy.suggested_trade_refresh_interval = interval
        
        # Save the updated configuration
        strategy.config['auto_execute_suggested_trades'] = enabled
        strategy.config['min_confidence_threshold'] = confidence
        strategy.config['suggested_trade_refresh_interval'] = interval
        strategy.save_config()
        
        # Save the updated state
        strategy.save_state()
        
        print(f"Auto-execution of suggested trades {'ENABLED' if enabled else 'DISABLED'}")
        if enabled:
            print(f"Minimum confidence threshold: {confidence:.2f}")
            print(f"Refresh interval: {interval} seconds")
        
        get_status(strategy)
        
    elif args.command == 'execute-trade':
        # Print all arguments for debugging
        print(f"Execute trade command received with args: {vars(args)}")
        
        symbol = args.symbol
        side = args.side
        confidence = args.confidence
        
        # Get price - if not provided, use current market price or a default
        if hasattr(args, 'price') and args.price is not None:
            try:
                price = float(args.price)
                print(f"Using provided price for {symbol}: {price}")
            except (ValueError, TypeError):
                print(f"Invalid price value: {args.price}, using default price")
                price = 100.0
        else:
            # Use last known price or a default price
            if symbol in strategy.last_prices and strategy.last_prices[symbol] is not None:
                price = strategy.last_prices[symbol]
                print(f"Using last known price for {symbol}: {price}")
            else:
                # Use a mock price for testing
                mock_prices = {
                    'BTCUSDT': 52768.34,
                    'ETHUSDT': 3164.56,
                    'SOLUSDT': 148.92,
                    'ADAUSDT': 0.52,
                    'DOGEUSDT': 0.15,
                    'BNBUSDT': 610.23
                }
                
                # Remove any / in the symbol
                clean_symbol = symbol.replace('/', '')
                
                if clean_symbol in mock_prices:
                    price = mock_prices[clean_symbol]
                else:
                    # Default price if symbol not found
                    price = 100.0
                print(f"Using mock price for {symbol}: {price}")
        
        # Default quantity calculation if not provided
        if hasattr(args, 'quantity') and args.quantity is not None:
            try:
                quantity = float(args.quantity)
                print(f"Using provided quantity: {quantity}")
            except (ValueError, TypeError):
                print(f"Invalid quantity value: {args.quantity}, calculating default quantity")
                # Calculate default quantity
                if side == 'BUY':
                    trade_amount = strategy.balance * 0.05
                    quantity = trade_amount / price
                else:
                    if symbol in strategy.holdings:
                        quantity = strategy.holdings[symbol] * 0.5
                    else:
                        print(f"Cannot sell {symbol}: no holdings found")
                        return
        else:
            # Calculate quantity based on a percentage of available balance
            if side == 'BUY':
                # Use 5% of available balance for each trade
                trade_amount = strategy.balance * 0.05
                quantity = trade_amount / price
                print(f"Calculated quantity for BUY: {quantity} (5% of balance)")
            else:
                # For sell, check if we have the asset
                if symbol in strategy.holdings:
                    # Sell 50% of holdings
                    quantity = strategy.holdings[symbol] * 0.5
                    print(f"Calculated quantity for SELL: {quantity} (50% of holdings)")
                else:
                    print(f"Cannot sell {symbol}: no holdings found")
                    return
        
        # Execute the trade
        try:
            from datetime import datetime
            
            # Ensure price and quantity are valid numbers
            if not isinstance(price, (int, float)) or price <= 0:
                print(f"Invalid price: {price}. Must be a positive number.")
                return
                
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                print(f"Invalid quantity: {quantity}. Must be a positive number.")
                return
            
            # Format the trade for logging
            trade = {
                'symbol': symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence
            }
            
            print(f"Executing trade: {trade}")
            
            # Ensure the symbol is in the correct format (no / character)
            clean_symbol = symbol.replace('/', '')
            
            # Update holdings and balance
            if side == 'BUY':
                cost = price * quantity
                if cost > strategy.balance:
                    print(f"Insufficient balance for trade: {cost} > {strategy.balance}")
                    return
                
                # Update balance
                strategy.balance -= cost
                
                # Update holdings
                if clean_symbol in strategy.holdings:
                    strategy.holdings[clean_symbol] += quantity
                else:
                    strategy.holdings[clean_symbol] = quantity
                    
                print(f"Bought {quantity} {clean_symbol} at {price} for {cost} {strategy.base_currency}")
                
            elif side == 'SELL':
                if clean_symbol not in strategy.holdings or strategy.holdings[clean_symbol] < quantity:
                    print(f"Insufficient holdings for trade: {quantity} > {strategy.holdings.get(clean_symbol, 0)}")
                    return
                
                # Calculate proceeds
                proceeds = price * quantity
                
                # Update balance and holdings
                strategy.balance += proceeds
                strategy.holdings[clean_symbol] -= quantity
                
                # Remove symbol from holdings if quantity is zero
                if strategy.holdings[clean_symbol] <= 0:
                    del strategy.holdings[clean_symbol]
                    
                print(f"Sold {quantity} {clean_symbol} at {price} for {proceeds} {strategy.base_currency}")
            
            # Update the trade with the clean symbol
            trade['symbol'] = clean_symbol
            
            # Add to trade history
            strategy.trade_history.append(trade)
            
            # Update last prices
            strategy.last_prices[clean_symbol] = price
            
            # Save the updated state
            strategy.save_state()
            
            # Show current status
            get_status(strategy)
            
        except Exception as e:
            print(f"Error executing trade: {str(e)}")
            import traceback
            traceback.print_exc()
    
    else:
        # Default to showing status and help
        get_status(strategy)
        parser.print_help()
    
    # Keep the script running if strategy is running
    if strategy.is_running:
        try:
            print("\nPress Ctrl+C to exit...\n")
            while strategy.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping trading and exiting...")
            strategy.stop()
            print("Done")

if __name__ == "__main__":
    main()
