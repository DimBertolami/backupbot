#!/usr/bin/env python3
"""
Trading Strategy Demonstration

This script demonstrates how to use the integrated trading strategy with advanced ML models.
It provides examples for both backtesting and real-time trading.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import json

# Import our trading strategy integration
from trading_strategy_integration import IntegratedTradingStrategy
from fetchall import fe_preprocess, fetch_binance_data, calculate_indicators
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("demo_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("demo_trading")


def run_backtest(days=60, initial_balance=10000, model_type=None):
    """
    Run a backtest of the trading strategy on historical data.
    
    Args:
        days (int): Number of days of historical data to use
        initial_balance (float): Initial balance for backtesting
        model_type (str, optional): Specific model type to use. If None, use best model.
        
    Returns:
        tuple: (predictions, trades, plot)
    """
    logger.info(f"Starting backtest with {days} days of data")
    
    # Create integrated strategy
    strategy = IntegratedTradingStrategy()
    
    # Fetch historical data
    try:
        # Try to get data from fe_preprocess first
        data = fe_preprocess(exch="binance")
        
        # If that doesn't work, fall back to direct fetch
        if data is None or len(data) < days:
            logger.info("Falling back to direct data fetch")
            data = fetch_binance_data(limit=days * 24)  # Assuming hourly data
            
            if data is not None:
                # Calculate indicators
                data = calculate_indicators(data)
        
        if data is None or len(data) == 0:
            logger.error("Failed to fetch data for backtesting")
            return None, None, None
        
        logger.info(f"Fetched {len(data)} data points for backtesting")
        
        # Run the backtest
        predictions, trades, plot = strategy.run(data, backtesting=True, initial_balance=initial_balance)
        
        # Save results
        if trades is not None and not trades.empty:
            trades.to_csv("backtest_trades.csv", index=False)
            
            # Calculate performance metrics
            total_trades = len(trades)
            win_trades = len(trades[trades['pnl'] > 0]) if 'pnl' in trades.columns else 0
            loss_trades = len(trades[trades['pnl'] < 0]) if 'pnl' in trades.columns else 0
            win_rate = win_trades / total_trades if total_trades > 0 else 0
            
            if 'pnl' in trades.columns:
                total_pnl = trades['pnl'].sum()
                avg_pnl = trades['pnl'].mean()
                max_win = trades['pnl'].max() if not trades.empty else 0
                max_loss = trades['pnl'].min() if not trades.empty else 0
                
                logger.info(f"Backtest Results:")
                logger.info(f"  Total Trades: {total_trades}")
                logger.info(f"  Win Rate: {win_rate:.2%}")
                logger.info(f"  Total P&L: ${total_pnl:.2f}")
                logger.info(f"  Avg P&L/Trade: ${avg_pnl:.2f}")
                logger.info(f"  Max Win: ${max_win:.2f}")
                logger.info(f"  Max Loss: ${max_loss:.2f}")
                
                # Save performance summary for React frontend
                perf_summary = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model_id": strategy.active_model_id or "traditional",
                    "metrics": {
                        "total_trades": total_trades,
                        "win_rate": win_rate,
                        "total_pnl": float(total_pnl),
                        "avg_pnl": float(avg_pnl),
                        "max_win": float(max_win),
                        "max_loss": float(max_loss),
                        "initial_balance": initial_balance,
                        "final_balance": float(trades.iloc[-1]['balance'] if 'balance' in trades.columns else 0)
                    }
                }
                
                os.makedirs("results", exist_ok=True)
                with open(f"results/backtest_summary.json", "w") as f:
                    json.dump(perf_summary, f, indent=2, default=str)
                
                # Copy to React frontend if available
                react_dir = "/opt/lampp/htdocs/bot/frontend/public/trading_data"
                if os.path.exists("/opt/lampp/htdocs/bot/frontend"):
                    os.makedirs(react_dir, exist_ok=True)
                    import shutil
                    shutil.copy(f"results/backtest_summary.json", f"{react_dir}/backtest_summary.json")
        
        if plot is not None:
            plot.savefig("backtest_results.png", dpi=300, bbox_inches='tight')
            
            # Copy to React frontend if available
            react_dir = "/opt/lampp/htdocs/bot/frontend/public/trading_data"
            if os.path.exists("/opt/lampp/htdocs/bot/frontend"):
                os.makedirs(react_dir, exist_ok=True)
                import shutil
                shutil.copy("backtest_results.png", f"{react_dir}/backtest_results.png")
        
        return predictions, trades, plot
        
    except Exception as e:
        logger.error(f"Error in backtesting: {e}", exc_info=True)
        return None, None, None


def compare_strategies():
    """
    Compare different trading strategies on the same dataset.
    
    Returns:
        matplotlib.figure.Figure: Comparison plot
    """
    logger.info("Starting strategy comparison")
    
    # Fetch data
    try:
        data = fe_preprocess(exch="binance")
        
        if data is None or len(data) == 0:
            logger.error("Failed to fetch data for comparison")
            return None
        
        logger.info(f"Fetched {len(data)} data points for strategy comparison")
        
        # Create strategies to compare
        # 1. Advanced ML strategy
        advanced_strategy = IntegratedTradingStrategy()
        
        # 2. Traditional strategy
        traditional_strategy = IntegratedTradingStrategy()
        traditional_strategy.active_model = None
        traditional_strategy.active_model_id = "traditional_only"
        
        # Run backtests
        advanced_results = advanced_strategy.run(data, backtesting=True, initial_balance=10000)
        traditional_results = traditional_strategy.run(data, backtesting=True, initial_balance=10000)
        
        # Extract trades
        advanced_trades = advanced_results[1] if advanced_results else None
        traditional_trades = traditional_results[1] if traditional_results else None
        
        # Create comparison plot
        if advanced_trades is not None and traditional_trades is not None:
            # Create a new figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Calculate portfolio values over time
            # For advanced strategy
            adv_portfolio = []
            balance = 10000
            position = 0
            entry_price = 0
            
            for i, row in advanced_trades.iterrows():
                if row['action'] == 'buy':
                    position = balance / row['price']
                    entry_price = row['price']
                    balance = 0
                    adv_portfolio.append((row['timestamp'], position * row['price']))
                elif row['action'] == 'sell':
                    balance = position * row['price']
                    position = 0
                    adv_portfolio.append((row['timestamp'], balance))
            
            # For traditional strategy
            trad_portfolio = []
            balance = 10000
            position = 0
            entry_price = 0
            
            for i, row in traditional_trades.iterrows():
                if row['action'] == 'buy':
                    position = balance / row['price']
                    entry_price = row['price']
                    balance = 0
                    trad_portfolio.append((row['timestamp'], position * row['price']))
                elif row['action'] == 'sell':
                    balance = position * row['price']
                    position = 0
                    trad_portfolio.append((row['timestamp'], balance))
            
            # Plot portfolio values
            if adv_portfolio:
                ax.plot([p[0] for p in adv_portfolio], [p[1] for p in adv_portfolio], 
                         label=f"Advanced ML ({advanced_strategy.active_model_id})", color='blue')
            
            if trad_portfolio:
                ax.plot([p[0] for p in trad_portfolio], [p[1] for p in trad_portfolio], 
                         label="Traditional Strategy", color='green')
            
            # Plot initial balance line
            ax.axhline(y=10000, color='gray', linestyle='--', label="Initial Balance")
            
            # Style the plot
            ax.set_title("Trading Strategy Comparison", fontsize=16)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Portfolio Value ($)", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            
            # Add performance metrics
            adv_final = adv_portfolio[-1][1] if adv_portfolio else 10000
            trad_final = trad_portfolio[-1][1] if trad_portfolio else 10000
            
            adv_return = (adv_final / 10000 - 1) * 100
            trad_return = (trad_final / 10000 - 1) * 100
            
            text = (
                f"Advanced Strategy Return: {adv_return:.2f}%\n"
                f"Traditional Strategy Return: {trad_return:.2f}%\n"
                f"Difference: {adv_return - trad_return:.2f}%"
            )
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.02, 0.97, text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig("strategy_comparison.png", dpi=300, bbox_inches='tight')
            
            # Copy to React frontend if available
            react_dir = "/opt/lampp/htdocs/bot/frontend/public/trading_data"
            if os.path.exists("/opt/lampp/htdocs/bot/frontend"):
                os.makedirs(react_dir, exist_ok=True)
                import shutil
                shutil.copy("strategy_comparison.png", f"{react_dir}/strategy_comparison.png")
            
            return fig
        
        logger.warning("No trade data available for comparison")
        return None
        
    except Exception as e:
        logger.error(f"Error in strategy comparison: {e}", exc_info=True)
        return None


def run_live_trading():
    """
    Run the trading strategy in real-time mode.
    Note: This is just a simulation - no real trades are executed.
    
    Returns:
        tuple: (predictions, signals)
    """
    logger.info("Starting live trading simulation")
    
    # Create integrated strategy
    strategy = IntegratedTradingStrategy()
    
    # Fetch latest data
    try:
        data = fe_preprocess(exch="binance")
        
        if data is None or len(data) == 0:
            logger.error("Failed to fetch data for live trading")
            return None, None
        
        logger.info(f"Fetched {len(data)} data points for live trading")
        
        # Get latest predictions
        predictions, _ = strategy.predict(data)
        
        if predictions is None:
            logger.error("Failed to generate predictions")
            return None, None
        
        # Extract latest signals
        latest_signals = predictions.tail(10)[['timestamp', 'close', 'final_signal']].copy()
        latest_signal = latest_signals.iloc[-1]['final_signal']
        
        signal_text = "BUY" if latest_signal == 1 else ("SELL" if latest_signal == -1 else "HOLD")
        
        logger.info(f"Latest signal: {signal_text} at price {latest_signals.iloc[-1]['close']}")
        
        # Save live trading status for React frontend
        live_status = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model_id": strategy.active_model_id or "traditional",
            "latest_signals": latest_signals.to_dict(orient='records'),
            "current_signal": signal_text,
            "current_price": float(latest_signals.iloc[-1]['close'])
        }
        
        os.makedirs("results", exist_ok=True)
        with open(f"results/live_trading_status.json", "w") as f:
            json.dump(live_status, f, indent=2, default=str)
        
        # Copy to React frontend if available
        react_dir = "/opt/lampp/htdocs/bot/frontend/public/trading_data"
        if os.path.exists("/opt/lampp/htdocs/bot/frontend"):
            os.makedirs(react_dir, exist_ok=True)
            import shutil
            shutil.copy(f"results/live_trading_status.json", f"{react_dir}/live_trading_status.json")
        
        # Generate and save plot
        fig = strategy.plot_trading_results(predictions.tail(100))
        
        if fig is not None:
            fig.savefig("live_trading.png", dpi=300, bbox_inches='tight')
            
            # Copy to React frontend if available
            if os.path.exists("/opt/lampp/htdocs/bot/frontend"):
                import shutil
                shutil.copy("live_trading.png", f"{react_dir}/live_trading.png")
        
        return predictions, latest_signals
        
    except Exception as e:
        logger.error(f"Error in live trading: {e}", exc_info=True)
        return None, None


def main():
    """Main function to run the demonstration."""
    parser = argparse.ArgumentParser(description='Trading Strategy Demonstration')
    parser.add_argument('--mode', type=str, default='backtest',
                        choices=['backtest', 'compare', 'live'],
                        help='Trading mode (backtest, compare, live)')
    parser.add_argument('--days', type=int, default=60,
                        help='Number of days for historical data (backtest mode)')
    parser.add_argument('--balance', type=float, default=10000,
                        help='Initial balance for backtesting')
    
    args = parser.parse_args()
    
    print(f"=== Cryptocurrency Trading Bot Demonstration ===")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'backtest':
        print(f"Running backtest with {args.days} days of data...")
        predictions, trades, plot = run_backtest(days=args.days, initial_balance=args.balance)
        
        if trades is not None and not trades.empty and 'pnl' in trades.columns:
            total_pnl = trades['pnl'].sum()
            print(f"\nBacktest Results:")
            print(f"  Total Trades: {len(trades)}")
            print(f"  Win Rate: {len(trades[trades['pnl'] > 0]) / len(trades):.2%}")
            print(f"  Total P&L: ${total_pnl:.2f}")
            print(f"  Return: {total_pnl / args.balance:.2%}")
            print(f"\nSee backtest_results.png for the performance chart")
        else:
            print("No trades were executed in the backtest")
    
    elif args.mode == 'compare':
        print("Comparing trading strategies...")
        fig = compare_strategies()
        
        if fig is not None:
            print("\nStrategy comparison complete")
            print("See strategy_comparison.png for the comparison chart")
        else:
            print("Strategy comparison failed")
    
    elif args.mode == 'live':
        print("Starting live trading simulation...")
        predictions, signals = run_live_trading()
        
        if signals is not None and not signals.empty:
            latest_signal = signals.iloc[-1]['final_signal']
            signal_text = "BUY" if latest_signal == 1 else ("SELL" if latest_signal == -1 else "HOLD")
            
            print(f"\nLive Trading Results:")
            print(f"  Current Price: ${signals.iloc[-1]['close']:.2f}")
            print(f"  Current Signal: {signal_text}")
            print(f"\nSee live_trading.png for the performance chart")
        else:
            print("Live trading simulation failed")
    
    print("\nResults have been exported to the React frontend")
    print("Check /opt/lampp/htdocs/bot/frontend/public/trading_data/ for the latest data")


if __name__ == "__main__":
    main()
