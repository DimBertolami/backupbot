#!/usr/bin/env python3
"""
Trading Database Module

This module manages the SQLite database for storing trading signals,
performance metrics, and bot learning data.
"""

import os
import json
import sqlite3
import datetime
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Database file location
DB_PATH = Path("trading_data.db")

# Output directory for frontend
FRONTEND_DIR = Path("/opt/lampp/htdocs/bot/frontend/public/trading_data")

def init_database() -> None:
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Trading signals table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trading_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        signal TEXT NOT NULL,
        confidence REAL NOT NULL,
        price REAL NOT NULL,
        price_change_24h REAL,
        model_id TEXT
    )
    ''')
    
    # Trading performance table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trading_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        symbol TEXT NOT NULL,
        initial_price REAL NOT NULL,
        final_price REAL NOT NULL,
        profit_loss REAL NOT NULL,
        signal TEXT NOT NULL,
        confidence REAL NOT NULL,
        duration_minutes INTEGER,
        success BOOLEAN
    )
    ''')
    
    # Bot thoughts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS bot_thoughts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        thought_type TEXT NOT NULL, 
        thought_content TEXT NOT NULL,
        symbol TEXT,
        confidence REAL,
        metrics TEXT
    )
    ''')
    
    # Learning metrics table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS learning_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        model_id TEXT NOT NULL,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1_score REAL,
        profit_factor REAL,
        sharpe_ratio REAL,
        win_rate REAL,
        dataset_size INTEGER,
        training_duration REAL
    )
    ''')
    
    conn.commit()
    conn.close()

def save_trading_signals(signals: List[Dict[str, Any]], model_id: str) -> None:
    """Save trading signals to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    
    for signal in signals:
        cursor.execute('''
        INSERT INTO trading_signals 
        (symbol, timestamp, signal, confidence, price, price_change_24h, model_id)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['symbol'],
            timestamp,
            signal['signal'],
            signal['confidence'],
            signal['currentPrice'],
            signal['priceChange24h'],
            model_id
        ))
    
    conn.commit()
    conn.close()

def save_bot_thought(thought_type: str, content: str, symbol: Optional[str] = None, 
                    confidence: Optional[float] = None, metrics: Optional[Dict] = None) -> None:
    """Save a bot thought to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    metrics_json = json.dumps(metrics) if metrics else None
    
    cursor.execute('''
    INSERT INTO bot_thoughts
    (timestamp, thought_type, thought_content, symbol, confidence, metrics)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,
        thought_type,
        content,
        symbol,
        confidence,
        metrics_json
    ))
    
    conn.commit()
    conn.close()
    
    # Also update the bot_thoughts.json file for frontend
    update_bot_thoughts_file()

def update_bot_thoughts_file() -> None:
    """Update the bot_thoughts.json file with recent thoughts for frontend"""
    conn = sqlite3.connect(DB_PATH)
    # Use pandas for easy data handling
    df = pd.read_sql_query('''
    SELECT * FROM bot_thoughts 
    ORDER BY timestamp DESC 
    LIMIT 20
    ''', conn)
    conn.close()
    
    if df.empty:
        return
    
    # Convert metrics from JSON string to dictionary
    df['metrics'] = df['metrics'].apply(lambda x: json.loads(x) if x else None)
    
    # Convert to list of dictionaries
    thoughts = df.to_dict(orient='records')
    
    # Prepare data object
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "thoughts": thoughts
    }
    
    # Save to file for frontend
    os.makedirs(FRONTEND_DIR, exist_ok=True)
    with open(FRONTEND_DIR / "bot_thoughts.json", "w") as f:
        json.dump(data, f, indent=2)

def get_performance_metrics() -> Dict[str, Any]:
    """Calculate performance metrics from trading history"""
    conn = sqlite3.connect(DB_PATH)
    
    # Get overall profit/loss
    cursor = conn.cursor()
    cursor.execute('''
    SELECT SUM(profit_loss) as total_profit, 
           COUNT(*) as total_trades,
           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_trades
    FROM trading_performance
    ''')
    
    result = cursor.fetchone()
    
    if not result or result[1] == 0:  # No trades yet
        conn.close()
        return {
            "total_profit": 0,
            "win_rate": 0,
            "total_trades": 0,
            "avg_profit_per_trade": 0,
            "best_performing_symbol": None,
            "best_success_rate": 0
        }
    
    total_profit, total_trades, successful_trades = result
    win_rate = successful_trades / total_trades if total_trades > 0 else 0
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    
    # Get best performing symbol
    cursor.execute('''
    SELECT symbol, SUM(profit_loss) as symbol_profit, 
           COUNT(*) as symbol_trades,
           SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) / COUNT(*) as success_rate
    FROM trading_performance
    GROUP BY symbol
    ORDER BY symbol_profit DESC
    LIMIT 1
    ''')
    
    best_symbol_result = cursor.fetchone()
    best_symbol = best_symbol_result[0] if best_symbol_result else None
    best_success_rate = best_symbol_result[3] if best_symbol_result else 0
    
    conn.close()
    
    return {
        "total_profit": total_profit,
        "win_rate": win_rate,
        "total_trades": total_trades,
        "avg_profit_per_trade": avg_profit,
        "best_performing_symbol": best_symbol,
        "best_success_rate": best_success_rate
    }

def generate_bot_thoughts(signals: List[Dict[str, Any]], model_id: str) -> List[str]:
    """Generate bot thoughts based on current trading signals and historical performance"""
    thoughts = []
    
    # Get overall performance metrics
    metrics = get_performance_metrics()
    
    # Generate thoughts about overall performance
    if metrics["total_trades"] > 0:
        if metrics["win_rate"] > 0.6:
            thoughts.append(f"My win rate is {metrics['win_rate']:.1%}. I'm getting quite good at this!")
        elif metrics["win_rate"] < 0.4:
            thoughts.append("My success rate isn't great. I need more training data...")
        
        if metrics["total_profit"] > 0:
            thoughts.append(f"Overall profit: ${metrics['total_profit']:.2f}. Making money, but I can do better.")
        else:
            thoughts.append(f"Currently at ${metrics['total_profit']:.2f}. Need to improve my predictions.")
    else:
        thoughts.append("Not enough historical data yet. Learning in progress...")
    
    # Generate thoughts about specific signals
    high_confidence_signals = [s for s in signals if s["confidence"] > 0.8]
    if high_confidence_signals:
        symbol = high_confidence_signals[0]["symbol"]
        signal = high_confidence_signals[0]["signal"]
        confidence = high_confidence_signals[0]["confidence"]
        thoughts.append(f"I'm {confidence:.1%} confident that {symbol} is a {signal} right now.")
    
    # Generate thoughts about market trends
    bullish_count = len([s for s in signals if s["signal"] == "BUY"])
    bearish_count = len([s for s in signals if s["signal"] == "SELL"])
    
    if bullish_count > bearish_count * 2:
        thoughts.append("Market seems bullish overall. More buy signals than sells.")
    elif bearish_count > bullish_count * 2:
        thoughts.append("Market seems bearish. Generating more sell signals than buys.")
    else:
        thoughts.append("Market is mixed. No clear trend direction yet.")
    
    # Generate thought about the model's performance
    thoughts.append(f"Using model: {model_id}. Analyzing price patterns...")
    
    # Save some thoughts to the database
    for thought in thoughts[:3]:  # Only save a subset to avoid database bloat
        save_bot_thought(
            thought_type="trading_analysis",
            content=thought,
            metrics=metrics
        )
    
    return thoughts

# Initialize DB when module is imported
init_database()

if __name__ == "__main__":
    # Test the database functions
    print("Initializing trading database...")
    init_database()
    print("Database initialized successfully.")
