#!/bin/bash
# Script to start paper trading in the background

# Create logs directory if it doesn't exist
mkdir -p /opt/lampp/htdocs/bot/logs

# Kill any existing paper trading processes
pkill -f "python.*paper_trading_cli.py start" || true

# Start paper trading in the background
nohup python /opt/lampp/htdocs/bot/paper_trading_cli.py start > /opt/lampp/htdocs/bot/logs/paper_trading.log 2>&1 &

# Store the PID in a file for future reference
echo $! > /opt/lampp/htdocs/bot/paper_trading.pid

echo "Paper trading started in background. Check logs at /opt/lampp/htdocs/bot/logs/paper_trading.log"
echo "Process ID: $(cat /opt/lampp/htdocs/bot/paper_trading.pid)"
