#!/bin/bash
# Script to stop paper trading background process

if [ -f /opt/lampp/htdocs/bot/paper_trading.pid ]; then
    PID=$(cat /opt/lampp/htdocs/bot/paper_trading.pid)
    if ps -p $PID > /dev/null; then
        echo "Stopping paper trading process (PID: $PID)"
        kill $PID
        # Run the stop command to ensure proper shutdown
        python /opt/lampp/htdocs/bot/paper_trading_cli.py stop
    else
        echo "Paper trading process not running"
    fi
    rm -f /opt/lampp/htdocs/bot/paper_trading.pid
else
    # Try to find and kill any running paper trading processes
    echo "Searching for paper trading processes..."
    PIDS=$(pgrep -f "python.*paper_trading_cli.py start")
    if [ -n "$PIDS" ]; then
        echo "Found paper trading processes: $PIDS"
        echo "Stopping all paper trading processes"
        pkill -f "python.*paper_trading_cli.py start"
        # Run the stop command to ensure proper shutdown
        python /opt/lampp/htdocs/bot/paper_trading_cli.py stop
    else
        echo "No paper trading processes found"
    fi
fi

echo "Paper trading stopped"
