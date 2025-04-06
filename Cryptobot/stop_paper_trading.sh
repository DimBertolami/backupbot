#!/bin/bash
# Helper script to stop the paper trading service from the main stop.sh

# Log function (mimicking the main stop.sh log functions)
log_msg() {
  echo -e "$1"
}

PAPER_TRADING_DIR="/opt/lampp/htdocs/bot"

# Check if the new service script exists
if [ -f "${PAPER_TRADING_DIR}/paper_trading_service.sh" ]; then
  log_msg "Stopping paper trading service using service script..."
  cd "${PAPER_TRADING_DIR}"
  ./paper_trading_service.sh stop
  
  if [ $? -eq 0 ]; then
    log_msg "Paper trading service stopped successfully"
    exit 0
  else
    log_msg "Failed to stop paper trading service with service script, falling back to direct method"
  fi
fi

# Fallback to the original direct method if service script doesn't exist or failed
log_msg "Stopping paper trading using direct method..."

# Find paper trading processes
PAPER_TRADING_PIDS=$(pgrep -f "python.*paper_trading_cli\.py")

if [ -n "$PAPER_TRADING_PIDS" ]; then
  # Run the CLI stop command first for a clean shutdown
  cd "${PAPER_TRADING_DIR}"
  python paper_trading_cli.py stop > /dev/null 2>&1
  
  # Wait a moment for graceful shutdown
  sleep 2
  
  # Kill any remaining processes
  REMAINING_PIDS=$(pgrep -f "python.*paper_trading_cli\.py")
  if [ -n "$REMAINING_PIDS" ]; then
    log_msg "Forcefully stopping paper trading processes: $REMAINING_PIDS"
    kill $REMAINING_PIDS
  fi
  
  log_msg "Paper trading service stopped"
else
  log_msg "No paper trading processes found running"
fi

# Clean up PID file if it exists
if [ -f "${PAPER_TRADING_DIR}/paper_trading.pid" ]; then
  rm "${PAPER_TRADING_DIR}/paper_trading.pid"
fi

exit 0
