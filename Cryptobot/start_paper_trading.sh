#!/bin/bash
# Helper script to start the paper trading service from the main startup.sh

# Log function (mimicking the main startup.sh log functions)
log_msg() {
  echo -e "$1"
}

PAPER_TRADING_DIR="/opt/lampp/htdocs/bot"
PAPER_TRADING_LOG_DIR="${PAPER_TRADING_DIR}/logs"

# Create logs directory if it doesn't exist
mkdir -p "${PAPER_TRADING_LOG_DIR}"

# Check if the new service script exists
if [ -f "${PAPER_TRADING_DIR}/paper_trading_service.sh" ]; then
  log_msg "Starting paper trading service using service script..."
  cd "${PAPER_TRADING_DIR}"
  ./paper_trading_service.sh start
  
  if [ $? -eq 0 ]; then
    log_msg "Paper trading service started successfully"
    exit 0
  else
    log_msg "Failed to start paper trading service with service script, falling back to direct method"
  fi
fi

# Fallback to the original direct method if service script doesn't exist or failed
log_msg "Starting paper trading using direct method..."
cd "${PAPER_TRADING_DIR}"
nohup python paper_trading_cli.py start > "${PAPER_TRADING_LOG_DIR}/paper_trading.log" 2>&1 &
PAPER_TRADING_PID=$!
log_msg "Paper trading service started with PID: ${PAPER_TRADING_PID}"

# Store PID in a file for future reference
echo "${PAPER_TRADING_PID}" > "${PAPER_TRADING_DIR}/paper_trading.pid"

exit 0
