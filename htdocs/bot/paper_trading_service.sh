#!/bin/bash
# Script for managing the paper trading service

# Base directory
BASE_DIR="/opt/lampp/htdocs/bot"
PID_FILE="$BASE_DIR/paper_trading.pid"
STATUS_FILE="$BASE_DIR/frontend/trading_data/backend_status.json"

# Function to update backend status file
update_backend_status() {
  local status=$1
  if [ -f "$STATUS_FILE" ]; then
    # Create a temporary file
    TMP_FILE=$(mktemp)
    
    # Use jq to update the status if jq is available
    if command -v jq >/dev/null 2>&1; then
      jq --arg status "$status" '.services.paper_trading.status = $status' "$STATUS_FILE" > "$TMP_FILE"
      mv "$TMP_FILE" "$STATUS_FILE"
    else
      # Fallback to sed if jq is not available
      sed -i 's/"status": "[^"]*"/"status": "'$status'"/' "$STATUS_FILE"
    fi
    
    echo "Updated backend status file - Paper trading status: $status"
  else
    echo "Backend status file not found at $STATUS_FILE"
  fi
}

case "$1" in
  start)
    echo "Starting paper trading service..."
    python "$BASE_DIR/paper_trading_daemon.py" start
    
    # Update backend status
    update_backend_status "active"
    ;;
    
  stop)
    echo "Stopping paper trading service..."
    python "$BASE_DIR/paper_trading_daemon.py" stop
    
    # Update backend status
    update_backend_status "inactive"
    ;;
    
  status)
    python "$BASE_DIR/paper_trading_daemon.py" status
    ;;
    
  restart)
    $0 stop
    sleep 2
    $0 start
    ;;
    
  *)
    echo "Usage: $0 {start|stop|status|restart}"
    exit 1
    ;;
esac

exit 0
