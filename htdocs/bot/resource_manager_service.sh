#!/bin/bash
# Resource Manager Service Controller
# Controls the resource manager daemon and auto-starts backend/frontend

# Base directories
BOT_DIR="/opt/lampp/htdocs/bot"
RESOURCE_MANAGER="$BOT_DIR/resource_manager.sh"
RESOURCE_MANAGER_PID="$BOT_DIR/resource_manager.pid"
LOG_FILE="$BOT_DIR/logs/resource_manager.log"

# Ensure log directory exists
mkdir -p "$BOT_DIR/logs"

# Function to check if resource manager is running
is_running() {
  if [ -f "$RESOURCE_MANAGER_PID" ]; then
    PID=$(cat "$RESOURCE_MANAGER_PID")
    if ps -p "$PID" > /dev/null; then
      return 0  # Running
    fi
  fi
  return 1  # Not running
}

# Check if backend is running
is_backend_running() {
  if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null && curl -s "http://localhost:5001/trading/status" > /dev/null 2>&1; then
    return 0  # Running
  fi
  return 1  # Not running
}

# Check if frontend is running
is_frontend_running() {
  for PORT in {5173..5179}; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
      return 0  # Running
    fi
  done
  return 1  # Not running
}

# Start the resource manager
start_manager() {
  if is_running; then
    echo "Resource manager is already running"
    return
  fi
  
  echo "Starting resource manager..."
  "$RESOURCE_MANAGER" &
  
  # Wait a moment to verify it started
  sleep 2
  if is_running; then
    echo "Resource manager started successfully"
  else
    echo "Failed to start resource manager. Check $LOG_FILE for details."
  fi
}

# Stop the resource manager
stop_manager() {
  if ! is_running; then
    echo "Resource manager is not running"
    return
  fi
  
  echo "Stopping resource manager..."
  PID=$(cat "$RESOURCE_MANAGER_PID")
  
  # Kill main process and any child cpulimit processes
  pkill -P "$PID" 2>/dev/null
  kill "$PID" 2>/dev/null
  
  # Clean up PID file
  rm -f "$RESOURCE_MANAGER_PID"
  
  echo "Resource manager stopped"
}

# Show resource manager status
status_manager() {
  echo "==== System Status ===="
  
  # Resource manager status
  if is_running; then
    PID=$(cat "$RESOURCE_MANAGER_PID")
    echo -e "\033[0;32m✓ Resource manager is running (PID: $PID)\033[0m"
  else
    echo -e "\033[0;31m✗ Resource manager is not running\033[0m"
  fi
  
  # Backend status
  if is_backend_running; then
    BACKEND_PID=$(lsof -t -i:5001 2>/dev/null)
    echo -e "\033[0;32m✓ Backend is running (PID: $BACKEND_PID)\033[0m"
  else
    echo -e "\033[0;31m✗ Backend is not running\033[0m"
  fi
  
  # Frontend status
  FRONTEND_PORT=""
  FRONTEND_PID=""
  for PORT in {5173..5179}; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
      FRONTEND_PORT=$PORT
      FRONTEND_PID=$(lsof -t -i:$PORT 2>/dev/null)
      break
    fi
  done
  
  if [ -n "$FRONTEND_PORT" ]; then
    echo -e "\033[0;32m✓ Frontend is running on port $FRONTEND_PORT (PID: $FRONTEND_PID)\033[0m"
  else
    echo -e "\033[0;31m✗ Frontend is not running\033[0m"
  fi
  
  echo -e "\nCurrent system resource usage:"
  echo "CPU: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')%"
  echo "Memory: $(free | grep Mem | awk '{print $3/$2 * 100.0}')%"
  echo -e "\nRecent log entries:"
  tail -n 10 "$LOG_FILE"
  
  echo -e "\nAccess the application at: http://localhost:$FRONTEND_PORT"
}

# Display usage if no arguments
if [ $# -eq 0 ]; then
  echo "Usage: $0 {start|stop|restart|status|autostart}"
  echo "Commands:"
  echo "  start      - Start the resource manager"
  echo "  stop       - Stop the resource manager"
  echo "  restart    - Restart the resource manager"
  echo "  status     - Show status of resource manager and services"
  echo "  autostart  - Configure autostart setting (on/off)"
  exit 1
fi

# Configure autostart
configure_autostart() {
  if [ "$1" = "on" ]; then
    sed -i 's/AUTO_START=false/AUTO_START=true/' "$RESOURCE_MANAGER"
    echo "Autostart enabled - resource manager will automatically start backend and frontend"
  elif [ "$1" = "off" ]; then
    sed -i 's/AUTO_START=true/AUTO_START=false/' "$RESOURCE_MANAGER"
    echo "Autostart disabled - resource manager will only monitor resources"
  else
    echo "Usage: $0 autostart {on|off}"
    exit 1
  fi
}

# Process command line arguments
case "$1" in
  start)
    start_manager
    ;;
  stop)
    stop_manager
    ;;
  restart)
    stop_manager
    sleep 2
    start_manager
    ;;
  status)
    status_manager
    ;;
  autostart)
    if [ $# -lt 2 ]; then
      echo "Usage: $0 autostart {on|off}"
      exit 1
    fi
    configure_autostart "$2"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|autostart}"
    exit 1
    ;;
esac

exit 0
