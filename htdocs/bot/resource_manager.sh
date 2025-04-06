#!/bin/bash
# Resource Manager for Crypto Bot
# This script controls CPU and memory usage for all bot processes
# and can automatically start the backend and frontend

# Configuration
MAX_CPU_PERCENT=70
MAX_MEM_PERCENT=70
CHECK_INTERVAL=10  # Seconds between checks
AUTO_START=true    # Automatically start backend and frontend

# Base directories from project structure
BOT_DIR="/opt/lampp/htdocs/bot"
BACKEND_DIR="/home/dim/git/Cryptobot"
FRONTEND_DIR="$BOT_DIR/frontend"
LOG_FILE="$BOT_DIR/logs/resource_manager.log"

# Process tracking files
BACKEND_PID_FILE="$BOT_DIR/backend.pid"
FRONTEND_PID_FILE="$BOT_DIR/frontend.pid"

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Log function
log_message() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
  echo "$1"
}

# Get system information
get_system_info() {
  # Get CPU cores for proper percentage calculation
  CPU_CORES=$(nproc)
  # Total memory in KB
  TOTAL_MEM=$(grep MemTotal /proc/meminfo | awk '{print $2}')
  
  log_message "System has $CPU_CORES CPU cores and $(($TOTAL_MEM / 1024)) MB total memory"
  log_message "Resource limits: CPU ${MAX_CPU_PERCENT}%, Memory ${MAX_MEM_PERCENT}%"
}

# Apply CPU limits to running processes
apply_cpu_limits() {
  # Find all processes related to the bot and Windsurf
  BACKEND_PIDS=$(pgrep -f "python.*$BACKEND_DIR" | tr '\n' ' ')
  PAPER_TRADING_PIDS=$(pgrep -f "python.*$BOT_DIR" | tr '\n' ' ')
  WINDSURF_PIDS=$(pgrep -f "windsurf" | tr '\n' ' ')
  
  if [ -z "$BACKEND_PIDS$PAPER_TRADING_PIDS$WINDSURF_PIDS" ]; then
    log_message "No bot or Windsurf processes found running"
    return
  fi
  
  # Calculate per-process limit based on total allowed percentage and number of processes
  PROCESS_COUNT=$(echo $BACKEND_PIDS $PAPER_TRADING_PIDS $WINDSURF_PIDS | wc -w)
  
  if [ $PROCESS_COUNT -gt 0 ]; then
    # Convert to per-core percentage for cpulimit
    PER_PROCESS_LIMIT=$(( (MAX_CPU_PERCENT * CPU_CORES) / PROCESS_COUNT ))
    
    log_message "Setting CPU limit of ${PER_PROCESS_LIMIT}% per process across $PROCESS_COUNT processes"
    
    # Apply CPU limits if cpulimit is available
    if [[ "$HAS_CPULIMIT" == "true" ]]; then
      # Apply limits to backend processes
      for PID in $BACKEND_PIDS $PAPER_TRADING_PIDS $WINDSURF_PIDS; do
        # Check if cpulimit is already running for this process
        if ! pgrep -f "cpulimit.*$PID" > /dev/null; then
          log_message "Applying CPU limit to PID $PID"
          cpulimit -p $PID -l $PER_PROCESS_LIMIT -b &
        fi
      done
    else
      # Without cpulimit, use nice values as fallback
      for PID in $BACKEND_PIDS $PAPER_TRADING_PIDS $WINDSURF_PIDS; do
        log_message "Setting nice value for PID $PID"
        renice 10 -p $PID > /dev/null 2>&1
      done
    fi
  fi
}

# Apply memory limits using process priority as we can't rely on root access
apply_memory_limits() {
  # Calculate maximum memory allowed in KB
  MAX_MEM_KB=$(( TOTAL_MEM * MAX_MEM_PERCENT / 100 ))
  
  log_message "Setting maximum memory target to $(( MAX_MEM_KB / 1024 )) MB"
  
  # Find all processes related to the bot and Windsurf
  BACKEND_PIDS=$(pgrep -f "python.*$BACKEND_DIR" | tr '\n' ' ')
  PAPER_TRADING_PIDS=$(pgrep -f "python.*$BOT_DIR" | tr '\n' ' ')
  FRONTEND_PIDS=$(pgrep -f "node.*$FRONTEND_DIR" | tr '\n' ' ')
  WINDSURF_PIDS=$(pgrep -f "windsurf" | tr '\n' ' ')
  
  # Use nice to lower priority for all bot processes
  for PID in $BACKEND_PIDS $PAPER_TRADING_PIDS $FRONTEND_PIDS $WINDSURF_PIDS; do
    if ps -p $PID > /dev/null; then
      log_message "Setting nice value for PID $PID to reduce resource usage"
      renice 10 -p $PID > /dev/null 2>&1
    fi
  done
  
  # For high memory usage, we'll simply monitor and alert
  TOTAL_USAGE_KB=0
  for PID in $BACKEND_PIDS $PAPER_TRADING_PIDS $FRONTEND_PIDS $WINDSURF_PIDS; do
    if ps -p $PID > /dev/null; then
      MEM_USAGE=$(ps -o rss= -p $PID)
      TOTAL_USAGE_KB=$((TOTAL_USAGE_KB + MEM_USAGE))
    fi
  done
  
  log_message "Current bot and Windsurf memory usage: $(( TOTAL_USAGE_KB / 1024 )) MB"
  
  if [ $TOTAL_USAGE_KB -gt $MAX_MEM_KB ]; then
    log_message "WARNING: Memory usage exceeds target limit"
  fi
}

# Monitor resource usage and adjust if needed
monitor_resources() {
  log_message "Starting resource monitoring..."
  
  while true; do
    # Get current CPU and memory usage percentage
    CURRENT_CPU=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    CURRENT_MEM=$(free | grep Mem | awk '{print $3/$2 * 100.0}')
    
    # Convert to integers for comparison
    CURRENT_CPU_INT=${CURRENT_CPU%.*}
    CURRENT_MEM_INT=${CURRENT_MEM%.*}
    
    log_message "Current usage: CPU ${CURRENT_CPU_INT}%, Memory ${CURRENT_MEM_INT}%"
    
    # Apply limits if usage exceeds thresholds
    if [ "$CURRENT_CPU_INT" -gt "$MAX_CPU_PERCENT" ]; then
      log_message "CPU usage too high (${CURRENT_CPU_INT}%), applying limits"
      apply_cpu_limits
    fi
    
    if [ "$CURRENT_MEM_INT" -gt "$MAX_MEM_PERCENT" ]; then
      log_message "Memory usage too high (${CURRENT_MEM_INT}%), applying limits"
      apply_memory_limits
    fi
    
    # Check if services are running once per minute
    if (( $(date +%s) % 60 < $CHECK_INTERVAL )); then
      auto_start_services
    fi
    
    # Wait before next check
    sleep $CHECK_INTERVAL
  done
}

# Check if cpulimit is installed
check_dependencies() {
  if ! command -v cpulimit &> /dev/null; then
    log_message "WARNING: cpulimit not found. CPU limiting will be less effective."
    log_message "TIP: Install cpulimit for better CPU control: sudo apt-get install -y cpulimit"
    HAS_CPULIMIT=false
  else
    HAS_CPULIMIT=true
    log_message "cpulimit is available for CPU limiting"
  fi
}

# Start the backend service
start_backend() {
  log_message "Starting backend server..."
  
  # Check if backend is already running
  if is_backend_running; then
    log_message "Backend is already running"
    return 0
  fi
  
  # Start the backend
  cd "$BACKEND_DIR"
  python paper_trading_api.py > "$BOT_DIR/backend.log" 2>&1 &
  BACKEND_PID=$!
  echo $BACKEND_PID > "$BACKEND_PID_FILE"
  
  # Wait for backend to come up
  log_message "Waiting for backend to start..."
  MAX_RETRIES=15
  for ((i=1; i<=MAX_RETRIES; i++)); do
    if curl -s "http://localhost:5001/trading/status" > /dev/null 2>&1; then
      log_message "Backend started successfully (PID: $BACKEND_PID)"
      return 0
    fi
    sleep 2
    
    # If backend process died, try with python3
    if [[ $i -eq 8 ]] && ! ps -p $BACKEND_PID > /dev/null; then
      log_message "Backend process died, trying with python3..."
      python3 paper_trading_api.py > "$BOT_DIR/backend.log" 2>&1 &
      BACKEND_PID=$!
      echo $BACKEND_PID > "$BACKEND_PID_FILE"
    fi
  done
  
  log_message "WARNING: Backend failed to start or respond"
  return 1
}

# Start the frontend service
start_frontend() {
  log_message "Starting frontend server..."
  
  # Check if frontend is already running
  if is_frontend_running; then
    log_message "Frontend is already running"
    return 0
  fi
  
  # Start the frontend
  cd "$FRONTEND_DIR"
  npm run dev > "$BOT_DIR/frontend.log" 2>&1 &
  FRONTEND_PID=$!
  echo $FRONTEND_PID > "$FRONTEND_PID_FILE"
  
  # Wait for frontend to come up
  log_message "Waiting for frontend to start..."
  sleep 5
  
  # Check if the frontend process is still running
  if ! ps -p $FRONTEND_PID > /dev/null; then
    log_message "WARNING: Frontend process died during startup"
    return 1
  fi
  
  # Find which port the frontend is running on
  FRONTEND_PORT=""
  for PORT in {5173..5179}; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
      FRONTEND_PORT=$PORT
      break
    fi
  done
  
  if [ -n "$FRONTEND_PORT" ]; then
    log_message "Frontend started successfully on port $FRONTEND_PORT (PID: $FRONTEND_PID)"
    return 0
  else
    log_message "WARNING: Frontend started but couldn't determine port"
    return 1
  fi
}

# Check if backend is running
is_backend_running() {
  # First check if there's a PID file
  if [ -f "$BACKEND_PID_FILE" ]; then
    PID=$(cat "$BACKEND_PID_FILE")
    if ps -p "$PID" > /dev/null && curl -s "http://localhost:5001/trading/status" > /dev/null 2>&1; then
      return 0  # Backend is running
    fi
  fi
  
  # Check if anything is listening on backend port
  if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null && curl -s "http://localhost:5001/trading/status" > /dev/null 2>&1; then
    return 0  # Backend is running on expected port
  fi
  
  return 1  # Backend is not running
}

# Check if frontend is running
is_frontend_running() {
  # First check if there's a PID file
  if [ -f "$FRONTEND_PID_FILE" ]; then
    PID=$(cat "$FRONTEND_PID_FILE")
    if ps -p "$PID" > /dev/null; then
      return 0  # Frontend is running
    fi
  fi
  
  # Check if anything is listening on frontend ports
  for PORT in {5173..5179}; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null; then
      return 0  # Frontend is running on one of the expected ports
    fi
  done
  
  return 1  # Frontend is not running
}

# Autostart the backend and frontend if not already running
auto_start_services() {
  if [[ "$AUTO_START" != "true" ]]; then
    return
  fi
  
  log_message "Performing auto-start check..."
  
  # Start backend if not running
  if ! is_backend_running; then
    log_message "Backend not detected, starting it..."
    start_backend
  else
    log_message "Backend already running"
  fi
  
  # Start frontend if not running
  if ! is_frontend_running; then
    log_message "Frontend not detected, starting it..."
    start_frontend
  else
    log_message "Frontend already running"
  fi
}

# Start the resource manager
start_manager() {
  log_message "=== Starting Crypto Bot Resource Manager ==="
  check_dependencies
  get_system_info
  
  # Auto-start services if configured
  if [[ "$AUTO_START" == "true" ]]; then
    auto_start_services
  fi
  
  # Begin resource monitoring
  monitor_resources
}

# Run as a daemon
start_manager </dev/null &>> "$LOG_FILE" &
echo $! > "$BOT_DIR/resource_manager.pid"
log_message "Resource manager started with PID $(cat "$BOT_DIR/resource_manager.pid")"
