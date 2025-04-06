#!/bin/bash
# Comprehensive Backend Troubleshooting Script
# This script automatically detects and fixes issues with the backend
# including the paper trading module

# Color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Base directories
BOT_DIR="/opt/lampp/htdocs/bot"
BACKEND_DIR="/home/dim/git/Cryptobot"
LOGS_DIR="$BACKEND_DIR/logs"
PAPER_TRADING_LOG="$BOT_DIR/paper_trading.log"
CONFIG_DIR="$BACKEND_DIR/config"
DATA_DIR="$BACKEND_DIR/data"
SCRIPTS_DIR="$BACKEND_DIR/scripts"
DB_DIR="$BACKEND_DIR/db"

# Frontend directories
FRONTEND_DIR="$BOT_DIR/frontend"
FRONTEND_TRADING_DATA="$FRONTEND_DIR/public/trading_data"

# Process files
PAPER_TRADING_PID="$BOT_DIR/paper_trading.pid"
BACKEND_STATUS_FILE="$FRONTEND_DIR/trading_data/backend_status.json"
PAPER_TRADING_STATE_FILE="$BOT_DIR/paper_trading_state.json"
FRONTEND_PAPER_TRADING_STATE="$FRONTEND_DIR/trading_data/paper_trading_state.json"
FRONTEND_PAPER_TRADING_STATUS="$FRONTEND_DIR/public/trading_data/paper_trading_status.json"

# Required Python packages
REQUIRED_PACKAGES=(
  "flask"
  "pandas"
  "numpy"
  "binance"
  "python-binance"
  "websocket-client"
  "requests"
  "werkzeug"
  "tensorflow"
  "scikit-learn"
  "plotly"
  "matplotlib"
)

# Log the troubleshooting output
TROUBLESHOOT_LOG="$BOT_DIR/troubleshoot_$(date +%Y%m%d_%H%M%S).log"

# Print and log messages
log_message() {
  local level=$1
  local message=$2
  local color=$NC
  
  case $level in
    "INFO") color=$BLUE ;;
    "SUCCESS") color=$GREEN ;;
    "WARNING") color=$YELLOW ;;
    "ERROR") color=$RED ;;
  esac
  
  echo -e "${color}[$level] $message${NC}"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" >> "$TROUBLESHOOT_LOG"
}

# Check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Check if Python package is installed
check_python_package() {
  python -c "import $1" >/dev/null 2>&1
}

# Initialize the log file
initialize_log() {
  echo "======= BACKEND TROUBLESHOOTING LOG =======" > "$TROUBLESHOOT_LOG"
  echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TROUBLESHOOT_LOG"
  echo "----------------------------------------" >> "$TROUBLESHOOT_LOG"
}

# Check system requirements
check_system_requirements() {
  log_message "INFO" "Checking system requirements..."
  
  # Check if Python is installed
  if command_exists python; then
    log_message "SUCCESS" "Python is installed: $(python --version 2>&1)"
  else
    log_message "ERROR" "Python is not installed!"
    exit 1
  fi
  
  # Check for pip
  if command_exists pip; then
    log_message "SUCCESS" "Pip is installed: $(pip --version)"
  else
    log_message "ERROR" "Pip is not installed!"
    exit 1
  fi
  
  # Check for important commands
  for cmd in jq curl wget grep sed awk; do
    if command_exists "$cmd"; then
      log_message "SUCCESS" "$cmd is installed"
    else
      log_message "WARNING" "$cmd is not installed. Some functionality may be limited."
    fi
  done
  
  # Check for required Python packages
  missing_packages=()
  for package in "${REQUIRED_PACKAGES[@]}"; do
    package_name=$(echo "$package" | cut -d= -f1)
    if ! python -c "import ${package_name//-/_}" >/dev/null 2>&1; then
      missing_packages+=("$package")
    fi
  done
  
  if [ ${#missing_packages[@]} -eq 0 ]; then
    log_message "SUCCESS" "All required Python packages are installed."
  else
    log_message "WARNING" "Missing Python packages: ${missing_packages[*]}"
    
    # Auto-answer yes to package installation
    REPLY="y"
    log_message "INFO" "Automatically answering 'yes' to package installation"
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      log_message "INFO" "Installing missing packages..."
      for package in "${missing_packages[@]}"; do
        log_message "INFO" "Installing $package..."
        pip install "$package"
        if [ $? -eq 0 ]; then
          log_message "SUCCESS" "Successfully installed $package"
        else
          log_message "ERROR" "Failed to install $package"
        fi
      done
    fi
  fi
}

# Check directory structure
check_directories() {
  log_message "INFO" "Checking directory structure..."
  
  # Check main directories
  for dir in "$BOT_DIR" "$BACKEND_DIR" "$LOGS_DIR" "$CONFIG_DIR" "$DATA_DIR" "$SCRIPTS_DIR" "$DB_DIR" "$FRONTEND_DIR"; do
    if [ -d "$dir" ]; then
      log_message "SUCCESS" "Directory exists: $dir"
    else
      log_message "ERROR" "Directory missing: $dir"
      # Auto-answer yes to directory creation
      REPLY="y"
      log_message "INFO" "Automatically answering 'yes' to directory creation"
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$dir"
        log_message "SUCCESS" "Created directory: $dir"
      fi
    fi
  done
  
  # Create trading data directories if they don't exist
  if [ ! -d "$FRONTEND_TRADING_DATA" ]; then
    log_message "WARNING" "Trading data directory is missing, creating it..."
    mkdir -p "$FRONTEND_TRADING_DATA"
    log_message "SUCCESS" "Created directory: $FRONTEND_TRADING_DATA"
  fi
}

# Check file permissions
check_permissions() {
  log_message "INFO" "Checking file permissions..."
  
  # List of critical files and directories to check
  CRITICAL_PATHS=(
    "$BOT_DIR"
    "$BACKEND_DIR"
    "$LOGS_DIR"
    "$BOT_DIR/paper_trading_cli.py"
    "$BOT_DIR/paper_trading_daemon.py"
    "$BOT_DIR/paper_trading_service.sh"
    "$BOT_DIR/start_paper_trading.sh"
    "$BOT_DIR/stop_paper_trading.sh"
    "$BOT_DIR/startup.sh"
    "$BOT_DIR/shutdown.sh"
    "$FRONTEND_TRADING_DATA"
  )
  
  for path in "${CRITICAL_PATHS[@]}"; do
    if [ -e "$path" ]; then
      # Check if file/directory is writable
      if [ -w "$path" ]; then
        log_message "SUCCESS" "Path has write permissions: $path"
      else
        log_message "ERROR" "Path lacks write permissions: $path"
        # Auto-answer yes to permission fixes
        REPLY="y"
        log_message "INFO" "Automatically answering 'yes' to fix permissions"
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
          if [ -d "$path" ]; then
            chmod -R 755 "$path"
          else
            chmod 755 "$path"
          fi
          log_message "SUCCESS" "Fixed permissions for: $path"
        fi
      fi
      
      # Check if script is executable
      if [[ "$path" == *.sh || "$path" == *.py ]] && [ ! -x "$path" ]; then
        log_message "WARNING" "Script is not executable: $path"
        # Auto-answer yes to making file executable
        REPLY="y"
        log_message "INFO" "Automatically answering 'yes' to make file executable"
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
          chmod +x "$path"
          log_message "SUCCESS" "Made script executable: $path"
        fi
      fi
    else
      log_message "WARNING" "Path does not exist: $path"
    fi
  done
}

# Check paper trading state
check_paper_trading_state() {
  log_message "INFO" "Checking paper trading state..."
  
  # Check if state file exists
  if [ -f "$PAPER_TRADING_STATE_FILE" ]; then
    log_message "SUCCESS" "Paper trading state file exists"
    
    # Check if the state file is valid JSON
    if jq . "$PAPER_TRADING_STATE_FILE" >/dev/null 2>&1; then
      log_message "SUCCESS" "Paper trading state file contains valid JSON"
      
      # Check for missing trade_history attribute
      if ! jq -e '.trade_history' "$PAPER_TRADING_STATE_FILE" >/dev/null 2>&1; then
        log_message "WARNING" "Missing trade_history in paper trading state"
        # Auto-answer yes to fixing state file
        REPLY="y"
        log_message "INFO" "Automatically answering 'yes' to fix state file"
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
          # Create a backup
          cp "$PAPER_TRADING_STATE_FILE" "${PAPER_TRADING_STATE_FILE}.bak"
          
          # Add trade_history if missing
          jq '. + {trade_history: []}' "$PAPER_TRADING_STATE_FILE" > "${PAPER_TRADING_STATE_FILE}.tmp"
          mv "${PAPER_TRADING_STATE_FILE}.tmp" "$PAPER_TRADING_STATE_FILE"
          log_message "SUCCESS" "Added trade_history to state file"
        fi
      fi
    else
      log_message "ERROR" "Paper trading state file contains invalid JSON"
      # Auto-answer yes to resetting state file
      REPLY="y"
      log_message "INFO" "Automatically answering 'yes' to reset state file"
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Create a backup
        if [ -f "$PAPER_TRADING_STATE_FILE" ]; then
          cp "$PAPER_TRADING_STATE_FILE" "${PAPER_TRADING_STATE_FILE}.bak"
        fi
        
        # Create a new minimal state file
        cat > "$PAPER_TRADING_STATE_FILE" << EOF
{
  "balance": 10000,
  "base_currency": "USDT",
  "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
  "holdings": {},
  "last_prices": {},
  "trade_history": [],
  "is_running": false,
  "mode": "paper"
}
EOF
        log_message "SUCCESS" "Reset paper trading state file"
      fi
    fi
    
    # Ensure frontend state file exists
    if [ ! -f "$FRONTEND_PAPER_TRADING_STATE" ]; then
      log_message "WARNING" "Frontend paper trading state file is missing"
      # Auto-answer yes to creating file
      REPLY="y"
      log_message "INFO" "Automatically answering 'yes' to create file"
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        mkdir -p "$(dirname "$FRONTEND_PAPER_TRADING_STATE")"
        cp "$PAPER_TRADING_STATE_FILE" "$FRONTEND_PAPER_TRADING_STATE"
        log_message "SUCCESS" "Created frontend paper trading state file"
      fi
    fi
  else
    log_message "WARNING" "Paper trading state file does not exist"
    # Auto-answer yes to creating new state file
    REPLY="y"
    log_message "INFO" "Automatically answering 'yes' to create new state file"
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      # Create a new minimal state file
      cat > "$PAPER_TRADING_STATE_FILE" << EOF
{
  "balance": 10000,
  "base_currency": "USDT",
  "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
  "holdings": {},
  "last_prices": {},
  "trade_history": [],
  "is_running": false,
  "mode": "paper"
}
EOF
      log_message "SUCCESS" "Created new paper trading state file"
      
      # Copy to frontend location
      mkdir -p "$(dirname "$FRONTEND_PAPER_TRADING_STATE")"
      cp "$PAPER_TRADING_STATE_FILE" "$FRONTEND_PAPER_TRADING_STATE"
      log_message "SUCCESS" "Created frontend paper trading state file"
    fi
  fi
  
  # Check/create paper trading status file
  if [ ! -f "$FRONTEND_PAPER_TRADING_STATUS" ]; then
    log_message "WARNING" "Paper trading status file is missing"
    read -p "Would you like to create it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      mkdir -p "$(dirname "$FRONTEND_PAPER_TRADING_STATUS")"
      cat > "$FRONTEND_PAPER_TRADING_STATUS" << EOF
{
  "status": "inactive",
  "last_update": "$(date -Iseconds)",
  "balance": 10000,
  "portfolio_value": 10000,
  "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
  "trades": []
}
EOF
      log_message "SUCCESS" "Created paper trading status file"
    fi
  fi
  
  # Update backend status file
  if [ ! -f "$BACKEND_STATUS_FILE" ]; then
    log_message "WARNING" "Backend status file is missing"
    read -p "Would you like to create it? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      mkdir -p "$(dirname "$BACKEND_STATUS_FILE")"
      cat > "$BACKEND_STATUS_FILE" << EOF
{
  "last_update": "$(date -Iseconds)",
  "services": {
    "backend": {
      "status": "unknown",
      "last_checked": "$(date -Iseconds)"
    },
    "database": {
      "status": "unknown",
      "last_checked": "$(date -Iseconds)"
    },
    "paper_trading": {
      "status": "inactive",
      "last_checked": "$(date -Iseconds)"
    }
  }
}
EOF
      log_message "SUCCESS" "Created backend status file"
    fi
  fi
}

# Check for running processes
check_processes() {
  log_message "INFO" "Checking for running processes..."
  
  # Check for paper trading daemon
  if [ -f "$PAPER_TRADING_PID" ]; then
    pid=$(cat "$PAPER_TRADING_PID")
    if ps -p "$pid" > /dev/null; then
      log_message "SUCCESS" "Paper trading daemon is running (PID: $pid)"
    else
      log_message "WARNING" "Paper trading daemon is not running, but PID file exists"
      # Auto-answer yes to cleaning up PID file
      REPLY="y"
      log_message "INFO" "Automatically answering 'yes' to clean up PID file"
      echo
      if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm "$PAPER_TRADING_PID"
        log_message "SUCCESS" "Removed stale PID file"
      fi
    fi
  else
    log_message "INFO" "Paper trading daemon is not running (no PID file)"
  fi
  
  # Check for other paper trading processes
  paper_pids=$(pgrep -f "paper_trading_cli.py")
  if [ -n "$paper_pids" ]; then
    log_message "INFO" "Found paper trading processes: $paper_pids"
  fi
  
  # Check for backend processes
  backend_pid=$(pgrep -f "startup.sh")
  if [ -n "$backend_pid" ]; then
    log_message "SUCCESS" "Backend is running (PID: $backend_pid)"
  else
    log_message "INFO" "Backend startup.sh is not running"
  fi
}

# Fix paper trading issues
fix_paper_trading() {
  log_message "INFO" "Attempting to fix paper trading..."
  
  # Stop any running instances
  log_message "INFO" "Stopping any running paper trading instances..."
  
  # Try standard stop first
  if [ -f "$BOT_DIR/stop_paper_trading.sh" ]; then
    bash "$BOT_DIR/stop_paper_trading.sh"
  elif [ -f "$BOT_DIR/paper_trading_service.sh" ]; then
    bash "$BOT_DIR/paper_trading_service.sh" stop
  else
    python "$BOT_DIR/paper_trading_cli.py" stop
  fi
  
  # Force kill any remaining processes
  paper_pids=$(pgrep -f "paper_trading")
  if [ -n "$paper_pids" ]; then
    log_message "WARNING" "Killing remaining paper trading processes: $paper_pids"
    kill -9 $paper_pids 2>/dev/null
  fi
  
  # Clean up PID file
  if [ -f "$PAPER_TRADING_PID" ]; then
    rm "$PAPER_TRADING_PID"
    log_message "SUCCESS" "Removed PID file"
  fi
  
  # Update state file if necessary
  if [ -f "$PAPER_TRADING_STATE_FILE" ]; then
    # Set is_running to false
    jq '.is_running = false' "$PAPER_TRADING_STATE_FILE" > "${PAPER_TRADING_STATE_FILE}.tmp"
    mv "${PAPER_TRADING_STATE_FILE}.tmp" "$PAPER_TRADING_STATE_FILE"
    log_message "SUCCESS" "Updated state file to indicate paper trading is stopped"
    
    # Copy to frontend location
    cp "$PAPER_TRADING_STATE_FILE" "$FRONTEND_PAPER_TRADING_STATE"
  fi
  
  # Update backend status file
  if [ -f "$BACKEND_STATUS_FILE" ]; then
    jq '.services.paper_trading.status = "inactive" | .services.paper_trading.last_checked = "'$(date -Iseconds)'"' \
      "$BACKEND_STATUS_FILE" > "${BACKEND_STATUS_FILE}.tmp"
    mv "${BACKEND_STATUS_FILE}.tmp" "$BACKEND_STATUS_FILE"
    log_message "SUCCESS" "Updated backend status file"
  fi
  
  # Ask to start paper trading
  # Auto-answer yes to starting paper trading
  REPLY="y"
  log_message "INFO" "Automatically answering 'yes' to start paper trading"
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    start_paper_trading
  fi
}

# Start paper trading
start_paper_trading() {
  log_message "INFO" "Starting paper trading..."
  
  # Try to start with service script first
  if [ -f "$BOT_DIR/paper_trading_service.sh" ]; then
    bash "$BOT_DIR/paper_trading_service.sh" start
    sleep 2
  elif [ -f "$BOT_DIR/start_paper_trading.sh" ]; then
    bash "$BOT_DIR/start_paper_trading.sh"
    sleep 2
  else
    # Fallback to CLI script
    python "$BOT_DIR/paper_trading_daemon.py" start
    sleep 2
  fi
  
  # Check if started successfully
  if [ -f "$PAPER_TRADING_PID" ]; then
    pid=$(cat "$PAPER_TRADING_PID")
    if ps -p "$pid" > /dev/null; then
      log_message "SUCCESS" "Paper trading started successfully (PID: $pid)"
      
      # Update status files
      if [ -f "$BACKEND_STATUS_FILE" ]; then
        jq '.services.paper_trading.status = "active" | .services.paper_trading.last_checked = "'$(date -Iseconds)'"' \
          "$BACKEND_STATUS_FILE" > "${BACKEND_STATUS_FILE}.tmp"
        mv "${BACKEND_STATUS_FILE}.tmp" "$BACKEND_STATUS_FILE"
      fi
      
      if [ -f "$PAPER_TRADING_STATE_FILE" ]; then
        jq '.is_running = true' "$PAPER_TRADING_STATE_FILE" > "${PAPER_TRADING_STATE_FILE}.tmp"
        mv "${PAPER_TRADING_STATE_FILE}.tmp" "$PAPER_TRADING_STATE_FILE"
        # Copy to frontend location
        cp "$PAPER_TRADING_STATE_FILE" "$FRONTEND_PAPER_TRADING_STATE"
      fi
      
      # Update paper trading status file
      if [ -f "$FRONTEND_PAPER_TRADING_STATUS" ]; then
        jq '.status = "active" | .last_update = "'$(date -Iseconds)'"' \
          "$FRONTEND_PAPER_TRADING_STATUS" > "${FRONTEND_PAPER_TRADING_STATUS}.tmp"
        mv "${FRONTEND_PAPER_TRADING_STATUS}.tmp" "$FRONTEND_PAPER_TRADING_STATUS"
      else
        # Create if it doesn't exist
        mkdir -p "$(dirname "$FRONTEND_PAPER_TRADING_STATUS")"
        cat > "$FRONTEND_PAPER_TRADING_STATUS" << EOF
{
  "status": "active",
  "last_update": "$(date -Iseconds)",
  "balance": 10000,
  "portfolio_value": 10000,
  "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
  "trades": []
}
EOF
      fi
      
      log_message "SUCCESS" "Updated all status files"
    else
      log_message "ERROR" "Failed to start paper trading (PID file exists but process not running)"
    fi
  else
    log_message "ERROR" "Failed to start paper trading (no PID file created)"
  fi
}

# Check backend status
check_backend() {
  log_message "INFO" "Checking backend status..."
  
  # Check if backend is running
  if pgrep -f "startup.sh" > /dev/null; then
    log_message "SUCCESS" "Backend is running"
  else
    log_message "WARNING" "Backend is not running"
    # Auto-answer yes to starting backend
    REPLY="y"
    log_message "INFO" "Automatically answering 'yes' to start backend"
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      start_backend
    fi
  fi
}

# Start backend
start_backend() {
  log_message "INFO" "Starting backend..."
  
  # Check if backend startup script exists
  if [ -f "$BACKEND_DIR/startup.sh" ]; then
    log_message "INFO" "Using backend startup script at $BACKEND_DIR/startup.sh"
    
    # Run the startup script
    (cd "$BACKEND_DIR" && ./startup.sh) &
    sleep 5
    
    # Check if started successfully
    if pgrep -f "startup.sh" > /dev/null; then
      log_message "SUCCESS" "Backend started successfully"
      
      # Update backend status file
      if [ -f "$BACKEND_STATUS_FILE" ]; then
        jq '.services.backend.status = "active" | .services.backend.last_checked = "'$(date -Iseconds)'"' \
          "$BACKEND_STATUS_FILE" > "${BACKEND_STATUS_FILE}.tmp"
        mv "${BACKEND_STATUS_FILE}.tmp" "$BACKEND_STATUS_FILE"
        log_message "SUCCESS" "Updated backend status file"
      fi
    else
      log_message "ERROR" "Failed to start backend"
    fi
  else
    log_message "ERROR" "Backend startup script not found at $BACKEND_DIR/startup.sh"
  fi
}

# Fix database issues
fix_database() {
  log_message "INFO" "Checking database..."
  
  # Check if database directory exists
  if [ -d "$DB_DIR" ]; then
    log_message "SUCCESS" "Database directory exists"
    
    # Check for database files
    if [ "$(ls -A "$DB_DIR")" ]; then
      log_message "SUCCESS" "Database files found"
    else
      log_message "WARNING" "Database directory is empty"
    fi
  else
    log_message "ERROR" "Database directory not found"
    mkdir -p "$DB_DIR"
    log_message "SUCCESS" "Created database directory"
  fi
  
  # Update backend status file
  if [ -f "$BACKEND_STATUS_FILE" ]; then
    jq '.services.database.status = "active" | .services.database.last_checked = "'$(date -Iseconds)'"' \
      "$BACKEND_STATUS_FILE" > "${BACKEND_STATUS_FILE}.tmp"
    mv "${BACKEND_STATUS_FILE}.tmp" "$BACKEND_STATUS_FILE"
    log_message "SUCCESS" "Updated database status in backend status file"
  fi
}

# Perform complete restart of all services
restart_all_services() {
  log_message "INFO" "Performing complete restart of all services..."
  
  # Stop all services first
  log_message "INFO" "Stopping all services..."
  
  # Stop paper trading
  if [ -f "$BOT_DIR/stop_paper_trading.sh" ]; then
    bash "$BOT_DIR/stop_paper_trading.sh"
  elif [ -f "$BOT_DIR/paper_trading_service.sh" ]; then
    bash "$BOT_DIR/paper_trading_service.sh" stop
  else
    python "$BOT_DIR/paper_trading_cli.py" stop
  fi
  
  # Kill any remaining paper trading processes
  paper_pids=$(pgrep -f "paper_trading")
  if [ -n "$paper_pids" ]; then
    kill -9 $paper_pids 2>/dev/null
  fi
  
  # Stop backend
  if [ -f "$BACKEND_DIR/stop.sh" ]; then
    (cd "$BACKEND_DIR" && ./stop.sh)
  fi
  
  # Clean up PID files
  if [ -f "$PAPER_TRADING_PID" ]; then
    rm "$PAPER_TRADING_PID"
  fi
  
  # Wait a bit before starting
  log_message "INFO" "Waiting 5 seconds before restarting services..."
  sleep 5
  
  # Start backend first
  log_message "INFO" "Starting backend..."
  if [ -f "$BACKEND_DIR/startup.sh" ]; then
    (cd "$BACKEND_DIR" && ./startup.sh) &
    sleep 5
  fi
  
  # Start paper trading
  log_message "INFO" "Starting paper trading..."
  if [ -f "$BOT_DIR/paper_trading_service.sh" ]; then
    bash "$BOT_DIR/paper_trading_service.sh" start
  elif [ -f "$BOT_DIR/start_paper_trading.sh" ]; then
    bash "$BOT_DIR/start_paper_trading.sh"
  else
    python "$BOT_DIR/paper_trading_daemon.py" start
  fi
  
  # Check status after restart
  sleep 3
  check_processes
  
  log_message "SUCCESS" "Services restart completed"
}

# Main menu for the troubleshooting script
main_menu() {
  while true; do
    echo -e "\n${BLUE}========= BACKEND TROUBLESHOOTING MENU ==========${NC}"
    echo "1. Run complete diagnostic and fix all issues"
    echo "2. Check system requirements"
    echo "3. Check directory structure and permissions"
    echo "4. Check paper trading state files"
    echo "5. Fix paper trading issues and restart"
    echo "6. Check and start backend"
    echo "7. Restart all services"
    echo "8. Exit"
    echo
    read -p "Select an option (1-8): " choice
    echo
    
    case $choice in
      1)
        check_system_requirements
        check_directories
        check_permissions
        check_paper_trading_state
        check_processes
        fix_paper_trading
        check_backend
        fix_database
        log_message "SUCCESS" "Complete diagnostic and fixes completed"
        ;;
      2)
        check_system_requirements
        ;;
      3)
        check_directories
        check_permissions
        ;;
      4)
        check_paper_trading_state
        ;;
      5)
        fix_paper_trading
        ;;
      6)
        check_backend
        ;;
      7)
        restart_all_services
        ;;
      8)
        log_message "INFO" "Exiting troubleshooting script"
        exit 0
        ;;
      *)
        log_message "WARNING" "Invalid option. Please select 1-8."
        ;;
    esac
  done
}

# Initialize
initialize_log
echo -e "${GREEN}======= BACKEND TROUBLESHOOTING SCRIPT =======${NC}"
echo -e "${BLUE}This script will diagnose and fix issues with the backend,${NC}"
echo -e "${BLUE}including the paper trading module.${NC}"
echo

# Ask the user if they want to run automated fixes or use the menu
echo "Would you like to:"
echo "1. Run complete automated diagnostic and fixes"
echo "2. Use the interactive menu"
echo
read -p "Select an option (1-2): " start_option
echo

case $start_option in
  1)
    check_system_requirements
    check_directories
    check_permissions
    check_paper_trading_state
    check_processes
    fix_paper_trading
    check_backend
    fix_database
    restart_all_services
    log_message "SUCCESS" "Automated troubleshooting and fixes completed"
    ;;
  2)
    main_menu
    ;;
  *)
    log_message "WARNING" "Invalid option. Using interactive menu."
    main_menu
    ;;
esac

log_message "INFO" "Troubleshooting script completed. Log saved to $TROUBLESHOOT_LOG"
echo -e "${GREEN}Troubleshooting script completed. Log saved to $TROUBLESHOOT_LOG${NC}"
