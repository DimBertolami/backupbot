#!/bin/bash

# Crypto Trading Bot Startup Script
# This script launches both the frontend React application and the Python backend
# It can be run manually or as a systemd service
#
# Usage:
#   ./startup.sh             - Run with minimal output
#   ./startup.sh --verbose   - Run with detailed debug output
#   ./startup.sh --service   - Run as a systemd service
# -------------------------------------------------------------------------------

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default to non-verbose output
VERBOSE=false

# Configuration (modify these as needed)
FRONTEND_DIR="/opt/lampp/htdocs/bot/frontend"
BACKEND_DIR="/home/dim/git/Cryptobot"
PYTHON_VENV_DIR="${BACKEND_DIR}/.venv"
LOG_DIR="${BACKEND_DIR}/logs"
TRADING_DATA_DIR="${FRONTEND_DIR}/trading_data"
PAPER_TRADING_DIR="/opt/lampp/htdocs/bot"
PAPER_TRADING_STATE_FILE="${PAPER_TRADING_DIR}/paper_trading_state.json"
PAPER_TRADING_LOG_DIR="${PAPER_TRADING_DIR}/logs"
PROXY_CONFIG_DIR="${FRONTEND_DIR}/src"
RUNNING_AS_SERVICE=false

# Process command line arguments
for arg in "$@"; do
  case $arg in
    --service)
      RUNNING_AS_SERVICE=true
      echo "Running as a system service"
      ;;
    --verbose)
      VERBOSE=true
      echo "Running in verbose mode"
      ;;
  esac
done

# Define logging functions
log_info() {
  if [ "$VERBOSE" = true ]; then
    echo -e "${BLUE}[INFO]${NC} $1"
  fi
}

log_success() {
  if [ "$VERBOSE" = true ]; then
    echo -e "${GREEN}[SUCCESS]${NC} $1"
  else
    # For critical success messages that should always be shown
    if [ "$2" = "always" ]; then
      echo -e "${GREEN}✓${NC} $1"
    fi
  fi
}

log_warning() {
  if [ "$VERBOSE" = true ]; then
    echo -e "${YELLOW}[WARNING]${NC} $1"
  else
    # For critical warnings that should always be shown
    if [ "$2" = "always" ]; then
      echo -e "${YELLOW}!${NC} $1"
    fi
  fi
}

log_error() {
  # Errors are always shown
  echo -e "${RED}[ERROR]${NC} $1"
}

# Create required directories if they don't exist
mkdir -p ${LOG_DIR}
mkdir -p ${TRADING_DATA_DIR}
mkdir -p ${PAPER_TRADING_LOG_DIR}

# Function to check and kill previous instances of backend processes
kill_previous_instances() {
    log_info "Checking for previous backend instances..."
    
    # Define patterns for backend processes
    local BACKEND_PATTERNS=(
        "python3.*app\.py" 
        "python3.*server\.py" 
        "python3.*main\.py" 
        "python3.*update_trading_signals\.py" 
        "python.*paper_trading_cli\.py"
    )
    
    local FOUND_PROCESSES=false
    local PIDS_TO_KILL=()
    
    # Check for each pattern
    for pattern in "${BACKEND_PATTERNS[@]}"; do
        local PIDS=$(pgrep -f "$pattern" || echo "")
        
        if [ -n "$PIDS" ]; then
            FOUND_PROCESSES=true
            log_warning "Found previous instances of ${pattern}:"
            if [ "$VERBOSE" = true ]; then
                ps -f -p $PIDS
            fi
            
            # Add PIDs to the list to kill
            for pid in $PIDS; do
                PIDS_TO_KILL+=("$pid")
            done
        fi
    done
    
    # If processes were found, kill them
    if [ "$FOUND_PROCESSES" = true ]; then
        log_warning "Stopping previous backend instances..." "always"
        
        # Try graceful termination first
        for pid in "${PIDS_TO_KILL[@]}"; do
            log_info "Stopping process $pid..."
            kill $pid 2>/dev/null
        done
        
        # Wait a moment for processes to terminate
        sleep 2
        
        # Check if any processes are still running and force kill if necessary
        local REMAINING_PIDS=()
        for pid in "${PIDS_TO_KILL[@]}"; do
            if ps -p $pid > /dev/null 2>&1; then
                REMAINING_PIDS+=("$pid")
            fi
        done
        
        if [ ${#REMAINING_PIDS[@]} -gt 0 ]; then
            log_warning "Some processes did not terminate gracefully. Forcing termination..."
            for pid in "${REMAINING_PIDS[@]}"; do
                log_info "Force stopping process $pid..."
                kill -9 $pid 2>/dev/null
            done
        fi
        
        log_success "All previous backend instances stopped" "always"
    else
        log_success "No previous backend instances found"
    fi
}

if [ "$VERBOSE" = true ]; then
    echo -e "${BLUE}=======================================================${NC}"
    echo -e "${BLUE}           Crypto Trading Bot - Startup Script         ${NC}"
    echo -e "${BLUE}=======================================================${NC}"
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Kill any previous instances of the backend
kill_previous_instances

# Check dependencies
log_info "Checking dependencies..."

# Check if Node.js is installed
if ! command_exists node; then
    log_error "Node.js is not installed. Please install Node.js to run the frontend."
    exit 1
else
    NODE_VERSION=$(node -v)
    log_success "Node.js ${NODE_VERSION} is installed"
fi

# Check if Python is installed
if ! command_exists python3; then
    log_error "Python 3 is not installed. Please install Python 3 to run the backend."
    exit 1
else
    PYTHON_VERSION=$(python3 --version)
    log_success "${PYTHON_VERSION} is installed"
fi

# Check if npm is installed
if ! command_exists npm; then
    log_error "npm is not installed. Please install npm to run the frontend."
    exit 1
else
    NPM_VERSION=$(npm -v)
    log_success "npm ${NPM_VERSION} is installed"
fi

# Start the Python virtual environment or create it if it doesn't exist
log_info "Setting up Python virtual environment..."
if [ ! -d "${PYTHON_VENV_DIR}" ]; then
    log_info "Creating new virtual environment in ${PYTHON_VENV_DIR}..."
    python3 -m venv "${PYTHON_VENV_DIR}"
    log_success "Virtual environment created"
fi

log_info "Activating virtual environment..."
source "${PYTHON_VENV_DIR}/bin/activate"
log_success "Virtual environment activated"

# Install Python dependencies
log_info "Installing Python dependencies..."
cd "${BACKEND_DIR}"
if [ -f "requirements.txt" ]; then
    if [ "$VERBOSE" = true ]; then
        pip install -r requirements.txt
    else
        pip install -r requirements.txt > /dev/null 2>&1
    fi
    log_success "Python dependencies installed"
else
    log_warning "requirements.txt not found. Installing essential packages..." "always"
    if [ "$VERBOSE" = true ]; then
        pip install pandas numpy scikit-learn tensorflow flask flask-cors sqlite3
    else
        pip install pandas numpy scikit-learn tensorflow flask flask-cors sqlite3 > /dev/null 2>&1
    fi
    echo -e "${GREEN}✓ Essential Python packages installed${NC}"
fi

# Install frontend dependencies
log_info "Installing frontend dependencies..."
cd "${FRONTEND_DIR}"
if [ "$VERBOSE" = true ]; then
    npm install
else
    npm install --silent > /dev/null 2>&1
fi
log_success "Frontend dependencies installed"

# Initialize the database and export existing data
log_info "Initializing database..."
if [ -f "/opt/lampp/htdocs/backend/backend_startup.sh" ]; then
    if [ "$VERBOSE" = true ]; then
        bash /opt/lampp/htdocs/backend/backend_startup.sh
    else
        bash /opt/lampp/htdocs/backend/backend_startup.sh > /dev/null 2>&1
    fi
    log_success "Database initialized"
else
    log_warning "Database initialization script not found"
fi

# Start the backend server
log_info "Starting the Python backend server..."
cd "${BACKEND_DIR}"

# Start paper trading backend
if [ -d "paper_trading/backend" ]; then
    log_info "Starting paper trading backend..."
    cd paper_trading/backend
    if [ "$VERBOSE" = true ]; then
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        python3 app.py > "${PAPER_TRADING_LOG_DIR}/paper_trading.log" 2>&1 &
    else
        python3 -m venv venv > /dev/null 2>&1
        source venv/bin/activate > /dev/null 2>&1
        pip install -r requirements.txt > /dev/null 2>&1
        python3 app.py > "${PAPER_TRADING_LOG_DIR}/paper_trading.log" 2>&1 &
    fi
    PAPER_TRADING_PID=$!
    log_success "Paper trading backend started with PID: ${PAPER_TRADING_PID}" "always"
fi

# Start the main backend server
log_info "Starting the main backend server..."
# Determine which script to run based on what exists
if [ -f "app.py" ]; then
    BACKEND_SCRIPT="app.py"
elif [ -f "server.py" ]; then
    BACKEND_SCRIPT="server.py"
elif [ -f "main.py" ]; then
    BACKEND_SCRIPT="main.py"
else
    # If no main script is found, use update_trading_signals.py
    BACKEND_SCRIPT="update_trading_signals.py"
fi

log_info "Launching backend script: ${BACKEND_SCRIPT}"
nohup python3 "${BACKEND_SCRIPT}" > "${LOG_DIR}/backend.log" 2>&1 &
BACKEND_PID=$!
log_success "Backend server started with PID: ${BACKEND_PID}" "always"

# Start the trading signal update service if it's not the main script
if [ "${BACKEND_SCRIPT}" != "update_trading_signals.py" ]; then
    log_info "Starting the trading signal update service..."
    nohup python3 update_trading_signals.py > "${LOG_DIR}/signals.log" 2>&1 &
    SIGNALS_PID=$!
    log_success "Trading signal service started with PID: ${SIGNALS_PID}" "always"
fi

# Start the frontend development server
log_info "Starting the frontend development server..."
cd "${FRONTEND_DIR}"

# Check if we're running as a service
if [ "$RUNNING_AS_SERVICE" = true ]; then
    # When running as a service, we don't start the frontend
    log_warning "Running as a service - frontend must be started separately" "always"
else
    # Start the frontend development server
    nohup npm run dev > "${LOG_DIR}/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    log_success "Frontend server started with PID: ${FRONTEND_PID}" "always"
fi

# Wait for servers to start up
log_info "Waiting for servers to initialize..."
sleep 5

# Check if servers are running
log_info "Checking server status..."
if ps -p $BACKEND_PID > /dev/null; then
    log_success "Backend server is running" "always"
else
    log_error "Backend server failed to start. Check ${LOG_DIR}/backend.log for details"
fi

if [ "${BACKEND_SCRIPT}" != "update_trading_signals.py" ]; then
    if ps -p $SIGNALS_PID > /dev/null; then
        log_success "Trading signal service is running" "always"
    else
        log_error "Trading signal service failed to start. Check ${LOG_DIR}/signals.log for details"
    fi
fi

# Check if paper trading was previously running and restore state if needed
log_info "Checking paper trading status..."

# Always start paper trading when running as a service, otherwise check previous state
if [ "$RUNNING_AS_SERVICE" = true ]; then
    log_info "Running as a service, ensuring paper trading is started..."
    cd "${PAPER_TRADING_DIR}"
    source "${PYTHON_VENV_DIR}/bin/activate"
    
    # Check if paper trading is already running
    PAPER_TRADING_RUNNING=$(pgrep -f "python.*paper_trading_cli\.py" || echo "")
    
    if [ -n "$PAPER_TRADING_RUNNING" ]; then
        log_success "Paper trading service is already running" "always"
    else
        # Start paper trading
        nohup python paper_trading_cli.py start > "${PAPER_TRADING_LOG_DIR}/paper_trading.log" 2>&1 &
        PAPER_TRADING_PID=$!
        log_success "Paper trading service started with PID: ${PAPER_TRADING_PID}" "always"
    fi
    
    # Copy the state file to the frontend trading_data directory for the proxy to find
    if [ -f "${PAPER_TRADING_STATE_FILE}" ]; then
        mkdir -p "${TRADING_DATA_DIR}"
        cp "${PAPER_TRADING_STATE_FILE}" "${TRADING_DATA_DIR}/paper_trading_status.json"
        log_success "Paper trading state file copied to frontend directory"
    fi
    
    # Start the paper trading API server
    log_info "Starting paper trading API server..."
    
    # Check if running as a service and use the appropriate path
    if [ "$RUNNING_AS_SERVICE" = true ]; then
        PAPER_TRADING_API_SCRIPT="${SCRIPT_DIR}/paper_trading_api.py"
    else
        PAPER_TRADING_API_SCRIPT="/opt/lampp/htdocs/bot/paper_trading_api.py"
    fi
    
    log_info "Using paper trading API script: ${PAPER_TRADING_API_SCRIPT}"
    
    if [ -f "$PAPER_TRADING_API_SCRIPT" ]; then
        # Check if the API server is already running
        PAPER_TRADING_API_RUNNING=$(pgrep -f "python.*paper_trading_api\.py" || echo "")
        
        if [ -n "$PAPER_TRADING_API_RUNNING" ]; then
            log_success "Paper trading API server is already running" "always"
        else
            # Start the paper trading API server
            if [ "$RUNNING_AS_SERVICE" = true ]; then
                cd "${SCRIPT_DIR}"
            else
                cd "/opt/lampp/htdocs/bot"
            fi
            
            nohup python "$PAPER_TRADING_API_SCRIPT" > "${PAPER_TRADING_LOG_DIR}/paper_trading_api.log" 2>&1 &
            PAPER_TRADING_API_PID=$!
            log_success "Paper trading API server started with PID: ${PAPER_TRADING_API_PID}" "always"
        fi
    else
        log_warning "Paper trading API script not found at ${PAPER_TRADING_API_SCRIPT}" "always"
    fi
elif [ -f "${PAPER_TRADING_STATE_FILE}" ]; then
    log_info "Paper trading state file found, checking previous status..."
    # Check if paper trading was previously running
    PAPER_TRADING_WAS_RUNNING=$(grep -o '"is_running": true' "${PAPER_TRADING_STATE_FILE}" || echo "")
    
    if [ -n "$PAPER_TRADING_WAS_RUNNING" ]; then
        log_info "Paper trading was previously running, restarting it..."
        cd "${PAPER_TRADING_DIR}"
        source "${PYTHON_VENV_DIR}/bin/activate"
        nohup python paper_trading_cli.py start > "${PAPER_TRADING_LOG_DIR}/paper_trading.log" 2>&1 &
        PAPER_TRADING_PID=$!
        log_success "Paper trading service started with PID: ${PAPER_TRADING_PID}" "always"
        
        # Copy the state file to the frontend trading_data directory for the proxy to find
        if [ -f "${PAPER_TRADING_STATE_FILE}" ]; then
            mkdir -p "${TRADING_DATA_DIR}"
            cp "${PAPER_TRADING_STATE_FILE}" "${TRADING_DATA_DIR}/paper_trading_status.json"
            log_success "Paper trading state file copied to frontend directory"
        fi
    else
        log_warning "Paper trading was not running previously, it can be started from the dashboard" "always"
    fi
else
    log_warning "No paper trading state found, it can be started from the dashboard" "always"
fi

if ps -p $FRONTEND_PID > /dev/null; then
    log_success "Frontend server is running" "always"
else
    log_error "Frontend server failed to start. Check ${LOG_DIR}/frontend.log for details"
fi

# Display URLs
FRONTEND_URL="http://localhost:5174"  # Vite default port
BACKEND_URL="http://localhost:5000"   # Flask default port

# Always show the final success message and URLs, regardless of verbose mode
echo -e "\n${BLUE}=======================================================${NC}"
echo -e "${GREEN}All services started successfully!${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo -e "\n${YELLOW}Frontend URL:${NC} ${FRONTEND_URL}"
echo -e "${YELLOW}Backend URL:${NC} ${BACKEND_URL}"
echo -e "\n${YELLOW}Log files:${NC}"
echo -e "  - Backend: ${LOG_DIR}/backend.log"
if [ "${BACKEND_SCRIPT}" != "update_trading_signals.py" ]; then
    echo -e "  - Trading signals: ${LOG_DIR}/signals.log"
fi
echo -e "  - Frontend: ${LOG_DIR}/frontend.log"

echo -e "\n${BLUE}To stop all services, run:${NC} ./stop.sh"

# Create a stop script
cat > "${BACKEND_DIR}/stop.sh" << 'EOF'
#!/bin/bash

# Crypto Trading Bot Stop Script
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}          Crypto Trading Bot - Shutdown Script         ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Kill processes by name
echo -e "\n${YELLOW}Stopping all bot processes...${NC}"

# Find and kill the frontend server (npm)
FRONTEND_PIDS=$(pgrep -f "node.*dev-server")
if [ -n "$FRONTEND_PIDS" ]; then
    echo -e "Stopping frontend server(s)..."
    kill $FRONTEND_PIDS
    echo -e "${GREEN}✓ Frontend server(s) stopped${NC}"
else
    echo -e "${YELLOW}No frontend server processes found${NC}"
fi

# Find and kill Python processes related to our app
BACKEND_PIDS=$(pgrep -f "python3.*app\.py|python3.*server\.py|python3.*main\.py|python3.*update_trading_signals\.py|python3.*paper_trading_cli\.py")
if [ -n "$BACKEND_PIDS" ]; then
    echo -e "Stopping backend server(s)..."
    kill $BACKEND_PIDS
    echo -e "${GREEN}✓ Backend server(s) stopped${NC}"
else
    echo -e "${YELLOW}No backend server processes found${NC}"
fi

echo -e "\n${GREEN}All services have been stopped.${NC}"
echo -e "${BLUE}=======================================================${NC}"
EOF

chmod +x "${BACKEND_DIR}/stop.sh"

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to view logs or run any of these commands to see logs:${NC}"
echo -e "  - Backend: tail -f ${LOG_DIR}/backend.log"
if [ "${BACKEND_SCRIPT}" != "update_trading_signals.py" ]; then
    echo -e "  - Trading signals: tail -f ${LOG_DIR}/signals.log"
fi
echo -e "  - Frontend: tail -f ${LOG_DIR}/frontend.log"
echo -e "  - Paper Trading: tail -f ${PAPER_TRADING_DIR}/logs/paper_trading.log"

# Show backend logs if not running as a service
if [ "$RUNNING_AS_SERVICE" = false ]; then
    tail -f "${LOG_DIR}/backend.log"
fi
