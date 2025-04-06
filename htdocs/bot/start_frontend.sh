#!/bin/bash

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default verbose setting
VERBOSE=false

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --verbose) VERBOSE=true ;;
        -v) VERBOSE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Logging functions
log_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    local always_show=${2:-""}
    if [ "$VERBOSE" = true ] || [ "$always_show" = "always" ]; then
        echo -e "${GREEN}[SUCCESS]${NC} $1"
    fi
}

log_warning() {
    local always_show=${2:-""}
    if [ "$VERBOSE" = true ] || [ "$always_show" = "always" ]; then
        echo -e "${YELLOW}[WARNING]${NC} $1"
    fi
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Define the frontend directory
FRONTEND_DIR="$(pwd)/frontend"
LOG_DIR="$(pwd)/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Function to kill previous frontend instances
kill_frontend_instances() {
    log_info "Checking for previous frontend instances..."
    
    # Find processes using Vite's default ports (5173-5199)
    for port in $(seq 5173 5199); do
        PID=$(lsof -t -i:$port 2>/dev/null)
        if [ -n "$PID" ]; then
            log_info "Found process using port $port (PID: $PID)"
            kill $PID 2>/dev/null
            if [ $? -eq 0 ]; then
                log_success "Killed process on port $port (PID: $PID)"
            else
                log_warning "Failed to kill process on port $port (PID: $PID)"
            fi
        fi
    done
    
    # Also find any npm/node processes related to Vite
    VITE_PIDS=$(pgrep -f "node.*vite" || echo "")
    if [ -n "$VITE_PIDS" ]; then
        log_info "Found Vite processes: $VITE_PIDS"
        kill $VITE_PIDS 2>/dev/null
        log_success "Killed Vite processes"
    fi
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Check if any processes are still running
    REMAINING_PIDS=""
    for port in $(seq 5173 5199); do
        PID=$(lsof -t -i:$port 2>/dev/null)
        if [ -n "$PID" ]; then
            REMAINING_PIDS="$REMAINING_PIDS $PID"
        fi
    done
    
    if [ -n "$REMAINING_PIDS" ]; then
        log_warning "Some processes are still running. Attempting force kill..." "always"
        kill -9 $REMAINING_PIDS 2>/dev/null
        log_success "Force killed remaining processes" "always"
    else
        log_success "All previous frontend instances terminated" "always"
    fi
}

# Main script execution
echo -e "\n${BLUE}=======================================================${NC}"
echo -e "${YELLOW}Starting Frontend Server${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Kill any previous frontend instances
kill_frontend_instances

# Start the frontend server
log_info "Starting frontend server..."
cd "$FRONTEND_DIR"

if [ "$VERBOSE" = true ]; then
    npm run dev
else
    echo -e "${GREEN}Starting frontend server...${NC}"
    nohup npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
    FRONTEND_PID=$!
    echo -e "${GREEN}Frontend server started with PID: $FRONTEND_PID${NC}"
    echo -e "${YELLOW}Frontend URL:${NC} http://localhost:5173 (or next available port)"
    echo -e "${YELLOW}Log file:${NC} $LOG_DIR/frontend.log"
fi

echo -e "\n${BLUE}=======================================================${NC}"
