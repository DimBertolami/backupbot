#!/bin/bash
# Paper Trading Bot Startup Script
# This script starts both the backend and frontend servers for the trading bot
# and ensures resource usage stays below 70% of system capacity

# Colors for terminal output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Paper Trading Bot System...${NC}"

# 1. Define directories
BOT_DIR="/opt/lampp/htdocs/bot"
BACKEND_DIR="$BOT_DIR/backend"
FRONTEND_DIR="$BOT_DIR/frontend"

# 1.1 Start resource manager to limit CPU/memory usage to 70%
if [ -f "$BOT_DIR/resource_manager_service.sh" ]; then
    echo -e "${YELLOW}Starting resource manager (70% limit)...${NC}"
    bash "$BOT_DIR/resource_manager_service.sh" start
    sleep 2
fi

# 2. Check for existing processes and stop them if necessary
echo -e "${YELLOW}Checking for existing processes...${NC}"

# Check backend port (5001)
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}Port 5001 is already in use. Stopping existing process...${NC}"
    PID=$(lsof -t -i:5001)
    kill -9 $PID
    sleep 2
    echo -e "${GREEN}✓ Stopped process on port 5001${NC}"
fi

# Check frontend port (default is 5173, but it might use up to 5179 if ports are taken)
for PORT in {5173..5179}; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}Port $PORT is already in use. Stopping existing process...${NC}"
        PID=$(lsof -t -i:$PORT)
        kill -9 $PID
        sleep 1
        echo -e "${GREEN}✓ Stopped process on port $PORT${NC}"
    fi
done

# 3. Make sure the config directories exist
CONFIG_DIR="$FRONTEND_DIR/trading_data"
if [ ! -d "$CONFIG_DIR" ]; then
    echo -e "${YELLOW}Creating config directory...${NC}"
    mkdir -p "$CONFIG_DIR"
fi

# 4. Check if API keys are configured and backup exists
API_KEYS_BACKUP="$CONFIG_DIR/api_keys_backup.json"
CONFIG_FILE="$CONFIG_DIR/trading_config.json"

if [ -f "$API_KEYS_BACKUP" ] && [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Restoring API keys from backup...${NC}"
    cp "$API_KEYS_BACKUP" "$CONFIG_FILE"
fi

# 5. Start backend server
echo -e "${GREEN}Starting backend server...${NC}"
cd "$BACKEND_DIR"
python paper_trading_api.py > "$BOT_DIR/backend.log" 2>&1 &
BACKEND_PID=$!

# Check if backend started successfully
sleep 3
if ! ps -p $BACKEND_PID > /dev/null; then
    echo -e "${RED}Failed to start backend server! Check $BOT_DIR/backend.log for details.${NC}"
    exit 1
fi

# Test if the backend server is actually responding
echo -e "${YELLOW}Verifying backend server connection...${NC}"
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "http://localhost:5001/trading/status" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend server is responding correctly${NC}"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT+1))
        if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
            echo -e "${RED}Backend server is not responding after $MAX_RETRIES attempts!${NC}"
            echo -e "${YELLOW}Checking backend log file for errors...${NC}"
            tail -n 20 "$BOT_DIR/backend.log"
            echo -e "${RED}Attempting to fix backend issues...${NC}"
            # Kill the non-responding backend process
            kill $BACKEND_PID 2>/dev/null
            sleep 2
            # Try alternate Python command (some systems use python3 instead of python)
            echo -e "${YELLOW}Trying alternate Python command...${NC}"
            cd "$BACKEND_DIR"
            python3 paper_trading_api.py > "$BOT_DIR/backend.log" 2>&1 &
            BACKEND_PID=$!
            sleep 5
            if curl -s "http://localhost:5001/trading/status" > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Backend server started with python3 and is responding (PID: $BACKEND_PID)${NC}"
                break
            else
                echo -e "${RED}Could not start backend server. Please check the configuration manually.${NC}"
                echo -e "${YELLOW}Last 20 lines of backend log:${NC}"
                tail -n 20 "$BOT_DIR/backend.log"
                exit 1
            fi
        fi
        echo -e "${YELLOW}Backend not responding yet, retrying in 2 seconds... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
        sleep 2
    fi
done

# 6. Start frontend server
echo -e "${GREEN}Starting frontend server...${NC}"
cd "$FRONTEND_DIR"

# Check if vite.config.ts exists
if [ ! -f "vite.config.ts" ]; then
    echo -e "${RED}vite.config.ts not found in $FRONTEND_DIR. Aborting.${NC}"
    exit 1
fi

# Make sure the vite.config.ts has the correct proxy configuration
if ! grep -q "'/trading'" vite.config.ts && ! grep -q "'/trading_data'" vite.config.ts; then
    echo -e "${YELLOW}Adding proxy configuration to vite.config.ts...${NC}"
    # Create a backup of the original file
    cp vite.config.ts vite.config.ts.bak
    
    # Add proxy configuration
    if grep -q "server: {" vite.config.ts; then
        # Server config exists, add proxy inside it
        sed -i "s|server: {|server: {\n    proxy: {\n      '/trading': 'http://localhost:5001',\n      '/trading_data': 'http://localhost:5001',\n    },|" vite.config.ts
    else
        # No server config, add the complete block
        sed -i "s|plugins: \[react()\],|plugins: [react()],\n  server: {\n    proxy: {\n      '/trading': 'http://localhost:5001',\n      '/trading_data': 'http://localhost:5001',\n    },\n  },|" vite.config.ts
    fi
    echo -e "${GREEN}✓ Added proxy configuration to vite.config.ts${NC}"
    echo -e "${YELLOW}Original file backed up as vite.config.ts.bak${NC}"
    # Add reminder to restart servers after change
    echo -e "${YELLOW}Proxy configuration added. Backend server may need to be restarted.${NC}"
else
    echo -e "${GREEN}✓ Proxy configuration already exists in vite.config.ts${NC}"
fi

# Start the frontend server
npm run dev > "$BOT_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!

# Check if frontend started successfully
sleep 5
if ! ps -p $FRONTEND_PID > /dev/null; then
    echo -e "${RED}Failed to start frontend server! Check $BOT_DIR/frontend.log for details.${NC}"
    echo -e "${YELLOW}Last 20 lines of frontend log:${NC}"
    tail -n 20 "$BOT_DIR/frontend.log"
    exit 1
fi

# Wait for Vite server to be ready (looking for the "ready in" message in logs)
echo -e "${YELLOW}Waiting for Vite server to be ready...${NC}"
MAX_WAIT=30
counter=0
while [ $counter -lt $MAX_WAIT ]; do
    if grep -q "ready in" "$BOT_DIR/frontend.log"; then
        # Extract the port from the frontend log
        FRONTEND_PORT=$(grep "Local:" "$BOT_DIR/frontend.log" | grep -oE 'http://localhost:[0-9]+' | cut -d ':' -f 3)
        if [ -z "$FRONTEND_PORT" ]; then
            FRONTEND_PORT=5173 # Default port if not found
        fi
        echo -e "${GREEN}✓ Frontend server started on port $FRONTEND_PORT (PID: $FRONTEND_PID)${NC}"
        break
    fi
    counter=$((counter+1))
    sleep 1
    if [ $counter -eq $MAX_WAIT ]; then
        echo -e "${YELLOW}Vite server may still be starting up. Continuing anyway.${NC}"
        echo -e "${YELLOW}Last 20 lines of frontend log:${NC}"
        tail -n 20 "$BOT_DIR/frontend.log"
        FRONTEND_PORT=5173 # Assume default port
    fi
done

# 7. Wait for frontend to be ready and determine the port
sleep 3
FRONTEND_PORT=""
for PORT in {5173..5179}; do
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
        FRONTEND_PORT=$PORT
        break
    fi
done

if [ -z "$FRONTEND_PORT" ]; then
    echo -e "${RED}Could not determine frontend port! Using default 5173.${NC}"
    FRONTEND_PORT=5173
fi

# 8. Print success message and URLs
echo -e "\n${GREEN}Trading Bot System is now running!${NC}"
echo -e "${GREEN}------------------------------------${NC}"
echo -e "${YELLOW}Resource Manager:${NC} Active (70% CPU/Memory limit)"
echo -e "Backend API: http://localhost:5001/trading"
echo -e "Frontend UI: http://localhost:$FRONTEND_PORT"
echo
echo -e "${YELLOW}To access the Paper Trading Dashboard, open your browser and navigate to:${NC}"
echo -e "${GREEN}http://localhost:$FRONTEND_PORT${NC}"
echo
echo -e "${YELLOW}To stop the system, run:${NC}"
echo -e "${GREEN}$BOT_DIR/shutdown.sh${NC}"
echo
echo -e "${YELLOW}If you encounter connection issues:${NC}"
echo -e "${GREEN}1. ${YELLOW}Restart services:${NC} ${GREEN}$BOT_DIR/shutdown.sh && $BOT_DIR/startup.sh${NC}"
echo -e "${GREEN}2. ${YELLOW}Check backend log:${NC} ${GREEN}tail -n 30 $BOT_DIR/backend.log${NC}"
echo -e "${GREEN}3. ${YELLOW}Check frontend log:${NC} ${GREEN}tail -n 30 $BOT_DIR/frontend.log${NC}"
echo
echo -e "${YELLOW}For 'Resource not found' errors, check:${NC}"
echo -e "${GREEN}1. ${YELLOW}Backend API is running:${NC} ${GREEN}curl http://localhost:5001/trading/status${NC}"
echo -e "${GREEN}2. ${YELLOW}Proxy configuration in vite.config.ts has:${NC}"
echo -e "   ${GREEN}'/trading': 'http://localhost:5001'${NC}"
echo -e "   ${GREEN}'/trading_data': 'http://localhost:5001'${NC}"

# Create improved shutdown script with better process cleanup
cat > "$BOT_DIR/shutdown.sh" << 'EOF'
#!/bin/bash
# Shutdown script for Trading Bot System

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Shutting down Trading Bot System...${NC}"

# Try to kill specific PIDs if they were saved
BACKEND_PID=$BACKEND_PID
FRONTEND_PID=$FRONTEND_PID

# Try to kill backend by PID first
if [ ! -z "$BACKEND_PID" ] && kill $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Backend server stopped${NC}"
else
    echo -e "${YELLOW}Searching for backend server process...${NC}"
    FOUND_PIDS=$(lsof -t -i:5001 2>/dev/null)
    if [ ! -z "$FOUND_PIDS" ]; then
        echo -e "${YELLOW}Found backend processes: $FOUND_PIDS${NC}"
        kill $FOUND_PIDS 2>/dev/null
        echo -e "${GREEN}✓ Backend server(s) stopped${NC}"
    else
        echo -e "${RED}No backend server found running on port 5001${NC}"
    fi
fi

# Try to kill frontend by PID first
if [ ! -z "$FRONTEND_PID" ] && kill $FRONTEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓ Frontend server stopped${NC}"
else
    echo -e "${YELLOW}Searching for frontend server processes...${NC}"
    for PORT in 5173 5174 5175 5176 5177 5178 5179 5180; do
        FOUND_PIDS=$(lsof -t -i:$PORT 2>/dev/null)
        if [ ! -z "$FOUND_PIDS" ]; then
            echo -e "${YELLOW}Found frontend on port $PORT: $FOUND_PIDS${NC}"
            kill $FOUND_PIDS 2>/dev/null
            echo -e "${GREEN}✓ Frontend server on port $PORT stopped${NC}"
        fi
    done
fi

# Stop resource manager
if [ -f "$BOT_DIR/resource_manager_service.sh" ]; then
    echo -e "${YELLOW}Stopping resource manager...${NC}"
    bash "$BOT_DIR/resource_manager_service.sh" stop
fi

echo -e "${GREEN}Trading Bot System has been stopped.${NC}"
EOF

# Fix variable substitution in the shutdown script
sed -i "s/\$BACKEND_PID/$BACKEND_PID/g" "$BOT_DIR/shutdown.sh"
sed -i "s/\$FRONTEND_PID/$FRONTEND_PID/g" "$BOT_DIR/shutdown.sh"

chmod +x "$BOT_DIR/shutdown.sh"

exit 0
