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
