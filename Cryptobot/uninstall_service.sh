#!/bin/bash

# Crypto Trading Bot - Service Uninstallation Script
# This script stops and removes the systemd service for the Crypto Trading Bot

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}     Crypto Trading Bot - Service Uninstallation       ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}This script must be run as root (use sudo)${NC}"
    exit 1
fi

# Service name
SERVICE_NAME="cryptobot"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Check if service exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo -e "${YELLOW}Service file not found. The service may not be installed.${NC}"
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Uninstallation cancelled.${NC}"
        exit 0
    fi
fi

# Stop the service if it's running
echo -e "Checking service status..."
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${YELLOW}Service is running. Stopping the service...${NC}"
    systemctl stop $SERVICE_NAME
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Service stopped successfully.${NC}"
    else
        echo -e "${RED}Failed to stop the service. Continuing anyway...${NC}"
    fi
else
    echo -e "${YELLOW}Service is not running.${NC}"
fi

# Disable the service
echo -e "Disabling service from startup..."
systemctl disable $SERVICE_NAME
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Service disabled from startup.${NC}"
else
    echo -e "${RED}Failed to disable the service. Continuing anyway...${NC}"
fi

# Remove the service file
echo -e "Removing service file..."
if [ -f "$SERVICE_FILE" ]; then
    rm "$SERVICE_FILE"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Service file removed.${NC}"
    else
        echo -e "${RED}Failed to remove service file.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Service file not found. Skipping removal.${NC}"
fi

# Reload systemd
echo -e "Reloading systemd..."
systemctl daemon-reload
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Systemd reloaded successfully.${NC}"
else
    echo -e "${RED}Failed to reload systemd.${NC}"
    exit 1
fi

# Check if any processes are still running
echo -e "\n${YELLOW}Checking for any remaining processes...${NC}"
BACKEND_PIDS=$(pgrep -f "python3.*app\.py|python3.*server\.py|python3.*main\.py|python3.*update_trading_signals\.py|python3.*paper_trading_cli\.py")

if [ -n "$BACKEND_PIDS" ]; then
    echo -e "${YELLOW}Some backend processes are still running:${NC}"
    ps -f -p $BACKEND_PIDS
    
    read -p "Do you want to stop these processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "Stopping remaining processes..."
        kill $BACKEND_PIDS 2>/dev/null
        sleep 2
        
        # Check if processes are still running
        REMAINING_PIDS=$(pgrep -f "python3.*app\.py|python3.*server\.py|python3.*main\.py|python3.*update_trading_signals\.py|python3.*paper_trading_cli\.py")
        if [ -n "$REMAINING_PIDS" ]; then
            echo -e "${YELLOW}Some processes are still running. Forcing termination...${NC}"
            kill -9 $REMAINING_PIDS 2>/dev/null
        fi
        
        echo -e "${GREEN}All processes stopped.${NC}"
    else
        echo -e "${YELLOW}Leaving processes running.${NC}"
    fi
else
    echo -e "${GREEN}No backend processes found.${NC}"
fi

echo -e "${BLUE}=======================================================${NC}"
echo -e "${GREEN}Service uninstallation complete!${NC}"
echo -e "${BLUE}=======================================================${NC}"

echo -e "\n${YELLOW}If you want to reinstall the service later, run:${NC}"
echo -e "  ${BLUE}sudo ./install_service.sh${NC}"
