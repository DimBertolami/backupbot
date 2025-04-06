#!/bin/bash

# Crypto Trading Bot - Service Installation Script
# This script installs the Crypto Trading Bot as a systemd service

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}     Crypto Trading Bot - Service Installation         ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}This script must be run as root (use sudo)${NC}"
  exit 1
fi

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
SERVICE_FILE="${SCRIPT_DIR}/cryptobot.service"
DEST_SERVICE_FILE="/etc/systemd/system/cryptobot.service"

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
  echo -e "${RED}Service file not found at ${SERVICE_FILE}${NC}"
  exit 1
fi

# Copy service file to systemd directory
echo -e "${YELLOW}Installing service file...${NC}"
cp "$SERVICE_FILE" "$DEST_SERVICE_FILE"
chmod 644 "$DEST_SERVICE_FILE"

# Get current user
CURRENT_USER=$(logname || echo $SUDO_USER)
if [ -z "$CURRENT_USER" ]; then
  echo -e "${YELLOW}Could not determine current user, using default in service file${NC}"
else
  echo -e "${YELLOW}Setting service to run as user: ${CURRENT_USER}${NC}"
  sed -i "s/User=dim/User=${CURRENT_USER}/g" "$DEST_SERVICE_FILE"
fi

# Reload systemd
echo -e "${YELLOW}Reloading systemd...${NC}"
systemctl daemon-reload

# Enable the service
echo -e "${YELLOW}Enabling service to start on boot...${NC}"
systemctl enable cryptobot.service

echo -e "${GREEN}Service installed successfully!${NC}"
echo -e "${YELLOW}You can now manage the service with these commands:${NC}"
echo -e "  Start:   ${BLUE}sudo systemctl start cryptobot${NC}"
echo -e "  Stop:    ${BLUE}sudo systemctl stop cryptobot${NC}"
echo -e "  Status:  ${BLUE}sudo systemctl status cryptobot${NC}"
echo -e "  Logs:    ${BLUE}sudo journalctl -u cryptobot -f${NC}"

echo -e "\n${YELLOW}Would you like to start the service now? (y/n)${NC}"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
  echo -e "${YELLOW}Starting Crypto Trading Bot service...${NC}"
  systemctl start cryptobot.service
  echo -e "${GREEN}Service started!${NC}"
  echo -e "${YELLOW}To view logs, run: ${BLUE}sudo journalctl -u cryptobot -f${NC}"
else
  echo -e "${YELLOW}Service not started. You can start it later with: ${BLUE}sudo systemctl start cryptobot${NC}"
fi

echo -e "${BLUE}=======================================================${NC}"
echo -e "${GREEN}Installation complete!${NC}"
echo -e "${BLUE}=======================================================${NC}"
