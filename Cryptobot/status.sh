#!/bin/bash

# Crypto Trading Bot Status Script
# This script shows the status of all backend services

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration (should match startup.sh)
FRONTEND_DIR="/opt/lampp/htdocs/bot/frontend"
BACKEND_DIR="/home/dim/git/Cryptobot"
LOG_DIR="${BACKEND_DIR}/logs"
PAPER_TRADING_DIR="/opt/lampp/htdocs/bot"
PAPER_TRADING_STATE_FILE="${PAPER_TRADING_DIR}/paper_trading_state.json"

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}           Crypto Trading Bot - Status Check           ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Function to check if a process is running
check_process() {
    local process_pattern=$1
    local name=$2
    local pid=$(pgrep -f "$process_pattern" | head -1) # Only get the first PID
    
    if [ -n "$pid" ]; then
        echo -e "${GREEN}✓ $name is running${NC} (PID: $pid)"
        return 0
    else
        echo -e "${RED}✗ $name is not running${NC}"
        return 1
    fi
}

# Function to check service status
check_service() {
    local service_name=$1
    local status=$(systemctl is-active $service_name 2>/dev/null)
    
    if [ "$status" = "active" ]; then
        echo -e "${GREEN}✓ $service_name service is active${NC}"
        return 0
    else
        echo -e "${RED}✗ $service_name service is not active${NC} (Status: $status)"
        return 1
    fi
}

# Function to get uptime of a process
get_process_uptime() {
    local pid=$1
    if [ -n "$pid" ]; then
        local process_start=$(ps -p $pid -o lstart= 2>/dev/null)
        if [ -n "$process_start" ]; then
            local start_seconds=$(date -d "$process_start" +%s)
            local current_seconds=$(date +%s)
            local uptime_seconds=$((current_seconds - start_seconds))
            
            local days=$((uptime_seconds / 86400))
            local hours=$(( (uptime_seconds % 86400) / 3600 ))
            local minutes=$(( (uptime_seconds % 3600) / 60 ))
            
            if [ $days -gt 0 ]; then
                echo "${days}d ${hours}h ${minutes}m"
            elif [ $hours -gt 0 ]; then
                echo "${hours}h ${minutes}m"
            else
                echo "${minutes}m"
            fi
        else
            echo "Unknown"
        fi
    else
        echo "Not running"
    fi
}

# Check if cryptobot service is running
echo -e "\n${CYAN}Systemd Service:${NC}"
SYSTEMD_ACTIVE=false
if check_service "cryptobot"; then
    SYSTEMD_ACTIVE=true
fi

# Check backend components
echo -e "\n${CYAN}Backend Components:${NC}"

# Check main backend - look for processes whether running via systemd or directly
BACKEND_PID=$(pgrep -f "python3.*app\.py|python3.*server\.py|python3.*main\.py")
if check_process "python3.*app\.py|python3.*server\.py|python3.*main\.py" "Main Backend Server"; then
    echo -e "  └─ Uptime: $(get_process_uptime $BACKEND_PID)"
    if [ "$SYSTEMD_ACTIVE" = true ]; then
        echo -e "  └─ ${BLUE}Running via systemd service${NC}"
    fi
fi

# Check trading signals service
SIGNALS_PID=$(pgrep -f "python3.*update_trading_signals\.py" | head -1) # Only get the first PID
if check_process "python3.*update_trading_signals\.py" "Trading Signals Service"; then
    echo -e "  └─ Uptime: $(get_process_uptime $SIGNALS_PID)"
    if [ "$SYSTEMD_ACTIVE" = true ]; then
        echo -e "  └─ ${BLUE}Running via systemd service${NC}"
    fi
fi

# Check paper trading
PAPER_TRADING_PID=$(pgrep -f "python.*paper_trading_cli\.py" | head -1) # Only get the first PID
if check_process "python.*paper_trading_cli\.py" "Paper Trading Service"; then
    echo -e "  └─ Uptime: $(get_process_uptime $PAPER_TRADING_PID)"
    
    # Check if paper trading is actually running (not just the process)
    if [ -f "$PAPER_TRADING_STATE_FILE" ]; then
        PAPER_TRADING_RUNNING=$(grep -o '"is_running": true' "$PAPER_TRADING_STATE_FILE" || echo "")
        if [ -n "$PAPER_TRADING_RUNNING" ]; then
            echo -e "  └─ ${GREEN}Trading is active${NC}"
        else
            echo -e "  └─ ${YELLOW}Trading is paused${NC}"
        fi
    fi
fi

# Check database
DB_PID=$(pgrep -f "sqlite3" | head -1) # Only get the first PID
check_process "sqlite3" "Database"

# Check frontend
FRONTEND_PID=$(pgrep -f "node.*dev-server" | head -1) # Only get the first PID
if check_process "node.*dev-server" "Frontend Server"; then
    echo -e "  └─ Uptime: $(get_process_uptime $FRONTEND_PID)"
fi

# Check for recent log activity
echo -e "\n${CYAN}Recent Log Activity:${NC}"

check_log_activity() {
    local log_file=$1
    local name=$2
    
    if [ -f "$log_file" ]; then
        local last_modified=$(stat -c %Y "$log_file")
        local current_time=$(date +%s)
        local time_diff=$((current_time - last_modified))
        
        if [ $time_diff -lt 300 ]; then  # Less than 5 minutes
            echo -e "${GREEN}✓ $name log has recent activity${NC} ($(date -d "@$last_modified" "+%H:%M:%S"))"
            echo -e "  └─ Last lines:"
            tail -n 3 "$log_file" | sed 's/^/      /'
        else
            local readable_time=$(date -d "@$last_modified" "+%Y-%m-%d %H:%M:%S")
            echo -e "${YELLOW}⚠ $name log last updated: $readable_time${NC}"
        fi
    else
        echo -e "${RED}✗ $name log file not found${NC}"
    fi
}

check_log_activity "${LOG_DIR}/backend.log" "Backend"
check_log_activity "${LOG_DIR}/signals.log" "Trading Signals"
check_log_activity "${PAPER_TRADING_DIR}/logs/paper_trading.log" "Paper Trading"

# Generate status JSON for frontend dashboard
echo -e "\n${CYAN}Generating Status JSON:${NC}"
STATUS_JSON="${FRONTEND_DIR}/trading_data/backend_status.json"

# Create the JSON structure
cat > "$STATUS_JSON" << EOL
{
  "timestamp": "$(date -Iseconds)",
  "services": {
    "systemd": {
      "name": "Systemd Service",
      "status": "$(systemctl is-active cryptobot 2>/dev/null || echo "inactive")",
      "pid": null
    },
    "backend": {
      "name": "Main Backend",
      "status": "$([ -n "$BACKEND_PID" ] && echo "running" || echo "stopped")",
      "managed_by_systemd": $([[ "$SYSTEMD_ACTIVE" = true && -n "$BACKEND_PID" ]] && echo "true" || echo "false"),
      "pid": $BACKEND_PID
    },
    "signals": {
      "name": "Trading Signals",
      "status": "$([ -n "$SIGNALS_PID" ] && echo "running" || echo "stopped")",
      "pid": $SIGNALS_PID
    },
    "paper_trading": {
      "name": "Paper Trading",
      "status": "$([ -n "$PAPER_TRADING_PID" ] && echo "running" || echo "stopped")",
      "trading_active": $([ -n "$PAPER_TRADING_RUNNING" ] && echo "true" || echo "false"),
      "pid": $PAPER_TRADING_PID
    },
    "database": {
      "name": "Database",
      "status": "$([ -n "$DB_PID" ] && echo "running" || echo "stopped")",
      "pid": $DB_PID
    },
    "frontend": {
      "name": "Frontend",
      "status": "$([ -n "$FRONTEND_PID" ] && echo "running" || echo "stopped")",
      "pid": $FRONTEND_PID
    }
  },
  "logs": {
    "backend": {
      "path": "${LOG_DIR}/backend.log",
      "last_modified": "$([ -f "${LOG_DIR}/backend.log" ] && date -r "${LOG_DIR}/backend.log" -Iseconds || echo "unknown")"
    },
    "signals": {
      "path": "${LOG_DIR}/signals.log",
      "last_modified": "$([ -f "${LOG_DIR}/signals.log" ] && date -r "${LOG_DIR}/signals.log" -Iseconds || echo "unknown")"
    },
    "paper_trading": {
      "path": "${PAPER_TRADING_DIR}/logs/paper_trading.log",
      "last_modified": "$([ -f "${PAPER_TRADING_DIR}/logs/paper_trading.log" ] && date -r "${PAPER_TRADING_DIR}/logs/paper_trading.log" -Iseconds || echo "unknown")"
    }
  }
}
EOL

echo -e "${GREEN}✓ Status JSON generated at ${STATUS_JSON}${NC}"
echo -e "${BLUE}=======================================================${NC}"
echo -e "${GREEN}Status check complete!${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Provide instructions for viewing logs
echo -e "\n${YELLOW}To view logs:${NC}"
echo -e "  Backend:        ${BLUE}tail -f ${LOG_DIR}/backend.log${NC}"
echo -e "  Trading Signals: ${BLUE}tail -f ${LOG_DIR}/signals.log${NC}"
echo -e "  Paper Trading:  ${BLUE}tail -f ${PAPER_TRADING_DIR}/logs/paper_trading.log${NC}"
echo -e "  Systemd Service: ${BLUE}sudo journalctl -u cryptobot -f${NC}"
