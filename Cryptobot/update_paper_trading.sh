#!/bin/bash

# This script updates the startup.sh and stop.sh scripts to integrate the new
# paper trading service management system.

CRYPTOBOT_DIR="/home/dim/git/Cryptobot"
STARTUP_SCRIPT="${CRYPTOBOT_DIR}/startup.sh"
STOP_SCRIPT="${CRYPTOBOT_DIR}/stop.sh"
PAPER_TRADING_DIR="/opt/lampp/htdocs/bot"

# Create backup of the original scripts
cp "${STARTUP_SCRIPT}" "${STARTUP_SCRIPT}.bak"
cp "${STOP_SCRIPT}" "${STOP_SCRIPT}.bak"

echo "Created backups of startup.sh and stop.sh"

# Update the paper trading section in startup.sh
# Replace the current paper trading start code with our new service script

# First section - replacing the part that checks if paper trading is already running
sed -i 's/# Check if paper trading is already running\n    PAPER_TRADING_RUNNING=\$(pgrep -f "python.*paper_trading_cli\.py" || echo "")\n    \n    if \[ -n "\$PAPER_TRADING_RUNNING" \]; then\n        log_success "Paper trading service is already running" "always"\n    else\n        # Start paper trading\n        nohup python paper_trading_cli\.py start > "\${PAPER_TRADING_LOG_DIR}\/paper_trading\.log" 2>\&1 \&\n        PAPER_TRADING_PID=\$!\n        log_success "Paper trading service started with PID: \${PAPER_TRADING_PID}" "always"\n    fi/# Check if paper trading is already running\n    PAPER_TRADING_RUNNING=\$(pgrep -f "python.*paper_trading_cli\.py" || echo "")\n    \n    if \[ -n "\$PAPER_TRADING_RUNNING" \]; then\n        log_success "Paper trading service is already running" "always"\n    else\n        # Start paper trading using our new service script\n        if \[ -f "\${PAPER_TRADING_DIR}\/paper_trading_service\.sh" \]; then\n            log_info "Starting paper trading service with service script..."\n            cd "\${PAPER_TRADING_DIR}"\n            .\/paper_trading_service\.sh start\n            log_success "Paper trading service started successfully" "always"\n        else\n            # Fallback to original method if service script doesn\'t exist\n            log_warning "Service script not found, using legacy method" "always"\n            nohup python paper_trading_cli\.py start > "\${PAPER_TRADING_LOG_DIR}\/paper_trading\.log" 2>\&1 \&\n            PAPER_TRADING_PID=\$!\n            log_success "Paper trading service started with PID: \${PAPER_TRADING_PID}" "always"\n        fi\n    fi/' "${STARTUP_SCRIPT}"

# Second section - replacing the part that checks if paper trading was previously running
sed -i 's/# Check if paper trading was previously running\n    PAPER_TRADING_WAS_RUNNING=\$(grep -o \'"is_running": true\' "\${PAPER_TRADING_STATE_FILE}" || echo "")\n    \n    if \[ -n "\$PAPER_TRADING_WAS_RUNNING" \]; then\n        log_info "Paper trading was previously running, restarting it..."\n        cd "\${PAPER_TRADING_DIR}"\n        source "\${PYTHON_VENV_DIR}\/bin\/activate"\n        nohup python paper_trading_cli\.py start > "\${PAPER_TRADING_LOG_DIR}\/paper_trading\.log" 2>\&1 \&\n        PAPER_TRADING_PID=\$!\n        log_success "Paper trading service started with PID: \${PAPER_TRADING_PID}" "always"/# Check if paper trading was previously running\n    PAPER_TRADING_WAS_RUNNING=\$(grep -o \'"is_running": true\' "\${PAPER_TRADING_STATE_FILE}" || echo "")\n    \n    if \[ -n "\$PAPER_TRADING_WAS_RUNNING" \]; then\n        log_info "Paper trading was previously running, restarting it..."\n        cd "\${PAPER_TRADING_DIR}"\n        if \[ -f "\${PAPER_TRADING_DIR}\/paper_trading_service\.sh" \]; then\n            log_info "Starting paper trading service with service script..."\n            source "\${PYTHON_VENV_DIR}\/bin\/activate"\n            .\/paper_trading_service\.sh start\n            log_success "Paper trading service started successfully" "always"\n        else\n            # Fallback to original method if service script doesn\'t exist\n            log_warning "Service script not found, using legacy method" "always"\n            source "\${PYTHON_VENV_DIR}\/bin\/activate"\n            nohup python paper_trading_cli\.py start > "\${PAPER_TRADING_LOG_DIR}\/paper_trading\.log" 2>\&1 \&\n            PAPER_TRADING_PID=\$!\n            log_success "Paper trading service started with PID: \${PAPER_TRADING_PID}" "always"\n        fi/' "${STARTUP_SCRIPT}"

# Update the stop.sh script to use our paper trading service for stopping
# Find the section that kills Python processes
sed -i 's/# Find and kill Python processes related to our app\nBACKEND_PIDS=\$(pgrep -f "python3.*app\\\.py|python3.*server\\\.py|python3.*main\\\.py|python3.*update_trading_signals\\\.py|python3.*paper_trading_cli\\\.py")\nif \[ -n "\$BACKEND_PIDS" \]; then\n    echo -e "Stopping backend server(s)..."\n    kill \$BACKEND_PIDS\n    echo -e "\${GREEN}✓ Backend server(s) stopped\${NC}"\nelse\n    echo -e "\${YELLOW}No backend server processes found\${NC}"\nfi/# Find and kill Python processes related to our app\nBACKEND_PIDS=\$(pgrep -f "python3.*app\\\.py|python3.*server\\\.py|python3.*main\\\.py|python3.*update_trading_signals\\\.py")\nif \[ -n "\$BACKEND_PIDS" \]; then\n    echo -e "Stopping backend server(s)..."\n    kill \$BACKEND_PIDS\n    echo -e "\${GREEN}✓ Backend server(s) stopped\${NC}"\nelse\n    echo -e "\${YELLOW}No backend server processes found\${NC}"\nfi\n\n# Stop paper trading service using the service script\nif \[ -f "\/opt\/lampp\/htdocs\/bot\/paper_trading_service.sh" \]; then\n    echo -e "Stopping paper trading service..."\n    cd "\/opt\/lampp\/htdocs\/bot"\n    .\/paper_trading_service.sh stop\n    echo -e "\${GREEN}✓ Paper trading service stopped\${NC}"\nelse\n    # Fallback to killing paper trading processes directly\n    PAPER_TRADING_PIDS=\$(pgrep -f "python.*paper_trading_cli\\\.py")\n    if \[ -n "\$PAPER_TRADING_PIDS" \]; then\n        echo -e "Stopping paper trading service..."\n        kill \$PAPER_TRADING_PIDS\n        echo -e "\${GREEN}✓ Paper trading service stopped\${NC}"\n    else\n        echo -e "\${YELLOW}No paper trading processes found\${NC}"\n    fi\nfi/' "${STOP_SCRIPT}"

echo "Updated startup.sh and stop.sh to use the new paper trading service management"
echo "Original files backed up to startup.sh.bak and stop.sh.bak"
echo "Please review the changes and test the scripts"
