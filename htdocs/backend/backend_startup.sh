#!/bin/bash

# Crypto Trading Bot - Backend Database Startup Script
# This script ensures the database is properly initialized and sets up scheduled data exports

# Color codes for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

BACKEND_DIR="/opt/lampp/htdocs/backend"
LOG_DIR="${BACKEND_DIR}/logs"

# Create logs directory if it doesn't exist
mkdir -p ${LOG_DIR}

echo -e "${BLUE}=======================================================${NC}"
echo -e "${BLUE}      Crypto Trading Bot - Database Initialization     ${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Activate virtual environment if it exists
if [ -d "${BACKEND_DIR}/venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source "${BACKEND_DIR}/venv/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
elif [ -d "/home/dim/git/Cryptobot/.venv" ]; then
    echo -e "${YELLOW}Activating virtual environment from Cryptobot directory...${NC}"
    source "/home/dim/git/Cryptobot/.venv/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
fi

# Move to the backend directory
cd "${BACKEND_DIR}"

# Make sure database directory exists
mkdir -p "${BACKEND_DIR}/data/db"

# Initialize database
echo -e "\n${YELLOW}Initializing database...${NC}"
python init_db.py > "${LOG_DIR}/db_init.log" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Database initialized successfully${NC}"
else
    echo -e "${RED}✗ Database initialization failed. Check ${LOG_DIR}/db_init.log for details${NC}"
fi

# Set up scheduled database maintenance
echo -e "\n${YELLOW}Setting up scheduled database maintenance...${NC}"

# Create a maintenance script for regular exports
cat > "${BACKEND_DIR}/maintenance.sh" << 'EOF'
#!/bin/bash

BACKEND_DIR="/opt/lampp/htdocs/backend"
LOG_FILE="${BACKEND_DIR}/logs/maintenance.log"

# Activate virtual environment if it exists
if [ -d "${BACKEND_DIR}/venv" ]; then
    source "${BACKEND_DIR}/venv/bin/activate"
elif [ -d "/home/dim/git/Cryptobot/.venv" ]; then
    source "/home/dim/git/Cryptobot/.venv/bin/activate"
fi

cd "${BACKEND_DIR}"

# Export bot thoughts to database
echo "$(date): Exporting bot thoughts to database..." >> $LOG_FILE
python -c "from data.bot_thoughts_exporter import export_bot_thoughts_to_db, export_trading_status_to_db; export_bot_thoughts_to_db(); export_trading_status_to_db()" >> $LOG_FILE 2>&1
EOF

chmod +x "${BACKEND_DIR}/maintenance.sh"

# Setup crontab entry for regular exports (every 15 minutes)
(crontab -l 2>/dev/null; echo "*/15 * * * * ${BACKEND_DIR}/maintenance.sh") | crontab -

echo -e "${GREEN}✓ Scheduled database maintenance setup complete${NC}"

# Run the initial bot thoughts export
echo -e "\n${YELLOW}Running initial data export...${NC}"
python -c "from data.bot_thoughts_exporter import export_bot_thoughts_to_db, export_trading_status_to_db; export_bot_thoughts_to_db(); export_trading_status_to_db()" > "${LOG_DIR}/initial_export.log" 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Initial data export completed successfully${NC}"
else
    echo -e "${RED}✗ Initial data export failed. Check ${LOG_DIR}/initial_export.log for details${NC}"
fi

echo -e "\n${BLUE}=======================================================${NC}"
echo -e "${GREEN}Database initialization complete!${NC}"
echo -e "${BLUE}=======================================================${NC}"

# Start the FastAPI server if not started by the main script
if [ "$1" == "run_server" ]; then
    echo -e "\n${YELLOW}Starting the FastAPI backend server...${NC}"
    cd "${BACKEND_DIR}"
    python run.py > "${LOG_DIR}/fastapi.log" 2>&1 &
    FASTAPI_PID=$!
    echo -e "${GREEN}✓ FastAPI server started with PID: ${FASTAPI_PID}${NC}"
fi

exit 0
