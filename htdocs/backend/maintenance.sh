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
