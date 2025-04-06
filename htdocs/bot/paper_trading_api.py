#!/usr/bin/env python3
"""
Paper Trading API
This script provides a simple API endpoint for the paper trading dashboard.
"""

import os
import sys
import json
import logging
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/paper_trading_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_trading_api")

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Path to the paper trading CLI script
PAPER_TRADING_CLI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trading_cli.py")

# Path to the paper trading state file
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend/trading_data/paper_trading_state.json")

def run_command(command, args=None):
    """Run a command on the paper trading CLI."""
    cmd = [sys.executable, PAPER_TRADING_CLI, command]
    if args:
        for key, value in args.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))
    
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e.stderr}")
        return {"status": "error", "message": e.stderr}
    except Exception as e:
        logger.error(f"Error running command: {str(e)}")
        return {"status": "error", "message": str(e)}

def get_status():
    """Get the current paper trading status."""
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return {"status": "success", "data": json.load(f)}
        else:
            return {"status": "error", "message": "State file not found"}
    except Exception as e:
        logger.error(f"Error getting status: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.route('/trading/paper', methods=['GET'])
def get_paper_trading_status():
    """GET endpoint to retrieve paper trading status."""
    return jsonify(get_status())

@app.route('/trading/paper', methods=['POST'])
def handle_paper_trading_command():
    """POST endpoint to handle paper trading commands."""
    try:
        data = request.json
        if not data or 'command' not in data:
            return jsonify({"status": "error", "message": "Missing command parameter"}), 400
        
        command = data.pop('command')
        
        # Handle special commands
        if command == 'api':
            if 'key' not in data or 'secret' not in data:
                return jsonify({"status": "error", "message": "Missing API key or secret"}), 400
            return jsonify(run_command(command, {'key': data['key'], 'secret': data['secret']}))
        
        # Handle auto-execute command
        if command == 'auto-execute':
            if 'enabled' not in data:
                return jsonify({"status": "error", "message": "Missing enabled parameter"}), 400
                
            args = {}
            args['enabled'] = str(data.get('enabled')).lower()
            
            if 'confidence' in data:
                confidence = float(data.get('confidence', 0.75))
                if confidence < 0 or confidence > 1:
                    return jsonify({"status": "error", "message": "Confidence must be between 0 and 1"}), 400
                args['confidence'] = str(confidence)
                
            if 'interval' in data:
                interval = int(data.get('interval', 60))
                if interval <= 0:
                    return jsonify({"status": "error", "message": "Interval must be positive"}), 400
                args['interval'] = str(interval)
                
            return jsonify(run_command(command, args))
        
        # Handle standard commands
        valid_commands = ['start', 'stop', 'status', 'switch', 'reset', 'export']
        if command not in valid_commands:
            return jsonify({"status": "error", "message": f"Invalid command: {command}"}), 400
        
        # For the switch command, we need a mode parameter
        if command == 'switch' and 'mode' not in data:
            return jsonify({"status": "error", "message": "Missing mode parameter for switch command"}), 400
        
        # Run the command
        result = run_command(command, data)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error handling command: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
