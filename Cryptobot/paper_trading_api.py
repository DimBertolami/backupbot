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

# Determine if running as a service or in development mode
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNNING_AS_SERVICE = os.path.basename(SCRIPT_DIR) == "Cryptobot"

# Configure paths based on environment
if RUNNING_AS_SERVICE:
    # Service mode - running from /home/dim/git/Cryptobot
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    PAPER_TRADING_CLI = os.path.join(SCRIPT_DIR, "paper_trading_cli.py")
    STATE_FILE = os.path.join(SCRIPT_DIR, "paper_trading_state.json")
    FRONTEND_DIR = "/opt/lampp/htdocs/bot/frontend"
    TRADING_DATA_DIR = os.path.join(FRONTEND_DIR, "trading_data")
    os.makedirs(TRADING_DATA_DIR, exist_ok=True)
    
    # Create a symbolic link to the paper trading CLI if it doesn't exist
    if not os.path.exists(PAPER_TRADING_CLI):
        try:
            original_cli = "/opt/lampp/htdocs/bot/paper_trading_cli.py"
            if os.path.exists(original_cli):
                os.symlink(original_cli, PAPER_TRADING_CLI)
        except Exception as e:
            print(f"Warning: Could not create symlink to paper_trading_cli.py: {e}")
else:
    # Development mode - running from /opt/lampp/htdocs/bot
    LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
    PAPER_TRADING_CLI = os.path.join(SCRIPT_DIR, "paper_trading_cli.py")
    STATE_FILE = os.path.join(SCRIPT_DIR, "frontend/trading_data/paper_trading_state.json")
    FRONTEND_DIR = os.path.join(SCRIPT_DIR, "frontend")
    TRADING_DATA_DIR = os.path.join(FRONTEND_DIR, "trading_data")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "paper_trading_api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("paper_trading_api")

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Log startup information
logger.info(f"Paper Trading API starting in {'service' if RUNNING_AS_SERVICE else 'development'} mode")
logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Paper Trading CLI: {PAPER_TRADING_CLI}")
logger.info(f"State file: {STATE_FILE}")
logger.info(f"Frontend directory: {FRONTEND_DIR}")
logger.info(f"Trading data directory: {TRADING_DATA_DIR}")

def run_command(command, args=None):
    """Run a command on the paper trading CLI."""
    # Ensure the CLI script exists
    if not os.path.exists(PAPER_TRADING_CLI):
        logger.error(f"Paper trading CLI script not found at: {PAPER_TRADING_CLI}")
        return {"status": "error", "message": f"Paper trading CLI script not found at: {PAPER_TRADING_CLI}"}
    
    # Build the command
    cmd = [sys.executable, PAPER_TRADING_CLI, command]
    
    # Add arguments with proper formatting
    if args:
        # Log the arguments we're processing
        logger.info(f"Processing arguments for {command}: {args}")
        
        # Special handling for auto-execute command
        if command == 'auto-execute':
            # Log the auto-execute arguments for debugging
            logger.info(f"Auto-execute command received with args: {args}")
            
            # Make sure enabled is 'true' or 'false' (lowercase)
            if 'enabled' in args:
                # Handle various input formats for the enabled parameter
                enabled_input = args['enabled']
                
                # Convert to lowercase string
                if isinstance(enabled_input, bool):
                    enabled_val = 'true' if enabled_input else 'false'
                else:
                    enabled_val = str(enabled_input).lower()
                
                # Validate the value
                if enabled_val not in ['true', 'false']:
                    error_msg = f"Invalid enabled value: {enabled_val}. Must be 'true' or 'false'."
                    logger.error(error_msg)
                    return {"status": "error", "message": error_msg}
                
                cmd.append("--enabled")
                cmd.append(enabled_val)
                logger.info(f"Using enabled value: {enabled_val}")
            else:
                error_msg = "Missing required parameter: enabled"
                logger.error(error_msg)
                return {"status": "error", "message": error_msg}
                
            # Add confidence if provided
            if 'confidence' in args:
                try:
                    # Handle the case where confidence might be a string or a number
                    confidence_input = args['confidence']
                    # Make sure we handle empty strings
                    if confidence_input == '':
                        confidence_val = 0.75  # Default value
                    else:
                        confidence_val = float(confidence_input)
                    cmd.append("--confidence")
                    cmd.append(str(confidence_val))
                    logger.info(f"Using confidence value: {confidence_val}")
                except (ValueError, TypeError) as e:
                    error_msg = f"Invalid confidence value: {args['confidence']}. Must be a number."
                    logger.error(f"{error_msg} Error: {str(e)}")
                    return {"status": "error", "message": error_msg}
            
            # Add interval if provided
            if 'interval' in args:
                try:
                    # Handle the case where interval might be a string or a number
                    interval_input = args['interval']
                    # Make sure we handle empty strings
                    if interval_input == '':
                        interval_val = 60  # Default value
                    else:
                        interval_val = int(float(interval_input))
                    cmd.append("--interval")
                    cmd.append(str(interval_val))
                    logger.info(f"Using interval value: {interval_val}")
                except (ValueError, TypeError) as e:
                    error_msg = f"Invalid interval value: {args['interval']}. Must be an integer."
                    logger.error(f"{error_msg} Error: {str(e)}")
                    return {"status": "error", "message": error_msg}
                    
            # Log the final command
            logger.info(f"Final auto-execute command: {cmd}")
                    
        # Special handling for execute-trade command
        elif command == 'execute-trade':
            # Log the arguments for debugging
            logger.info(f"Execute trade command received with args: {args}")
            
            # Required parameters: symbol and side
            if 'symbol' not in args:
                return {"status": "error", "message": "Missing required parameter: symbol"}
            
            # Add symbol parameter
            symbol_val = str(args['symbol'])
            cmd.append("--symbol")
            cmd.append(symbol_val)
            
            # Add side parameter (must be BUY or SELL)
            if 'side' not in args:
                return {"status": "error", "message": "Missing required parameter: side"}
                
            side_val = str(args['side']).upper()
            if side_val not in ['BUY', 'SELL']:
                return {"status": "error", "message": f"Invalid side value: {side_val}. Must be 'BUY' or 'SELL'."}
            cmd.append("--side")
            cmd.append(side_val)
            
            # Add price if provided
            if 'price' in args:
                try:
                    # Handle the case where price might be a string or a number
                    price_val = args['price']
                    if isinstance(price_val, str):
                        price_val = float(price_val)
                    else:
                        price_val = float(price_val)
                    cmd.append("--price")
                    cmd.append(str(price_val))
                    logger.info(f"Using price: {price_val}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid price value: {args['price']}, error: {str(e)}")
                    return {"status": "error", "message": f"Invalid price value: {args['price']}. Must be a number."}
            
            # Add quantity if provided
            if 'quantity' in args:
                try:
                    quantity_val = float(args['quantity'])
                    cmd.append("--quantity")
                    cmd.append(str(quantity_val))
                    logger.info(f"Using quantity: {quantity_val}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid quantity value: {args['quantity']}, error: {str(e)}")
                    return {"status": "error", "message": f"Invalid quantity value: {args['quantity']}. Must be a number."}
            
            # Add confidence if provided
            if 'confidence' in args:
                try:
                    confidence_val = float(args['confidence'])
                    cmd.append("--confidence")
                    cmd.append(str(confidence_val))
                    logger.info(f"Using confidence: {confidence_val}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Invalid confidence value: {args['confidence']}, error: {str(e)}")
                    return {"status": "error", "message": f"Invalid confidence value: {args['confidence']}. Must be a number."}
                    
            # Log the final command
            logger.info(f"Final execute-trade command: {cmd}")
        else:
            # Standard argument handling for other commands
            for key, value in args.items():
                cmd.append(f"--{key}")
                cmd.append(str(value))
    
    # Log the full command we're about to run
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Set the working directory to the script directory
        logger.info(f"Running command in directory: {os.path.dirname(PAPER_TRADING_CLI)}")
        
        # Convert all arguments to strings
        cmd_str = [str(arg) for arg in cmd]
        
        # Run the command with shell=False for better security and argument handling
        result = subprocess.run(cmd_str, capture_output=True, text=True, check=False, cwd=os.path.dirname(PAPER_TRADING_CLI))
        
        # Log the complete output for debugging
        logger.info(f"Command stdout: {result.stdout}")
        logger.info(f"Command stderr: {result.stderr}")
        logger.info(f"Command exit code: {result.returncode}")
        
        if result.returncode != 0:
            logger.error(f"Command failed with exit code {result.returncode}: {result.stderr}")
            return {"status": "error", "message": result.stderr or "Unknown error occurred"}
            
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
        # Try multiple possible locations for the state file
        possible_paths = [
            STATE_FILE,
            os.path.join(TRADING_DATA_DIR, "paper_trading_status.json"),
            "/opt/lampp/htdocs/bot/paper_trading_state.json",
            "/opt/lampp/htdocs/bot/frontend/trading_data/paper_trading_state.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found state file at: {path}")
                with open(path, 'r') as f:
                    data = json.load(f)
                    
                    # Copy the state file to the frontend trading_data directory if needed
                    if path != os.path.join(TRADING_DATA_DIR, "paper_trading_status.json"):
                        try:
                            os.makedirs(TRADING_DATA_DIR, exist_ok=True)
                            with open(os.path.join(TRADING_DATA_DIR, "paper_trading_status.json"), 'w') as dest_f:
                                json.dump(data, dest_f, indent=2)
                            logger.info(f"Copied state file to frontend directory")
                        except Exception as copy_err:
                            logger.warning(f"Could not copy state file to frontend directory: {copy_err}")
                    
                    return {"status": "success", "data": data}
        
        # If no file is found, return a default status
        logger.warning("No state file found, returning default status")
        default_status = {
            "is_running": False,
            "mode": "paper",
            "balance": 10000,
            "holdings": {},
            "base_currency": "USDT",
            "message": "Paper trading state file not found. Please start paper trading with: python paper_trading_cli.py start"
        }
        return {"status": "success", "data": default_status}
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
        
        # Handle standard commands
        valid_commands = ['start', 'stop', 'status', 'switch', 'reset', 'export', 'auto-execute', 'execute-trade']
        if command not in valid_commands:
            return jsonify({"status": "error", "message": f"Invalid command: {command}"}), 400
        
        # For the switch command, we need a mode parameter
        if command == 'switch' and 'mode' not in data:
            return jsonify({"status": "error", "message": "Missing mode parameter for switch command"}), 400
            
        # For the auto-execute command, we need the enabled parameter
        if command == 'auto-execute' and 'enabled' not in data:
            return jsonify({"status": "error", "message": "Missing enabled parameter for auto-execute command"}), 400
            
        # Validate enabled parameter for auto-execute command
        if command == 'auto-execute' and data['enabled'] not in ['true', 'false']:
            return jsonify({
                "status": "error", 
                "message": f"Invalid enabled parameter: {data['enabled']}. Must be 'true' or 'false'."
            }), 400
            
        # For the execute-trade command, we need symbol, side, and price parameters
        if command == 'execute-trade':
            required_params = ['symbol', 'side']
            missing_params = [param for param in required_params if param not in data]
            
            if missing_params:
                return jsonify({
                    "status": "error", 
                    "message": f"Missing required parameters for execute-trade command: {', '.join(missing_params)}"
                }), 400
                
            # Validate side parameter
            if data['side'] not in ['BUY', 'SELL']:
                return jsonify({
                    "status": "error", 
                    "message": f"Invalid side parameter: {data['side']}. Must be 'BUY' or 'SELL'."
                }), 400
                
            # Convert price to float if it's a string
            if 'price' in data and isinstance(data['price'], str):
                try:
                    data['price'] = float(data['price'])
                except ValueError:
                    return jsonify({
                        "status": "error", 
                        "message": f"Invalid price parameter: {data['price']}. Must be a number."
                    }), 400
            
            # If price is not provided, use current market price (mock for now)
            if 'price' not in data:
                # Mock price data for testing
                mock_prices = {
                    'BTCUSDT': 52768.34,
                    'ETHUSDT': 3164.56,
                    'SOLUSDT': 148.92,
                    'ADAUSDT': 0.52,
                    'DOGEUSDT': 0.15,
                    'BNBUSDT': 610.23
                }
                
                symbol = data['symbol']
                if '/' in symbol:
                    symbol = symbol.replace('/', '')
                    
                if symbol in mock_prices:
                    data['price'] = mock_prices[symbol]
                else:
                    # Default price if symbol not found
                    data['price'] = 100.0
                    
            # Add confidence if not provided
            if 'confidence' not in data:
                data['confidence'] = 0.85  # Default high confidence
        
        # Run the command
        result = run_command(command, data)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error handling command: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    # Log startup information
    logger.info("Paper Trading API server starting")
    logger.info(f"Running in {'service' if RUNNING_AS_SERVICE else 'development'} mode")
    logger.info(f"Listening on port 5001")
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)
