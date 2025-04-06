#!/usr/bin/env python3
"""
Paper Trading Daemon - A background service for paper trading

This script runs the paper trading service in the background and properly maintains state.
It manages the PID file and ensures status checking works correctly.
"""

import os
import sys
import time
import signal
import subprocess
import json
from datetime import datetime

# Base directory for the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(BASE_DIR, 'paper_trading.pid')
LOG_FILE = os.path.join(BASE_DIR, 'logs', 'paper_trading.log')
STATE_FILE = os.path.join(BASE_DIR, 'frontend', 'trading_data', 'paper_trading_state.json')

def write_pid_file(pid):
    """Write the PID to a file for future reference"""
    with open(PID_FILE, 'w') as f:
        f.write(str(pid))
    print(f"PID file created at {PID_FILE}")

def read_pid_file():
    """Read the PID from the file"""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return None
    return None

def is_process_running(pid):
    """Check if a process with the given PID is running"""
    try:
        # Sending signal 0 to a process will raise an Exception if the process is not running
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def update_state_file(running=True):
    """Update the state file to indicate if the service is running"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
            
            state['is_running'] = running
            
            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error updating state file: {str(e)}")
            return False
    return False

def start_paper_trading():
    """Start the paper trading service in the background"""
    # Check if already running
    pid = read_pid_file()
    if pid and is_process_running(pid):
        print(f"Paper trading is already running (PID: {pid})")
        return
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    
    # Start the paper trading script
    print("Starting paper trading service...")
    
    # First run the stop command to ensure clean state
    subprocess.run([sys.executable, os.path.join(BASE_DIR, 'paper_trading_cli.py'), 'stop'], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Now start the service in background
    process = subprocess.Popen(
        [sys.executable, os.path.join(BASE_DIR, 'paper_trading_cli.py'), 'start'],
        stdout=open(LOG_FILE, 'a'),
        stderr=subprocess.STDOUT,
        start_new_session=True
    )
    
    # Write PID file
    write_pid_file(process.pid)
    
    # Update state file
    update_state_file(True)
    
    print(f"Paper trading started with PID: {process.pid}")
    print(f"Log file: {LOG_FILE}")

def stop_paper_trading():
    """Stop the paper trading service"""
    pid = read_pid_file()
    
    if not pid:
        print("No PID file found. Paper trading might not be running.")
        # Try to stop via the CLI anyway
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'paper_trading_cli.py'), 'stop'])
        return
    
    if is_process_running(pid):
        print(f"Stopping paper trading process (PID: {pid})...")
        
        # First try to stop cleanly using the CLI
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'paper_trading_cli.py'), 'stop'])
        
        # Give it a moment to stop gracefully
        time.sleep(2)
        
        # If still running, terminate it
        if is_process_running(pid):
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                if is_process_running(pid):
                    os.kill(pid, signal.SIGKILL)
            except OSError as e:
                print(f"Error stopping process: {str(e)}")
    else:
        print(f"Paper trading process (PID: {pid}) is not running")
    
    # Clean up PID file
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)
    
    # Update state file
    update_state_file(False)
    
    print("Paper trading stopped")

def check_status():
    """Check the status of the paper trading service"""
    pid = read_pid_file()
    
    if pid and is_process_running(pid):
        print(f"Paper trading is running (PID: {pid})")
        # Update state file if needed
        update_state_file(True)
        
        # Run the CLI status command to show details
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'paper_trading_cli.py'), 'status'])
    else:
        print("Paper trading is not running")
        # Update state file if needed
        update_state_file(False)
        
        # Clean up stale PID file if it exists
        if pid and os.path.exists(PID_FILE):
            os.remove(PID_FILE)
            
        # Still show status from CLI for information
        subprocess.run([sys.executable, os.path.join(BASE_DIR, 'paper_trading_cli.py'), 'status'])

def main():
    """Main function to process commands"""
    if len(sys.argv) < 2 or sys.argv[1] not in ['start', 'stop', 'status', 'restart']:
        print("Usage: python paper_trading_daemon.py [start|stop|status|restart]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'start':
        start_paper_trading()
    elif command == 'stop':
        stop_paper_trading()
    elif command == 'status':
        check_status()
    elif command == 'restart':
        stop_paper_trading()
        time.sleep(2)
        start_paper_trading()

if __name__ == "__main__":
    main()
