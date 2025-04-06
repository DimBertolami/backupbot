<?php
/**
 * API Service for Cryptobot
 * This script provides API endpoints for controlling the backend services from the frontend
 */

// Set headers for JSON response
header('Content-Type: application/json');

// Define constants
define('SCRIPT_DIR', __DIR__);
define('PAPER_TRADING_CLI', SCRIPT_DIR . '/paper_trading_cli.py');
define('STATUS_FILE', SCRIPT_DIR . '/frontend/trading_data/backend_status.json');
define('LOG_DIR', SCRIPT_DIR . '/logs');
define('PYTHON_BIN', 'python');

// Ensure log directory exists
if (!is_dir(LOG_DIR)) {
    mkdir(LOG_DIR, 0755, true);
}

// Get request method and URI
$method = $_SERVER['REQUEST_METHOD'];
$uri = $_SERVER['REQUEST_URI'];

// Process endpoint
if (preg_match('/\/api\/restart-service\/([a-z_]+)/', $uri, $matches)) {
    if ($method === 'POST') {
        $service = $matches[1];
        restartService($service);
    } else {
        http_response_code(405);
        echo json_encode(['error' => 'Method not allowed']);
    }
} elseif (preg_match('/\/api\/service-status\/([a-z_]+)/', $uri, $matches)) {
    if ($method === 'GET') {
        $service = $matches[1];
        getServiceStatus($service);
    } else {
        http_response_code(405);
        echo json_encode(['error' => 'Method not allowed']);
    }
} else {
    http_response_code(404);
    echo json_encode(['error' => 'Endpoint not found']);
}

/**
 * Restart a specific service
 * @param string $service The service to restart
 */
function restartService($service) {
    $result = ['success' => false, 'message' => ''];
    
    switch ($service) {
        case 'paper_trading':
            // First stop the service if it's running
            exec(PYTHON_BIN . ' ' . PAPER_TRADING_CLI . ' stop 2>&1', $stopOutput, $stopExitCode);
            
            // Wait a moment to ensure it's stopped
            sleep(1);
            
            // Then start the service
            exec(PYTHON_BIN . ' ' . PAPER_TRADING_CLI . ' start 2>&1', $startOutput, $startExitCode);
            
            if ($startExitCode === 0) {
                $result['success'] = true;
                $result['message'] = 'Paper trading restarted successfully';
                
                // Update the status file
                updateStatusFile($service, 'active');
            } else {
                $result['message'] = 'Failed to restart paper trading: ' . implode("\n", $startOutput);
                error_log("Error restarting paper trading: " . implode("\n", $startOutput));
            }
            break;
            
        case 'backend':
            exec('/home/dim/git/Cryptobot/scripts/restart_backend.sh 2>&1', $output, $exitCode);
            
            if ($exitCode === 0) {
                $result['success'] = true;
                $result['message'] = 'Backend restarted successfully';
                updateStatusFile($service, 'active');
            } else {
                $result['message'] = 'Failed to restart backend: ' . implode("\n", $output);
                error_log("Error restarting backend: " . implode("\n", $output));
            }
            break;
            
        case 'signals':
            exec('/home/dim/git/Cryptobot/scripts/restart_signals.sh 2>&1', $output, $exitCode);
            
            if ($exitCode === 0) {
                $result['success'] = true;
                $result['message'] = 'Signals service restarted successfully';
                updateStatusFile($service, 'active');
            } else {
                $result['message'] = 'Failed to restart signals service: ' . implode("\n", $output);
                error_log("Error restarting signals: " . implode("\n", $output));
            }
            break;
            
        case 'database':
            exec('/home/dim/git/Cryptobot/scripts/restart_database.sh 2>&1', $output, $exitCode);
            
            if ($exitCode === 0) {
                $result['success'] = true;
                $result['message'] = 'Database restarted successfully';
                updateStatusFile($service, 'active');
            } else {
                $result['message'] = 'Failed to restart database: ' . implode("\n", $output);
                error_log("Error restarting database: " . implode("\n", $output));
            }
            break;
            
        default:
            $result['message'] = 'Unknown service: ' . $service;
            http_response_code(400);
            break;
    }
    
    echo json_encode($result);
}

/**
 * Get the status of a specific service
 * @param string $service The service to check
 */
function getServiceStatus($service) {
    $result = ['status' => 'unknown'];
    
    switch ($service) {
        case 'paper_trading':
            exec(PYTHON_BIN . ' ' . PAPER_TRADING_CLI . ' status 2>&1', $output, $exitCode);
            
            if ($exitCode === 0) {
                $status = strpos(implode("\n", $output), 'Status: Running') !== false ? 'active' : 'inactive';
                $result['status'] = $status;
            }
            break;
            
        // Additional services can be added here
            
        default:
            $result['message'] = 'Unknown service: ' . $service;
            http_response_code(400);
            break;
    }
    
    echo json_encode($result);
}

/**
 * Update the status file for a service
 * @param string $service The service to update
 * @param string $status The new status
 */
function updateStatusFile($service, $status) {
    if (!file_exists(STATUS_FILE)) {
        error_log("Status file not found: " . STATUS_FILE);
        return false;
    }
    
    try {
        // Read the current status file
        $statusData = json_decode(file_get_contents(STATUS_FILE), true);
        
        // Update the service status if it exists
        if (isset($statusData['services'][$service])) {
            $statusData['services'][$service]['status'] = $status;
            
            // Update timestamp
            $statusData['timestamp'] = date('c');
            
            // Save the updated status back to file
            file_put_contents(STATUS_FILE, json_encode($statusData, JSON_PRETTY_PRINT));
            return true;
        }
    } catch (Exception $e) {
        error_log("Error updating service status: " . $e->getMessage());
    }
    
    return false;
}
