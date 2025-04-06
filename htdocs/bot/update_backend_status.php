<?php
/**
 * Backend Status Updater
 * This script receives service update requests from the frontend and updates
 * the backend_status.json file to persist resolved service statuses.
 */

// Set headers for JSON response
header('Content-Type: application/json');

// Define constants
define('STATUS_FILE', __DIR__ . '/frontend/trading_data/backend_status.json');
define('BACKUP_FILE', __DIR__ . '/frontend/trading_data/backend_status.backup.json');

// Get request method and body
$method = $_SERVER['REQUEST_METHOD'];
$input = json_decode(file_get_contents('php://input'), true);

// Handle different request types
if ($method === 'POST' && isset($input['action'])) {
    switch ($input['action']) {
        case 'resolve_service':
            if (isset($input['service'])) {
                $result = resolveService($input['service']);
                echo json_encode(['success' => $result, 'message' => $result ? 'Service resolved successfully' : 'Failed to resolve service']);
            } else {
                http_response_code(400);
                echo json_encode(['success' => false, 'message' => 'Missing service parameter']);
            }
            break;
            
        case 'resolve_all':
            $result = resolveAllServices();
            echo json_encode(['success' => $result, 'message' => $result ? 'All services resolved successfully' : 'Failed to resolve services']);
            break;
            
        default:
            http_response_code(400);
            echo json_encode(['success' => false, 'message' => 'Invalid action']);
    }
} else {
    http_response_code(405);
    echo json_encode(['success' => false, 'message' => 'Method not allowed or missing action parameter']);
}

/**
 * Updates a specific service's status to 'active' in the backend_status.json file
 * 
 * @param string $service The service key to update
 * @return bool Whether the update was successful
 */
function resolveService($service) {
    if (!file_exists(STATUS_FILE)) {
        error_log("Backend status file not found: " . STATUS_FILE);
        return false;
    }
    
    try {
        // Create a backup
        copy(STATUS_FILE, BACKUP_FILE);
        
        // Read the current status file
        $status = json_decode(file_get_contents(STATUS_FILE), true);
        
        // Update the service status if it exists
        if (isset($status['services'][$service])) {
            // Update status to active
            $status['services'][$service]['status'] = 'active';
            
            // Update timestamp
            $status['timestamp'] = date('c');
            
            // Save the updated status back to file
            $result = file_put_contents(STATUS_FILE, json_encode($status, JSON_PRETTY_PRINT));
            return $result !== false;
        } else {
            error_log("Service not found in status file: " . $service);
            return false;
        }
    } catch (Exception $e) {
        error_log("Error updating service status: " . $e->getMessage());
        // Restore from backup if available
        if (file_exists(BACKUP_FILE)) {
            copy(BACKUP_FILE, STATUS_FILE);
        }
        return false;
    }
}

/**
 * Updates all services' status to 'active' in the backend_status.json file
 * 
 * @return bool Whether the update was successful
 */
function resolveAllServices() {
    if (!file_exists(STATUS_FILE)) {
        error_log("Backend status file not found: " . STATUS_FILE);
        return false;
    }
    
    try {
        // Create a backup
        copy(STATUS_FILE, BACKUP_FILE);
        
        // Read the current status file
        $status = json_decode(file_get_contents(STATUS_FILE), true);
        
        // Update all service statuses
        foreach ($status['services'] as $key => $service) {
            $status['services'][$key]['status'] = 'active';
        }
        
        // Update timestamp
        $status['timestamp'] = date('c');
        
        // Save the updated status back to file
        $result = file_put_contents(STATUS_FILE, json_encode($status, JSON_PRETTY_PRINT));
        return $result !== false;
    } catch (Exception $e) {
        error_log("Error updating service statuses: " . $e->getMessage());
        // Restore from backup if available
        if (file_exists(BACKUP_FILE)) {
            copy(BACKUP_FILE, STATUS_FILE);
        }
        return false;
    }
}
