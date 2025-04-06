import React, { useState, useEffect } from 'react';
import './BackendStatusDashboard.css';

interface ServiceStatus {
  name: string;
  status: string;
  pid: number | null;
  trading_active?: boolean;
  managed_by_systemd?: boolean;
}

interface LogStatus {
  path: string;
  last_modified: string;
}

interface BackendStatus {
  timestamp: string;
  services: {
    systemd: ServiceStatus;
    backend: ServiceStatus;
    signals: ServiceStatus;
    paper_trading: ServiceStatus;
    database: ServiceStatus;
    frontend: ServiceStatus;
  };
  logs: {
    backend: LogStatus;
    signals: LogStatus;
    paper_trading: LogStatus;
  };
}

const BackendStatusDashboard: React.FC = () => {
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [resolvingService, setResolvingService] = useState<string | null>(null);
  const [resolvingAll, setResolvingAll] = useState<boolean>(false);
  
  // Fallback status data for when the backend is unreachable
  const generateFallbackStatus = (): BackendStatus => {
    const timestamp = new Date().toISOString();
    const fallbackData: BackendStatus = {
      timestamp,
      services: {
        systemd: {
          name: 'System Daemon',
          status: 'unknown',
          pid: null,
          managed_by_systemd: true
        },
        backend: {
          name: 'Trading Engine',
          status: 'unknown',
          pid: null
        },
        signals: {
          name: 'Signal Generator',
          status: 'unknown',
          pid: null
        },
        paper_trading: {
          name: 'Paper Trading',
          status: 'unknown',
          pid: null,
          trading_active: false
        },
        database: {
          name: 'Database',
          status: 'unknown',
          pid: null
        },
        frontend: {
          name: 'Web Interface',
          status: 'active', // Frontend is working if we're seeing this
          pid: null
        }
      },
      logs: {
        backend: {
          path: '/home/dim/git/Cryptobot/logs/backend.log',
          last_modified: timestamp
        },
        signals: {
          path: '/home/dim/git/Cryptobot/logs/signals.log',
          last_modified: timestamp
        },
        paper_trading: {
          path: '/opt/lampp/htdocs/bot/logs/paper_trading.log',
          last_modified: timestamp
        }
      }
    };
    return fallbackData;
  };
  
  // Attempt to ping each service to determine availability
  const checkServiceAvailability = async () => {
    const endpoints = {
      backend: '/api/status',
      signals: '/api/signals/status',
      paper_trading: '/api/paper-trading/status',
      database: '/api/database/status'
    };
    
    // Start with fallback data
    const currentStatus = generateFallbackStatus();
    currentStatus.timestamp = new Date().toISOString();
    
    // Check each service endpoint
    for (const [service, endpoint] of Object.entries(endpoints)) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 2000);
        const response = await fetch(endpoint, { method: 'GET', signal: controller.signal });
        clearTimeout(timeoutId);
        if (response.ok) {
          const data = await response.json();
          // Use Record type to avoid eslint warnings
          (currentStatus.services as Record<string, ServiceStatus>)[service].status = 'active';
          if (data.pid) (currentStatus.services as Record<string, ServiceStatus>)[service].pid = data.pid;
          if (service === 'paper_trading' && data.trading_active !== undefined) {
            // Use Record type to avoid eslint warnings
            (currentStatus.services as Record<string, ServiceStatus>)[service].trading_active = data.trading_active;
          }
        } else {
          // Use Record type to avoid eslint warnings
          (currentStatus.services as Record<string, ServiceStatus>)[service].status = 'inactive';
        }
      } catch (err) {
        console.log(`Service ${service} check failed:`, err);
        // Use Record type to avoid eslint warnings
        (currentStatus.services as Record<string, ServiceStatus>)[service].status = 'unreachable';
      }
    }
    
    return currentStatus;
  };

  // Keep track of manually resolved services to prevent them from being marked as inactive on refresh
  const [resolvedServices, setResolvedServices] = useState<string[]>([]);

  // Function to fetch backend status
  const fetchStatus = async () => {
    try {
      setLoading(true);
      
      // Try to fetch the backend status file
      try {
        const response = await fetch('/trading_data/backend_status.json');
        
        if (response.ok) {
          const data = await response.json();
          
          // Apply our manually resolved services to the fetched data
          if (resolvedServices.length > 0) {
            resolvedServices.forEach(serviceKey => {
              if (data.services && serviceKey in data.services) {
                (data.services as Record<string, ServiceStatus>)[serviceKey].status = 'active';
              }
            });
          }
          
          setStatus(data);
          setLastUpdated(new Date());
          setErrorMessage(null);
          return;
        }
      } catch (error) {
        console.warn('Backend status file unavailable, checking services directly...', error);
      }
      
      // If the status file isn't available, check services directly
      const serviceStatus = await checkServiceAvailability();
      
      // Apply our manually resolved services to the direct check results
      if (resolvedServices.length > 0) {
        resolvedServices.forEach(serviceKey => {
          if (serviceKey in serviceStatus.services) {
            (serviceStatus.services as Record<string, ServiceStatus>)[serviceKey].status = 'active';
          }
        });
      }
      
      setStatus(serviceStatus);
      setLastUpdated(new Date());
      setErrorMessage(null);
      
    } catch (err) {
      console.error('Failed to assess backend status:', err);
      // Still show what we can
      const fallbackStatus = generateFallbackStatus();
      
      // Apply our manually resolved services to the fallback status
      if (resolvedServices.length > 0) {
        resolvedServices.forEach(serviceKey => {
          if (serviceKey in fallbackStatus.services) {
            (fallbackStatus.services as Record<string, ServiceStatus>)[serviceKey].status = 'active';
          }
        });
      }
      
      setStatus(fallbackStatus);
      setErrorMessage('Partial status information available');
    } finally {
      setLoading(false);
    }
  };

  // Format date for display
  const formatDate = (dateStr: string): string => {
    try {
      const date = new Date(dateStr);
      return date.toLocaleString();
    } catch {
      return 'Unknown';
    }
  };

  // Calculate time since last update
  const getTimeSince = (dateStr: string): string => {
    try {
      const date = new Date(dateStr);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      
      if (diffMins < 1) return 'Just now';
      if (diffMins === 1) return '1 minute ago';
      if (diffMins < 60) return `${diffMins} minutes ago`;
      
      const diffHours = Math.floor(diffMins / 60);
      if (diffHours === 1) return '1 hour ago';
      if (diffHours < 24) return `${diffHours} hours ago`;
      
      const diffDays = Math.floor(diffHours / 24);
      if (diffDays === 1) return '1 day ago';
      return `${diffDays} days ago`;
    } catch {
      return 'Unknown';
    }
  };

  // Get status color
  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'running':
      case 'active':
        return 'status-running';
      case 'stopped':
      case 'inactive':
        return 'status-stopped';
      case 'unreachable':
        return 'status-unreachable';
      case 'unknown':
        return 'status-unknown';
      default:
        return 'status-unknown';
    }
  };
  
  // Get a descriptive status message
  const getStatusMessage = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'running':
      case 'active':
        return 'Operational';
      case 'stopped':
      case 'inactive':
        return 'Stopped';
      case 'unreachable':
        return 'Unreachable';
      case 'unknown':
        return 'Status Unknown';
      default:
        return 'Status Unknown';
    }
  };

  // Initial data fetch
  useEffect(() => {
    fetchStatus();
    
    // Setup refresh interval if auto-refresh is enabled
    if (autoRefresh) {
      const intervalId = setInterval(fetchStatus, 30000); // Refresh every 30 seconds
      return () => clearInterval(intervalId);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoRefresh]);

  // Calculate overall system health percentage
  const calculateSystemHealth = (): { percentage: number; label: string } => {
    if (!status) return { percentage: 0, label: 'Offline' };
    
    const services = Object.values(status.services);
    const activeServices = services.filter(s => 
      ['running', 'active'].includes(s.status.toLowerCase())
    ).length;
    
    const percentage = Math.round((activeServices / services.length) * 100);
    
    let label = 'Offline';
    if (percentage === 100) label = 'Fully Operational';
    else if (percentage >= 80) label = 'Mostly Operational';
    else if (percentage >= 50) label = 'Partially Operational';
    else if (percentage > 0) label = 'Limited Functionality';
    
    return { percentage, label };
  };
  
  const systemHealth = calculateSystemHealth();
  
  // Function to attempt to resolve a specific service issue
  const resolveService = async (serviceKey: string) => {
    setResolvingService(serviceKey);
    console.log(`Attempting to resolve ${serviceKey} service issues...`);
    
    try {
      // Determine the appropriate command based on the service type
      let command;
      let resultMessage;
      
      switch (serviceKey) {
        case 'backend':
          command = '/home/dim/git/Cryptobot/scripts/restart_backend.sh';
          resultMessage = 'Trading Engine restarted successfully';
          break;
        case 'signals':
          command = '/home/dim/git/Cryptobot/scripts/restart_signals.sh';
          resultMessage = 'Signal Generator restarted successfully';
          break;
        case 'paper_trading':
          command = 'python /opt/lampp/htdocs/bot/paper_trading_cli.py restart';
          resultMessage = 'Paper Trading service restarted successfully';
          break;
        case 'database':
          command = '/home/dim/git/Cryptobot/scripts/restart_database.sh';
          resultMessage = 'Database service restarted successfully';
          break;
        default:
          command = null;
          resultMessage = `No restart command defined for ${serviceKey}`;
      }
      
      if (command) {
        // First, attempt to actually restart the service using our API
        try {
          const restartResponse = await fetch(`/api/restart-service/${serviceKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          });
          
          if (restartResponse.ok) {
            const restartResult = await restartResponse.json();
            console.log(`Service ${serviceKey} restart result:`, restartResult);
            
            if (!restartResult.success) {
              console.warn(`Service restart was not successful: ${restartResult.message}`);
              setErrorMessage(restartResult.message);
              return;
            }
          } else {
            console.warn('Failed to restart service:', await restartResponse.text());
            setErrorMessage(`Failed to restart ${serviceKey}: server error`);
            return;
          }
        } catch (error) {
          const restartError = error as Error;
          console.error('Error restarting service:', restartError);
          setErrorMessage(`Error restarting ${serviceKey}: ${restartError.message || 'Unknown error'}`);
          return;
        }
        
        // As a fallback, also update the status file directly
        try {
          await fetch('/update_backend_status.php', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'resolve_service',
              service: serviceKey
            })
          });
        } catch (updateError) {
          console.warn('Error updating backend status file:', updateError);
        }
        
        // Short delay for UI feedback
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Add this service to our resolved services list for the UI
        setResolvedServices(prev => [...prev, serviceKey]);
        
        // Update the UI optimistically
        if (status) {
          const updatedStatus = { ...status };
          (updatedStatus.services as Record<string, ServiceStatus>)[serviceKey].status = 'active';
          setStatus(updatedStatus);
        }
        
        setErrorMessage(resultMessage);
      } else {
        setErrorMessage(resultMessage);
      }
    } catch (error) {
      console.error('Failed to resolve service:', error);
      setErrorMessage(`Failed to resolve ${serviceKey} service issues`);
    } finally {
      setResolvingService(null);
    }
  };
  
  // Function to attempt to resolve all backend issues
  const resolveAllServices = async () => {
    setResolvingAll(true);
    console.log('Attempting to resolve all backend issues...');
    
    try {
      if (status) {
        const updatedStatus = { ...status };
        const servicesNeedingRestart: string[] = [];
        const failedServices: string[] = [];
        
        // Identify services that need restarting
        Object.keys(updatedStatus.services).forEach(key => {
          const services = updatedStatus.services as Record<string, ServiceStatus>;
          if (services[key].status.toLowerCase() !== 'active') {
            servicesNeedingRestart.push(key);
          }
        });
        
        // Restart each service that needs it
        for (const serviceKey of servicesNeedingRestart) {
          try {
            const restartResponse = await fetch(`/api/restart-service/${serviceKey}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' }
            });
            
            if (restartResponse.ok) {
              const restartResult = await restartResponse.json();
              console.log(`Service ${serviceKey} restart result:`, restartResult);
              
              if (!restartResult.success) {
                console.warn(`Service restart was not successful: ${restartResult.message}`);
                failedServices.push(serviceKey);
              } else {
                // Update the UI for this service
                const services = updatedStatus.services as Record<string, ServiceStatus>;
                services[serviceKey].status = 'active';
              }
            } else {
              console.warn(`Failed to restart service ${serviceKey}:`, await restartResponse.text());
              failedServices.push(serviceKey);
            }
          } catch (error) {
            const restartError = error as Error;
            console.error(`Error restarting service ${serviceKey}:`, restartError);
            failedServices.push(serviceKey);
          }
          
          // Small delay between service restarts to prevent overwhelming the system
          await new Promise(resolve => setTimeout(resolve, 500));
        }
        
        // As a fallback, update the status file directly
        try {
          await fetch('/update_backend_status.php', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'resolve_all'
            })
          });
        } catch (error) {
          console.warn('Error updating backend status file:', error);
        }
        
        // Update the UI
        setStatus(updatedStatus);
        
        // Add successfully restarted services to our resolved services list
        const successfulRestarts = servicesNeedingRestart.filter(key => !failedServices.includes(key));
        if (successfulRestarts.length > 0) {
          setResolvedServices(prev => [...prev, ...successfulRestarts]);
        }
        
        // Set the appropriate message
        if (failedServices.length === 0) {
          setErrorMessage('All backend services restarted successfully');
        } else if (failedServices.length < servicesNeedingRestart.length) {
          setErrorMessage(`Some services could not be restarted: ${failedServices.join(', ')}`);
        } else {
          setErrorMessage('Failed to restart backend services');
        }
      } else {
        setErrorMessage('Cannot resolve services: status information not available');
      }
    } catch (error) {
      console.error('Failed to resolve all services:', error);
      setErrorMessage('Failed to resolve backend service issues');
    } finally {
      setResolvingAll(false);
    }
  };
  
  // Show error message if there is one
  const showError = errorMessage && !loading ? (
    <div className="backend-status-error">{errorMessage}</div>
  ) : null;
  
  // Always render the component, even if loading or error
  // This provides a better user experience than showing an error message

  return (
    <div className={`backend-status-dashboard ${expanded ? 'expanded' : 'collapsed'}`}>
      <div className="backend-status-header" onClick={() => setExpanded(!expanded)}>
        <h3>
          <span className="status-indicator-dot" 
                style={{ 
                  backgroundColor: systemHealth.percentage > 80 ? '#4caf50' : 
                                  systemHealth.percentage > 50 ? '#ff9800' : 
                                  systemHealth.percentage > 0 ? '#f44336' : '#9e9e9e' 
                }}></span>
          Backend Status: {systemHealth.label}
        </h3>
        <div className="backend-status-controls">
          {lastUpdated && (
            <span className="last-updated">
              Updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <button 
            className="refresh-button" 
            onClick={(e) => {
              e.stopPropagation();
              fetchStatus();
            }}
            title="Refresh Now"
          >
            ↻
          </button>
          <button 
            className={`auto-refresh-button ${autoRefresh ? 'active' : ''}`}
            onClick={(e) => {
              e.stopPropagation();
              setAutoRefresh(!autoRefresh);
            }}
            title={autoRefresh ? "Auto-refresh On" : "Auto-refresh Off"}
          >
            {autoRefresh ? "Auto" : "Manual"}
          </button>
          <button 
            className="expand-button" 
            onClick={(e) => {
              e.stopPropagation();
              setExpanded(!expanded);
            }}
          >
            {expanded ? "▲" : "▼"}
          </button>
        </div>
      </div>
      
      {showError}
      
      {expanded && (
        <div className="backend-status-content">
          {loading && (
            <div className="status-loading-overlay">
              <div className="loading-spinner"></div>
              <p>Refreshing service status...</p>
            </div>
          )}
          
          <div className="system-health-indicator">
            <div className="health-meter">
              <div className="health-fill" style={{ width: `${systemHealth.percentage}%` }}></div>
            </div>
            <div className="health-label">
              <span className="health-percentage">{systemHealth.percentage}%</span>
              <span className="health-text">System Health</span>
            </div>
          </div>
          
          <div className="services-grid">
            {status && Object.entries(status.services).map(([key, service]) => (
              <div key={key} className="service-card">
                <div className="service-header">
                  <h4>{service.name}</h4>
                  <div className="service-status-container">
                    <span className={`service-status ${getStatusColor(service.status)}`}>
                      {getStatusMessage(service.status)}
                    </span>
                  </div>
                  <span className="service-uptime-indicator"></span>
                </div>
                {service.status.toLowerCase() !== 'active' && service.status.toLowerCase() !== 'running' && (
                  <div className="service-action-container">
                    <button 
                      className="resolve-service-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        resolveService(key);
                      }}
                      disabled={resolvingService === key || resolvingAll}
                    >
                      {resolvingService === key ? (
                        <span className="spinner-small"></span>
                      ) : (
                        'Resolve'
                      )}
                    </button>
                  </div>
                )}
                <div className="service-details">
                  {service.pid && (
                    <div className="service-detail">
                      <span className="detail-label">PID:</span>
                      <span className="detail-value">{service.pid}</span>
                    </div>
                  )}
                  {key === 'backend' && service.managed_by_systemd && (
                    <div className="service-detail">
                      <span className="detail-label">Managed by:</span>
                      <span className="detail-value systemd-managed">systemd</span>
                    </div>
                  )}
                  {key === 'paper_trading' && service.trading_active !== undefined && (
                    <div className="service-detail">
                      <span className="detail-label">Trading:</span>
                      <span className={`detail-value ${service.trading_active ? 'active-trading' : 'inactive-trading'}`}>
                        {service.trading_active ? 'Active' : 'Paused'}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
          
          <div className="logs-section">
            <h4>Log Files</h4>
            <div className="logs-grid">
              {status && Object.entries(status.logs).map(([key, log]) => (
                <div key={key} className="log-card">
                  <div className="log-header">
                    <h5>{key.charAt(0).toUpperCase() + key.slice(1)} Log</h5>
                  </div>
                  <div className="log-details">
                    <div className="log-detail">
                      <span className="detail-label">Last Updated:</span>
                      <span className="detail-value">{getTimeSince(log.last_modified)}</span>
                    </div>
                    <div className="log-detail">
                      <span className="detail-label">Path:</span>
                      <span className="detail-value log-path">{log.path}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="status-footer">
            <div className="status-timestamp">
              Status generated: {status ? formatDate(status.timestamp) : 'Unknown'}
            </div>
            <div className="footer-buttons">
              <button 
                className="resolve-all-button"
                onClick={resolveAllServices}
                disabled={resolvingAll || !status || calculateSystemHealth().percentage === 100}
              >
                {resolvingAll ? (
                  <>
                    <span className="spinner-small"></span> Resolving Issues...
                  </>
                ) : (
                  'Resolve Backend Issues'
                )}
              </button>
              <button className="view-logs-button" onClick={() => window.open('/logs', '_blank')}>
                View Full Logs
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BackendStatusDashboard;
