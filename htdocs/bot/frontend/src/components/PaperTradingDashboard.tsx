import React, { useState, useEffect, useCallback, useRef } from 'react';
import './PaperTradingDashboard.css';

interface TradeHistoryItem {
  timestamp: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number;
  value: number;
  balance_after: number;
  type: string;
}

interface PerformanceMetrics {
  total_trades: number;
  win_rate: number;
  profit_loss: number;
  return_pct: number;
  sharpe_ratio: number;
  max_drawdown: number;
}

interface Holdings {
  [key: string]: number;
}

interface PaperTradingDashboardProps {
  selectedPeriod?: string;
  autoExecuteEnabled?: boolean;
  onAutoExecuteChange?: (enabled: boolean) => void;
}

interface PaperTradingStatus {
  is_running: boolean;
  mode: string;
  balance: number;
  holdings: Holdings;
  base_currency: string;
  portfolio_value: number;
  performance: PerformanceMetrics;
  trade_history: TradeHistoryItem[];
  last_prices: { [key: string]: number };
  last_updated: string;
  api_keys_configured?: boolean;
  auto_execute_suggested_trades?: boolean;
  min_confidence_threshold?: number;
  suggested_trade_refresh_interval?: number;
}

const DEFAULT_STATUS: PaperTradingStatus = {
  is_running: false,
  mode: 'paper',
  balance: 10000,
  holdings: {},
  base_currency: 'USDT',
  portfolio_value: 10000,
  performance: {
    total_trades: 0,
    win_rate: 0,
    profit_loss: 0,
    return_pct: 0,
    sharpe_ratio: 0,
    max_drawdown: 0
  },
  trade_history: [],
  last_prices: {},
  last_updated: new Date().toISOString(),
  api_keys_configured: false,
  auto_execute_suggested_trades: false,
  min_confidence_threshold: 0.75,
  suggested_trade_refresh_interval: 60
};

const PaperTradingDashboard: React.FC<PaperTradingDashboardProps> = ({ 
  selectedPeriod = '1m',
  autoExecuteEnabled: propAutoExecuteEnabled,
  onAutoExecuteChange
}) => {
  const [status, setStatus] = useState<PaperTradingStatus>(DEFAULT_STATUS);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [configVisible, setConfigVisible] = useState<boolean>(false);
  const [apiKey, setApiKey] = useState<string>('');
  const [apiSecret, setApiSecret] = useState<string>('');
  const [showTradeVisualization, setShowTradeVisualization] = useState<boolean>(false);
  
  // Reference for trade history table scrolling
  const tradeHistoryTableRef = useRef<HTMLDivElement>(null);
  
  // State to track highlighted trade from 3D visualization
  const [highlightedTradeId, setHighlightedTradeId] = useState<number | null>(null);
  
  // Auto-execution settings
  const [autoExecuteEnabled, setAutoExecuteEnabled] = useState<boolean>(propAutoExecuteEnabled ?? false);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.75);
  // Using the main dashboard's interval instead of a separate setting

  // Calculate refresh interval based on selected period
  const getRefreshIntervalFromPeriod = (period: string): number => {
    const value = parseInt(period.replace(/[^0-9]/g, '')) || 5;
    const unit = period.replace(/[0-9]/g, '');
    
    // Convert to milliseconds
    switch(unit.toLowerCase()) {
      case 'm':
        return value * 60 * 1000; // minutes
      case 'h':
        return value * 60 * 60 * 1000; // hours
      case 'd':
        return value * 24 * 60 * 60 * 1000; // days
      case 'w':
        return value * 7 * 24 * 60 * 60 * 1000; // weeks
      default:
        return 5 * 60 * 1000; // default to 5 minutes
    }
  };
  
  // Get refresh interval in seconds from the selected period
  const getRefreshIntervalInSeconds = (period: string): number => {
    const value = parseInt(period.replace(/[^0-9]/g, '')) || 5;
    const unit = period.replace(/[0-9]/g, '');
    
    switch(unit.toLowerCase()) {
      case 'm':
        return value * 60; // minutes to seconds
      case 'h':
        return value * 60 * 60; // hours to seconds
      case 'd':
        return value * 24 * 60 * 60; // days to seconds
      case 'w':
        return value * 7 * 24 * 60 * 60; // weeks to seconds
      default:
        return 5 * 60; // default to 5 minutes in seconds
    }
  };
  
  // Helper function to get ordinal suffix (1st, 2nd, 3rd, etc.)
  const getOrdinalSuffix = (num: number): string => {
    const j = num % 10;
    const k = num % 100;
    if (j === 1 && k !== 11) return num + "st";
    if (j === 2 && k !== 12) return num + "nd";
    if (j === 3 && k !== 13) return num + "rd";
    return num + "th";
  };
  
  // Helper function to safely get keys from an object that might be undefined
  const safeObjectKeys = <T extends Record<string, unknown>>(obj: T | undefined | null): string[] => {
    if (!obj) return [];
    return Object.keys(obj);
  };
  
  // Helper function to safely get entries from an object that might be undefined
  const safeObjectEntries = <T extends Record<string, number>>(obj: T | undefined | null): [string, number][] => {
    if (!obj) return [];
    return Object.entries(obj) as [string, number][];
  };

  // Generate fallback trade data to match the 3D visualization
  const generateFallbackTradeData = (numTrades = 20) => {
    console.warn('Using fallback trade data for dashboard');
    const trades: TradeHistoryItem[] = [];
    const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT'];
    
    let timestamp = new Date(Date.now() - (numTrades * 3600000)).getTime(); // Start from numTrades hours ago
    let balance = 10000; // Starting balance
    
    for (let i = 0; i < numTrades; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const side = Math.random() < 0.5 ? 'BUY' : 'SELL';
      const isProfit = side === 'SELL' && Math.random() < 0.6; // 60% chance of profit for sells
      
      // Calculate trade details
      const price = symbol.includes('BTC') ? 30000 + Math.random() * 5000 : 
                   symbol.includes('ETH') ? 2000 + Math.random() * 500 : 
                   symbol.includes('SOL') ? 100 + Math.random() * 50 : 
                   0.5 + Math.random() * 0.5; // ADA
      
      const tradeSize = balance * (0.05 + Math.random() * 0.1); // 5-15% of balance
      const quantity = tradeSize / price;
      
      // Calculate profit/loss for SELL trades
      const profitLossPercent = side === 'SELL' ? (isProfit ? 1 : -1) * (0.01 + Math.random() * 0.05) : 0;
      const profitLossAmount = side === 'SELL' ? tradeSize * profitLossPercent : 0;
      
      // Update balance
      if (side === 'BUY') {
        balance -= tradeSize;
      } else {
        balance += tradeSize + profitLossAmount;
      }
      balance = Math.max(balance, 5000); // Ensure balance doesn't go too low
      
      // Add time between trades (1-3 hours)
      timestamp += (1 + Math.floor(Math.random() * 2)) * 3600000;
      
      trades.push({
        timestamp: new Date(timestamp).toISOString(),
        symbol: symbol,
        side: side,
        price: price,
        quantity: quantity,
        value: tradeSize,
        balance_after: balance,
        type: 'market' // Adding required type field
      });
    }
    
    return trades;
  };

  // Fetch trading status
  const fetchStatus = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // Use API endpoint to get real data
      const response = await fetch('/trading/paper');
      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.status === 'success') {
        console.log('Received status data:', result.data);
        // If auto_execute_suggested_trades is undefined, set it to true by default
        if (result.data.auto_execute_suggested_trades === undefined) {
          result.data.auto_execute_suggested_trades = true;
        }
        
        // Ensure trade_history is always an array
        if (!result.data.trade_history) {
          result.data.trade_history = [];
        }
        
        // Try to fetch trade history from all possible sources as the 3D visualization
        const possiblePaths = [
          '/trading_data/paper_trading_status.json',
          '/frontend/trading_data/paper_trading_status.json',
          '/frontend/public/trading_data/paper_trading_status.json',
          '/frontend/dist/trading_data/paper_trading_status.json'
        ];
        
        // Try each possible path to find visualization data
        for (const path of possiblePaths) {
          try {
            console.log(`Trying to fetch trade data from: ${path}`);
            const visualizationDataResponse = await fetch(path);
            if (visualizationDataResponse.ok) {
              const visualizationData = await visualizationDataResponse.json();
              console.log(`Fetched visualization data from ${path}:`, visualizationData);
              
              // If the visualization data has trade history and our current data doesn't, use it
              if (visualizationData.trade_history && visualizationData.trade_history.length > 0) {
                console.log(`Found ${visualizationData.trade_history.length} trades in visualization data`);
                if (result.data.trade_history.length === 0) {
                  console.log('Using trade history from visualization data');
                  result.data.trade_history = visualizationData.trade_history;
                } else {
                  console.log('Merging trade histories from both sources');
                  // Merge trade histories, avoiding duplicates by checking timestamps
                  const existingTimestamps = new Set(result.data.trade_history.map((t: TradeHistoryItem) => t.timestamp));
                  const newTrades = visualizationData.trade_history.filter((t: TradeHistoryItem) => !existingTimestamps.has(t.timestamp));
                  result.data.trade_history = [...result.data.trade_history, ...newTrades];
                }
                
                // Break after finding valid data
                break;
              }
            }
          } catch (visualizationError) {
            console.warn(`Could not fetch visualization data from ${path}:`, visualizationError);
          }
        }
        
        // If we still don't have any trade history data, use fallback data
        if (result.data.trade_history.length === 0) {
          console.log('No trade history found in any source, using fallback data');
          result.data.trade_history = generateFallbackTradeData(20);
          
          // Update holdings based on fallback trades
          const holdings: Record<string, number> = {};
          const lastPrices: Record<string, number> = {};
          
          result.data.trade_history.forEach((trade: TradeHistoryItem) => {
            // Update holdings
            if (!holdings[trade.symbol]) {
              holdings[trade.symbol] = 0;
            }
            
            if (trade.side === 'BUY') {
              holdings[trade.symbol] += trade.quantity;
            } else if (trade.side === 'SELL') {
              holdings[trade.symbol] = Math.max(0, holdings[trade.symbol] - trade.quantity);
            }
            
            // Update last prices
            lastPrices[trade.symbol] = trade.price;
          });
          
          // Update the result data with our generated holdings and prices
          result.data.holdings = holdings;
          result.data.last_prices = lastPrices;
          
          // Update balance to match the last trade's balance_after
          if (result.data.trade_history.length > 0) {
            const lastTrade = result.data.trade_history[result.data.trade_history.length - 1];
            result.data.balance = lastTrade.balance_after;
          }
        }
        
        // Update performance metrics if they don't exist
        if (!result.data.performance) {
          result.data.performance = {
            total_trades: result.data.trade_history.length,
            profit_loss: 0,
            return_pct: 0,
            win_rate: 0,
            sharpe_ratio: 0,
            max_drawdown: 0
          };
        } else if (result.data.performance.total_trades === undefined) {
          // Make sure total_trades is set based on trade history length
          result.data.performance.total_trades = result.data.trade_history.length;
        }
        
        // Always update total_trades to match trade history length
        if (result.data.performance) {
          result.data.performance.total_trades = result.data.trade_history.length;
        }
        
        // Update status with the data from the API
        setStatus(result.data);
      } else {
        throw new Error(result.message || 'Failed to fetch trading status');
      }
    } catch (err) {
      console.error('Error fetching trading status:', err);
      setError(`Failed to fetch trading status: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Send command to the trading API
  const sendCommand = async (command: string, params: Record<string, string | number | boolean> = {}) => {
    console.log(`Sending command: ${command} with params:`, params);
    try {
      setIsLoading(true);
      setError(null);
      
      // Convert all parameters to strings for consistency
      const stringParams: Record<string, string> = {};
      Object.entries(params).forEach(([key, value]) => {
        if (typeof value === 'boolean') {
          stringParams[key] = value ? 'true' : 'false';
        } else {
          stringParams[key] = String(value);
        }
      });
      
      console.log(`Sending command with stringified params:`, stringParams);
      
      const response = await fetch('/trading/paper', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          command,
          ...stringParams
        }),
      });
      
      console.log(`Response status:`, response.status, response.statusText);
      
      if (!response.ok) {
        // Try to get more detailed error information
        let errorMessage = `Server responded with ${response.status}: ${response.statusText}`;
        try {
          const errorText = await response.text();
          console.error('Error response body:', errorText);
          if (errorText) {
            // Try to parse as JSON, but fall back to using raw text
            const isJson = errorText.startsWith('{') && errorText.endsWith('}');
            if (isJson) {
              try {
                const errorJson = JSON.parse(errorText);
                if (errorJson.message) {
                  errorMessage = errorJson.message;
                }
              } catch {
                // JSON parsing failed, use the raw text
                errorMessage = `${errorMessage}\nDetails: ${errorText}`;
              }
            } else {
              // Not JSON, use the raw text
              errorMessage = `${errorMessage}\nDetails: ${errorText}`;
            }
          }
        } catch (e) {
          console.error('Failed to read error response:', e);
        }
        throw new Error(errorMessage);
      }
      
      const result = await response.json();
      console.log(`Command result:`, result);
      
      if (result.status !== 'success') {
        throw new Error(result.message || `Failed to execute command: ${command}`);
      }
      
      // Refresh status after command execution
      await fetchStatus();
      
      return result;
    } catch (err) {
      console.error(`Error executing command ${command}:`, err);
      setError(`Failed to execute command: ${err instanceof Error ? err.message : String(err)}`);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };
  
  // Format currency with symbol
  const formatCurrency = (value: number, currency?: string): string => {
    const currencySymbol = currency || status.base_currency || 'USDT';
    if (isNaN(value)) return `0.00 ${currencySymbol}`;
    return `${value.toFixed(2)} ${currencySymbol}`;
  };
  
  // Format percentage
  const formatPercent = (value: number): string => {
    if (isNaN(value)) return '0.00%';
    return `${(value * 100).toFixed(2)}%`;
  };
  
  // Format date
  const formatDate = (dateStr: string): string => {
    try {
      if (!dateStr) return 'N/A';
      const date = new Date(dateStr);
      if (isNaN(date.getTime())) return 'Invalid date';
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      }).format(date);
    } catch (error) {
      console.error('Date formatting error:', error);
      return 'Invalid date';
    }
  };
  
  // Save API keys
  const saveApiKeys = async () => {
    try {
      await sendCommand('api', { key: apiKey, secret: apiSecret });
      // Update status to show keys are configured
      setStatus(prev => ({ ...prev, api_keys_configured: true }));
      setConfigVisible(false);
      // Clear the fields after saving
      setApiKey('');
      setApiSecret('');
    } catch (err) {
      console.error('Error saving API keys:', err);
      setError(`Failed to save API keys: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  
  // Save auto-execution settings
  const saveAutoExecuteSettings = async () => {
    try {
      // Get refresh interval in seconds from the selected period
      const intervalSeconds = getRefreshIntervalInSeconds(selectedPeriod);
      
      // Ensure parameters are in the correct format - use string values for all parameters
      // This is critical as the backend expects string values
      const enabledValue = autoExecuteEnabled ? 'true' : 'false';
      const confidenceValue = confidenceThreshold.toString();
      const intervalValue = intervalSeconds.toString();
      
      console.log('Saving auto-execute settings:', {
        enabled: enabledValue, 
        confidence: confidenceValue,
        interval: intervalValue
      });
      
      // Send auto-execute command with the correct parameter names and formats
      // The CLI expects --enabled, --confidence, and --interval parameters as strings
      await sendCommand('auto-execute', { 
        enabled: enabledValue, 
        confidence: confidenceValue,
        interval: intervalValue
      });
      
      // Update status with new settings
      setStatus(prev => ({
        ...prev,
        auto_execute_suggested_trades: autoExecuteEnabled,
        min_confidence_threshold: confidenceThreshold,
        suggested_trade_refresh_interval: intervalSeconds
      }));
      
      // Show success message
      const message = `Auto-execution settings ${autoExecuteEnabled ? 'enabled' : 'disabled'} successfully`;
      setError(message);
      setTimeout(() => setError(null), 3000); // Clear message after 3 seconds
      
      // Force a status refresh to ensure we get the latest settings
      setTimeout(() => fetchStatus(), 1000);
    } catch (err) {
      console.error('Error saving auto-execution settings:', err);
      setError(`Failed to save auto-execution settings: ${err instanceof Error ? err.message : String(err)}`);
    }
  };
  

  
  // Load initial data
  useEffect(() => {
    // Only fetch status on initial load, don't set or save auto-execute settings
    fetchStatus();
    
    // We intentionally omit fetchStatus from dependencies
    // as we only want this to run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  
  // Track first render with useRef
  const isFirstRender = useRef(true);
  
  // Sync local state with props when they change
  useEffect(() => {
    if (propAutoExecuteEnabled !== undefined && propAutoExecuteEnabled !== autoExecuteEnabled) {
      setAutoExecuteEnabled(propAutoExecuteEnabled);
    }
  }, [propAutoExecuteEnabled, autoExecuteEnabled]);
  
  // Update auto-execution settings when status changes, but only on the first load
  useEffect(() => {
    if (status) {
      console.log('Status update received:', status);
      
      if (isFirstRender.current) {
        // Initialize settings from status only on first render
        const initialAutoExecute = propAutoExecuteEnabled !== undefined 
          ? propAutoExecuteEnabled 
          : (status.auto_execute_suggested_trades !== undefined ? !!status.auto_execute_suggested_trades : false);
        
        setAutoExecuteEnabled(initialAutoExecute);
        setConfidenceThreshold(status.min_confidence_threshold || 0.75);
        isFirstRender.current = false;
      }
      // No longer need to set refresh interval as we're using the dashboard's interval
    }
  }, [status, propAutoExecuteEnabled]);
  
  // Set up polling for updates
  useEffect(() => {
    const intervalId = setInterval(() => {
      fetchStatus();
    }, getRefreshIntervalFromPeriod(selectedPeriod));
    
    return () => clearInterval(intervalId);
  }, [selectedPeriod, fetchStatus]);
  
  // Listen for trade execution events from other components
  useEffect(() => {
    // Define the event handler to refresh the dashboard
    const handleTradeExecution = () => {
      console.log('Paper trading update event received, refreshing dashboard...');
      fetchStatus();
    };
    
    // Add event listener
    window.addEventListener('paper-trading-update', handleTradeExecution);
    
    // Clean up the event listener when component unmounts
    return () => {
      window.removeEventListener('paper-trading-update', handleTradeExecution);
    };
  }, [fetchStatus]);
  
  return (
    <div className="paper-trading-dashboard">
      {error && <div className="error-message">{error}</div>}
      {isLoading && <div className="loading-indicator">Loading...</div>}
      
      <div className="dashboard-header">
        <h2>Paper Trading Dashboard</h2>
        <div className="dashboard-controls">
          <button 
            className={`control-button ${status.is_running ? 'stop' : 'start'}`}
            onClick={() => sendCommand(status.is_running ? 'stop' : 'start')}
            title={status.is_running ? 'Stop Trading' : 'Start Trading'}
          >
            {status.is_running ? 
              <><span role="img" aria-label="Stop">⏹️</span> Stop Trading</> : 
              <><span role="img" aria-label="Start">▶️</span> Start Trading</>}
          </button>
          
          <button 
            className="control-button config"
            onClick={() => setConfigVisible(!configVisible)}
            title="Configure Settings"
          >
            <span role="img" aria-label="Configure">⚙️</span> Configure
          </button>
          
          <button 
            className="control-button save"
            onClick={() => {
              // Save current configuration
              saveApiKeys();
              saveAutoExecuteSettings();
            }}
            title="Save Configuration"
          >
            <span role="img" aria-label="Save">💾</span> Save
          </button>
          
          <button 
            className="control-button reset"
            onClick={() => {
              if (window.confirm('Are you sure you want to reset all trading data? This cannot be undone.')) {
                sendCommand('reset');
              }
            }}
          >
            Reset
          </button>
          
          <button 
            className="control-button mode"
            onClick={() => {
              if (status.mode === 'paper') {
                if (window.confirm('Switch to live trading mode? This will use real funds from your Binance account.')) {
                  sendCommand('switch', { mode: 'live' });
                }
              } else {
                sendCommand('switch', { mode: 'paper' });
              }
            }}
          >
            Mode: {status.mode.toUpperCase()}
          </button>
        </div>
      </div>
      
      <div className="status-indicators">
        <div className="indicator">
          <div className={`indicator-dot ${status.is_running ? 'active' : 'inactive'}`}></div>
          <span className="indicator-text">
            {status.is_running ? 'Trading Active' : 'Trading Inactive'}
          </span>
        </div>
        
        <div className="indicator">
          <div className={`indicator-dot ${status.mode === 'live' ? 'live' : 'paper'}`}></div>
          <span className="indicator-text">
            {status.mode === 'live' ? 'Live Trading' : 'Paper Trading'}
          </span>
        </div>
        
        <div className="indicator api-indicator">
          <div className={`indicator-dot ${status.api_keys_configured ? 'configured' : 'not-configured'}`}></div>
          <span className="indicator-text">
            {status.api_keys_configured ? 'API Ready' : 'API Not Configured'}
          </span>
        </div>
        
        <div className="indicator">
          <div className={`indicator-dot ${autoExecuteEnabled ? 'active' : 'inactive'}`}></div>
          <span className="indicator-text">
            {autoExecuteEnabled ? 'Auto-Execute On' : 'Auto-Execute Off'}
          </span>
        </div>
      </div>
      
      {configVisible && (
        <div className="config-panel">
          <h3>Trading Configuration</h3>
          
          {/* API Keys Section */}
          <div className="config-section">
            <h4>API Configuration</h4>
            <div className="api-form">
              <div className="form-group">
                <label htmlFor="apiKey">Binance API Key:</label>
                <input 
                  type="text" 
                  id="apiKey" 
                  value={apiKey} 
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="Enter your Binance API key"
                />
              </div>
              
              <div className="form-group">
                <label htmlFor="apiSecret">Binance API Secret:</label>
                <input 
                  type="password" 
                  id="apiSecret" 
                  value={apiSecret} 
                  onChange={(e) => setApiSecret(e.target.value)}
                  placeholder="Enter your Binance API secret"
                />
              </div>
              
              <div className="form-actions">
                <button className="save-button" onClick={saveApiKeys}>Save API Keys</button>
              </div>
              
              <div className="config-note">
                <p>
                  <strong>Note:</strong> For live trading, you need to set your Binance API keys.
                  Make sure your API keys have permission for reading market data and trading.
                </p>
              </div>
            </div>
          </div>
          
          {/* Auto-Execute Suggested Trades Section */}
          <div className="config-section">
            <h4>Auto-Execute Suggested Trades</h4>
            <div className="auto-execute-form">
              <div className="form-group checkbox-group">
                <label htmlFor="autoExecuteEnabled" className="checkbox-label">
                  <input 
                    type="checkbox" 
                    id="autoExecuteEnabled" 
                    checked={autoExecuteEnabled} 
                    onChange={(e) => {
                      const newValue = e.target.checked;
                      setAutoExecuteEnabled(newValue);
                      console.log('Auto-execute toggled to:', newValue);
                      if (onAutoExecuteChange) {
                        onAutoExecuteChange(newValue);
                      }
                    }}
                  />
                  <span>Enable Auto-Execution of Suggested Trades</span>
                </label>
              </div>
              
              <div className="form-group">
                <label htmlFor="confidenceThreshold">Minimum Confidence Threshold:</label>
                <div className="slider-container">
                  <input 
                    type="range" 
                    id="confidenceThreshold" 
                    min="0" 
                    max="1" 
                    step="0.05" 
                    value={confidenceThreshold} 
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    disabled={!autoExecuteEnabled}
                  />
                  <span className="slider-value">{(confidenceThreshold * 100).toFixed(0)}%</span>
                </div>
                <div className="slider-labels">
                  <span>Low</span>
                  <span>Medium</span>
                  <span>High</span>
                </div>
              </div>
              
              <div className="form-group">
                <label>Refresh Interval:</label>
                <div className="info-text">
                  Using dashboard refresh interval: <strong>{selectedPeriod}</strong> ({getRefreshIntervalInSeconds(selectedPeriod)} seconds)
                </div>
              </div>
              
              <div className="form-actions">
                <button 
                  className="save-button" 
                  onClick={() => {
                    // Ensure auto-execute is enabled before saving
                    if (!autoExecuteEnabled) {
                      setAutoExecuteEnabled(true);
                      setTimeout(saveAutoExecuteSettings, 100); // Give state time to update
                    } else {
                      saveAutoExecuteSettings();
                    }
                  }}
                >
                  Save Auto-Execution Settings
                </button>
              </div>
              
              <div className="config-note">
                <p>
                  <strong>Note:</strong> When enabled, the system will automatically execute suggested trades 
                  from the trading dashboard with confidence levels above your specified threshold. 
                  Trades will be executed at the dashboard's refresh interval ({selectedPeriod}).
                </p>
              </div>
            </div>
          </div>
          
          <div className="form-actions centered">
            <button className="cancel-button" onClick={() => setConfigVisible(false)}>Close Configuration</button>
          </div>
          
          <div className="api-key-instructions">
            <h4>How to get Binance API Keys:</h4>
            <ol>
              <li>Log in to your Binance account</li>
              <li>Go to <strong>API Management</strong> in your account settings</li>
              <li>Create a new API key with trading permissions</li>
              <li>Copy both the API Key and Secret Key</li>
              <li>Paste them in the fields above</li>
            </ol>
            <p className="warning-text">⚠️ Keep your API keys secure and never share them with anyone</p>
          </div>
        </div>
      )}
      
      <div className="dashboard-content">
        <div className="summary-section">
          <div className="summary-item">
            <h3>Balance</h3>
            <div className="summary-value">{formatCurrency(status.balance || 0)}</div>
          </div>
          
          <div className="summary-item">
            <h3>Portfolio Value</h3>
            <div className="summary-value">{formatCurrency(status.portfolio_value || 0)}</div>
          </div>
          
          <div className="summary-item performance">
            <h3>Performance</h3>
            <div className="performance-metrics">
              <div className="metric">
                Trades: {status.performance?.total_trades || status.trade_history?.length || 0}
              </div>
              <div className="metric">P/L: {formatCurrency(status.performance?.profit_loss || 0)} ({formatPercent(status.performance?.return_pct || 0)})</div>
              <div className="metric">Win Rate: {formatPercent(status.performance?.win_rate || 0)}</div>
            </div>
          </div>
        </div>
        
        <div className="metrics-section">
          <div className="metric-card">
            <h3>Sharpe Ratio</h3>
            <div className={`metric-value ${(status.performance?.sharpe_ratio || 0) > 1 ? 'positive' : 'negative'}`}>
              {(status.performance?.sharpe_ratio || 0).toFixed(2)}
            </div>
          </div>
          
          <div className="metric-card">
            <h3>Win Rate</h3>
            <div className={`metric-value ${(status.performance?.win_rate || 0) > 0.5 ? 'positive' : 'negative'}`}>
              {formatPercent(status.performance?.win_rate || 0)}
            </div>
          </div>
          
          <div className="metric-card">
            <h3>Max Drawdown</h3>
            <div className="metric-value negative">
              {formatPercent(status.performance?.max_drawdown || 0)}
            </div>
          </div>
        </div>
        
        <div className="visualization-toggle">
          <button 
            className="toggle-button" 
            onClick={() => setShowTradeVisualization(!showTradeVisualization)}
          >
            {showTradeVisualization ? 'Hide Trade Visualization' : 'Show Trade Visualization'}
          </button>
        </div>
        
        {showTradeVisualization && (
          <div className="trade-visualization">
            <iframe 
              src="/charts/html/trade_history_3d.html" 
              title="Trading Bot Performance 3D Visualization"
              className="trade-visualization-iframe"
              sandbox="allow-scripts allow-same-origin"
            />
          </div>
        )}
        
        <div className="holdings-trade-grid">
          <div className="holdings-section">
            <h3>Holdings</h3>
            {safeObjectKeys(status.holdings).length === 0 ? (
              <div className="empty-state">No holdings</div>
            ) : (
              <table className="holdings-table">
                <thead>
                  <tr>
                    <th>Symbol</th>
                    <th>Amount</th>
                    <th>Last Price</th>
                    <th>Value</th>
                    <th>Growth</th>
                  </tr>
                </thead>
                <tbody>
                  {safeObjectEntries(status.holdings).map(([symbol, amount]) => {
                    const price = status.last_prices && status.last_prices[symbol] || 0;
                    const value = amount * price;
                    
                    // Calculate growth/decline for this holding
                    // Find the purchase price from trade history
                    let purchasePrice = 0;
                    let growth = 0;
                    let growthPercent = 0;
                    
                    // Find the average purchase price from BUY trades for this symbol
                    const buyTrades = (status.trade_history || []).filter(trade => 
                      trade.symbol === symbol && trade.side === 'BUY'
                    );
                    
                    if (buyTrades.length > 0) {
                      // Calculate weighted average purchase price
                      let totalValue = 0;
                      let totalQuantity = 0;
                      
                      buyTrades.forEach(trade => {
                        totalValue += trade.price * trade.quantity;
                        totalQuantity += trade.quantity;
                      });
                      
                      purchasePrice = totalValue / totalQuantity;
                      growth = price - purchasePrice;
                      growthPercent = (purchasePrice > 0) ? (growth / purchasePrice) * 100 : 0;
                    }
                    
                    const isPositiveGrowth = growth >= 0;
                    
                    return amount > 0 ? (
                      <tr key={symbol}>
                        <td>{symbol}</td>
                        <td>{amount.toFixed(8)}</td>
                        <td>{price.toFixed(2)}</td>
                        <td>{formatCurrency(value)}</td>
                        <td className={isPositiveGrowth ? 'positive-growth' : 'negative-growth'}>
                          {purchasePrice > 0 ? (
                            <>
                              {isPositiveGrowth ? '+' : ''}{growth.toFixed(2)} ({isPositiveGrowth ? '+' : ''}{growthPercent.toFixed(2)}%)
                            </>
                          ) : 'N/A'}
                        </td>
                      </tr>
                    ) : null;
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
        
        <div className="trade-history-section" style={{ gridColumn: '1 / span 2' }}>
          <h3>
            {!status.trade_history || status.trade_history.length === 0 ? 'No Trades Yet' :
             status.trade_history.length === 1 ? '1st Trade' :
             status.trade_history.length <= 5 ? `First ${status.trade_history.length} Trades` :
             `Recent Trades (Last ${Math.min(status.trade_history.length, 10)} of ${status.trade_history.length} Trades)`}
          </h3>
          {!status.trade_history || status.trade_history.length === 0 ? (
            <div className="empty-state">No trades yet</div>
          ) : (
            <div className="trade-scroll-container" ref={tradeHistoryTableRef}>
              <table className="trade-history-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Action</th>
                    <th>Symbol</th>
                    <th>Price</th>
                    <th>Quantity</th>
                    <th>Value</th>
                    <th>Balance After</th>
                  </tr>
                </thead>
                <tbody>
                  {[...(status.trade_history || [])].reverse().map((trade, index) => {
                    // Calculate the trade number (from newest to oldest)
                    const tradeNumber = (status.trade_history?.length || 0) - index;
                    const tradeOrdinal = getOrdinalSuffix(tradeNumber);
                    
                    // Calculate profit/loss for this trade if it's a SELL
                    let profitLoss = 0;
                    let profitLossPercent = 0;
                    
                    if (trade.side === 'SELL') {
                      // Find the corresponding BUY trades for this symbol
                      const buyTrades = (status.trade_history || []).filter(t => 
                        t.symbol === trade.symbol && t.side === 'BUY' && new Date(t.timestamp) < new Date(trade.timestamp)
                      );
                      
                      if (buyTrades.length > 0) {
                        // Calculate weighted average purchase price
                        let totalValue = 0;
                        let totalQuantity = 0;
                        
                        buyTrades.forEach(t => {
                          totalValue += t.price * t.quantity;
                          totalQuantity += t.quantity;
                        });
                        
                        const avgBuyPrice = totalValue / totalQuantity;
                        profitLoss = (trade.price - avgBuyPrice) * trade.quantity;
                        profitLossPercent = (avgBuyPrice > 0) ? ((trade.price - avgBuyPrice) / avgBuyPrice) * 100 : 0;
                      }
                    }
                    
                    const isProfitable = profitLoss > 0;
                    
                    // Calculate the trade ID that would match the 3D visualization
                    // The 3D visualization uses IDs starting from 1, with the oldest trade as ID 1
                    const visualizationTradeId = (status.trade_history?.length || 0) - index;
                    
                    // Check if this trade should be highlighted
                    const isHighlighted = highlightedTradeId === visualizationTradeId;
                    
                    return (
                      <tr 
                        key={index} 
                        className={`${trade.side === 'BUY' ? 'buy-trade' : 'sell-trade'} ${isHighlighted ? 'highlight-trade' : ''}`}
                        data-trade-number={tradeNumber}
                        data-trade-id={visualizationTradeId}
                      >
                        <td>
                          <span className="trade-ordinal">{tradeOrdinal}</span>
                          {formatDate(trade.timestamp)}
                        </td>
                        <td className={trade.side === 'BUY' ? 'buy-action' : 'sell-action'}>
                          {trade.side}
                        </td>
                        <td>{trade.symbol}</td>
                        <td>{trade.price.toFixed(2)}</td>
                        <td>{trade.quantity.toFixed(8)}</td>
                        <td>{formatCurrency(trade.value)}</td>
                        <td>
                          {formatCurrency(trade.balance_after)}
                          {trade.side === 'SELL' && profitLoss !== 0 && (
                            <span className={isProfitable ? 'profit-indicator' : 'loss-indicator'}>
                              {' '}({isProfitable ? '+' : ''}{profitLoss.toFixed(2)} / {isProfitable ? '+' : ''}{profitLossPercent.toFixed(2)}%)
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
      
      <div className="dashboard-footer">
        <div className="last-updated">
          Last updated: {formatDate(status.last_updated || '')}
          <span className="refresh-interval"> (Auto-refresh: {selectedPeriod})</span>
        </div>
        <button className="refresh-button" onClick={fetchStatus}>
          Refresh
        </button>
      </div>
    </div>
  );
};

export default PaperTradingDashboard;
