import axios from 'axios';

interface Trade {
  id: string;
  type: 'buy' | 'sell';
  symbol: string;
  amount: number;
  price: number;
  total: number;
  timestamp: string;
  status: 'pending' | 'completed' | 'failed';
  note?: string;
}

interface TradeResponse {
  success: boolean;
  data?: Trade;
  error?: string;
}

export class TradeTracker {
  private static instance: TradeTracker;
  private apiUrl = '/api/trades';
  private offlineStorage: Trade[] = [];
  private isApiAvailable = true;
  
  private constructor() {
    // Try to load any cached trades from localStorage
    try {
      const savedTrades = localStorage.getItem('offlineTrades');
      if (savedTrades) {
        this.offlineStorage = JSON.parse(savedTrades);
      }
    } catch (_) {
      console.warn('Could not load offline trades from localStorage');
    }
  }

  public static getInstance(): TradeTracker {
    if (!TradeTracker.instance) {
      TradeTracker.instance = new TradeTracker();
    }
    return TradeTracker.instance;
  }

  async logTrade(trade: Omit<Trade, 'id' | 'timestamp' | 'status'>): Promise<TradeResponse> {
    // If we already know the API is unavailable, go straight to offline mode
    if (!this.isApiAvailable) {
      return this.logTradeOffline(trade);
    }

    try {
      const response = await axios.post(`${this.apiUrl}/log`, {
        ...trade,
        timestamp: new Date().toISOString(),
        status: 'pending'
      });
      
      // API is available
      this.isApiAvailable = true;
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error('Error logging trade:', error);
      
      // Mark API as unavailable and use offline mode
      this.isApiAvailable = false;
      return this.logTradeOffline(trade);
    }
  }
  
  private logTradeOffline(trade: Omit<Trade, 'id' | 'timestamp' | 'status'>): TradeResponse {
    try {
      // Create a new trade with local ID and timestamp
      const newTrade: Trade = {
        ...trade,
        id: `offline-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        status: 'completed'
      };
      
      // Add to offline storage
      this.offlineStorage.push(newTrade);
      
      // Save to localStorage
      localStorage.setItem('offlineTrades', JSON.stringify(this.offlineStorage));
      
      return {
        success: true,
        data: newTrade
      };
    } catch (_) {
      return {
        success: false,
        error: 'Failed to save trade offline'
      };
    }
  }

  async updateTradeStatus(tradeId: string, status: Trade['status']): Promise<TradeResponse> {
    // If offline mode or trade ID starts with 'offline-', use offline update
    if (!this.isApiAvailable || tradeId.startsWith('offline-')) {
      return this.updateTradeStatusOffline(tradeId, status);
    }
    
    try {
      const response = await axios.put(`${this.apiUrl}/${tradeId}/status`, { status });
      this.isApiAvailable = true;
      return {
        success: true,
        data: response.data
      };
    } catch (error: unknown) {
      console.error('Error updating trade status:', error);
      this.isApiAvailable = false;
      return this.updateTradeStatusOffline(tradeId, status);
    }
  }
  
  private updateTradeStatusOffline(tradeId: string, status: Trade['status']): TradeResponse {
    try {
      // Find the trade in offline storage and update its status
      const tradeIndex = this.offlineStorage.findIndex(t => t.id === tradeId);
      
      if (tradeIndex >= 0) {
        this.offlineStorage[tradeIndex].status = status;
        localStorage.setItem('offlineTrades', JSON.stringify(this.offlineStorage));
        
        return {
          success: true,
          data: this.offlineStorage[tradeIndex]
        };
      }
      
      return {
        success: false,
        error: 'Trade not found in offline storage'
      };
    } catch (_) {
      return {
        success: false,
        error: 'Failed to update trade status offline'
      };
    }
  }

  async getTradeHistory(): Promise<TradeResponse> {
    try {
      const response = await axios.get(`${this.apiUrl}/history`);
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error('Error fetching trade history:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  async getTradeById(tradeId: string): Promise<TradeResponse> {
    try {
      const response = await axios.get(`${this.apiUrl}/${tradeId}`);
      return {
        success: true,
        data: response.data
      };
    } catch (error) {
      console.error('Error fetching trade:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}

// Export a singleton instance
export const tradeTracker = TradeTracker.getInstance();
