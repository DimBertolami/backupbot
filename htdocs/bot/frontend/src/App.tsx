import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Trash2,
  ArrowRight,
  Coins,
  X,
  Bitcoin,
  Feather as Ethereum,
  Gem,
  Menu,
  Sun,
  Moon,
  Settings,
  History,
  LineChart,
  Wallet,
  LayoutDashboard,
  TrendingUp,
  DollarSign
} from 'lucide-react';

import axios from 'axios';
import CryptoCharts from './components/CryptoCharts';
import BackendStatusDashboard from './components/BackendStatusDashboard';
import TradingStrategyResults from './components/TradingStrategyResults';
import PaperTradingDashboard from './components/PaperTradingDashboard';

// Interface for coin data from CoinGecko API
interface CoinGeckoData {
  id: string;
  symbol: string;
  name: string;
  image: string;
  current_price: number;
  market_cap: number;
  market_cap_rank: number;
  price_change_percentage_24h: number;
  price_change_24h: number;
  total_volume: number;
}


// Interface for cryptocurrency data
interface CryptoData {
  name: string;
  symbol: string;
  icon: React.ReactNode;
  amount: string;
  value: string;
  profit: string;
  prediction: string;
  imageUrl: string;
  price: number;
  price_change_24h_percentage: number;
}

// Initial crypto data with icons
const initialCryptoData: CryptoData[] = [
  {
    name: "Bitcoin",
    symbol: "BTC",
    icon: <Bitcoin className="text-yellow-500" />,
    amount: "1.245",
    value: "$0.00",
    profit: "0.0%",
    prediction: "Loading...",
    imageUrl: "https://cryptologos.cc/logos/bitcoin-btc-logo.png",
    price: 0,
    price_change_24h_percentage: 0
  },
  {
    name: "Ethereum",
    symbol: "ETH",
    icon: <Ethereum className="text-blue-500" />,
    amount: "12.54",
    value: "$0.00",
    profit: "0.0%",
    prediction: "Loading...",
    imageUrl: "https://cryptologos.cc/logos/ethereum-eth-logo.png",
    price: 0,
    price_change_24h_percentage: 0
  },
  {
    name: "Cardano",
    symbol: "ADA",
    icon: <Gem className="text-blue-400" />,
    amount: "5,420",
    value: "$0.00",
    profit: "0.0%",
    prediction: "Loading...",
    imageUrl: "https://cryptologos.cc/logos/cardano-ada-logo.png",
    price: 0,
    price_change_24h_percentage: 0
  },
  {
    name: "Solana",
    symbol: "SOL",
    icon: <Coins className="text-purple-500" />,
    amount: "156.8",
    value: "$0.00",
    profit: "0.0%",
    prediction: "Loading...",
    imageUrl: "https://cryptologos.cc/logos/solana-sol-logo.png",
    price: 0,
    price_change_24h_percentage: 0
  },
  {
    name: "Ripple",
    symbol: "XRP",
    icon: <Gem className="text-blue-500" />,
    amount: "10,500",
    value: "$0.00",
    profit: "0.0%",
    prediction: "Loading...",
    imageUrl: "https://cryptologos.cc/logos/xrp-xrp-logo.png",
    price: 0,
    price_change_24h_percentage: 0
  },
  {
    name: "Polkadot",
    symbol: "DOT",
    icon: <Coins className="text-pink-500" />,
    amount: "520",
    value: "$0.00",
    profit: "0.0%",
    prediction: "Loading...",
    imageUrl: "https://cryptologos.cc/logos/polkadot-new-dot-logo.png",
    price: 0,
    price_change_24h_percentage: 0
  },
  {
    name: "Avalanche",
    symbol: "AVAX",
    icon: <Coins className="text-red-500" />,
    amount: "85",
    value: "$0.00",
    profit: "0.0%",
    prediction: "Loading...",
    imageUrl: "https://cryptologos.cc/logos/avalanche-avax-logo.png",
    price: 0,
    price_change_24h_percentage: 0
  }
];

function App() {
  const [cryptoData, setCryptoData] = useState<CryptoData[]>(initialCryptoData);
  const [botThought, setBotThought] = useState({
    isVisible: false,
    message: ''
  });
  const [isDarkMode, setIsDarkMode] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [isSpinning, setIsSpinning] = useState(false);
  const [isAssetsExpanded, setIsAssetsExpanded] = useState(true);
  const [isDragging, setIsDragging] = useState(false);
  const [startY, setStartY] = useState(0);
  const [offsetY, setOffsetY] = useState(0);
  const [currentIndex, setCurrentIndex] = useState(0);
  const spinTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const slotRef = useRef<HTMLDivElement>(null);

  // Bot thought helper function
  const showBotThought = (message: string) => {
    setBotThought({
      isVisible: true,
      message
    });
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
      setBotThought(prev => ({
        ...prev,
        isVisible: false
      }));
    }, 5000);
  };

  // Set dark mode as default
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  // Handle dark mode changes
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);



  const [selectedPeriod, setSelectedPeriod] = useState('1h');
  const [autoExecuteEnabled, setAutoExecuteEnabled] = useState(false);
  

  
  // Function to update the trading configuration file
  const updateTradingConfig = useCallback(async (period: string) => {
    try {
      // Create a POST request to update the config
      const response = await fetch('/api/update-trading-config', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ update_interval: period }),
      });
      
      if (!response.ok) {
        console.warn('Could not update trading config:', response.statusText);
      }
    } catch (error) {
      console.error('Error updating trading config:', error);
      // The backend script will still read the config file periodically
    }
  }, []);

  // Fetch real cryptocurrency data
  useEffect(() => {
    const fetchCryptoData = async () => {
      try {
        setIsLoading(true);
        // Using CoinGecko API to fetch real crypto data
        const response = await axios.get('https://api.coingecko.com/api/v3/coins/markets', {
          params: {
            vs_currency: 'usd',
            ids: 'bitcoin,ethereum,cardano,solana,ripple,polkadot,avalanche',
            order: 'market_cap_desc',
            per_page: 10,
            page: 1,
            sparkline: false,
            price_change_percentage: '24h'
          }
        });
        
        // Map API data to our CryptoData format
        const mappedData = response.data.map((coin: CoinGeckoData) => {
          // Find corresponding entry in our initialData to keep icons and amounts
          const existingCoin = initialCryptoData.find(c => c.symbol.toLowerCase() === coin.symbol.toLowerCase());
          
          // Set prediction based on price change
          let prediction = "Neutral";
          if (coin.price_change_percentage_24h > 5) prediction = "Highly Bullish";
          else if (coin.price_change_percentage_24h > 1) prediction = "Bullish";
          else if (coin.price_change_percentage_24h < -5) prediction = "Highly Bearish";
          else if (coin.price_change_percentage_24h < -1) prediction = "Bearish";
          
          return {
            name: coin.name,
            symbol: coin.symbol.toUpperCase(),
            icon: existingCoin?.icon || <Coins className="text-gray-500" />,
            amount: existingCoin?.amount || "0",
            value: `$${coin.current_price.toLocaleString()}`,
            profit: `${coin.price_change_percentage_24h >= 0 ? '+' : ''}${coin.price_change_percentage_24h.toFixed(2)}%`,
            prediction: prediction,
            imageUrl: coin.image,
            price: coin.current_price,
            price_change_24h_percentage: coin.price_change_percentage_24h
          };
        });
        
        setCryptoData(mappedData);
        setIsLoading(false);
      } catch (error) {
        console.error('Error fetching crypto data:', error);
        // If API fails, fallback to our static data but with a notice
        const fallbackData = initialCryptoData.map(coin => ({
          ...coin,
          prediction: "API Error - Using Fallback Data"
        }));
        setCryptoData(fallbackData);
        setIsLoading(false);
      }
    };
    
    fetchCryptoData();
    
    // Refresh data every 5 minutes
    const intervalId = setInterval(fetchCryptoData, 5 * 60 * 1000);
    
    return () => clearInterval(intervalId);
  }, []);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [isDarkMode]);

  const spinSlot = () => {
    // Clear any existing spin timer
    if (spinTimerRef.current) {
      clearTimeout(spinTimerRef.current);
    }
    
    setIsSpinning(true);
    const spins = Math.random() * 10 + 15; // More spins before stopping
    let count = 0;
    
    const spin = () => {
      setCurrentIndex(prev => (prev + 1) % cryptoData.length);
      count++;
      
      if (count < spins) {
        // Much slower spinning (300ms base + slower acceleration)
        spinTimerRef.current = setTimeout(spin, 300 + (count * 25)); 
      } else {
        setIsSpinning(false);
        spinTimerRef.current = undefined;
      }
    };
    
    spin();
  };
  
  // Function to stop spinning immediately
  const stopSpinning = () => {
    if (isSpinning && spinTimerRef.current) {
      clearTimeout(spinTimerRef.current);
      spinTimerRef.current = undefined;
      setIsSpinning(false);
    }
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (isSpinning) return;
    setIsDragging(true);
    setStartY(e.clientY);
    document.addEventListener('mousemove', handleMouseMoveDoc);
    document.addEventListener('mouseup', handleMouseUpDoc);
  };

  const handleMouseMoveDoc = (e: Event) => {
    // Cast to DOM MouseEvent, not React's MouseEvent
    const mouseEvent = e as globalThis.MouseEvent;
    if (!isDragging) return;
    const deltaY = mouseEvent.clientY - startY;
    setOffsetY(deltaY);

    // Calculate index based on drag distance
    const slotHeight = 120; // Height of one slot item
    const newIndex = Math.floor(Math.abs(deltaY) / slotHeight) % cryptoData.length;
    
    if (deltaY > 0) {
      setCurrentIndex((currentIndex - newIndex + cryptoData.length) % cryptoData.length);
    } else {
      setCurrentIndex((currentIndex + newIndex) % cryptoData.length);
    }
  };
  
  // Handler removed as it's not used

  const handleMouseUpDoc = () => {
    setIsDragging(false);
    setOffsetY(0);
    document.removeEventListener('mousemove', handleMouseMoveDoc);
    document.removeEventListener('mouseup', handleMouseUpDoc);
  };

  // Handle mouse wheel scrolling for the crypto roller
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    if (isSpinning) return; // Don't allow wheel scrolling while spinning
    
    e.preventDefault(); // Prevent the page from scrolling
    
    // Determine scroll direction (positive deltaY = scroll down, negative = scroll up)
    if (e.deltaY > 0) {
      // Scroll down - move to next crypto
      setCurrentIndex((prev) => (prev + 1) % cryptoData.length);
    } else {
      // Scroll up - move to previous crypto
      setCurrentIndex((prev) => (prev - 1 + cryptoData.length) % cryptoData.length);
    }
  };

  // Handler removed as it's not used

  // Define SidebarLink component inside App function to prevent error
  const SidebarLink = ({ icon, text, active = false, collapsed = false, onClick }: {
    icon: React.ReactNode;
    text: string;
    active?: boolean;
    collapsed?: boolean;
    onClick?: () => void;
  }) => {
    return (
      <div 
        onClick={onClick}
        className={`
          flex items-center gap-4 p-3 rounded-lg mb-2 transition-all duration-300
          ${active 
            ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/20' 
            : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
          }
        `}
      >
        {icon}
        {!collapsed && <span className="font-medium">{text}</span>}
      </div>
    );
  };

  // Define StatCard component
  const StatCard = ({ title, value, change, positive, icon }: {
    title: string;
    value: string;
    change: string;
    positive: boolean;
    icon: React.ReactNode;
  }) => {
    return (
      <div className="theme-card rounded-xl p-6 relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-20 transform transition-transform group-hover:scale-110">
          {icon}
        </div>
        <h3 className="text-gray-600 dark:text-gray-400 mb-2">{title}</h3>
        <div className="text-2xl font-bold mb-2 text-gray-900 dark:text-white">
          {value}
        </div>
        <div className={`text-sm ${positive ? 'text-green-500' : 'text-red-500'}`}>
          {change}
        </div>
      </div>
    );
  };
  
  // Define AssetRow component
  const AssetRow = ({
    icon, name, symbol, price, holdings, value, change, positive, onBuy, onSell, onTransfer, onRemove
  }: {
    icon: React.ReactNode;
    name: string;
    symbol: string;
    price: string;
    holdings: string;
    value: string;
    change: string;
    positive: boolean;
    onBuy: () => void;
    onSell: () => void;
    onTransfer: () => void;
    onRemove: () => void;
  }) => {
    return (
      <tr className="transition-colors hover:bg-gray-50 dark:hover:bg-gray-700">
        <td className="p-4">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-gray-100 dark:bg-gray-700">
              {icon}
            </div>
            <div>
              <div className="font-medium text-gray-900 dark:text-white">{name}</div>
              <div className="text-gray-500 dark:text-gray-400 text-sm">{symbol}</div>
            </div>
          </div>
        </td>
        <td className="p-4">{price}</td>
        <td className="p-4">{holdings}</td>
        <td className="p-4">{value}</td>
        <td className={`p-4 ${positive ? 'text-green-500' : 'text-red-500'}`}>
          {change}
        </td>
        <td className="p-4">
          <div className="flex gap-2">
            <button 
              onClick={onBuy}
              className="px-3 py-1 text-sm rounded bg-green-500 text-white hover:bg-green-600"
              title="Buy"
            >
              Buy
            </button>
            <button 
              onClick={onSell}
              className="px-3 py-1 text-sm rounded bg-red-500 text-white hover:bg-red-600"
              title="Sell"
            >
              Sell
            </button>
            <button 
              onClick={onTransfer}
              className="p-1 rounded bg-blue-500 text-white hover:bg-blue-600 flex items-center justify-center"
              title="Transfer"
            >
              <ArrowRight size={16} />
            </button>
            <button 
              onClick={onRemove}
              className="p-1 rounded bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-500 flex items-center justify-center"
              title="Remove Asset"
            >
              <Trash2 size={16} />
            </button>
          </div>
        </td>
      </tr>
    );
  };

  // Define BotThought component
  const BotThought = ({ isVisible, message, onClose }: {
    isVisible: boolean;
    message: string;
    onClose: () => void;
  }) => {
    if (!isVisible) return null;

    return (
      <div className="fixed bottom-4 right-4 max-w-md w-full animate-slide-up z-50">
        <div className="theme-card rounded-xl p-4 shadow-lg border border-gray-200 dark:border-gray-700">
          <div className="flex items-start gap-3">
            {/* Bot Avatar */}
            <div className="w-10 h-10 rounded-full overflow-hidden flex-shrink-0 bg-gray-100 dark:bg-gray-700 flex items-center justify-center">
              <img 
                src="/dimbot-avatar.png" 
                alt="Dimbot"
                className="w-full h-full object-cover"
              />
            </div>
            
            {/* Content */}
            <div className="flex-1">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-gray-900 dark:text-white">
                  Dimbot:
                </h3>
                <button 
                  onClick={onClose}
                  className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400"
                >
                  <X size={16} />
                </button>
              </div>
              <p className="text-gray-700 dark:text-gray-300 text-sm">
                {message}
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  };
  
  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 text-gray-900 dark:text-gray-100 flex font-cinzel transition-colors duration-300">
      {/* Sidebar */}
      <div 
        className={`
          ${isSidebarOpen ? 'w-64' : 'w-20'} 
          theme-card
          p-4 transition-all duration-300 ease-in-out
          relative
        `}
      >
        <div className="flex items-center justify-between mb-4">
          <h1 className={`${!isSidebarOpen && 'hidden'} text-xl font-bold text-gray-900 dark:text-white`}>
            CryptoTracker
          </h1>
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)} 
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
          >
            {isSidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
        
        {/* Theme Switcher - Moved to top */}
        <div className="theme-switcher mb-6">
          <button
            onClick={() => setIsDarkMode(!isDarkMode)}
            className="w-full flex items-center justify-center gap-2 p-2 rounded-lg
                     bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600
                     transition-colors duration-200"
          >
            {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
            {isSidebarOpen && <span>{isDarkMode ? 'Light Mode' : 'Dark Mode'}</span>}
          </button>
        </div>
        
        <nav>
          <SidebarLink 
            icon={<LayoutDashboard />} 
            text="Dashboard" 
            active={activeTab === 'dashboard'} 
            collapsed={!isSidebarOpen} 
            onClick={() => setActiveTab('dashboard')}
          />
          <SidebarLink 
            icon={<DollarSign />} 
            text="Paper Trading" 
            active={activeTab === 'paperTrading'} 
            collapsed={!isSidebarOpen}
            onClick={() => setActiveTab('paperTrading')}
          />
          <SidebarLink 
            icon={<Wallet />} 
            text="Portfolio" 
            active={activeTab === 'portfolio'} 
            collapsed={!isSidebarOpen}
            onClick={() => setActiveTab('portfolio')}
          />
          <SidebarLink 
            icon={<LineChart />} 
            text="Analytics" 
            active={activeTab === 'analytics'} 
            collapsed={!isSidebarOpen}
            onClick={() => setActiveTab('analytics')}
          />
          <SidebarLink 
            icon={<History />} 
            text="History" 
            active={activeTab === 'history'} 
            collapsed={!isSidebarOpen}
            onClick={() => setActiveTab('history')}
          />
          <SidebarLink 
            icon={<Settings />} 
            text="Settings" 
            active={activeTab === 'settings'} 
            collapsed={!isSidebarOpen}
            onClick={() => setActiveTab('settings')}
          />
        </nav>

        {/* Theme switcher moved to top */}
      </div>

      {/* Main Content */}
      <div className="flex-1 p-8">
        {activeTab === 'paperTrading' ? (
          <PaperTradingDashboard 
            selectedPeriod={selectedPeriod} 
            autoExecuteEnabled={autoExecuteEnabled}
            onAutoExecuteChange={setAutoExecuteEnabled}
          />
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              {/* Slot Machine Card */}
              <div className="theme-card rounded-xl p-6 col-span-3 md:col-span-1">
            <div className="slot-machine-container">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-bold text-gray-900 dark:text-white">Crypto Roller</h3>
                <div className="space-x-2">
                  {isSpinning && (
                    <button 
                      onClick={stopSpinning}
                      className="theme-button-secondary px-4 py-2 rounded-lg"
                    >
                      Stop
                    </button>
                  )}
                  <button 
                    onClick={spinSlot}
                    disabled={isDragging || isLoading}
                    className="theme-button px-4 py-2 rounded-lg"
                  >
                    {isSpinning ? 'Spinning...' : 'Auto Spin'}
                  </button>
                </div>
              </div>
              <div 
                ref={slotRef}
                className={`slot-window bg-gray-800 dark:bg-gray-700 rounded-lg p-4 mb-4 cursor-ns-resize
                          ${isDragging ? 'dragging' : ''}`}
                onMouseDown={handleMouseDown}
                onWheel={handleWheel}
                onClick={isSpinning ? stopSpinning : undefined} /* Click to stop spinning */
              >
                <div 
                  className={`slot-item ${isSpinning ? 'spinning' : ''}`}
                  style={{
                    transform: isDragging ? `translateY(${offsetY}px)` : 'none'
                  }}
                >
                  <div className="flex items-center gap-4 p-4 bg-gray-700 dark:bg-gray-600 rounded-lg">
                    <img 
                      src={cryptoData[currentIndex].imageUrl} 
                      alt={cryptoData[currentIndex].name}
                      className="w-12 h-12 object-contain"
                      draggable="false"
                    />
                    <div>
                      <div className="text-xl font-bold text-white">
                        {cryptoData[currentIndex].name}
                      </div>
                      <div className="text-gray-400">
                        {cryptoData[currentIndex].symbol}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="crypto-details bg-gray-50 dark:bg-gray-800 rounded-lg p-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">Amount</div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {cryptoData[currentIndex].amount} {cryptoData[currentIndex].symbol}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">Current Price</div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      {cryptoData[currentIndex].value}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">24h Change</div>
                    <div className={`text-lg font-bold ${
                      cryptoData[currentIndex].profit.startsWith('+') 
                        ? 'text-green-500' 
                        : 'text-red-500'
                    }`}>
                      {cryptoData[currentIndex].profit}
                    </div>
                  </div>
                  <div>
                    <div className="text-sm text-gray-500 dark:text-gray-400">Market Sentiment</div>
                    <div className={`text-lg font-bold ${
                      cryptoData[currentIndex].prediction.includes('Bull') 
                        ? 'text-green-600 dark:text-green-400' 
                        : cryptoData[currentIndex].prediction.includes('Bear')
                          ? 'text-red-600 dark:text-red-400'
                          : 'text-indigo-600 dark:text-indigo-400'
                    }`}>
                      {cryptoData[currentIndex].prediction}
                    </div>
                  </div>
                  <div className="col-span-2 pt-2 mt-2 border-t border-gray-200 dark:border-gray-700">
                    <div className="text-sm text-gray-500 dark:text-gray-400">Portfolio Value</div>
                    <div className="text-lg font-bold text-gray-900 dark:text-white">
                      ${((parseFloat(cryptoData[currentIndex].amount.replace(/,/g, '')) * 
                         cryptoData[currentIndex].price) || 0).toLocaleString(undefined, {maximumFractionDigits: 2})}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <StatCard 
            title="24h Change" 
            value="$1,248.90" 
            change="+1.2%" 
            positive={true}
            icon={<TrendingUp size={24} />}
          />
          <StatCard 
            title="Total Profit" 
            value="$12,483.45" 
            change="+24.8%" 
            positive={true}
            icon={<LineChart size={24} />}
          />
        </div>

        {/* Advanced Crypto Charts Section */}
        <div className="theme-card rounded-xl p-6 mb-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Crypto Analysis
            </h2>
            <div className="flex gap-2 overflow-x-auto pb-2 theme-scroll">
              {['1m','5m','10m','30m','1h','1D','1W','1M','1Y'].map((period) => (
                <button 
                  key={period}
                  onClick={() => {
                    setSelectedPeriod(period);
                    updateTradingConfig(period); // Sync with trading signals
                  }}
                  className={`
                    px-4 py-2 rounded-lg transition-all duration-300 whitespace-nowrap
                    ${period === selectedPeriod ? 'theme-button' : 'theme-button-secondary'}
                  `}
                >
                  {period}
                </button>
              ))}
            </div>
          </div>
          
          {/* Integrated CryptoCharts Component */}
          <div className="crypto-charts-wrapper">
            <BackendStatusDashboard />
            <CryptoCharts refreshInterval={300000} /> {/* Refresh every 5 minutes */}
          </div>
        </div>

        {/* Assets Table */}
        <div className="theme-card rounded-xl p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Your Assets
            </h2>
            <button 
              onClick={() => setIsAssetsExpanded(!isAssetsExpanded)}
              className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              aria-label={isAssetsExpanded ? 'Collapse' : 'Expand'}
            >
              {isAssetsExpanded ? (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="18 15 12 9 6 15"></polyline>
                </svg>
              ) : (
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
              )}
            </button>
          </div>
          {isAssetsExpanded && <div className="overflow-x-auto theme-scroll">
            <table className="w-full theme-table">
              <thead>
                <tr>
                  <th className="text-left p-4 rounded-tl-lg">Asset</th>
                  <th className="text-left p-4">Price</th>
                  <th className="text-left p-4">Holdings</th>
                  <th className="text-left p-4">Value</th>
                  <th className="text-left p-4 rounded-tr-lg">24h Change</th>
                </tr>
              </thead>
              <tbody>
                {cryptoData.map((crypto) => (
                  <AssetRow 
                    key={crypto.symbol}
                    icon={crypto.icon}
                    name={crypto.name}
                    symbol={crypto.symbol}
                    price={crypto.value}
                    holdings={`${crypto.amount} ${crypto.symbol}`}
                    value={crypto.value}
                    change={crypto.profit}
                    positive={crypto.profit.startsWith('+')}
                    onBuy={() => {
                      showBotThought(`Analyzing market conditions for buying ${crypto.symbol}...`);
                      const amount = prompt(`How much ${crypto.symbol} do you want to buy?`);
                      if (amount && !isNaN(parseFloat(amount))) {
                        showBotThought(`Successfully bought ${amount} ${crypto.symbol}!`);
                      }
                    }}
                    onSell={() => {
                      showBotThought(`Checking your ${crypto.symbol} balance and current market price...`);
                      const amount = prompt(`How much ${crypto.symbol} do you want to sell?`);
                      if (amount && !isNaN(parseFloat(amount))) {
                        showBotThought(`Successfully sold ${amount} ${crypto.symbol}!`);
                      }
                    }}
                    onTransfer={() => {
                      showBotThought(`Preparing to transfer ${crypto.symbol} between wallets...`);
                    }}
                    onRemove={() => {
                      showBotThought(`Removing ${crypto.symbol} from your portfolio...`);
                      setCryptoData(prev => prev.filter(c => c.symbol !== crypto.symbol));
                    }}
                  />
                ))}
              </tbody>
            </table>
          </div>}
            </div>
            
            {/* Trading Strategy Results */}
            <div className="theme-card rounded-xl p-6 mt-8">
              <h2 className="text-xl font-semibold mb-6 text-gray-900 dark:text-white">
                Trading Strategy Performance
              </h2>
              <TradingStrategyResults 
                selectedPeriod={selectedPeriod}
                autoExecuteEnabled={autoExecuteEnabled}
                onExecuteTrade={async (trade) => {
                  try {
                    // Format the symbol (remove any / character)
                    const formattedSymbol = trade.symbol.replace('/', '');
                    
                    // Prepare the request payload
                    const payload = {
                      command: 'execute-trade',
                      symbol: formattedSymbol,
                      side: trade.action,
                      price: trade.price,
                      confidence: trade.confidence
                    };
                    
                    console.log('Sending trade execution request with payload:', payload);
                    
                    // Call the paper trading API to execute the trade
                    const response = await fetch('/trading/paper', {
                      method: 'POST',
                      headers: {
                        'Content-Type': 'application/json',
                      },
                      body: JSON.stringify(payload),
                    });
                    
                    // Log the raw response
                    console.log('Raw response status:', response.status, response.statusText);
                    
                    if (!response.ok) {
                      const errorText = await response.text();
                      console.error('Error response body:', errorText);
                      throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
                    }
                    
                    const result = await response.json();
                    console.log('Parsed response:', result);
                    
                    if (result.status !== 'success') {
                      console.error('API reported error:', result.message);
                      throw new Error(result.message || 'Failed to execute trade');
                    }
                    
                    console.log('Trade executed successfully:', trade);
                    console.log('API response:', result);
                    
                    // Force a refresh of the paper trading dashboard
                    setTimeout(() => {
                      // Dispatch a custom event to notify the PaperTradingDashboard to refresh
                      const refreshEvent = new CustomEvent('paper-trading-update');
                      window.dispatchEvent(refreshEvent);
                    }, 1000);
                    
                    return true;
                  } catch (err) {
                    console.error('Error executing trade:', err);
                    return false;
                  }
                }}
              />
            </div>
          </>
        )}
      </div>
      
      {/* Bot Thought component for displaying bot feedback */}
      {botThought.isVisible && (
        <BotThought 
          isVisible={botThought.isVisible} 
          message={botThought.message} 
          onClose={() => setBotThought(prev => ({ ...prev, isVisible: false }))} 
        />
      )}
    </div>
  );
};
export default App;