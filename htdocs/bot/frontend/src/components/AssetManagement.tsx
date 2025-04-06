import React, { useState, useEffect, useMemo } from 'react';
import { Plus, Eye, EyeOff, Wallet, Lock, Star, Trash, Sun, Moon, Copy, QrCode } from 'lucide-react';
import { tradeTracker } from '../services/tradeTracker';
import { ethers } from 'ethers';
import { QRCodeSVG } from 'qrcode.react';
import copy from 'copy-to-clipboard';

interface AssetManagementProps {
  onAddAsset: (asset: CryptoData) => void;
  isDarkMode?: boolean;
  onDarkModeChange?: (isDark: boolean) => void;
  onSelectCrypto?: (symbol: string) => void;
  portfolioAssets?: CryptoData[];
}

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
  change?: string; // For displaying formatted 24h change in the UI
}

interface CryptoCurrency {
  name: string;
  symbol: string;
  image: string;
  isSuspicious?: boolean;
  warning?: string;
}

const AssetManagement: React.FC<AssetManagementProps> = ({ 
  onAddAsset,
  isDarkMode = true,
  onDarkModeChange = () => {},
  onSelectCrypto = () => {},
  portfolioAssets = []
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState<CryptoCurrency[]>([]);
  const [showBlocked, setShowBlocked] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // State for wallet and beneficiary management
  const [showWalletSection, setShowWalletSection] = useState(false);
  const [showBeneficiariesSection, setShowBeneficiariesSection] = useState(false);
  const [walletName, setWalletName] = useState('');
  const [walletType, setWalletType] = useState('hot');
  const [walletCreated, setWalletCreated] = useState(false);
  const [beneficiaryName, setBeneficiaryName] = useState('');
  const [beneficiaryAddress, setBeneficiaryAddress] = useState('');
  const [seedPhrase, setSeedPhrase] = useState('');
  const [walletAddress, setWalletAddress] = useState('');
  const [privateKey, setPrivateKey] = useState('');
  const [showSeedPhrase, setShowSeedPhrase] = useState(false);
  const [showQRCode, setShowQRCode] = useState(false);
  const [seedCopied, setSeedCopied] = useState(false);
  const [walletMode, setWalletMode] = useState<'create' | 'import'>('create');
  const [importMethod, setImportMethod] = useState<'seed' | 'private'>('seed');
  const [existingWalletAddress, setExistingWalletAddress] = useState('');
  const [existingWalletName, setExistingWalletName] = useState('');
  const [existingWalletType, setExistingWalletType] = useState('hot');
  const [importedWalletSuccess, setImportedWalletSuccess] = useState(false);
  const [beneficiaries, setBeneficiaries] = useState<{name: string; address: string}[]>([
    { name: 'Charity Fund', address: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e' },
    { name: 'Investment Pool', address: '0x1234567890123456789012345678901234567890' },
  ]);
  const [expandedList, setExpandedList] = useState(false);

  // Popular cryptocurrencies
  const popularCryptos = useMemo(() => [
    { name: 'Bitcoin', symbol: 'BTC', image: 'https://cryptologos.cc/logos/bitcoin-btc-logo.png' },
    { name: 'Ethereum', symbol: 'ETH', image: 'https://cryptologos.cc/logos/ethereum-eth-logo.png' },
    { name: 'Solana', symbol: 'SOL', image: 'https://cryptologos.cc/logos/solana-sol-logo.png' },
    { name: 'Cardano', symbol: 'ADA', image: 'https://cryptologos.cc/logos/cardano-ada-logo.png' },
    { name: 'Ripple', symbol: 'XRP', image: 'https://cryptologos.cc/logos/xrp-xrp-logo.png' },
    { name: 'Polkadot', symbol: 'DOT', image: 'https://cryptologos.cc/logos/polkadot-new-dot-logo.png' },
    { name: 'Avalanche', symbol: 'AVAX', image: 'https://cryptologos.cc/logos/avalanche-avax-logo.png' },
    { name: 'Dogecoin', symbol: 'DOGE', image: 'https://cryptologos.cc/logos/dogecoin-doge-logo.png' },
    { name: 'Shiba Inu', symbol: 'SHIB', image: 'https://cryptologos.cc/logos/shiba-inu-shib-logo.png' },
    { name: 'Polygon', symbol: 'MATIC', image: 'https://cryptologos.cc/logos/polygon-matic-logo.png' },
  ] as CryptoCurrency[], []);

  // List of all cryptocurrencies including suspicious ones
  const allCryptocurrencies = useMemo(() => [
    // Legitimate cryptocurrencies
    ...popularCryptos,
    { name: 'Chainlink', symbol: 'LINK', image: 'https://cryptologos.cc/logos/chainlink-link-logo.png' },
    { name: 'Uniswap', symbol: 'UNI', image: 'https://cryptologos.cc/logos/uniswap-uni-logo.png' },
    { name: 'Litecoin', symbol: 'LTC', image: 'https://cryptologos.cc/logos/litecoin-ltc-logo.png' },
    { name: 'Stellar', symbol: 'XLM', image: 'https://cryptologos.cc/logos/stellar-xlm-logo.png' },
    { name: 'Cosmos', symbol: 'ATOM', image: 'https://cryptologos.cc/logos/cosmos-atom-logo.png' },
    { name: 'Algorand', symbol: 'ALGO', image: 'https://cryptologos.cc/logos/algorand-algo-logo.png' },
    { name: 'Tezos', symbol: 'XTZ', image: 'https://cryptologos.cc/logos/tezos-xtz-logo.png' },
    
    // Suspicious cryptocurrencies with detailed warnings
    { 
      name: 'SafeMoon', 
      symbol: 'SAFEMOON', 
      image: 'https://cryptologos.cc/logos/safemoon-safemoon-logo.png',
      isSuspicious: true,
      warning: "This token has a high risk of being a scam. It features:\n- High transfer fees\n- Burn mechanism that can be manipulated\n- No verifiable team members\n- Multiple reported incidents of rug pulls"
    },
    { 
      name: 'ElonDoge', 
      symbol: 'EDOGE', 
      image: 'https://via.placeholder.com/32',
      isSuspicious: true,
      warning: "This token is named after a celebrity and mimics a popular meme coin. It has:\n- No verifiable development team\n- No clear use case\n- Extremely high volatility\n- Multiple reported pump-and-dump schemes"
    },
    { 
      name: 'MoonRocket', 
      symbol: 'MOON', 
      image: 'https://via.placeholder.com/32',
      isSuspicious: true,
      warning: "This token has multiple red flags:\n- No whitepaper or technical documentation\n- No verifiable developers\n- Extremely low market cap\n- Multiple reported scams under similar names"
    },
    { 
      name: 'BabyDoge', 
      symbol: 'BABYDOGE', 
      image: 'https://via.placeholder.com/32',
      isSuspicious: true,
      warning: "This token is a copycat of a popular meme coin. It features:\n- No clear development team\n- No unique value proposition\n- Extremely high volatility\n- Multiple reported scams targeting new investors"
    },
    { 
      name: 'SafeElon', 
      symbol: 'SAFELON', 
      image: 'https://via.placeholder.com/32',
      isSuspicious: true,
      warning: "This token is named after a celebrity and has:\n- No verifiable team members\n- No clear use case\n- Extremely high volatility\n- Multiple reported pump-and-dump schemes"
    },
    { 
      name: 'RocketMoon', 
      symbol: 'RMOON', 
      image: 'https://via.placeholder.com/32',
      isSuspicious: true,
      warning: "This token has multiple red flags:\n- No whitepaper or technical documentation\n- No verifiable developers\n- Extremely low market cap\n- Multiple reported scams under similar names"
    },
    { 
      name: 'ElonMars', 
      symbol: 'ELONMARS', 
      image: 'https://via.placeholder.com/32',
      isSuspicious: true,
      warning: "This token is named after a celebrity and has:\n- No verifiable team members\n- No clear use case\n- Extremely high volatility\n- Multiple reported pump-and-dump schemes"
    },
    { 
      name: 'DogeElonMars', 
      symbol: 'DOGELON', 
      image: 'https://via.placeholder.com/32',
      isSuspicious: true,
      warning: "This token combines multiple red flags:\n- No verifiable development team\n- No clear use case\n- Extremely high volatility\n- Multiple reported scams targeting new investors"
    }
  ] as CryptoCurrency[], [popularCryptos]);

  const searchCryptocurrencies = (term: string) => {
    if (!term.trim()) {
      // When no search term, clear the list
      setSearchResults([]);
      return;
    }
    
    // Filter cryptos based on search term and showBlocked state
    const results = allCryptocurrencies.filter(crypto => 
      (crypto.name.toLowerCase().includes(term.toLowerCase()) || 
      crypto.symbol.toLowerCase().includes(term.toLowerCase())) &&
      (showBlocked || !crypto.isSuspicious)
    );
    
    setSearchResults(results);
  };

  const showAllCryptocurrencies = () => {
    const filteredResults = showBlocked 
      ? allCryptocurrencies 
      : allCryptocurrencies.filter(crypto => !crypto.isSuspicious);
    setSearchResults(filteredResults);
    setSearchTerm('');
    setExpandedList(true);
  };

  const handleAddAsset = async (crypto: CryptoCurrency) => {
    if (crypto.isSuspicious && !showBlocked) {
      alert(`Warning: ${crypto.name} (${crypto.symbol}) appears to be suspicious or fraudulent.\n\nIt's recommended to review its risk factors before adding to your portfolio.`);
      return;
    }

    try {
      // Fetch current market price for the crypto (using a dummy price range for demo)
      // In a real app, you would fetch this from an API
      const randomPrice = Math.random() * (100000 - 1) + 1;
      const price = crypto.symbol === 'BTC' ? randomPrice : 
                   crypto.symbol === 'ETH' ? randomPrice / 10 : 
                   randomPrice / 100;
      
      // Format the price for display in the prompt
      const formattedPrice = price.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
      });
      
      // Prompt for amount to buy
      const amountPrompt = prompt(
        `How much ${crypto.name} (${crypto.symbol}) would you like to buy?\n\nCurrent price: $${formattedPrice} per ${crypto.symbol}`,
        '0.01'
      );
      
      // Check if user cancelled
      if (amountPrompt === null) {
        return;
      }
      
      // Parse and validate amount
      const amount = parseFloat(amountPrompt);
      if (isNaN(amount) || amount <= 0) {
        alert('Please enter a valid positive number.');
        return;
      }
      
      setLoading(true);
      
      // Calculate total value
      const total = price * amount;
      
      // Create the asset
      const newAsset: CryptoData = {
        name: crypto.name,
        symbol: crypto.symbol,
        icon: <img src={crypto.image} alt={crypto.symbol} className="w-5 h-5" />,
        amount: amount.toString(),
        value: `$${total.toLocaleString(undefined, {
          minimumFractionDigits: 2,
          maximumFractionDigits: 2
        })}`,
        profit: '+0.0%',
        prediction: 'Neutral',
        imageUrl: crypto.image,
        price: price,
        price_change_24h_percentage: 0
      };
      
      // Add asset to portfolio and show confirmation
      onAddAsset(newAsset);
      console.log('Asset added to portfolio:', newAsset);
      
      // Show success message to confirm the purchase was added
      alert(`Successfully added ${amount} ${crypto.symbol} to your portfolio!`);
      
      try {
        // Log the trade with valid values
        const tradeResult = await tradeTracker.logTrade({
          type: 'buy',
          symbol: crypto.symbol,
          price: price,
          amount: amount,
          total: total,
          note: `Bought ${amount} ${crypto.symbol} at $${formattedPrice} each`
        });

        // Update trade status if successful
        if (tradeResult.success && tradeResult.data) {
          await tradeTracker.updateTradeStatus(tradeResult.data.id, 'completed');
        }
      } catch (tradeError) {
        console.warn('Trade logging failed, but asset was added to portfolio:', tradeError);
        // Don't throw the error since we've already added the asset
      }

    } catch (error) {
      console.error('Error adding asset:', error);
      alert(`Failed to add ${crypto.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Keep list empty by default, only populate when explicitly requested
    if (!expandedList && searchTerm === '') {
      setSearchResults([]);
    } else if (expandedList) {
      // Only show cryptos when in expanded view (Show All was clicked)
      const filteredResults = showBlocked 
        ? allCryptocurrencies 
        : allCryptocurrencies.filter(crypto => !crypto.isSuspicious);
      setSearchResults(filteredResults);
    }
    // When searching, the searchCryptocurrencies function handles population
  }, [showBlocked, allCryptocurrencies, expandedList, searchTerm]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-48">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-indigo-500"></div>
      </div>
    );
  }

  // Functions for wallet and beneficiary management

  // Create a new wallet with ethers.js
  const createWallet = () => {
    if (!walletName) {
      alert('Please enter a wallet name');
      return;
    }

    try {
      // Generate a random mnemonic (seed phrase)
      const wallet = ethers.Wallet.createRandom();
      
      // Get the mnemonic, address, and private key
      const mnemonic = wallet.mnemonic.phrase;
      const address = wallet.address;
      const key = wallet.privateKey;
      
      // Set the wallet information in the state
      setSeedPhrase(mnemonic);
      setWalletAddress(address);
      setPrivateKey(key);
      setWalletCreated(true);
      setShowSeedPhrase(false); // Initially hide the seed phrase for security
      
      // Log to console for development (remove in production)
      console.log('Wallet created successfully');
      console.log('Address:', address);
      console.log('Type:', walletType);
    } catch (error) {
      console.error('Error creating wallet:', error);
      alert('Error creating wallet. Please try again.');
    }
  };
  
  // Import an existing wallet
  const importExistingWallet = () => {
    if (!existingWalletName) {
      alert('Please enter a wallet name');
      return;
    }

    if (!existingWalletAddress) {
      alert('Please enter a wallet address');
      return;
    }

    // Basic address validation
    if (!/^0x[a-fA-F0-9]{40}$/.test(existingWalletAddress)) {
      alert('Please enter a valid Ethereum address (0x + 40 hex characters)');
      return;
    }

    try {
      let wallet;
      
      // Create wallet from seed phrase or private key
      if (importMethod === 'seed' && seedPhrase) {
        // Validate seed phrase (this is a basic check, should be more robust in production)
        const words = seedPhrase.trim().split(/\s+/);
        if (words.length !== 12 && words.length !== 24) {
          alert('Seed phrase must contain 12 or 24 words');
          return;
        }
        wallet = ethers.Wallet.fromMnemonic(seedPhrase);
      } else if (importMethod === 'private' && privateKey) {
        // Validate private key format
        if (!/^0x[a-fA-F0-9]{64}$/.test(privateKey)) {
          alert('Please enter a valid private key (0x + 64 hex characters)');
          return;
        }
        wallet = new ethers.Wallet(privateKey);
      } else {
        alert('Please enter a valid seed phrase or private key');
        return;
      }
      
      // Verify that the derived address matches the provided address
      if (wallet.address.toLowerCase() !== existingWalletAddress.toLowerCase()) {
        alert('The provided key/seed phrase does not match the wallet address');
        return;
      }
      
      // Set success state
      setImportedWalletSuccess(true);
      setWalletAddress(existingWalletAddress);
      
      // Log to console for development (remove in production)
      console.log('Wallet imported successfully');
      console.log('Address:', existingWalletAddress);
      console.log('Type:', existingWalletType);
    } catch (error) {
      console.error('Error importing wallet:', error);
      alert(`Error importing wallet: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  };
  
  // Copy seed phrase to clipboard
  const copySeedPhrase = () => {
    if (seedPhrase) {
      copy(seedPhrase);
      setSeedCopied(true);
      setTimeout(() => setSeedCopied(false), 3000);
    }
  };

  // Add a new beneficiary wallet
  const addBeneficiary = () => {
    if (!beneficiaryName || !beneficiaryAddress) {
      alert('Please enter both name and address for the beneficiary');
      return;
    }
    
    // Validate address format (basic check for demo purposes)
    if (!/^0x[a-fA-F0-9]{40}$/.test(beneficiaryAddress)) {
      alert('Please enter a valid Ethereum address (0x + 40 hex characters)');
      return;
    }
    
    // Add to beneficiaries list
    setBeneficiaries([...beneficiaries, {
      name: beneficiaryName,
      address: beneficiaryAddress
    }]);
    
    // Clear form
    setBeneficiaryName('');
    setBeneficiaryAddress('');
  };

  // Remove a beneficiary
  const removeBeneficiary = (name: string) => {
    setBeneficiaries(beneficiaries.filter(b => b.name !== name));
  };

  // Calculate simple portfolio metrics without using React hooks
  const portfolioMetrics = {
    totalValue: portfolioAssets && portfolioAssets.length > 0 
      ? portfolioAssets.reduce((sum, asset) => {
          const value = asset.value ? parseFloat(asset.value.replace(/[$,]/g, '')) || 0 : 0;
          return sum + value;
        }, 0).toLocaleString('en-US', {
          style: 'currency',
          currency: 'USD'
        })
      : '$0.00',
    totalChange: portfolioAssets && portfolioAssets.length > 0 ? '+$348.25 (+1.37%)' : '+$0.00 (0.00%)', // Placeholder
    isPositiveChange: true,
    assetsCount: portfolioAssets ? portfolioAssets.length : 0
  };

  return (
    <div className="p-4">
      <div className="flex gap-8">
        {/* Left column - 30% width - Add New Assets */}
        <div className="w-[30%]">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Add New Assets</h2>
            <button
              onClick={() => onDarkModeChange(!isDarkMode)}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
            >
              {isDarkMode ? <Sun className="h-5 w-5 text-yellow-400" /> : <Moon className="h-5 w-5 text-gray-500" />}
            </button>
          </div>
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowBlocked(!showBlocked)}
                className="flex items-center p-2 bg-gray-100 rounded hover:bg-gray-200"
                title={showBlocked ? "Hide potentially fake cryptocurrencies" : "Show potentially fake cryptocurrencies"}
              >
                {showBlocked ? (
                  <EyeOff size={16} className="text-red-500" />
                ) : (
                  <Eye size={16} className="text-gray-500" />
                )}
                <span className="text-sm text-gray-600">{showBlocked ? "Showing blocked" : "Hiding blocked"}</span>
              </button>
            </div>
          </div>

          <div className="relative mb-6">
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-1 p-2 border rounded"
                placeholder="Search cryptocurrencies..."
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value);
                  searchCryptocurrencies(e.target.value);
                  setExpandedList(false);
                }}
              />
              <button
                onClick={showAllCryptocurrencies}
                className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                title="Show all cryptocurrencies"
              >
                Show All
              </button>
              {expandedList && (
                <button
                  onClick={() => setExpandedList(false)}
                  className="p-2 bg-gray-500 text-white rounded hover:bg-gray-600"
                  title="Collapse the list"
                >
                  Collapse
                </button>
              )}
            </div>
          </div>

          <div className={`space-y-4 ${expandedList ? 'pr-2' : 'max-h-[65vh] overflow-y-auto pr-2 theme-scroll'}`}>
            {searchResults.length > 0 ? (
              searchResults.map((crypto) => (
                <div 
                  key={crypto.symbol} 
                  className="flex items-center p-2 border rounded relative"
                  title={crypto.isSuspicious ? crypto.warning : undefined}
                >
                  <img 
                    src={crypto.image} 
                    alt={crypto.symbol} 
                    className="w-6 h-6 mr-2"
                  />
                  <div className="flex-1">
                    <div className="font-medium">{crypto.name}</div>
                    <div className="text-sm text-gray-500">{crypto.symbol}</div>
                  </div>
                  
                  {/* Show warning for suspicious cryptocurrencies */}
                  {crypto.isSuspicious && (
                    <div className="flex items-center text-red-500 ml-2 cursor-help">
                      <span className="text-xs">⚠️ Suspicious</span>
                    </div>
                  )}
                  
                  <button
                    onClick={() => handleAddAsset(crypto)}
                    className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600"
                    disabled={crypto.isSuspicious && !showBlocked}
                    title={crypto.isSuspicious && !showBlocked ? "This cryptocurrency is blocked due to suspicious activity" : "Add to portfolio"}
                  >
                    <Plus size={16} />
                  </button>
                </div>
              ))
            ) : (
              <div className="text-center py-4 text-gray-500">
                {searchTerm ? `No cryptocurrencies found matching "${searchTerm}"` : 
                showBlocked ? 'Search or click Show All to view all cryptocurrencies' : 
                'No cryptocurrencies found. Suspicious cryptocurrencies are automatically hidden. Click Show All to view them.'}
              </div>
            )}
          </div>
        </div>

        {/* Right column - 67% width - Your Assets and Wallet Management */}
        <div className="w-[67%]">
          {/* Wallet Management Buttons */}
          <div className="flex gap-4 mb-4">
            <button
              onClick={() => {
                setShowWalletSection(!showWalletSection);
                if (showWalletSection) setWalletCreated(false);
              }}
              className="flex items-center gap-2 px-3 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
            >
              <Wallet size={16} />
              <span>{showWalletSection ? 'Hide Wallet Options' : 'Manage Wallets'}</span>
            </button>
            
            <button
              onClick={() => setShowBeneficiariesSection(!showBeneficiariesSection)}
              className="flex items-center gap-2 px-3 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 text-sm"
            >
              <Star size={16} />
              <span>{showBeneficiariesSection ? 'Hide Beneficiaries' : 'Manage Beneficiaries'}</span>
            </button>
          </div>
        
          {/* Wallet Creation Section */}
          {showWalletSection && (
            <div className="theme-card rounded-xl p-4 mb-4 border-2 border-blue-200 dark:border-blue-800">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                <div className="flex items-center gap-2">
                  <Lock size={18} className="text-blue-500" />
                  <span>Wallet Management</span>
                </div>
              </h2>
              
              {/* Wallet Mode Selector */}
              <div className="mb-4">
                <div className="flex w-full rounded-md overflow-hidden">
                  <button
                    onClick={() => {
                      setWalletMode('create');
                      setWalletCreated(false);
                      setImportedWalletSuccess(false);
                    }}
                    className={`flex-1 py-2 text-center text-sm ${walletMode === 'create' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'}`}
                  >
                    Create New Wallet
                  </button>
                  <button
                    onClick={() => {
                      setWalletMode('import');
                      setWalletCreated(false);
                      setImportedWalletSuccess(false);
                    }}
                    className={`flex-1 py-2 text-center text-sm ${walletMode === 'import' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'}`}
                  >
                    Add Existing Wallet
                  </button>
                </div>
              </div>
              
              {walletMode === 'create' && !walletCreated ? (
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Wallet Name</label>
                    <input
                      type="text"
                      className="w-full p-2 border rounded text-sm"
                      placeholder="My Primary Wallet"
                      value={walletName}
                      onChange={(e) => setWalletName(e.target.value)}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Wallet Type</label>
                    <div className="flex gap-4">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="walletType"
                          value="hot"
                          checked={walletType === 'hot'}
                          onChange={() => setWalletType('hot')}
                          className="text-blue-500"
                        />
                        <span className="text-sm">Hot Wallet (Online)</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="walletType"
                          value="cold"
                          checked={walletType === 'cold'}
                          onChange={() => setWalletType('cold')}
                          className="text-blue-500"
                        />
                        <span className="text-sm">Cold Wallet (Offline)</span>
                      </label>
                    </div>
                  </div>
                  
                  <div className="pt-2">
                    <button
                      onClick={createWallet}
                      className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                    >
                      Create Secure Wallet
                    </button>
                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      Your keys will be encrypted and stored securely. Always backup your seed phrase.
                    </p>
                  </div>
                </div>
              ) : walletMode === 'import' && !importedWalletSuccess ? (
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Wallet Name</label>
                    <input
                      type="text"
                      className="w-full p-2 border rounded text-sm"
                      placeholder="My Existing Wallet"
                      value={existingWalletName}
                      onChange={(e) => setExistingWalletName(e.target.value)}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Wallet Address</label>
                    <input
                      type="text"
                      className="w-full p-2 border rounded text-sm"
                      placeholder="0x..."
                      value={existingWalletAddress}
                      onChange={(e) => setExistingWalletAddress(e.target.value)}
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Import Method</label>
                    <div className="flex gap-4">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="importMethod"
                          value="seed"
                          checked={importMethod === 'seed'}
                          onChange={() => setImportMethod('seed')}
                          className="text-blue-500"
                        />
                        <span className="text-sm">Seed Phrase</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="importMethod"
                          value="private"
                          checked={importMethod === 'private'}
                          onChange={() => setImportMethod('private')}
                          className="text-blue-500"
                        />
                        <span className="text-sm">Private Key</span>
                      </label>
                    </div>
                  </div>
                  
                  {importMethod === 'seed' ? (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Seed Phrase</label>
                      <textarea
                        className="w-full p-2 border rounded text-sm font-mono"
                        placeholder="Enter 12 or 24 word seed phrase..."
                        rows={3}
                        value={seedPhrase}
                        onChange={(e) => setSeedPhrase(e.target.value)}
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Enter your 12 or 24 word seed phrase separated by spaces
                      </p>
                    </div>
                  ) : (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Private Key</label>
                      <input
                        type="password"
                        className="w-full p-2 border rounded text-sm font-mono"
                        placeholder="0x..."
                        value={privateKey}
                        onChange={(e) => setPrivateKey(e.target.value)}
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Enter your wallet's private key
                      </p>
                    </div>
                  )}
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Wallet Type</label>
                    <div className="flex gap-4">
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="existingWalletType"
                          value="hot"
                          checked={existingWalletType === 'hot'}
                          onChange={() => setExistingWalletType('hot')}
                          className="text-blue-500"
                        />
                        <span className="text-sm">Hot Wallet (Online)</span>
                      </label>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="radio"
                          name="existingWalletType"
                          value="cold"
                          checked={existingWalletType === 'cold'}
                          onChange={() => setExistingWalletType('cold')}
                          className="text-blue-500"
                        />
                        <span className="text-sm">Cold Wallet (Offline)</span>
                      </label>
                    </div>
                  </div>
                  
                  <div className="pt-2">
                    <button
                      onClick={importExistingWallet}
                      className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
                    >
                      Import Wallet
                    </button>
                    <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                      Your wallet details will be securely stored. We never transmit your private keys or seed phrases.
                    </p>
                  </div>
                </div>
              ) : importedWalletSuccess ? (
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                  <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    <span className="font-medium">Wallet Imported Successfully!</span>
                  </div>
                  
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                    Your {existingWalletType === 'hot' ? 'Hot' : 'Cold'} wallet "{existingWalletName}" has been imported and is ready to use.
                  </p>
                  
                  <div className="my-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Wallet Address:</span>
                    </div>
                    <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-md overflow-x-auto text-xs font-mono break-all">
                      {existingWalletAddress}
                    </div>
                  </div>
                  
                  <p className="mt-1 text-xs text-orange-600 dark:text-orange-400 font-medium">
                    Your wallet has been successfully imported. You can now use it for transactions.
                  </p>
                </div>
              ) : (
                <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded">
                  <div className="flex items-center gap-2 text-green-600 dark:text-green-400">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path>
                    </svg>
                    <span className="font-medium">Wallet Created Successfully!</span>
                  </div>
                  
                  <p className="mt-2 text-sm text-gray-600 dark:text-gray-300">
                    Your {walletType === 'hot' ? 'Hot' : 'Cold'} wallet "{walletName}" has been created and is ready to use.
                  </p>
                  
                  <div className="my-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Wallet Address:</span>
                    </div>
                    <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-md overflow-x-auto text-xs font-mono break-all">
                      {walletAddress}
                    </div>
                  </div>
                  
                  <div className="my-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Seed Phrase:</span>
                      <div className="flex space-x-2">
                        <button
                          className="flex items-center gap-1 text-xs py-1 px-2 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors"
                          onClick={() => setShowSeedPhrase(!showSeedPhrase)}
                        >
                          {showSeedPhrase ? <EyeOff size={12} /> : <Eye size={12} />}
                          <span>{showSeedPhrase ? 'Hide' : 'Show'}</span>
                        </button>
                        <button
                          className="flex items-center gap-1 text-xs py-1 px-2 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 rounded hover:bg-purple-200 dark:hover:bg-purple-900/50 transition-colors"
                          onClick={copySeedPhrase}
                        >
                          <Copy size={12} />
                          <span>{seedCopied ? 'Copied!' : 'Copy'}</span>
                        </button>
                        <button
                          className="flex items-center gap-1 text-xs py-1 px-2 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded hover:bg-green-200 dark:hover:bg-green-900/50 transition-colors"
                          onClick={() => setShowQRCode(!showQRCode)}
                        >
                          <QrCode size={12} />
                          <span>{showQRCode ? 'Hide QR' : 'Show QR'}</span>
                        </button>
                      </div>
                    </div>
                    {showSeedPhrase ? (
                      <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-900 rounded-md overflow-x-auto font-mono text-sm break-all">
                        {seedPhrase}
                      </div>
                    ) : (
                      <div className="p-3 bg-gray-100 dark:bg-gray-800 rounded-md overflow-hidden text-center">
                        <span className="text-xs text-gray-500 dark:text-gray-400">Hidden for security - click "Show" to reveal</span>
                      </div>
                    )}
                  </div>
                  
                  {showQRCode && (
                    <div className="my-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Wallet QR Code:</span>
                      </div>
                      <div className="flex justify-center p-4 bg-white rounded-md">
                        <QRCodeSVG 
                          value={JSON.stringify({
                            address: walletAddress,
                            seedPhrase: seedPhrase
                          })}
                          size={200}
                          level="H"
                          includeMargin={true}
                        />
                      </div>
                      <div className="mt-2 text-xs text-center text-gray-500 dark:text-gray-400">
                        Scan this QR code with a compatible wallet app to import
                      </div>
                    </div>
                  )}
                  
                  <p className="mt-1 text-xs text-red-600 dark:text-red-400 font-medium">
                    IMPORTANT: Securely store your seed phrase offline. Never share it with anyone.
                  </p>
                </div>
              )}
            </div>
          )}
          
          {/* Beneficiaries Management Section */}
          {showBeneficiariesSection && (
            <div className="theme-card rounded-xl p-4 mb-4 border-2 border-purple-200 dark:border-purple-800">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                <div className="flex items-center gap-2">
                  <Star size={18} className="text-purple-500" />
                  <span>Manage Beneficiary Wallets</span>
                </div>
              </h2>
              
              <div className="space-y-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Beneficiary Name</label>
                    <input
                      type="text"
                      className="w-full p-2 border rounded text-sm"
                      placeholder="E.g., Family Fund"
                      value={beneficiaryName}
                      onChange={(e) => setBeneficiaryName(e.target.value)}
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Wallet Address</label>
                    <input
                      type="text"
                      className="w-full p-2 border rounded text-sm"
                      placeholder="0x..."
                      value={beneficiaryAddress}
                      onChange={(e) => setBeneficiaryAddress(e.target.value)}
                    />
                  </div>
                </div>
                
                <div>
                  <button
                    onClick={addBeneficiary}
                    className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 text-sm"
                  >
                    Add Beneficiary
                  </button>
                </div>
                
                {/* Beneficiary List */}
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">Saved Beneficiaries</h3>
                  
                  {beneficiaries.length > 0 ? (
                    <div className="space-y-2 max-h-[20vh] overflow-y-auto pr-2 theme-scroll">
                      {beneficiaries.map((ben) => (
                        <div 
                          key={ben.name}
                          className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded text-sm"
                        >
                          <div>
                            <div className="font-medium text-gray-900 dark:text-gray-100">{ben.name}</div>
                            <div className="text-xs text-gray-500 truncate max-w-[250px]">{ben.address}</div>
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={() => {
                                // In a real app, this would copy the address to clipboard
                                alert(`Address for ${ben.name} is ready to use: ${ben.address}`);
                              }}
                              className="p-1 text-blue-500 hover:text-blue-600"
                              title="Use this address"
                            >
                              <Star size={16} />
                            </button>
                            <button
                              onClick={() => removeBeneficiary(ben.name)}
                              className="p-1 text-red-500 hover:text-red-600"
                              title="Remove"
                            >
                              <Trash size={16} />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-sm text-gray-500 dark:text-gray-400 p-2 text-center">
                      No beneficiaries added yet. Add your first one above.
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          
          <div className="theme-card rounded-xl p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
                Your Assets
              </h2>
              <button 
                className="p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
                aria-label="Settings"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="3"></circle>
                  <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
              </button>
            </div>

            <div className="mb-4">
              <div className="flex justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Total Value</div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">{portfolioMetrics.totalValue}</div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">24h Change</div>
                  <div className={`text-lg font-bold ${portfolioMetrics.isPositiveChange ? 'text-green-500' : 'text-red-500'}`}>
                    {portfolioMetrics.totalChange}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">Assets</div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">{portfolioMetrics.assetsCount}</div>
                </div>
              </div>
            </div>

            <div>
              <table className="w-full theme-table text-sm">
                <thead>
                  <tr>
                    <th className="text-left p-2 rounded-tl-lg text-gray-600 dark:text-gray-300">Asset</th>
                    <th className="text-left p-2 text-gray-600 dark:text-gray-300">Price</th>
                    <th className="text-left p-2 text-gray-600 dark:text-gray-300">Holdings</th>
                    <th className="text-left p-2 text-gray-600 dark:text-gray-300">Value</th>
                    <th className="text-left p-2 rounded-tr-lg text-gray-600 dark:text-gray-300">24h Change</th>
                  </tr>
                </thead>
                <tbody>
                  {portfolioAssets.map((asset) => (
                    <tr 
                      key={asset.symbol}
                      className="transition-colors hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                      onClick={() => onSelectCrypto(asset.symbol)}
                      title={`View ${asset.name} technical analysis`}
                    >
                      <td className="p-2 text-gray-900 dark:text-gray-100">
                        <div className="flex items-center gap-2">
                          <div className="p-1 rounded-lg bg-gray-100 dark:bg-gray-700">
                            <img 
                              src={asset.imageUrl} 
                              alt={asset.symbol}
                              className="w-5 h-5"
                            />
                          </div>
                          <div>
                            <div className="font-medium text-gray-900 dark:text-gray-100 text-sm">{asset.name}</div>
                            <div className="text-gray-500 dark:text-gray-300 text-xs">{asset.symbol}</div>
                          </div>
                        </div>
                      </td>
                      <td className="p-2 text-gray-900 dark:text-gray-100 text-sm">${asset.price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</td>
                      <td className="p-2 text-gray-900 dark:text-gray-100 text-sm">{asset.amount} {asset.symbol}</td>
                      <td className="p-2 text-gray-900 dark:text-gray-100 text-sm">{asset.value}</td>
                      <td className={`p-2 text-sm ${asset.change && asset.change.startsWith('+') ? 'text-green-500 dark:text-green-300' : 'text-red-500 dark:text-red-300'}`}>
                        {asset.change || (asset.price_change_24h_percentage >= 0 ? `+${asset.price_change_24h_percentage.toFixed(2)}%` : `${asset.price_change_24h_percentage.toFixed(2)}%`)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AssetManagement;
