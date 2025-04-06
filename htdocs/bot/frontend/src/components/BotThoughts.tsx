import React, { useState, useEffect, useRef, useCallback } from 'react';
import './BotThoughts.css';

interface MetricsData {
  total_profit: number;
  win_rate: number;
  total_trades: number;
  avg_profit_per_trade: number;
  best_performing_symbol: string | null;
  best_success_rate: number;
}

interface BotThought {
  id: number;
  timestamp: string;
  thought_type: string;
  thought_content: string;
  symbol?: string;
  confidence?: number;
  metrics?: MetricsData;
}

interface BotJoke {
  id: number;
  joke_text: string;
  category: string;
  created_at: string;
  last_used_at?: string;
  use_count: number;
  active: boolean;
}

interface BotThoughtsProps {
  refreshInterval: number;
}

// Bot name constant
const BOT_NAME = "Dimbot";

const BotThoughts: React.FC<BotThoughtsProps> = ({ refreshInterval }) => {
  const [thoughts, setThoughts] = useState<BotThought[]>([]);
  const [visibleThought, setVisibleThought] = useState<BotThought | null>(null);
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [isHidden, setIsHidden] = useState<boolean>(false);
  const [autoPlay, setAutoPlay] = useState<boolean>(true);
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [lastFetchTime, setLastFetchTime] = useState<Date>(new Date());
  const [jokes, setJokes] = useState<BotJoke[]>([]);
  const [currentJoke, setCurrentJoke] = useState<BotJoke | null>(null);
  const [showingJoke, setShowingJoke] = useState<boolean>(false);
  const [errorDisplayTime, setErrorDisplayTime] = useState<number>(0); // Track how long error is displayed
  const thoughtTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const thinkingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const jokeTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const errorTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Initialize showThoughtFn reference to break circular dependency
  const showThoughtFn = useRef<() => void>();
  
  // Function to fetch jokes from the API
  const fetchJokes = useCallback(async () => {
    try {
      const response = await fetch('/trading/jokes');
      if (response.ok) {
        const data = await response.json();
        if (data.status === 'success' && Array.isArray(data.data)) {
          setJokes(data.data);
        }
      }
    } catch (error) {
      console.error('Error fetching jokes:', error);
    }
  }, []);

  // Get a random joke to display
  const getRandomJoke = useCallback(() => {
    if (jokes.length === 0) return null;
    
    // Weight jokes that have been used less
    const weightedJokes: BotJoke[] = [];
    jokes.forEach(joke => {
      // Inversely proportional to use count - less used jokes appear more often
      const weight = Math.max(1, 10 - (joke.use_count || 0));
      for (let i = 0; i < weight; i++) {
        weightedJokes.push(joke);
      }
    });
    
    if (weightedJokes.length === 0) return null;
    return weightedJokes[Math.floor(Math.random() * weightedJokes.length)];
  }, [jokes]);
  
  // Initialize thought display cycle - declare early to break dependency cycle
  const startThoughtCycle = useCallback(() => {
    if (thoughts.length === 0) return;
    
    // Clear any existing timeouts
    if (thoughtTimeoutRef.current) {
      clearTimeout(thoughtTimeoutRef.current);
      thoughtTimeoutRef.current = null;
    }
    
    if (thinkingTimeoutRef.current) {
      clearTimeout(thinkingTimeoutRef.current);
      thinkingTimeoutRef.current = null;
    }
    
    if (jokeTimeoutRef.current) {
      clearTimeout(jokeTimeoutRef.current);
      jokeTimeoutRef.current = null;
    }
    
    // Don't start a new cycle if we're already thinking or showing a thought
    if (isThinking || visibleThought) return;
    
    // Start the thinking animation
    setIsThinking(true);
    setVisibleThought(null);
    setShowingJoke(false);
    
    // Show thinking indicator for a random time between 2-4 seconds
    const thinkingTime = Math.floor(Math.random() * 2000) + 2000; // 2-4 seconds
    
    thinkingTimeoutRef.current = setTimeout(() => {
      setIsThinking(false);
      
      // Decide whether to show a joke instead of a thought (10% chance)
      const showJoke = Math.random() < 0.10 && jokes.length > 0;
      
      if (showJoke) {
        const joke = getRandomJoke();
        if (joke) {
          setCurrentJoke(joke);
          setShowingJoke(true);
          setVisibleThought(null);
          
          // Track joke usage on the server
          try {
            fetch(`/trading/jokes/${joke.id}`, {
              method: 'PUT',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                use_count: (joke.use_count || 0) + 1,
              }),
            });
          } catch (error) {
            console.error('Error updating joke usage:', error);
          }
        } else {
          // Fallback to thought if no joke is available
          const randomIndex = Math.floor(Math.random() * thoughts.length);
          setVisibleThought(thoughts[randomIndex]);
          setShowingJoke(false);
        }
      } else {
        // Select a random thought
        const randomIndex = Math.floor(Math.random() * thoughts.length);
        setVisibleThought(thoughts[randomIndex]);
        setShowingJoke(false);
      }
      
      // Display the thought or joke for 6-12 seconds (longer display time as requested)
      const displayTime = Math.floor(Math.random() * 6000) + 6000; // 6-12 seconds
      
      thoughtTimeoutRef.current = setTimeout(() => {
        // Fade out effect is handled by CSS animation
        // After the CSS animation completes, remove the thought
        const fadeOutTime = 700; // This should match the CSS bubbleOut animation duration
        
        // Wait for fade out animation to complete before removing
        setTimeout(() => {
          setVisibleThought(null);
          
          // Wait a bit before starting the next cycle
          const pauseTime = Math.floor(Math.random() * 6000) + 4000; // 4-10 seconds pause
          
          thoughtTimeoutRef.current = setTimeout(() => {
            startThoughtCycle();
          }, pauseTime);
        }, fadeOutTime);
      }, displayTime);
    }, thinkingTime);
  }, [thoughts, isThinking, visibleThought, jokes, getRandomJoke]);
  
  // Store the startThoughtCycle in the ref to break circular dependency
  showThoughtFn.current = startThoughtCycle;
  
  // Function to fetch bot thoughts with retry logic
  const fetchBotThoughts = useCallback(async () => {
    try {
      // Update last fetch attempt time
      setLastFetchTime(new Date());
      
      // First try to get thoughts from the specific bot_thoughts.json file
      let response = await fetch('/trading_data/bot_thoughts.json', { 
        // Add timeout to detect connection issues faster
        signal: AbortSignal.timeout(5000) 
      });
      
      // If that fails, get thoughts from the live_trading_status.json which also contains thoughts
      if (!response.ok) {
        response = await fetch('/trading_data/live_trading_status.json', {
          signal: AbortSignal.timeout(5000)
        });
        
        if (!response.ok) {
          console.error('Failed to fetch bot thoughts');
          if (response.status === 404) {
            setConnectionError('Resource not found. The backend might be restarting.');
          } else {
            setConnectionError(`Connection issue (${response.status}). The backend might be unavailable.`);
          }

          // Even with error, we should continue with any existing thoughts
          // Only return if we have no thoughts at all
          if (thoughts.length === 0) {
            // Schedule a retry with exponential backoff
            setTimeout(() => {
              fetchBotThoughts();
            }, 5000);
            return;
          }
          return;
        }
        
        const data = await response.json();
        
        // Extract thoughts from live_trading_status.json
        if (data && data.thoughts && Array.isArray(data.thoughts)) {
          // Convert to the expected format
          const convertedThoughts = data.thoughts.map((thought: string, index: number) => ({
            id: index,
            timestamp: data.timestamp,
            thought_type: 'trading_analysis',
            thought_content: thought,
            symbol: null,
            confidence: null,
            metrics: null
          }));
          
          setThoughts(convertedThoughts);
          // Clear any previous error since we got valid data
          setConnectionError(null);
        }
      } else {
        // Process the dedicated thoughts file
        const data = await response.json();
        if (data && data.thoughts && Array.isArray(data.thoughts)) {
          setThoughts(data.thoughts);
          // Clear any previous error since we got valid data
          setConnectionError(null);
        }
      }
      
      // If we have thoughts and not currently showing one, start the cycle
      if (thoughts.length > 0 && !visibleThought && !isThinking && showThoughtFn.current) {
        showThoughtFn.current();
      }
    } catch (error) {
      console.error('Error fetching bot thoughts:', error);
      // Check if it's an abort error (timeout)
      if (error instanceof DOMException && error.name === 'AbortError') {
        setConnectionError('Connection timed out. The backend might be restarting.');
      } else if (error instanceof TypeError && error.message.includes('NetworkError')) {
        setConnectionError('Network error. The backend might be unavailable.');
      } else {
        setConnectionError(`Connection issue: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
      
      // Reset error display time when setting a new error
      setErrorDisplayTime(0);
      
      // Clear any existing error timeout
      if (errorTimeoutRef.current) {
        clearTimeout(errorTimeoutRef.current);
        errorTimeoutRef.current = null;
      }
      
      // Even with error, if we have existing thoughts, continue the cycle
      if (thoughts.length > 0 && !visibleThought && !isThinking && showThoughtFn.current) {
        showThoughtFn.current();
      }
    }
  }, [thoughts, visibleThought, isThinking]);
  

  
  // Navigate to a specific thought
  const navigateToThought = useCallback((index: number) => {
    // Clear any existing timeouts
    if (thoughtTimeoutRef.current) {
      clearTimeout(thoughtTimeoutRef.current);
    }
    
    if (thinkingTimeoutRef.current) {
      clearTimeout(thinkingTimeoutRef.current);
    }
    
    if (jokeTimeoutRef.current) {
      clearTimeout(jokeTimeoutRef.current);
    }
    
    // Update current index
    setCurrentIndex(index);
    
    // Show thinking indicator briefly
    setIsThinking(true);
    setVisibleThought(null);
    setShowingJoke(false); // Always show thoughts when manually navigating
    
    // Show thinking for a short time
    setTimeout(() => {
      setIsThinking(false);
      setVisibleThought(thoughts[index]);
    }, 500);
    
    // If autoplay is on, schedule the next thought
    if (autoPlay) {
      thoughtTimeoutRef.current = setTimeout(() => {
        navigateToThought((index + 1) % thoughts.length);
      }, 8000); // Show for 8 seconds before moving to next
    }
  }, [thoughts, autoPlay]);
  
  // Navigate to previous thought
  const prevThought = useCallback(() => {
    if (thoughts.length === 0) return;
    const newIndex = currentIndex > 0 ? currentIndex - 1 : thoughts.length - 1;
    navigateToThought(newIndex);
  }, [currentIndex, thoughts, navigateToThought]);
  
  // Navigate to next thought
  const nextThought = useCallback(() => {
    if (thoughts.length === 0) return;
    const newIndex = (currentIndex + 1) % thoughts.length;
    navigateToThought(newIndex);
  }, [currentIndex, thoughts, navigateToThought]);
  
  // Toggle hide/show
  const toggleHide = useCallback(() => {
    setIsHidden(!isHidden);
  }, [isHidden]);
  
  // Toggle autoplay
  const toggleAutoPlay = useCallback(() => {
    setAutoPlay(!autoPlay);
    
    // If turning autoplay on, start from current thought
    if (!autoPlay && thoughts.length > 0) {
      navigateToThought(currentIndex);
    }
  }, [autoPlay, currentIndex, thoughts, navigateToThought]);
  
  // Handle error display time tracking
  useEffect(() => {
    // Only run this effect when there's an active connection error
    if (connectionError) {
      // Set minimum display time for error messages (6 seconds)
      const minErrorDisplayTime = 6000; // 6 seconds minimum display time
      const errorDisplayInterval = 100; // Update progress every 100ms
      
      // Start tracking error display time
      const errorTrackingId = setInterval(() => {
        setErrorDisplayTime(prev => prev + errorDisplayInterval);
      }, errorDisplayInterval);
      
      // Set timer to allow fetching again after minimum display time
      errorTimeoutRef.current = setTimeout(() => {
        // After minimum display time, retry fetch if auto-play is on
        if (autoPlay) {
          fetchBotThoughts();
        }
      }, minErrorDisplayTime);
      
      return () => {
        clearInterval(errorTrackingId);
        if (errorTimeoutRef.current) {
          clearTimeout(errorTimeoutRef.current);
          errorTimeoutRef.current = null;
        }
        // Reset error display time when cleaning up
        setErrorDisplayTime(0);
      };
    }
  }, [connectionError, fetchBotThoughts, autoPlay]);

  // Fetch thoughts initially and set up refresh interval
  useEffect(() => {
    fetchBotThoughts();
    fetchJokes(); // Fetch jokes as well
    
    // Set up interval to fetch new thoughts with progressive backoff
    let retryCount = 0;
    const maxRetryCount = 5;
    const baseRetryInterval = Math.min(refreshInterval, 10000); // Use 10s as max base interval
    
    const thoughtsIntervalId = setInterval(() => {
      // Only proceed if we're not currently displaying an error message
      // or if we've displayed it for the minimum time
      if (!connectionError || errorDisplayTime >= 6000) {
        // Only clear connection error if we have a successful fetch
        fetchBotThoughts().then(() => {
          // Reset retry count on successful fetch
          retryCount = 0;
        }).catch(() => {
          // Increment retry count on failure
          retryCount = Math.min(retryCount + 1, maxRetryCount);
        });
      }
    }, refreshInterval);
    
    // Set up additional recovery interval with exponential backoff
    const recoveryIntervalId = setInterval(() => {
      // If we have an error state and no thoughts or we haven't shown a thought in a while
      // Only try to recover if we've shown the error message for at least 6 seconds
      const timeSinceLastFetch = new Date().getTime() - lastFetchTime.getTime();
      if (connectionError && errorDisplayTime >= 6000 && 
          (thoughts.length === 0 || timeSinceLastFetch > refreshInterval * 2)) {
        fetchBotThoughts();
      }
    }, baseRetryInterval);
    
    // Set up interval to fetch jokes (less frequently)
    const jokesIntervalId = setInterval(fetchJokes, refreshInterval * 5); // Refresh jokes less frequently
    
    // Store the current timeout references to avoid closure issues in cleanup
    // This fixes the ESLint warning about stale ref values
    const currentThoughtTimeout = thoughtTimeoutRef.current;
    const currentThinkingTimeout = thinkingTimeoutRef.current;
    const currentJokeTimeout = jokeTimeoutRef.current;
    const currentErrorTimeout = errorTimeoutRef.current;
    
    return () => {
      clearInterval(thoughtsIntervalId);
      clearInterval(jokesIntervalId);
      clearInterval(recoveryIntervalId);
      
      // Clear any existing timeouts with the captured refs
      if (currentThoughtTimeout) {
        clearTimeout(currentThoughtTimeout);
      }
      
      if (currentThinkingTimeout) {
        clearTimeout(currentThinkingTimeout);
      }
      
      if (currentJokeTimeout) {
        clearTimeout(currentJokeTimeout);
      }
      
      if (currentErrorTimeout) {
        clearTimeout(currentErrorTimeout);
      }
    };
  }, [fetchBotThoughts, fetchJokes, refreshInterval, connectionError, lastFetchTime, thoughts.length, errorDisplayTime]);
  
  // Start showing thoughts when they're loaded
  useEffect(() => {
    if (thoughts.length > 0) {
      // Show the latest thought (at the end of the array) if available
      navigateToThought(thoughts.length - 1);
      setCurrentIndex(thoughts.length - 1);
    }
  }, [thoughts, navigateToThought]);
  
  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleString();
    } catch {
      // If date parsing fails, return the original timestamp or fallback text
      return timestamp || 'Unknown time';
    }
  };
  
  if ((thoughts.length === 0 && !isThinking && !connectionError) || isHidden) {
    // If hidden or no thoughts, show only the compact eye button with timestamp
    const currentTimestamp = thoughts.length > 0 ? thoughts[currentIndex]?.timestamp : new Date().toISOString();
    
    return (
      <div className="bot-thoughts-container">
        <div className="carousel-controls">
          <button 
            className="control-button compact" 
            onClick={toggleHide}
            title={`Show ${BOT_NAME}'s thoughts`}
          >
            👁️
            <span className="timestamp-tooltip">
              {formatTimestamp(currentTimestamp)}
            </span>
          </button>
        </div>
      </div>
    );
  }
  
  // Show thinking, visible thought, or connection error
  const renderThoughts = () => {
    // If there's a connection error, show it in a bubble
    if (connectionError && !isHidden) {
      // Note: No longer setting automatic timeout here - we use the dedicated error display time handling
      
      return (
        <div className="error-bubble" style={{ animation: 'none' }}> {/* Stop any animations that might cause flickering */}
          <div className="bot-icon">
            <div className="bot-avatar"></div>
          </div>
          <div className="error-content">
            <div className="bot-name">{BOT_NAME}</div>
            <div className="error-message">{connectionError}</div>
            <div className="error-time">Last attempt: {formatTimestamp(lastFetchTime.toISOString())}</div>
            {/* Progress bar to show how long until auto-retry */}
            <div style={{ 
              marginTop: '10px', 
              height: '3px', 
              width: '100%',
              background: 'rgba(200,200,200,0.3)', 
              borderRadius: '2px',
              overflow: 'hidden'
            }}>
              <div 
                style={{ 
                  height: '100%', 
                  width: `${Math.min(100, (errorDisplayTime / 6000) * 100)}%`, 
                  background: 'rgba(255,100,100,0.6)', 
                  transition: 'width 0.1s linear'
                }} 
              />
            </div>
            <div style={{ fontSize: '12px', marginTop: '5px', opacity: 0.7 }}>
              {errorDisplayTime < 6000 ? 
                `Waiting ${Math.ceil((6000 - errorDisplayTime) / 1000)}s before retry...` : 
                'Attempting to reconnect...'
              }
            </div>
            <button 
              onClick={() => {
                // First clear the error display time tracking
                setErrorDisplayTime(0);
                if (errorTimeoutRef.current) {
                  clearTimeout(errorTimeoutRef.current);
                  errorTimeoutRef.current = null;
                }
                
                // Set thinking state to show loading indicator
                setIsThinking(true);
                setVisibleThought(null);
                
                // Then attempt to fetch again
                setConnectionError(null);
                fetchBotThoughts();
              }}
              className="retry-button"
              style={{ marginTop: '10px' }}
            >
              Retry Now
            </button>
          </div>
        </div>
      );
    }
    
    // If thinking animation is active
    if (isThinking) {
      return (
        <div className="thinking-indicator">
          <div className="bot-name">{BOT_NAME}</div>
          <div className="thinking-dots">
            <span>.</span><span>.</span><span>.</span>
          </div>
        </div>
      );
    }
    
    // If showing a joke
    if (showingJoke && currentJoke) {
      return (
        <div className="thought-bubble joke">
          <div className="thought-wrapper">
            <div className="bot-name">{BOT_NAME}</div>
            <div>{currentJoke.joke_text}</div>
          </div>
        </div>
      );
    }
    
    // If showing a regular thought
    if (visibleThought) {
      return (
        <div className="thought-bubble">
          <div className="thought-wrapper">
            <div className="bot-name">{BOT_NAME}</div>
            <div>{visibleThought.thought_content}</div>
            {visibleThought.symbol && (
              <div className="thought-symbol">{visibleThought.symbol}</div>
            )}
          </div>
        </div>
      );
    }
    
    return null;
  };
  
  // Show connection error or thoughts
  if (!isHidden) {
    return (
      <div className="bot-thoughts-container">
        <div className="bot-thoughts">
          {renderThoughts()}
          
          <div className="carousel-controls">
            <button 
              className="control-button error compact" 
              onClick={() => {
                // First clear the error display time tracking
                setErrorDisplayTime(0);
                if (errorTimeoutRef.current) {
                  clearTimeout(errorTimeoutRef.current);
                  errorTimeoutRef.current = null;
                }
                
                // Set thinking state to show loading indicator
                setIsThinking(true);
                setVisibleThought(null);
                
                // Then attempt to fetch again after a short delay
                setTimeout(() => {
                  // When manually retrying, clear error state and force a new fetch
                  setConnectionError(null);
                  fetchBotThoughts()
                    .then(() => {
                      // On success, start the thought cycle
                      if (thoughts.length > 0 && showThoughtFn.current) {
                        showThoughtFn.current();
                      }
                    })
                    .catch(() => {
                      // On continued error, ensure the error message stays visible
                      setIsThinking(false);
                    });
                }, 500); // Short delay for UX clarity
              }}
              title="Retry connection"
            >
              🔄
            </button>
            <button 
              className="control-button compact" 
              onClick={toggleHide}
              title="Hide message"
            >
              👁️
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bot-thoughts-container">
      <div className="bot-thoughts">
        {isThinking && (
          <div className="thinking-indicator">
            <div className="bot-icon">
              <div className="bot-avatar"></div>
            </div>
            <span className="bot-name">{BOT_NAME}</span>
            <span className="dot-animation">
              <span className="dot">.</span>
              <span className="dot">.</span>
              <span className="dot">.</span>
            </span>
          </div>
        )}
        
        {visibleThought && !showingJoke && (
          <div className="thought-bubble">
            <div className="bot-icon">
              <div className="bot-avatar"></div>
            </div>
            <div className="thought-wrapper">
              <div className="bot-name">{BOT_NAME}</div>
              <span className="thought-content">{visibleThought.thought_content}</span>
            </div>
          </div>
        )}
        
        {showingJoke && currentJoke && (
          <div className="thought-bubble joke-bubble">
            <div className="bot-icon">
              <div className="bot-avatar"></div>
              <span className="joke-emoji">😂</span>
            </div>
            <div className="thought-wrapper">
              <div className="bot-name">{BOT_NAME}</div>
              <span className="thought-content">{currentJoke.joke_text}</span>
              <span className="joke-tag">#{currentJoke.category}</span>
            </div>
          </div>
        )}
        
        <div className="carousel-controls">
          <button 
            className="control-button" 
            onClick={prevThought}
            title="Previous thought"
          >
            ◀
          </button>
          
          <div className="position-indicator">
            {thoughts.map((_, index) => (
              <span 
                key={index} 
                className={`position-dot ${index === currentIndex ? 'active' : ''}`}
                onClick={() => navigateToThought(index)}
              />
            ))}
          </div>
          
          <button 
            className="control-button" 
            onClick={nextThought}
            title="Next thought"
          >
            ▶
          </button>
          
          <button 
            className={`control-button ${autoPlay ? 'active' : ''}`} 
            onClick={toggleAutoPlay}
            title={autoPlay ? "Pause auto-scroll" : "Resume auto-scroll"}
          >
            {autoPlay ? "⏸" : "▶️"}
          </button>
          
          <button 
            className="control-button" 
            onClick={toggleHide}
            title={`Hide ${BOT_NAME}'s thoughts`}
          >
            {isHidden ? "👁️" : "👁️‍🗨️"}
            <span className="timestamp-tooltip">
              {formatTimestamp(visibleThought?.timestamp || '')}
            </span>
          </button>
        </div>
      </div>
    </div>
  );
};

export default BotThoughts;
