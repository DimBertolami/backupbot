import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Provider } from 'react-redux';
import store from '../../store';
import BacktestingDashboard from '../BacktestingDashboard';
import { BacktestMetrics } from '../../types/backtest';

// Properly typed mock response
const mockFetchResponse = <T,>(data: T) => ({
  ok: true,
  json: () => Promise.resolve(data),
});

describe('BacktestingDashboard', () => {
  const mockMetrics: BacktestMetrics = {
    totalReturn: 0.3,
    maxDrawdown: 0.15,
    sharpeRatio: 1.2,
    winRate: 0.7,
    averageWin: 150,
    averageLoss: -50,
    averageLossPerTrade: -50,
    totalTrades: 100,
    profitableTrades: 70,
    losingTrades: 30,
    longestWinningStreak: 3,
    longestLosingStreak: 2,
    averageWinningStreak: 2.5,
    averageLosingStreak: 1.5,
    averageTradeDuration: "2h",
    maxTradeDuration: "24h",
    minTradeDuration: "15m",
    averageProfitPerTrade: 120,
    totalProfit: 12000,
    totalLoss: 4000,
    netProfit: 8000,
    profitFactor: 3.0,
    expectancy: 0.7,
    riskRewardRatio: 3.0,
    recoveryFactor: 2.5,
    riskOfRuin: 0.05,
    averageDailyReturn: 0.01,
    volatility: 0.1,
    correlationWithMarket: 0.6,
    beta: 0.8,
    alpha: 0.05,
    informationRatio: 1.5,
    trackingError: 0.08,
    downsideDeviation: 0.06,
    sortinoRatio: 2.0,
    calmarRatio: 2.5,
    omegaRatio: 1.8,
    tailRatio: 1.2,
    valueAtRisk: -0.05,
    conditionalValueAtRisk: -0.07,
    skewness: 0.5,
    kurtosis: 3.2,
    maxDailyLoss: -0.1,
    maxDailyGain: 0.15,
    averageDailyLoss: -0.03,
    averageDailyGain: 0.05,
    dailyWinRate: 0.6,
    dailyLossRate: 0.4,
    consecutiveWinningDays: 3,
    consecutiveLosingDays: 2,
    averageWinningDays: 2.5,
    averageLosingDays: 1.5,
    dailyProfitFactor: 2.5,
    dailyExpectancy: 0.7,
    dailyRiskRewardRatio: 3.0,
    dailyRecoveryFactor: 2.5,
    dailyRiskOfRuin: 0.05,
    dailyAverageReturn: 0.01,
    dailyVolatility: 0.1,
    dailyCorrelationWithMarket: 0.6,
    dailyBeta: 0.8,
    dailyAlpha: 0.05,
    dailyInformationRatio: 1.5,
    dailyTrackingError: 0.08,
    dailyDownsideDeviation: 0.06,
    dailySortinoRatio: 2.0,
    dailyCalmarRatio: 2.5,
    dailyOmegaRatio: 1.8,
    dailyTailRatio: 1.2,
    dailyValueAtRisk: -0.05,
    dailyConditionalValueAtRisk: -0.07,
    dailySkewness: 0.5,
    dailyKurtosis: 3.2,
    dailyMaxLoss: -0.1,
    dailyMaxGain: 0.15,
    dailyAverageLoss: -0.03,
    dailyAverageGain: 0.05
  };

  const mockBacktestResult = {
    strategy: 'RSI + MACD',
    timeframe: '1h',
    startDate: '2024-01-01',
    endDate: '2024-03-31',
    initialCapital: 10000,
    metrics: mockMetrics,
    trades: [
      {
        timestamp: '2024-01-15T14:00:00Z',
        type: 'buy',
        price: 25000,
        amount: 0.04,
        fee: 0.00075,
        pnl: 0,
        duration: '0h',
        strategy: 'RSI + MACD'
      },
      {
        timestamp: '2024-01-15T16:00:00Z',
        type: 'sell',
        price: 25500,
        amount: 0.04,
        fee: 0.00075,
        pnl: 199.8,
        duration: '2h',
        strategy: 'RSI + MACD'
      }
    ]
  };

  const mocks = {
    success: {
      backtestResult: mockBacktestResult,
      fetchResponse: mockFetchResponse(mockBacktestResult),
    },
    error: {
      backtestResult: null,
      fetchResponse: mockFetchResponse(null),
    },
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders component with default values', () => {
    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    expect(screen.getByText('Backtesting Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Select Strategy')).toBeInTheDocument();
    expect(screen.getByText('Select Timeframe')).toBeInTheDocument();
    expect(screen.getByText('Select Date Range')).toBeInTheDocument();
    expect(screen.getByText('Initial Capital')).toBeInTheDocument();
    expect(screen.getByText('Backtest Results')).toBeInTheDocument();
  });

  it('handles strategy selection', async () => {
    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const strategySelect = screen.getByLabelText('Select Strategy');
    fireEvent.change(strategySelect, { target: { value: 'RSI + MACD' } });

    await waitFor(() => {
      expect(strategySelect).toHaveValue('RSI + MACD');
    });
  });

  it('handles timeframe selection', async () => {
    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const timeframeSelect = screen.getByLabelText('Select Timeframe');
    fireEvent.change(timeframeSelect, { target: { value: '4h' } });

    await waitFor(() => {
      expect(timeframeSelect).toHaveValue('4h');
    });
  });

  it('handles date range selection', async () => {
    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const startDateInput = screen.getByLabelText('Start Date');
    const endDateInput = screen.getByLabelText('End Date');

    fireEvent.change(startDateInput, { target: { value: '2024-01-01' } });
    fireEvent.change(endDateInput, { target: { value: '2024-03-31' } });

    await waitFor(() => {
      expect(startDateInput).toHaveValue('2024-01-01');
      expect(endDateInput).toHaveValue('2024-03-31');
    });
  });

  it('handles initial capital input', async () => {
    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const capitalInput = screen.getByLabelText('Initial Capital');
    fireEvent.change(capitalInput, { target: { value: '10000' } });

    await waitFor(() => {
      expect(capitalInput).toHaveValue('10000');
    });
  });

  it('handles backtest execution', async () => {
    global.fetch = jest.fn().mockResolvedValue(mocks.success.fetchResponse);

    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const backtestButton = screen.getByText('Run Backtest');
    fireEvent.click(backtestButton);

    await waitFor(() => {
      expect(screen.getByText('Strategy: RSI + MACD')).toBeInTheDocument();
      expect(screen.getByText('Timeframe: 1h')).toBeInTheDocument();
      expect(screen.getByText('Start Date: 2024-01-01')).toBeInTheDocument();
      expect(screen.getByText('End Date: 2024-03-31')).toBeInTheDocument();
      expect(screen.getByText('Initial Capital: $10,000')).toBeInTheDocument();
      expect(screen.getByText('Total Return: 0.30%')).toBeInTheDocument();
    });
  });

  it('handles error case', async () => {
    global.fetch = jest.fn().mockRejectedValue(new Error('API Error'));

    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const backtestButton = screen.getByText('Run Backtest');
    fireEvent.click(backtestButton);

    await waitFor(() => {
      expect(screen.getByText('Error: API Error')).toBeInTheDocument();
    });
  });

  it('renders trade analysis correctly', async () => {
    global.fetch = jest.fn().mockResolvedValue(mocks.success.fetchResponse);

    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const backtestButton = screen.getByText('Run Backtest');
    fireEvent.click(backtestButton);

    await waitFor(() => {
      expect(screen.getByText('Trade Analysis')).toBeInTheDocument();
      expect(screen.getByText('Total Trades: 100')).toBeInTheDocument();
      expect(screen.getByText('Win Rate: 70%')).toBeInTheDocument();
      expect(screen.getByText('Average Win: 150%')).toBeInTheDocument();
      expect(screen.getByText('Average Loss: -50%')).toBeInTheDocument();
    });
  });

  it('renders performance metrics correctly', async () => {
    global.fetch = jest.fn().mockResolvedValue(mocks.success.fetchResponse);

    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const backtestButton = screen.getByText('Run Backtest');
    fireEvent.click(backtestButton);

    await waitFor(() => {
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
      expect(screen.getByText('Sharpe Ratio: 1.20')).toBeInTheDocument();
      expect(screen.getByText('Max Drawdown: 0.15%')).toBeInTheDocument();
    });
  });

  it('handles loading state', async () => {
    global.fetch = jest.fn().mockImplementation(() => {
      return new Promise((resolve) => {
        setTimeout(() => resolve(mocks.success.fetchResponse), 2000);
      });
    });

    render(
      <Provider store={store}>
        <BacktestingDashboard />
      </Provider>
    );

    const backtestButton = screen.getByText('Run Backtest');
    fireEvent.click(backtestButton);

    await waitFor(() => {
      expect(screen.getByText('Loading...')).toBeInTheDocument();
    });
  });
});
