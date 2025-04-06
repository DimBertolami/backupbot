import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardBody, CardText, Button, FormGroup, Label, Input, Table } from 'reactstrap';
import Chart from 'chart.js/auto';
import { Line } from 'react-chartjs-2';
import { BACKTEST_API_URL } from '../config/api';

const BacktestingDashboard = () => {
  const [strategy, setStrategy] = useState('');
  const [timeframe, setTimeframe] = useState('1h');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [backtestResults, setBacktestResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState(null);

  const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'];
  const strategies = [
    'SMA_Crossover',
    'RSI_Momentum',
    'MACD_Strategy',
    'Bollinger_Bands',
    'Moving_Average_Envelope',
    'Volatility_Breakout',
    'Machine_Learning_Predictor'
  ];

  const handleBacktest = async () => {
    if (!strategy || !timeframe || !startDate || !endDate) {
      alert('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch(`${BACKTEST_API_URL}/backtest`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy,
          timeframe,
          startDate,
          endDate,
          initialCapital,
        }),
      });
      
      const data = await response.json();
      setBacktestResults(data);
      updateChart(data);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to run backtest');
    } finally {
      setLoading(false);
    }
  };

  const updateChart = (results) => {
    if (!results) return;

    const chartData = {
      labels: results.dates,
      datasets: [
        {
          label: 'Strategy Returns',
          data: results.strategyReturns,
          borderColor: 'rgb(75, 192, 192)',
          tension: 0.1,
        },
        {
          label: 'Benchmark Returns',
          data: results.benchmarkReturns,
          borderColor: 'rgb(255, 99, 132)',
          tension: 0.1,
        },
      ],
    };

    setChartData(chartData);
  };

  const generateReport = () => {
    if (!backtestResults) return null;

    const { totalReturn, maxDrawdown, sharpeRatio, winRate, avgWin, avgLoss } = backtestResults.metrics;

    return (
      <div className="mt-4">
        <h4>Performance Metrics</h4>
        <Table>
          <tbody>
            <tr>
              <td>Total Return</td>
              <td>{totalReturn.toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Maximum Drawdown</td>
              <td>{maxDrawdown.toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Sharpe Ratio</td>
              <td>{sharpeRatio.toFixed(2)}</td>
            </tr>
            <tr>
              <td>Win Rate</td>
              <td>{winRate.toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Average Win</td>
              <td>{avgWin.toFixed(2)}%</td>
            </tr>
            <tr>
              <td>Average Loss</td>
              <td>{avgLoss.toFixed(2)}%</td>
            </tr>
          </tbody>
        </Table>
        
        <h4>Trade Analysis</h4>
        <Table>
          <thead>
            <tr>
              <th>Date</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>Return</th>
              <th>Duration</th>
            </tr>
          </thead>
          <tbody>
            {backtestResults.trades.map((trade, index) => (
              <tr key={index}>
                <td>{trade.date}</td>
                <td>{trade.entryPrice}</td>
                <td>{trade.exitPrice}</td>
                <td>{trade.return}%</td>
                <td>{trade.duration}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </div>
    );
  };

  return (
    <div className="container mt-4">
      <Card>
        <CardHeader>
          <h3>Backtesting Dashboard</h3>
        </CardHeader>
        <CardBody>
          <FormGroup>
            <Label for="strategy">Strategy</Label>
            <Input
              type="select"
              name="strategy"
              id="strategy"
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
            >
              {strategies.map((strat, index) => (
                <option key={index} value={strat}>
                  {strat}
                </option>
              ))}
            </Input>
          </FormGroup>

          <FormGroup>
            <Label for="timeframe">Timeframe</Label>
            <Input
              type="select"
              name="timeframe"
              id="timeframe"
              value={timeframe}
              onChange={(e) => setTimeframe(e.target.value)}
            >
              {timeframes.map((tf, index) => (
                <option key={index} value={tf}>
                  {tf}
                </option>
              ))}
            </Input>
          </FormGroup>

          <FormGroup>
            <Label for="startDate">Start Date</Label>
            <Input
              type="date"
              name="startDate"
              id="startDate"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
            />
          </FormGroup>

          <FormGroup>
            <Label for="endDate">End Date</Label>
            <Input
              type="date"
              name="endDate"
              id="endDate"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
            />
          </FormGroup>

          <FormGroup>
            <Label for="initialCapital">Initial Capital</Label>
            <Input
              type="number"
              name="initialCapital"
              id="initialCapital"
              value={initialCapital}
              onChange={(e) => setInitialCapital(e.target.value)}
            />
          </FormGroup>

          <Button color="primary" onClick={handleBacktest} disabled={loading}>
            {loading ? 'Running Backtest...' : 'Run Backtest'}
          </Button>

          {backtestResults && (
            <div className="mt-4">
              <h4>Performance Chart</h4>
              {chartData && (
                <div style={{ height: '400px' }}>
                  <Line data={chartData} />
                </div>
              )}
              {generateReport()}
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
};

export default BacktestingDashboard;
