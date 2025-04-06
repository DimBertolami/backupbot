"""
Advanced Cryptocurrency Plotting Module

This module provides enhanced visualization functions for cryptocurrency data
including interactive candlestick charts, technical indicators, and trading signals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import mplfinance as mpf
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def validate_plot_data(data, required_cols):
    """Validate data before plotting"""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a pandas DataFrame")
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")
    if 'timestamp' in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data

def plot_candlestick(data, title='Bitcoin Price', save_path=None, return_base64=False):
    """
    Create a candlestick chart with OHLC data.
    
    Args:
        data (pd.DataFrame): DataFrame with 'timestamp', 'open', 'high', 'low', 'close' columns
        title (str): Chart title
        save_path (str): Path to save the plot image
        return_base64 (bool): Whether to return base64 encoded image for web embedding
        
    Returns:
        str or None: Base64 encoded image if return_base64=True, otherwise None
    """
    try:
        # Validate data
        required_cols = ['open', 'high', 'low', 'close']
        data = validate_plot_data(data, required_cols)
        
        # Ensure data is sorted by timestamp
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp').copy()
            # Convert to datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data = data.set_index('timestamp')
        
        # Create the figure for mpf to use
        fig = plt.figure(figsize=(14, 10))
        
        # Define style
        mc = mpf.make_marketcolors(
            up='green', down='red',
            wick={'up': 'green', 'down': 'red'},
            edge={'up': 'green', 'down': 'red'},
            volume={'up': 'green', 'down': 'red'}
        )
        
        s = mpf.make_mpf_style(
            marketcolors=mc, 
            gridstyle='--', 
            y_on_right=False,
            figcolor='white',
            facecolor='white'
        )
        
        # Close the figure to avoid overlap
        plt.close(fig)
        
        # Use mpf.plot to create a new figure
        fig, axs = mpf.plot(
            data,
            type='candle',
            style=s,
            volume='volume' in data.columns,
            figsize=(14, 10),
            title=title,
            tight_layout=True,
            returnfig=True
        )
        
        # Save figure if path is provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Candlestick chart saved to {save_path}")
        
        # Return base64 encoded image if requested
        if return_base64:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return image_base64
        else:
            plt.show()
            return None
            
    except Exception as e:
        print(f"Error creating candlestick chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_technical_indicators(data, indicators=None, title='Bitcoin Technical Indicators', 
                            save_path=None, return_base64=False):
    """
    Create a technical analysis chart with price and indicators.
    
    Args:
        data (pd.DataFrame): DataFrame with price data and technical indicators
        indicators (list): List of indicator column names to include
        title (str): Chart title
        save_path (str): Path to save the plot image
        return_base64 (bool): Whether to return base64 encoded image for web embedding
        
    Returns:
        str or None: Base64 encoded image if return_base64=True, otherwise None
    """
    try:
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp').copy()
        
        # Set default indicators if none provided
        if indicators is None:
            # Look for common indicator names
            possible_indicators = [
                'SMA', 'SMA14', 'EMA', 'EMA14', 'RSI', 'MACD', 
                'UpperBand', 'MiddleBand', 'LowerBand'
            ]
            indicators = [ind for ind in possible_indicators if ind in data.columns]
        
        # Filter to only include indicators that exist in the data
        indicators = [ind for ind in indicators if ind in data.columns]
        
        if not indicators:
            print("No valid indicators found in data")
            return None
            
        # Determine how many indicator subplots we need
        # Group some indicators together, others separate
        subplot_groups = {
            'price': ['close', 'SMA', 'SMA14', 'EMA', 'EMA14', 'UpperBand', 'MiddleBand', 'LowerBand'],
            'momentum': ['RSI'],
            'trend': ['MACD', 'MACD_signal', 'MACD_hist']
        }
        
        # Count how many subplots we need
        active_groups = []
        for group, inds in subplot_groups.items():
            if any(ind in indicators for ind in inds):
                active_groups.append(group)
        
        n_subplots = len(active_groups)
        if n_subplots == 0:
            n_subplots = 1
            active_groups = ['price']
        
        # Create the subplots
        fig, axs = plt.subplots(n_subplots, 1, figsize=(14, 4*n_subplots), sharex=True)
        
        # If only one subplot, wrap axs in a list for iteration
        if n_subplots == 1:
            axs = [axs]
        
        # Plot each group of indicators
        for i, group in enumerate(active_groups):
            ax = axs[i]
            group_indicators = [ind for ind in indicators if ind in subplot_groups[group]]
            
            # Plot price data on first subplot
            if group == 'price':
                # Plot candlesticks for price data if we have OHLC
                if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                    # Convert to mplfinance format
                    ohlc_data = data.copy()
                    ohlc_data['timestamp'] = pd.to_datetime(ohlc_data['timestamp'])
                    ohlc_data = ohlc_data.set_index('timestamp')
                    
                    # Plot on this axis
                    mpf.plot(
                        ohlc_data,
                        type='candle',
                        ax=ax,
                        volume=False,
                        show_nontrading=True,
                        style='yahoo'
                    )
                else:
                    # Just plot close price as a line
                    ax.plot(data['timestamp'], data['close'], label='Price', color='black', linewidth=2)
                
                # Plot moving averages and Bollinger Bands
                if 'SMA' in indicators:
                    ax.plot(data['timestamp'], data['SMA'], label='SMA', linestyle='--', color='blue')
                if 'SMA14' in indicators:
                    ax.plot(data['timestamp'], data['SMA14'], label='SMA14', linestyle='--', color='green')
                if 'EMA' in indicators:
                    ax.plot(data['timestamp'], data['EMA'], label='EMA', linestyle=':', color='purple')
                if 'EMA14' in indicators:
                    ax.plot(data['timestamp'], data['EMA14'], label='EMA14', linestyle=':', color='orange')
                
                # Plot Bollinger Bands
                if 'UpperBand' in indicators and 'LowerBand' in indicators:
                    ax.fill_between(
                        data['timestamp'],
                        data['UpperBand'],
                        data['LowerBand'],
                        alpha=0.2,
                        color='gray',
                        label='Bollinger Bands'
                    )
                    if 'MiddleBand' in indicators:
                        ax.plot(data['timestamp'], data['MiddleBand'], label='Middle Band', color='red', alpha=0.6)
                
                ax.set_ylabel('Price')
                ax.set_title('Price and Overlays')
                ax.legend(loc='upper left')
                ax.grid(True)
            
            # Plot RSI
            elif group == 'momentum':
                ax.plot(data['timestamp'], data['RSI'], label='RSI', color='purple')
                ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)
                ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)
                ax.set_ylabel('RSI')
                ax.set_title('Relative Strength Index')
                ax.set_ylim(0, 100)
                ax.legend(loc='upper left')
                ax.grid(True)
            
            # Plot MACD
            elif group == 'trend':
                if 'MACD' in indicators and 'MACD_signal' in indicators:
                    ax.plot(data['timestamp'], data['MACD'], label='MACD', color='blue')
                    ax.plot(data['timestamp'], data['MACD_signal'], label='Signal', color='red', linestyle='--')
                    
                    if 'MACD_hist' in indicators:
                        # Plot histogram using bar chart
                        for j, (t, hist) in enumerate(zip(data['timestamp'], data['MACD_hist'])):
                            color = 'green' if hist > 0 else 'red'
                            ax.bar(t, hist, width=1.0, color=color, alpha=0.5)
                            
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                ax.set_ylabel('MACD')
                ax.set_title('Moving Average Convergence Divergence')
                ax.legend(loc='upper left')
                ax.grid(True)
        
        # Format x-axis dates
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        for label in axs[-1].get_xticklabels():
            label.set_rotation(45)
        
        # Add main title
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Technical indicator chart saved to {save_path}")
        
        # Return base64 encoded image if requested
        if return_base64:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return image_base64
        else:
            plt.show()
            return None
            
    except Exception as e:
        print(f"Error creating technical indicators chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_trading_signals(data, model_predictions=None, title='Trading Signals', 
                        save_path=None, return_base64=False):
    """
    Create a chart with price data and trading signals.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        model_predictions (np.array): Array of model predictions (1 for buy, -1 for sell)
        title (str): Chart title
        save_path (str): Path to save the plot image
        return_base64 (bool): Whether to return base64 encoded image for web embedding
        
    Returns:
        str or None: Base64 encoded image if return_base64=True, otherwise None
    """
    try:
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp').copy()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot candlestick chart if OHLC data available
        if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            # Convert to mplfinance format
            ohlc_data = data.copy()
            ohlc_data['timestamp'] = pd.to_datetime(ohlc_data['timestamp'])
            ohlc_data = ohlc_data.set_index('timestamp')
            
            # Plot on this axis
            mpf.plot(
                ohlc_data,
                type='candle',
                ax=ax,
                volume=False,
                show_nontrading=True,
                style='yahoo'
            )
        else:
            # Just plot close price as a line
            ax.plot(data['timestamp'], data['close'], label='Price', color='black')
        
        # Add trading signals if provided
        if model_predictions is not None:
            # If predictions length doesn't match data, use only the last section
            if len(model_predictions) < len(data):
                signals = np.full(len(data), np.nan)
                signals[-len(model_predictions):] = model_predictions
            elif len(model_predictions) > len(data):
                signals = model_predictions[-len(data):]
            else:
                signals = model_predictions
            
            # Plot buy signals (1)
            buy_signals = np.where(signals == 1)[0]
            if len(buy_signals) > 0:
                ax.scatter(
                    data.iloc[buy_signals]['timestamp'],
                    data.iloc[buy_signals]['close'],
                    marker='^',
                    color='green',
                    s=100,
                    label='Buy Signal'
                )
            
            # Plot sell signals (-1)
            sell_signals = np.where(signals == -1)[0]
            if len(sell_signals) > 0:
                ax.scatter(
                    data.iloc[sell_signals]['timestamp'],
                    data.iloc[sell_signals]['close'],
                    marker='v',
                    color='red',
                    s=100,
                    label='Sell Signal'
                )
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
        
        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.legend(loc='best')
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trading signals chart saved to {save_path}")
        
        # Return base64 encoded image if requested
        if return_base64:
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)
            return image_base64
        else:
            plt.show()
            return None
            
    except Exception as e:
        print(f"Error creating trading signals chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_3d_surface(data, title='3D Cryptocurrency Analysis', save_path=None, return_html=False):
    """
    Create a 3D surface plot of cryptocurrency data.
    
    Args:
        data (pd.DataFrame): DataFrame with price data and indicators
        title (str): Chart title
        save_path (str): Path to save the plot HTML file
        return_html (bool): Whether to return HTML string for web embedding
        
    Returns:
        str or None: HTML string if return_html=True, otherwise None
    """
    try:
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp').copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Create a meshgrid for the 3D surface
        # X-axis: Time
        # Y-axis: Technical indicators (e.g., RSI, MACD)
        # Z-axis: Price
        
        # Get the technical indicators we want to display
        indicators = ['RSI', 'MACD', 'SMA', 'EMA']
        indicators = [ind for ind in indicators if ind in data.columns]
        
        if not indicators:
            print("No valid indicators found in data")
            return None
        
        # Create the 3D figure
        fig = go.Figure()
        
        # Add price surface
        times = np.arange(len(data))
        prices = data['close'].values
        
        fig.add_trace(
            go.Surface(
                z=np.tile(prices.reshape(-1, 1), (1, 2)),  # Repeat prices to create a surface
                x=np.tile(times.reshape(-1, 1), (1, 2)),    # Time indices
                y=[[0, 1]] * len(data),                     # Two points to create width
                colorscale='Viridis',
                opacity=0.8,
                name='Price Surface',
                showscale=True
            )
        )
        
        # Add indicator lines floating above the surface
        for i, indicator in enumerate(indicators):
            # Normalize the indicator values to a reasonable range
            normalized = (data[indicator] - data[indicator].min()) / (data[indicator].max() - data[indicator].min())
            # Scale and offset for visualization
            z_values = prices + (normalized * prices.mean() * 0.3)
            
            # Pick a color based on the indicator
            colors = {
                'RSI': 'red',
                'MACD': 'blue',
                'SMA': 'green',
                'EMA': 'purple',
                'SMA14': 'teal',
                'EMA14': 'orange'
            }
            color = colors.get(indicator, 'yellow')
            
            # Add a 3D line for this indicator
            fig.add_trace(
                go.Scatter3d(
                    x=times,
                    y=[1.5] * len(data),  # Offset from the surface
                    z=z_values,
                    mode='lines',
                    line=dict(color=color, width=4),
                    name=indicator
                )
            )
        
        # Create time labels for hover text (showing every 10th point)
        time_labels = [data['timestamp'].iloc[i].strftime('%Y-%m-%d %H:%M') for i in range(0, len(data), max(1, len(data)//10))]
        time_indices = list(range(0, len(data), max(1, len(data)//10)))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Dimension',
                zaxis_title='Price',
                xaxis=dict(
                    tickmode='array',
                    tickvals=time_indices,
                    ticktext=time_labels,
                    tickangle=45,
                    nticks=10
                ),
                yaxis=dict(
                    showticklabels=False
                ),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=0.8)
                ),
                aspectratio=dict(x=2, y=0.5, z=1)
            ),
            height=700,
            width=1000,
            margin=dict(t=50, l=25, r=25, b=25),
            template='plotly_dark'
        )
        
        # Save to HTML file if path is provided
        if save_path:
            if not save_path.endswith('.html'):
                save_path += '.html'
            fig.write_html(save_path)
            print(f"3D surface chart saved to {save_path}")
        
        # Return HTML string if requested
        if return_html:
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
        else:
            fig.show()
            return None
            
    except Exception as e:
        print(f"Error creating 3D surface chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_3d_trading_signals(data, model_predictions=None, title='3D Trading Signals', save_path=None, return_html=False):
    """
    Create a 3D visualization of trading signals.
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        model_predictions (np.array): Array of model predictions (1 for buy, -1 for sell, 0 for hold)
        title (str): Chart title
        save_path (str): Path to save the plot HTML file
        return_html (bool): Whether to return HTML string for web embedding
        
    Returns:
        str or None: HTML string if return_html=True, otherwise None
    """
    try:
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp').copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # If predictions are not provided, create a placeholder array of holds (0)
        if model_predictions is None or len(model_predictions) != len(data):
            model_predictions = np.zeros(len(data))
        
        # Create a 3D figure
        fig = go.Figure()
        
        # Extract data for plotting
        times = np.arange(len(data))
        prices = data['close'].values
        volumes = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
        
        # Normalize volume for better visualization
        normalized_volumes = (volumes - volumes.min()) / (volumes.max() - volumes.min()) * 100
        
        # Plot the price trajectory as a 3D line
        fig.add_trace(
            go.Scatter3d(
                x=times,
                y=normalized_volumes,
                z=prices,
                mode='lines',
                line=dict(color='gray', width=3),
                name='Price Path'
            )
        )
        
        # Add buy signals
        buy_indices = np.where(model_predictions == 1)[0]
        if len(buy_indices) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=times[buy_indices],
                    y=normalized_volumes[buy_indices],
                    z=prices[buy_indices],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='green',
                        symbol='circle'
                    ),
                    name='Buy Signals'
                )
            )
        
        # Add sell signals
        sell_indices = np.where(model_predictions == -1)[0]
        if len(sell_indices) > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=times[sell_indices],
                    y=normalized_volumes[sell_indices],
                    z=prices[sell_indices],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='x'
                    ),
                    name='Sell Signals'
                )
            )
        
        # Create time labels for hover text (showing every 10th point)
        time_labels = [data['timestamp'].iloc[i].strftime('%Y-%m-%d %H:%M') for i in range(0, len(data), max(1, len(data)//10))]
        time_indices = list(range(0, len(data), max(1, len(data)//10)))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Volume (normalized)',
                zaxis_title='Price',
                xaxis=dict(
                    tickmode='array',
                    tickvals=time_indices,
                    ticktext=time_labels,
                    tickangle=45,
                    nticks=10
                )
            ),
            height=700,
            width=1000,
            margin=dict(t=50, l=25, r=25, b=25),
            template='plotly_dark'
        )
        
        # Save to HTML file if path is provided
        if save_path:
            if not save_path.endswith('.html'):
                save_path += '.html'
            fig.write_html(save_path)
            print(f"3D trading signals chart saved to {save_path}")
        
        # Return HTML string if requested
        if return_html:
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
        else:
            fig.show()
            return None
            
    except Exception as e:
        print(f"Error creating 3D trading signals chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def plot_combined_indicators_plotly(data, indicators=None, title='Bitcoin Price with Indicators', save_path=None, return_html=False):
    """
    Create a combined interactive Plotly chart with price and all indicators.
    
    Args:
        data (pd.DataFrame): DataFrame with price data and technical indicators
        indicators (list): List of indicator column names to include
        title (str): Chart title
        save_path (str): Path to save the plot HTML file
        return_html (bool): Whether to return HTML string for web embedding
        
    Returns:
        str or None: HTML string if return_html=True, otherwise None
    """
    try:
        # Ensure data is sorted by timestamp
        data = data.sort_values('timestamp').copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Set default indicators if none provided
        if indicators is None:
            # Look for common indicator names
            possible_indicators = [
                'SMA', 'SMA14', 'EMA', 'EMA14', 'RSI', 
                'MACD', 'MACD_signal', 'MACD_hist',
                'UpperBand', 'MiddleBand', 'LowerBand'
            ]
            indicators = [ind for ind in possible_indicators if ind in data.columns]
        
        # Filter to only include indicators that exist in the data
        indicators = [ind for ind in indicators if ind in data.columns]
        
        if not indicators:
            print("No valid indicators found in data")
            return None
            
        # Create a subplot with 2 rows - main price chart and RSI/momentum indicators
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           row_heights=[0.7, 0.3],
                           vertical_spacing=0.05,
                           subplot_titles=[title, 'Momentum Indicators (RSI)'])
        
        # Plot the actual price as a black line in the main chart
        fig.add_trace(
            go.Scatter(x=data['timestamp'], y=data['close'],
                      mode='lines', name='Price', line=dict(color='black', width=2)),
            row=1, col=1
        )
        
        # Plot all other indicators on the main chart except RSI (which goes on bottom chart)
        color_map = {
            'SMA': 'blue',
            'SMA14': 'green',
            'EMA': 'purple',
            'EMA14': 'orange',
            'UpperBand': 'red',
            'MiddleBand': 'grey',
            'LowerBand': 'red',
        }
        
        # Add all non-RSI indicators to main chart
        for indicator in indicators:
            if indicator not in ['RSI', 'MACD', 'MACD_signal', 'MACD_hist'] and 'RSI' not in indicator:
                color = color_map.get(indicator, 'cyan')
                fig.add_trace(
                    go.Scatter(x=data['timestamp'], y=data[indicator],
                            mode='lines', name=indicator, line=dict(color=color)),
                    row=1, col=1
                )

        # Add RSI to the second subplot
        if 'RSI' in indicators:
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['RSI'],
                        mode='lines', name='RSI', line=dict(color='blue')),
                row=2, col=1
            )
            # Add overbought/oversold levels for RSI
            fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=2, col=1)
            fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=2, col=1)
        
        # Add MACD components to the second subplot if available
        if 'MACD' in indicators and 'MACD_signal' in indicators:
            if 'RSI' not in indicators:
                # If no RSI, change subtitle
                fig.layout.annotations[1].text = 'MACD Indicator'
            
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['MACD'],
                        mode='lines', name='MACD', line=dict(color='blue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=data['timestamp'], y=data['MACD_signal'],
                        mode='lines', name='Signal', line=dict(color='red')),
                row=2, col=1
            )
            
            # Add MACD histogram if available
            if 'MACD_hist' in indicators:
                # Create a custom MACD histogram
                colors = ['green' if val >= 0 else 'red' for val in data['MACD_hist']]
                fig.add_trace(
                    go.Bar(x=data['timestamp'], y=data['MACD_hist'],
                          name='MACD Hist', marker_color=colors),
                    row=2, col=1
                )
        
        # Update layout for better appearance
        fig.update_layout(
            height=800,
            template='plotly_white',
            hovermode='x unified',
            legend=dict(orientation='h', y=1.02),
            margin=dict(t=60, l=60, r=30, b=30)
        )
        
        fig.update_yaxes(title_text='Price', row=1, col=1)
        
        # Determine what to show on the second subplot y-axis
        if 'RSI' in indicators:
            fig.update_yaxes(title_text='RSI', row=2, col=1)
        elif any(ind in indicators for ind in ['MACD', 'MACD_signal', 'MACD_hist']):
            fig.update_yaxes(title_text='MACD', row=2, col=1)
        
        # Save to HTML file if path is provided
        if save_path:
            if not save_path.endswith('.html'):
                save_path += '.html'
            fig.write_html(save_path)
            print(f"Combined indicator chart saved to {save_path}")
        
        # Return HTML string if requested
        if return_html:
            return pio.to_html(fig, include_plotlyjs=True, full_html=True)
        else:
            fig.show()
            return None
            
    except Exception as e:
        print(f"Error creating combined indicator chart: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_all_charts(data, predictions=None, save_dir='charts', return_base64=False):
    """
    Generate all types of charts (candlestick, technical, signals) and save them.
    
    Args:
        data (pd.DataFrame): DataFrame with price and indicator data
        predictions (np.array): Array of model predictions
        save_dir (str): Directory to save chart images
        return_base64 (bool): Whether to return base64 encoded images
        
    Returns:
        dict: Dictionary of chart types and their base64 encodings if return_base64=True
    """
    try:
        # Create the directory if it doesn't exist
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Dictionary to store base64 encoded images and HTML content
        charts = {}
        
        # Generate 3D price surface chart (replacing candlestick)
        price_3d_path = os.path.join(save_dir, '3d_price.html') if save_dir else None
        price_3d_html = plot_3d_surface(
            data, 
            title='3D Cryptocurrency Analysis',
            save_path=price_3d_path,
            return_html=True
        )
        
        # Save the 3D price chart HTML content for the frontend to use
        if price_3d_html:
            # Save the HTML content to a file that will be accessible to the frontend
            price_3d_file = os.path.join(save_dir, '3d_price.html')
            with open(price_3d_file, 'w') as f:
                f.write(price_3d_html)
                
            # Store the path for frontend reference
            charts['price_3d_html'] = price_3d_file
            
            # For compatibility, also generate the traditional candlestick chart
            if return_base64:
                candlestick_path = os.path.join(save_dir, 'candlestick.png') if save_dir else None
                candlestick_b64 = plot_candlestick(
                    data, 
                    title='Bitcoin Price Chart',
                    save_path=candlestick_path,
                    return_base64=return_base64
                )
                if candlestick_b64:
                    charts['candlestick'] = candlestick_b64
        
        # Generate the combined indicators chart with Plotly
        indicators_path = os.path.join(save_dir, 'indicators.html') if save_dir else None
        combined_html = plot_combined_indicators_plotly(
            data,
            title='Crypto Price with All Indicators',
            save_path=indicators_path,
            return_html=True  # Return HTML instead of base64 image
        )
        
        # Save the combined indicators HTML content for the frontend to use
        if combined_html:
            # Save the HTML content to a file that will be accessible to the frontend
            indicators_file = os.path.join(save_dir, 'indicators.html')
            with open(indicators_file, 'w') as f:
                f.write(combined_html)
            
            # Store the path for frontend reference
            charts['indicators_html'] = indicators_file
            
            # For compatibility with the existing system, also generate the static image
            if return_base64:
                indicators_static_path = os.path.join(save_dir, 'indicators.png') if save_dir else None
                indicators_b64 = plot_technical_indicators(
                    data,
                    title='Technical Indicators (Static Version)',
                    save_path=indicators_static_path,
                    return_base64=return_base64
                )
                if indicators_b64:
                    charts['indicators'] = indicators_b64
        
        # Generate 3D trading signals chart (replacing traditional signals chart)
        if predictions is not None:
            signals_3d_path = os.path.join(save_dir, '3d_signals.html') if save_dir else None
            signals_3d_html = plot_3d_trading_signals(
                data,
                model_predictions=predictions,
                title='3D Trading Signals',
                save_path=signals_3d_path,
                return_html=True
            )
            
            # Save the 3D signals HTML content for the frontend to use
            if signals_3d_html:
                # Save the HTML content to a file that will be accessible to the frontend
                signals_3d_file = os.path.join(save_dir, '3d_signals.html')
                with open(signals_3d_file, 'w') as f:
                    f.write(signals_3d_html)
                
                # Store the path for frontend reference
                charts['signals_3d_html'] = signals_3d_file
                
                # For compatibility, also generate the traditional signals chart
                if return_base64:
                    signals_path = os.path.join(save_dir, 'signals.png') if save_dir else None
                    signals_b64 = plot_trading_signals(
                        data,
                        model_predictions=predictions,
                        title='Trading Signals',
                        save_path=signals_path,
                        return_base64=return_base64
                    )
                    if signals_b64:
                        charts['signals'] = signals_b64
        
        return charts if return_base64 else None
        
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage when running this module directly
if __name__ == "__main__":
    try:
        # Try to import and use data from the main cryptocurrency bot
        from fetchall import fe_preprocess
        
        print("Fetching and preprocessing cryptocurrency data...")
        data = fe_preprocess(exch="binance")
        
        if data is not None:
            print("Generating charts...")
            # Create a directory for charts
            charts_dir = os.path.join(os.getcwd(), 'charts')
            os.makedirs(charts_dir, exist_ok=True)
            
            # Generate all charts
            generate_all_charts(data, save_dir=charts_dir)
            
            print(f"Charts saved to {charts_dir}")
        else:
            print("Failed to fetch cryptocurrency data. Cannot generate charts.")
            
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
