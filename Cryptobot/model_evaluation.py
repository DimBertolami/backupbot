"""
Model Evaluation Module for Cryptocurrency Trading Bot
This module provides functions to evaluate and compare different trading models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

# Check if TensorFlow is available for deep learning model evaluation
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

def evaluate_classification_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a classification model with detailed metrics and visualizations.
    
    Args:
        model: Trained classification model
        X_test: Test features
        y_test: True target values
        model_name: Name of the model for reporting
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For models that provide probability estimates
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            try:
                y_proba = model.predict_proba(X_test)
                # For binary classification
                if y_proba.shape[1] == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    # For multi-class
                    roc_auc = roc_auc_score(
                        pd.get_dummies(y_test), 
                        y_proba, 
                        multi_class='ovr', 
                        average='weighted'
                    )
            except:
                has_proba = False
                roc_auc = None
        else:
            roc_auc = None
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        
        # Convert labels for precision/recall if -1 is used
        y_test_adj = y_test.copy() if isinstance(y_test, np.ndarray) else y_test.values.copy()
        y_pred_adj = y_pred.copy()
        
        if -1 in y_test_adj:
            # Convert -1 to 0 for sklearn metrics
            y_test_adj = np.where(y_test_adj == -1, 0, y_test_adj)
            y_pred_adj = np.where(y_pred_adj == -1, 0, y_pred_adj)
        
        prec = precision_score(y_test_adj, y_pred_adj, average='weighted', zero_division=0)
        rec = recall_score(y_test_adj, y_pred_adj, average='weighted', zero_division=0)
        f1 = f1_score(y_test_adj, y_pred_adj, average='weighted', zero_division=0)
        
        # Print results
        print(f"\n--- {model_name} Evaluation ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print(f"\nClassification Report:\n")
        print(classification_report(y_test_adj, y_pred_adj, zero_division=0))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Determine labels based on unique values
        unique_values = np.unique(np.concatenate([y_test, y_pred]))
        if np.array_equal(unique_values, [-1, 1]) or np.array_equal(unique_values, [0, 1]):
            labels = ['Sell', 'Buy'] if -1 in unique_values else ['Hold', 'Buy']
        else:
            labels = [str(v) for v in unique_values]
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels,
                   yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name} Confusion Matrix')
        plt.tight_layout()
        plt.show()
        
        # Return metrics dictionary
        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_trading_performance(predictions, actual_prices, initial_balance=10000, commission=0.001, save_path=None):
    """
    Evaluate trading performance based on predictions and actual price data.
    
    Args:
        predictions: Array of predictions (1 for Buy, -1 for Sell, 0 for Hold)
        actual_prices: Series or array of closing prices
        initial_balance: Starting balance for simulation
        
    Returns:
        dict: Trading performance metrics
    """
    try:
        results = pd.DataFrame({
            'price': actual_prices,
            'signal': predictions
        })
        
        # Calculate returns
        results['price_change'] = results['price'].pct_change()
        
        # Calculate strategy returns based on previous signal
        results['strategy_returns'] = results['price_change'] * results['signal'].shift(1)
        
        # Calculate cumulative returns
        results['cumulative_market_returns'] = (1 + results['price_change']).cumprod() - 1
        results['cumulative_strategy_returns'] = (1 + results['strategy_returns']).cumprod() - 1
        
        # Calculate drawdown
        results['market_peak'] = results['cumulative_market_returns'].cummax()
        results['strategy_peak'] = results['cumulative_strategy_returns'].cummax()
        results['market_drawdown'] = results['market_peak'] - results['cumulative_market_returns']
        results['strategy_drawdown'] = results['strategy_peak'] - results['cumulative_strategy_returns']
        
        # Calculate portfolio value
        results['market_portfolio'] = initial_balance * (1 + results['cumulative_market_returns'])
        results['strategy_portfolio'] = initial_balance * (1 + results['cumulative_strategy_returns'])
        
        # Calculate trading metrics
        total_trades = (results['signal'].shift(1) != results['signal']).sum()
        winning_trades = (results['strategy_returns'] > 0).sum()
        losing_trades = (results['strategy_returns'] < 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Final portfolio value
        final_market_value = results['market_portfolio'].iloc[-1]
        final_strategy_value = results['strategy_portfolio'].iloc[-1]
        
        # Calculate annualized returns
        n_days = len(results)
        market_annual_return = (final_market_value / initial_balance) ** (252 / n_days) - 1
        strategy_annual_return = (final_strategy_value / initial_balance) ** (252 / n_days) - 1
        
        # Maximum drawdown
        max_market_drawdown = results['market_drawdown'].max()
        max_strategy_drawdown = results['strategy_drawdown'].max()
        
        # Calculate additional metrics
        # Apply commission costs
        if commission > 0:
            # Calculate position changes
            position_changes = (results['signal'].shift(1) != results['signal']).astype(int)
            # Apply commission to each trade
            commission_costs = position_changes * commission
            # Adjust strategy returns
            results['strategy_returns_after_commission'] = results['strategy_returns'] - commission_costs
            # Recalculate cumulative returns with commission
            results['cumulative_strategy_returns_after_commission'] = (1 + results['strategy_returns_after_commission']).cumprod() - 1
            # Update portfolio value
            results['strategy_portfolio_after_commission'] = initial_balance * (1 + results['cumulative_strategy_returns_after_commission'])
            # Update final values
            final_strategy_value_after_commission = results['strategy_portfolio_after_commission'].iloc[-1]
            strategy_annual_return_after_commission = (final_strategy_value_after_commission / initial_balance) ** (252 / n_days) - 1
        else:
            final_strategy_value_after_commission = final_strategy_value
            strategy_annual_return_after_commission = strategy_annual_return
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0)
        strategy_daily_returns = results['strategy_returns']
        market_daily_returns = results['price_change']
        
        strategy_sharpe = np.sqrt(252) * (strategy_daily_returns.mean() / strategy_daily_returns.std())
        market_sharpe = np.sqrt(252) * (market_daily_returns.mean() / market_daily_returns.std())
        
        # Print results
        print(f"\n--- Trading Performance Evaluation ---")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades} ({win_rate:.2%})")
        print(f"Losing Trades: {losing_trades} ({1-win_rate:.2%})")
        print(f"Initial Portfolio: ${initial_balance:.2f}")
        print(f"Final Market Value: ${final_market_value:.2f} ({results['cumulative_market_returns'].iloc[-1]:.2%} return)")
        print(f"Final Strategy Value: ${final_strategy_value:.2f} ({results['cumulative_strategy_returns'].iloc[-1]:.2%} return)")
        if commission > 0:
            print(f"Final Strategy Value After {commission*100:.2f}% Commission: ${final_strategy_value_after_commission:.2f}")
        print(f"Market Annual Return: {market_annual_return:.2%}")
        print(f"Strategy Annual Return: {strategy_annual_return:.2%}")
        if commission > 0:
            print(f"Strategy Annual Return After Commission: {strategy_annual_return_after_commission:.2%}")
        print(f"Market Sharpe Ratio: {market_sharpe:.2f}")
        print(f"Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"Maximum Market Drawdown: {max_market_drawdown:.2%}")
        print(f"Maximum Strategy Drawdown: {max_strategy_drawdown:.2%}")
        
        # Create figure with subplots
        fig, axs = plt.subplots(3, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [3, 2, 1]})
        
        # Plot portfolio performance
        axs[0].plot(results.index, results['market_portfolio'], label='Buy & Hold', color='blue')
        axs[0].plot(results.index, results['strategy_portfolio'], label='Strategy', color='green')
        if commission > 0:
            axs[0].plot(results.index, results['strategy_portfolio_after_commission'], 
                      label=f'Strategy After {commission*100:.2f}% Commission', color='orange', linestyle='--')
        axs[0].set_title('Portfolio Performance', fontsize=14)
        axs[0].set_ylabel('Portfolio Value ($)', fontsize=12)
        axs[0].legend()
        axs[0].grid(True)
        
        # Plot cumulative returns
        axs[1].plot(results.index, results['cumulative_market_returns'], label='Buy & Hold', color='blue')
        axs[1].plot(results.index, results['cumulative_strategy_returns'], label='Strategy', color='green')
        if commission > 0:
            axs[1].plot(results.index, results['cumulative_strategy_returns_after_commission'], 
                      label=f'Strategy After Commission', color='orange', linestyle='--')
        axs[1].set_title('Cumulative Returns', fontsize=14)
        axs[1].set_ylabel('Cumulative Return (%)', fontsize=12)
        axs[1].legend()
        axs[1].grid(True)
        
        # Plot drawdowns
        axs[2].fill_between(results.index, 0, results['market_drawdown'], alpha=0.3, color='blue', label='Market Drawdown')
        axs[2].fill_between(results.index, 0, results['strategy_drawdown'], alpha=0.3, color='red', label='Strategy Drawdown')
        axs[2].set_title('Drawdowns', fontsize=14)
        axs[2].set_ylabel('Drawdown (%)', fontsize=12)
        axs[2].set_xlabel('Date', fontsize=12)
        axs[2].legend()
        axs[2].grid(True)
        
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
            
        plt.show()
        
        # Plot drawdown
        plt.figure(figsize=(12, 6))
        plt.plot(results.index, results['market_drawdown'] * 100, label='Market Drawdown', alpha=0.7)
        plt.plot(results.index, results['strategy_drawdown'] * 100, label='Strategy Drawdown', alpha=0.7)
        plt.title('Drawdown Over Time')
        plt.xlabel('Date')
        plt.ylabel('Drawdown (%)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Return performance metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'final_market_value': final_market_value,
            'final_strategy_value': final_strategy_value,
            'market_return': results['cumulative_market_returns'].iloc[-1],
            'strategy_return': results['cumulative_strategy_returns'].iloc[-1],
            'market_annual_return': market_annual_return,
            'strategy_annual_return': strategy_annual_return,
            'max_market_drawdown': max_market_drawdown,
            'max_strategy_drawdown': max_strategy_drawdown,
            'results_df': results
        }
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating trading performance: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_deep_learning_model(model, X_test, y_test, sequence_length=60, features=None, actual_prices=None, model_name="Deep Learning Model", save_dir="models"):
    """
    Evaluate a deep learning model for cryptocurrency price prediction.
    
    Args:
        model: Trained TensorFlow/Keras model
        X_test: Test features (3D for sequence models)
        y_test: True target values
        sequence_length: Length of input sequences for reshaping if needed
        features: List of feature names
        actual_prices: Optional series of closing prices for trading simulation
        model_name: Name of the model for reporting
        save_dir: Directory to save output figures
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    if not HAS_TENSORFLOW:
        print("TensorFlow is not available. Cannot evaluate deep learning models.")
        return None
        
    try:
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Reshape data if needed
        if len(X_test.shape) == 2 and model.input_shape[1] > 1:  # Need to reshape to 3D
            n_features = X_test.shape[1] if features is None else len(features)
            X_test_reshaped = []
            for i in range(len(X_test) - sequence_length + 1):
                X_test_reshaped.append(X_test[i:i+sequence_length])
            X_test = np.array(X_test_reshaped)
        
        # Get predictions
        y_pred_prob = model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred = y_pred.flatten()
            y_pred_prob = y_pred_prob.flatten()
            
        # Convert to trading signals (-1 for sell, 1 for buy)
        trade_signals = np.where(y_pred == 0, -1, 1)
        
        # Basic classification metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Ensure y_test and y_pred are in correct format for metrics
        y_test_binary = y_test.copy()
        y_pred_binary = y_pred.copy()
        
        if -1 in y_test_binary:
            # Convert -1 to 0 for sklearn metrics
            y_test_binary = np.where(y_test_binary == -1, 0, y_test_binary)
            y_pred_binary = np.where(y_pred_binary == -1, 0, y_pred_binary)
        
        precision = precision_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
        recall = recall_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
        f1 = f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
        
        try:
            roc_auc = roc_auc_score(y_test_binary, y_pred_prob)
        except:
            roc_auc = None
        
        # Print results
        print(f"\n--- {model_name} Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC AUC: {roc_auc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        
        # Determine labels based on unique values
        unique_values = np.unique(np.concatenate([y_test, y_pred]))
        if np.array_equal(unique_values, [-1, 1]) or np.array_equal(unique_values, [0, 1]):
            labels = ['Sell', 'Buy'] if -1 in unique_values else ['Hold', 'Buy']
        else:
            labels = [str(v) for v in unique_values]
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels,
                   yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name} Confusion Matrix')
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Evaluate trading performance if prices are provided
        trading_metrics = None
        if actual_prices is not None and len(actual_prices) >= len(trade_signals):
            print("Evaluating trading performance...")
            # Use only the relevant portion of prices that match the predictions
            relevant_prices = actual_prices[-len(trade_signals):]
            
            # Get performance metrics
            save_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}_trading_performance.png")
            trading_metrics = evaluate_trading_performance(
                trade_signals, relevant_prices, save_path=save_path
            )
        
        # Compile and return all metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_prob,
            'trading_metrics': trading_metrics
        }
        
        return metrics
    
    except Exception as e:
        print(f"Error evaluating deep learning model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_models(models, X_test, y_test, actual_prices=None, model_names=None):
    """
    Compare multiple models and visualize their performance.
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test: True target values
        actual_prices: Optional series of prices for trading performance evaluation
        model_names: Optional list of model names
        
    Returns:
        dict: Comparison results
    """
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    
    results = {}
    performance_metrics = ['accuracy', 'precision', 'recall', 'f1']
    comparison = {metric: [] for metric in performance_metrics}
    
    # Evaluate each model
    for model, name in zip(models, model_names):
        print(f"\nEvaluating {name}...")
        metrics = evaluate_classification_model(model, X_test, y_test, name)
        if metrics:
            results[name] = metrics
            for metric in performance_metrics:
                comparison[metric].append(metrics[metric])
    
    # Create comparison dataframe
    comp_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': comparison['accuracy'],
        'Precision': comparison['precision'],
        'Recall': comparison['recall'],
        'F1 Score': comparison['f1']
    })
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    metrics_df = pd.melt(comp_df, id_vars=['Model'], var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=metrics_df)
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Print best model for each metric
    print("\n--- Best Models ---")
    for metric in performance_metrics:
        best_idx = np.argmax(comparison[metric])
        print(f"Best {metric.capitalize()}: {model_names[best_idx]} ({comparison[metric][best_idx]:.4f})")
    
    return {'individual_results': results, 'comparison': comp_df}

def backtest_ensemble(models, X_test, y_test, actual_prices, ensemble_method='majority_vote'):
    """
    Backtest an ensemble of models against historical price data.
    
    Args:
        models: List of trained models
        X_test: Test features
        y_test: True target values
        actual_prices: Series of closing prices
        ensemble_method: Method to combine predictions ('majority_vote' or 'average')
        
    Returns:
        dict: Backtesting results
    """
    # Get predictions from each model
    all_predictions = []
    for i, model in enumerate(models):
        try:
            preds = model.predict(X_test)
            all_predictions.append(preds)
        except Exception as e:
            print(f"Error getting predictions from model {i}: {e}")
    
    if not all_predictions:
        print("No valid predictions from any model")
        return None
    
    # Convert to numpy array
    all_predictions = np.array(all_predictions)
    
    # Combine predictions based on ensemble method
    if ensemble_method == 'majority_vote':
        # For each sample, count occurrences of each unique prediction
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            sample_preds = all_predictions[:, i]
            values, counts = np.unique(sample_preds, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])
    else:  # average
        final_predictions = np.mean(all_predictions, axis=0)
        final_predictions = np.round(final_predictions).astype(int)
    
    # Evaluate ensemble predictions
    print("\n--- Ensemble Model Evaluation ---")
    acc = accuracy_score(y_test, final_predictions)
    
    # Adjust for precision/recall if -1 is used
    y_test_adj = y_test.copy() if isinstance(y_test, np.ndarray) else y_test.values.copy()
    final_predictions_adj = final_predictions.copy()
    
    if -1 in y_test_adj:
        y_test_adj = np.where(y_test_adj == -1, 0, y_test_adj)
        final_predictions_adj = np.where(final_predictions_adj == -1, 0, final_predictions_adj)
    
    prec = precision_score(y_test_adj, final_predictions_adj, average='weighted', zero_division=0)
    rec = recall_score(y_test_adj, final_predictions_adj, average='weighted', zero_division=0)
    f1 = f1_score(y_test_adj, final_predictions_adj, average='weighted', zero_division=0)
    
    print(f"Ensemble Accuracy: {acc:.4f}")
    print(f"Ensemble Precision: {prec:.4f}")
    print(f"Ensemble Recall: {rec:.4f}")
    print(f"Ensemble F1 Score: {f1:.4f}")
    
    # Evaluate trading performance
    trading_performance = evaluate_trading_performance(final_predictions, actual_prices)
    
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'predictions': final_predictions,
        'trading_performance': trading_performance
    }
