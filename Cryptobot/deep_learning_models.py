"""
Deep Learning Models for Cryptocurrency Trading

This module implements various deep learning models for time series forecasting
and trading signal generation in cryptocurrency markets.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def prepare_sequences(data, features, seq_length=60, target_col='target'):
    """
    Prepare sequence data for time series models.
    
    Args:
        data (pd.DataFrame): DataFrame with features and target
        features (list): List of feature column names
        seq_length (int): Sequence length (lookback window)
        target_col (str): Name of the target column
        
    Returns:
        tuple: X_sequences, y_targets
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[features].iloc[i:i+seq_length].values)
        y.append(data[target_col].iloc[i+seq_length])
    
    return np.array(X), np.array(y)


def build_lstm_model(input_shape, output_units=1, dropout_rate=0.2):
    """
    Build an LSTM model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units (1 for binary classification)
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        LSTM(50, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(25, activation='relu'),
        BatchNormalization(),
        
        Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_gru_model(input_shape, output_units=1, dropout_rate=0.2):
    """
    Build a GRU model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units (1 for binary classification)
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled GRU model
    """
    model = Sequential([
        GRU(100, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        GRU(50, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(25, activation='relu'),
        BatchNormalization(),
        
        Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_cnn_lstm_model(input_shape, output_units=1, dropout_rate=0.2):
    """
    Build a hybrid CNN-LSTM model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units (1 for binary classification)
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled CNN-LSTM model
    """
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_rate),
        
        LSTM(50, return_sequences=False),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(25, activation='relu'),
        BatchNormalization(),
        
        Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_bidirectional_lstm_model(input_shape, output_units=1, dropout_rate=0.2):
    """
    Build a Bidirectional LSTM model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units (1 for binary classification)
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled Bidirectional LSTM model
    """
    model = Sequential([
        Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Bidirectional(LSTM(50, return_sequences=False)),
        BatchNormalization(),
        Dropout(dropout_rate),
        
        Dense(25, activation='relu'),
        BatchNormalization(),
        
        Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_advanced_lstm_model(input_shape, output_units=1, dropout_rate=0.3):
    """
    Build an advanced LSTM model with residual connections.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units (1 for binary classification)
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled advanced LSTM model
    """
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First LSTM layer with residual connection
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Second LSTM layer
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Third LSTM layer
    x = LSTM(32, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Dense layers
    x = Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(x)
    x = BatchNormalization()(x)
    
    # Output layer
    if output_units == 1:
        outputs = Dense(1, activation='sigmoid')(x)
        loss_func = 'binary_crossentropy'
    else:
        outputs = Dense(output_units, activation='softmax')(x)
        loss_func = 'categorical_crossentropy'
    
    # Build and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss_func,
        metrics=['accuracy']
    )
    
    return model


class DeepLearningTrader:
    """
    A class for training, evaluating, and deploying deep learning models
    for cryptocurrency trading.
    """
    
    def __init__(self, model_type='lstm', sequence_length=60, batch_size=32,
                 epochs=50, patience=10, save_dir='models'):
        """
        Initialize the DeepLearningTrader.
        
        Args:
            model_type (str): Type of model ('lstm', 'gru', 'cnn_lstm', 'bilstm', 'advanced_lstm')
            sequence_length (int): Length of input sequences
            batch_size (int): Training batch size
            epochs (int): Maximum training epochs
            patience (int): Early stopping patience
            save_dir (str): Directory to save models
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.save_dir = save_dir
        self.model = None
        self.history = None
        self.features = None
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def prepare_data(self, data, features, target_col='target', test_size=0.2, val_size=0.2):
        """
        Prepare data for training.
        
        Args:
            data (pd.DataFrame): Preprocessed DataFrame with features and target
            features (list): Feature columns to use
            target_col (str): Target column name
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of training data for validation
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.features = features
        
        # Prepare sequences
        X, y = prepare_sequences(data, features, self.sequence_length, target_col)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Split training data into training and validation
        train_samples = len(X_train)
        val_samples = int(train_samples * val_size)
        
        X_val = X_train[-val_samples:]
        y_val = y_train[-val_samples:]
        X_train = X_train[:-val_samples]
        y_train = y_train[:-val_samples]
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self, input_shape, output_units=1):
        """
        Build the deep learning model.
        
        Args:
            input_shape (tuple): Shape of input data (sequence_length, n_features)
            output_units (int): Number of output units
            
        Returns:
            tf.keras.Model: Built model
        """
        if self.model_type == 'lstm':
            self.model = build_lstm_model(input_shape, output_units)
        elif self.model_type == 'gru':
            self.model = build_gru_model(input_shape, output_units)
        elif self.model_type == 'cnn_lstm':
            self.model = build_cnn_lstm_model(input_shape, output_units)
        elif self.model_type == 'bilstm':
            self.model = build_bidirectional_lstm_model(input_shape, output_units)
        elif self.model_type == 'advanced_lstm':
            self.model = build_advanced_lstm_model(input_shape, output_units)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, output_units=1):
        """
        Train the deep learning model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features
            y_val (np.array): Validation targets
            output_units (int): Number of output units
            
        Returns:
            tf.keras.Model: Trained model
        """
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        # Build the model
        if self.model is None:
            self.model = self.build_model(input_shape, output_units)
        
        # Define callbacks
        model_path = os.path.join(self.save_dir, f"{self.model_type}_model.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
        ]
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Get predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Convert to binary class for metrics if needed
        y_test_binary = (y_test > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        # Calculate metrics
        metrics = {
            'loss': loss,
            'accuracy': accuracy_score(y_test_binary, y_pred_binary),
            'precision': precision_score(y_test_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test_binary, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_test_binary, y_pred_binary, zero_division=0),
            'roc_auc': roc_auc_score(y_test_binary, y_pred_prob)
        }
        
        print("\nModel Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def plot_training_history(self):
        """
        Plot the training history.
        
        Returns:
            matplotlib.figure.Figure: Figure with plots
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet")
        
        # Create figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the figure
        fig_path = os.path.join(self.save_dir, f"{self.model_type}_training_history.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_model(self, model_name=None):
        """
        Save the trained model.
        
        Args:
            model_name (str): Custom model name
            
        Returns:
            str: Path to saved model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if model_name is None:
            model_name = f"{self.model_type}_model"
        
        # Save model
        model_path = os.path.join(self.save_dir, f"{model_name}.h5")
        self.model.save(model_path)
        
        # Save model architecture JSON
        json_path = os.path.join(self.save_dir, f"{model_name}_architecture.json")
        with open(json_path, 'w') as json_file:
            json_file.write(self.model.to_json())
        
        # Save feature list
        if self.features is not None:
            features_path = os.path.join(self.save_dir, f"{model_name}_features.joblib")
            joblib.dump(self.features, features_path)
        
        print(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
            
        Returns:
            tf.keras.Model: Loaded model
        """
        self.model = tf.keras.models.load_model(model_path)
        
        # Try to load feature list
        features_path = model_path.replace('.h5', '_features.joblib')
        if os.path.exists(features_path):
            self.features = joblib.load(features_path)
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions with the trained model.
        
        Args:
            X (np.array): Input data
            
        Returns:
            np.array: Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def backtest(self, data, features, initial_cash=10000, commission=0.001, plot=True):
        """
        Backtest the model on historical data.
        
        Args:
            data (pd.DataFrame): Historical price data with features
            features (list): Feature columns to use
            initial_cash (float): Initial investment amount
            commission (float): Trading commission (as a fraction)
            plot (bool): Whether to plot the results
            
        Returns:
            tuple: (pd.DataFrame with backtest results, backtest metrics)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Prepare sequences for the entire dataset
        X_full = []
        for i in range(len(data) - self.sequence_length):
            X_full.append(data[features].iloc[i:i+self.sequence_length].values)
        
        X_full = np.array(X_full)
        
        # Get predictions for the entire dataset
        predictions = self.model.predict(X_full)
        
        # Create a copy of the data for backtesting
        backtest_data = data.iloc[self.sequence_length:].copy().reset_index(drop=True)
        backtest_data['prediction'] = predictions.flatten()
        backtest_data['signal'] = (backtest_data['prediction'] > 0.5).astype(int)
        
        # Calculate returns
        backtest_data['return'] = backtest_data['close'].pct_change()
        backtest_data['strategy_return'] = backtest_data['return'] * backtest_data['signal'].shift(1)
        
        # Account for commission
        backtest_data['position_change'] = backtest_data['signal'].diff().abs()
        backtest_data['commission'] = backtest_data['position_change'] * commission
        backtest_data['strategy_return'] = backtest_data['strategy_return'] - backtest_data['commission']
        
        # Calculate cumulative returns
        backtest_data['cum_return'] = (1 + backtest_data['return']).cumprod()
        backtest_data['cum_strategy_return'] = (1 + backtest_data['strategy_return']).cumprod()
        
        # Calculate portfolio value
        backtest_data['portfolio_value'] = initial_cash * backtest_data['cum_strategy_return']
        
        # Calculate backtest metrics
        total_trades = backtest_data['position_change'].sum()
        profitable_trades = backtest_data[
            (backtest_data['position_change'] == 1) & 
            (backtest_data['strategy_return'] > 0)
        ].shape[0]
        
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate drawdowns
        backtest_data['peak'] = backtest_data['portfolio_value'].cummax()
        backtest_data['drawdown'] = (backtest_data['portfolio_value'] - backtest_data['peak']) / backtest_data['peak']
        max_drawdown = backtest_data['drawdown'].min()
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = (backtest_data['strategy_return'].mean() / backtest_data['strategy_return'].std()) * np.sqrt(252)
        
        metrics = {
            'total_return': backtest_data['cum_strategy_return'].iloc[-1] - 1,
            'annualized_return': backtest_data['cum_strategy_return'].iloc[-1] ** (252 / len(backtest_data)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'final_portfolio_value': backtest_data['portfolio_value'].iloc[-1],
        }
        
        # Plot backtest results
        if plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
            
            # Plot price and signals
            ax1.plot(backtest_data.index, backtest_data['close'])
            ax1.scatter(
                backtest_data[backtest_data['signal'] == 1].index,
                backtest_data[backtest_data['signal'] == 1]['close'],
                color='green', marker='^', label='Buy'
            )
            ax1.scatter(
                backtest_data[backtest_data['signal'] == 0].index,
                backtest_data[backtest_data['signal'] == 0]['close'],
                color='red', marker='v', label='Sell'
            )
            ax1.set_title('Price and Trading Signals')
            ax1.set_ylabel('Price')
            ax1.legend()
            
            # Plot cumulative returns
            ax2.plot(backtest_data.index, backtest_data['cum_return'], label='Buy and Hold')
            ax2.plot(backtest_data.index, backtest_data['cum_strategy_return'], label='Trading Strategy')
            ax2.set_title('Cumulative Returns')
            ax2.set_ylabel('Cumulative Return')
            ax2.legend()
            
            # Plot drawdowns
            ax3.fill_between(
                backtest_data.index,
                backtest_data['drawdown'],
                0,
                color='red', alpha=0.3
            )
            ax3.set_title('Drawdowns')
            ax3.set_ylabel('Drawdown')
            ax3.set_xlabel('Time')
            
            plt.tight_layout()
            
            # Save the figure
            fig_path = os.path.join(self.save_dir, f"{self.model_type}_backtest_results.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        
        # Print metrics
        print("\nBacktest Metrics:")
        for metric, value in metrics.items():
            if metric in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.2f}")
        
        return backtest_data, metrics


# Example usage
if __name__ == "__main__":
    # This is just a demonstration of how to use the module
    print("Deep Learning Models for Cryptocurrency Trading")
    print("Import this module to use the models in your trading system")
