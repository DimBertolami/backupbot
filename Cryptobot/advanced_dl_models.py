"""
Advanced Deep Learning Models for Cryptocurrency Trading

This module implements cutting-edge deep learning architectures optimized for 
cryptocurrency price prediction and trading signal generation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Bidirectional, Concatenate
from tensorflow.keras.layers import Attention, MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import TimeDistributed, GlobalAveragePooling1D
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import joblib

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def build_transformer_model(input_shape, output_units=1, dropout_rate=0.2, 
                           num_heads=4, ff_dim=256, num_transformer_blocks=4):
    """
    Build a Transformer-based model for time series prediction.
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units
        dropout_rate (float): Dropout rate for regularization
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward network dimension
        num_transformer_blocks (int): Number of transformer blocks
        
    Returns:
        tf.keras.Model: Compiled Transformer model
    """
    inputs = Input(shape=input_shape)
    x = inputs
    
    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=input_shape[1], dropout=dropout_rate
        )(x, x)
        
        # Skip connection 1
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(input_shape[1]),
        ])
        ffn_output = ffn(x)
        
        # Skip connection 2
        x = Add()([x, ffn_output])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    
    # Final layers
    x = Dense(64, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_inception_time_model(input_shape, output_units=1, num_filters=32, 
                              kernel_sizes=[1, 3, 5, 8, 12], dropout_rate=0.2):
    """
    Build an InceptionTime model for time series classification.
    Based on the architecture from https://arxiv.org/abs/1909.04939
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units
        num_filters (int): Number of filters in Conv1D layers
        kernel_sizes (list): List of kernel sizes for inception modules
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        tf.keras.Model: Compiled InceptionTime model
    """
    inputs = Input(shape=input_shape)
    
    # Inception module function
    def inception_module(input_tensor, stride=1):
        conv_list = []
        for kernel_size in kernel_sizes:
            conv = Conv1D(filters=num_filters, kernel_size=kernel_size, 
                         strides=stride, padding='same', activation='relu')(input_tensor)
            conv = BatchNormalization()(conv)
            conv_list.append(conv)
        
        # Add bottleneck layer (1x1 convolution)
        bottleneck = Conv1D(filters=num_filters, kernel_size=1, 
                           padding='same', activation='relu')(input_tensor)
        bottleneck = BatchNormalization()(bottleneck)
        conv_list.append(bottleneck)
        
        # Concatenate filters
        x = Concatenate(axis=-1)(conv_list)
        x = Dropout(dropout_rate)(x)
        return x
    
    # Initial convolution
    x = Conv1D(filters=num_filters, kernel_size=7, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    
    # Stack inception modules with residual connections
    num_inception_modules = 3
    for i in range(num_inception_modules):
        inception_out = inception_module(x)
        
        # Residual connection
        if i > 0:  # Skip the first module for residual
            # Match dimensions with 1x1 conv if needed
            shortcut = Conv1D(filters=num_filters * (len(kernel_sizes) + 1), 
                             kernel_size=1, padding='same')(x)
            x = Add()([inception_out, shortcut])
        else:
            x = inception_out
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Final classification
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_temporal_fusion_transformer(input_shape, output_units=1, num_heads=4, 
                                    dropout_rate=0.2, ff_dim=256):
    """
    Build a simplified Temporal Fusion Transformer for time series forecasting.
    Based on the architecture from https://arxiv.org/abs/1912.09363
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units
        num_heads (int): Number of attention heads
        dropout_rate (float): Dropout rate for regularization
        ff_dim (int): Feed-forward network dimension
        
    Returns:
        tf.keras.Model: Compiled Temporal Fusion Transformer model
    """
    inputs = Input(shape=input_shape)
    
    # Feature-wise processing with LSTM
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    
    # Self-attention layer
    attn_out = MultiHeadAttention(
        num_heads=num_heads, key_dim=128, dropout=dropout_rate
    )(x, x)
    x = Add()([x, attn_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ffn = Sequential([
        Dense(ff_dim, activation="relu"),
        Dense(128),
    ])
    ffn_out = TimeDistributed(ffn)(x)
    x = Add()([x, ffn_out])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Final processing
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def build_ensemble_model(models, input_shape, output_units=1):
    """
    Build an ensemble model that combines predictions from multiple models.
    
    Args:
        models (list): List of pre-trained models
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        output_units (int): Number of output units
        
    Returns:
        tf.keras.Model: Compiled ensemble model
    """
    inputs = Input(shape=input_shape)
    
    # Get outputs from each model
    model_outputs = []
    for i, model in enumerate(models):
        # Create a clone of the model to avoid weight sharing
        cloned_model = tf.keras.models.clone_model(model)
        cloned_model.set_weights(model.get_weights())
        
        # Freeze the model weights
        cloned_model.trainable = False
        
        # Get outputs
        outputs = cloned_model(inputs)
        model_outputs.append(outputs)
    
    # Merge outputs
    if len(model_outputs) > 1:
        x = Concatenate()(model_outputs)
    else:
        x = model_outputs[0]
    
    # Meta-learner
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    outputs = Dense(output_units, activation='sigmoid' if output_units == 1 else 'softmax')(x)
    
    ensemble_model = Model(inputs=inputs, outputs=outputs)
    ensemble_model.compile(
        optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for fine-tuning
        loss='binary_crossentropy' if output_units == 1 else 'categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return ensemble_model


# Example usage
if __name__ == "__main__":
    # Sample data shape
    sample_input_shape = (60, 20)  # 60 time steps, 20 features
    
    # Create a Transformer model
    transformer_model = build_transformer_model(sample_input_shape)
    print("Transformer model summary:")
    transformer_model.summary()
    
    # Create an InceptionTime model
    inception_model = build_inception_time_model(sample_input_shape)
    print("\nInceptionTime model summary:")
    inception_model.summary()
    
    # Create a TFT model
    tft_model = build_temporal_fusion_transformer(sample_input_shape)
    print("\nTemporal Fusion Transformer model summary:")
    tft_model.summary()
