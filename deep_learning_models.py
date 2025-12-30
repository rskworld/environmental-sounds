"""
Environmental Sound Dataset - Deep Learning Models Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides deep learning models (CNN, LSTM, Transformer) for audio classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from typing import Tuple, Optional, List
import librosa
import os


class AudioCNN:
    """
    Convolutional Neural Network for audio classification.
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int):
        """
        Initialize CNN model.
        
        Args:
            input_shape: Shape of input (time_steps, features)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build CNN architecture."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        callbacks_list: Optional[List] = None
    ) -> keras.callbacks.History:
        """Train the model."""
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
                callbacks.ModelCheckpoint('best_cnn_model.h5', save_best_only=True, monitor='val_loss')
            ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Save the model."""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)


class AudioLSTM:
    """
    LSTM model for audio sequence classification.
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int):
        """
        Initialize LSTM model.
        
        Args:
            input_shape: Shape of input (time_steps, features)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
    
    def _build_model(self) -> keras.Model:
        """Build LSTM architecture."""
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            # Bidirectional LSTM layers
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dropout(0.3),
            
            # Dense layers
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> keras.callbacks.History:
        """Train the model."""
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            callbacks.ModelCheckpoint('best_lstm_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Save the model."""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)


class AudioTransformer:
    """
    Transformer model for audio classification.
    """
    
    def __init__(self, input_shape: Tuple[int, int], num_classes: int, d_model: int = 128):
        """
        Initialize Transformer model.
        
        Args:
            input_shape: Shape of input (time_steps, features)
            num_classes: Number of output classes
            d_model: Model dimension
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.d_model = d_model
        self.model = self._build_model()
    
    def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """Transformer encoder block."""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        attention_output = layers.Dropout(dropout)(attention_output)
        out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attention_output)
        
        # Feed forward
        ffn_output = layers.Dense(ff_dim, activation="relu")(out1)
        ffn_output = layers.Dense(inputs.shape[-1])(ffn_output)
        ffn_output = layers.Dropout(dropout)(ffn_output)
        out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
        
        return out2
    
    def _build_model(self) -> keras.Model:
        """Build Transformer architecture."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Embedding
        x = layers.Dense(self.d_model)(inputs)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Transformer blocks
        x = self._transformer_encoder(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.3)
        x = self._transformer_encoder(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.3)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32
    ) -> keras.callbacks.History:
        """Train the model."""
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            callbacks.ModelCheckpoint('best_transformer_model.h5', save_best_only=True, monitor='val_loss')
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def save(self, filepath: str):
        """Save the model."""
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """Load a saved model."""
        self.model = keras.models.load_model(filepath)


def prepare_features_for_dl(
    audio_list: List[np.ndarray],
    feature_type: str = 'mel',
    n_mels: int = 128,
    hop_length: int = 512
) -> np.ndarray:
    """
    Prepare features for deep learning models.
    
    Args:
        audio_list: List of audio waveforms
        feature_type: Type of features ('mel', 'mfcc', 'spectrogram')
        n_mels: Number of mel bands
        hop_length: Hop length for STFT
    
    Returns:
        Feature array with shape (n_samples, time_steps, features)
    """
    features = []
    
    for audio in audio_list:
        if feature_type == 'mel':
            feat = librosa.feature.melspectrogram(
                y=audio, n_mels=n_mels, hop_length=hop_length
            )
            feat = librosa.power_to_db(feat, ref=np.max)
        elif feature_type == 'mfcc':
            feat = librosa.feature.mfcc(y=audio, n_mfcc=13, hop_length=hop_length)
        elif feature_type == 'spectrogram':
            stft = librosa.stft(audio, hop_length=hop_length)
            feat = np.abs(stft)
            feat = librosa.power_to_db(feat, ref=np.max)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        # Transpose to (time_steps, features)
        feat = feat.T
        features.append(feat)
    
    # Pad sequences to same length
    max_len = max(f.shape[0] for f in features)
    padded_features = []
    
    for feat in features:
        if feat.shape[0] < max_len:
            pad_width = max_len - feat.shape[0]
            feat = np.pad(feat, ((0, pad_width), (0, 0)), mode='constant')
        padded_features.append(feat)
    
    return np.array(padded_features)


if __name__ == '__main__':
    print("Deep Learning Models Module")
    print("=" * 50)
    print("Available models:")
    print("  - AudioCNN: Convolutional Neural Network")
    print("  - AudioLSTM: Long Short-Term Memory network")
    print("  - AudioTransformer: Transformer-based model")
    print()
    print("Example usage:")
    print("  model = AudioCNN(input_shape=(128, 128), num_classes=10)")
    print("  model.train(X_train, y_train, X_val, y_val)")

