"""
Environmental Sound Dataset - Real-time Audio Classification Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides real-time audio classification capabilities.
"""

import numpy as np
import librosa
import sounddevice as sd
from typing import Callable, Optional
import queue
import threading
import time
from collections import deque


class RealTimeClassifier:
    """
    Real-time audio classification from microphone input.
    """
    
    def __init__(
        self,
        model,
        scaler,
        label_encoder,
        sample_rate: int = 22050,
        chunk_duration: float = 1.0,
        overlap: float = 0.5
    ):
        """
        Initialize real-time classifier.
        
        Args:
            model: Trained classification model
            scaler: Fitted feature scaler
            label_encoder: Fitted label encoder
            sample_rate: Audio sample rate
            chunk_duration: Duration of each audio chunk in seconds
            overlap: Overlap between chunks (0.0 to 1.0)
        """
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.overlap_size = int(self.chunk_size * overlap)
        self.hop_size = self.chunk_size - self.overlap_size
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.predictions = deque(maxlen=10)
        self.callback = None
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input."""
        if status:
            print(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract features from audio chunk."""
        from load_data import prepare_features
        features = prepare_features([audio], feature_type='mfcc', n_mfcc=13)
        return features
    
    def predict_chunk(self, audio: np.ndarray) -> tuple:
        """
        Predict class for an audio chunk.
        
        Args:
            audio: Audio waveform
        
        Returns:
            Tuple of (predicted_class, confidence)
        """
        # Ensure correct length
        if len(audio) < self.chunk_size:
            audio = np.pad(audio, (0, self.chunk_size - len(audio)), mode='constant')
        elif len(audio) > self.chunk_size:
            audio = audio[:self.chunk_size]
        
        # Extract features
        features = self.extract_features(audio)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class_idx])
        predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        return predicted_class, confidence
    
    def process_audio_stream(self):
        """Process audio stream in real-time."""
        buffer = np.zeros(self.chunk_size)
        
        while self.is_recording:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                chunk = chunk.flatten()
                
                # Update buffer
                buffer = np.roll(buffer, -len(chunk))
                buffer[-len(chunk):] = chunk
                
                # Predict when we have enough data
                if len(buffer) >= self.chunk_size:
                    prediction, confidence = self.predict_chunk(buffer)
                    self.predictions.append((prediction, confidence, time.time()))
                    
                    # Call user callback if provided
                    if self.callback:
                        self.callback(prediction, confidence)
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def start(self, callback: Optional[Callable] = None):
        """
        Start real-time classification.
        
        Args:
            callback: Optional callback function(prediction, confidence)
        """
        self.callback = callback
        self.is_recording = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=int(self.sample_rate * 0.1)  # 100ms blocks
        )
        
        self.stream.start()
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_audio_stream)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        print("Real-time classification started. Press Ctrl+C to stop.")
    
    def stop(self):
        """Stop real-time classification."""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("Real-time classification stopped.")
    
    def get_latest_prediction(self) -> Optional[tuple]:
        """Get the latest prediction."""
        if len(self.predictions) > 0:
            return self.predictions[-1]
        return None
    
    def get_prediction_history(self, n: int = 5) -> list:
        """Get last n predictions."""
        return list(self.predictions)[-n:]


class AudioFileClassifier:
    """
    Classify audio from file with streaming approach.
    """
    
    def __init__(
        self,
        model,
        scaler,
        label_encoder,
        window_size: float = 1.0,
        hop_size: float = 0.5
    ):
        """
        Initialize file classifier.
        
        Args:
            model: Trained classification model
            scaler: Fitted feature scaler
            label_encoder: Fitted label encoder
            window_size: Analysis window size in seconds
            hop_size: Hop size in seconds
        """
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.window_size = window_size
        self.hop_size = hop_size
    
    def classify_file(
        self,
        filepath: str,
        sample_rate: int = 22050
    ) -> list:
        """
        Classify audio file with sliding window.
        
        Args:
            filepath: Path to audio file
            sample_rate: Sample rate
        
        Returns:
            List of (time, prediction, confidence) tuples
        """
        # Load audio
        audio, sr = librosa.load(filepath, sr=sample_rate)
        
        window_samples = int(sample_rate * self.window_size)
        hop_samples = int(sample_rate * self.hop_size)
        
        results = []
        
        for start in range(0, len(audio) - window_samples, hop_samples):
            chunk = audio[start:start + window_samples]
            time_sec = start / sample_rate
            
            # Extract features
            from load_data import prepare_features
            features = prepare_features([chunk], feature_type='mfcc', n_mfcc=13)
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            predicted_class = self.label_encoder.inverse_transform([predicted_class_idx])[0]
            
            results.append((time_sec, predicted_class, confidence))
        
        return results
    
    def get_dominant_class(self, filepath: str) -> tuple:
        """
        Get the dominant class in an audio file.
        
        Args:
            filepath: Path to audio file
        
        Returns:
            Tuple of (dominant_class, average_confidence, class_distribution)
        """
        results = self.classify_file(filepath)
        
        if not results:
            return None, 0.0, {}
        
        # Count predictions
        class_counts = {}
        class_confidences = {}
        
        for _, pred_class, confidence in results:
            if pred_class not in class_counts:
                class_counts[pred_class] = 0
                class_confidences[pred_class] = []
            class_counts[pred_class] += 1
            class_confidences[pred_class].append(confidence)
        
        # Find dominant class
        dominant_class = max(class_counts, key=class_counts.get)
        avg_confidence = np.mean(class_confidences[dominant_class])
        
        # Calculate distribution
        total = sum(class_counts.values())
        distribution = {cls: count / total for cls, count in class_counts.items()}
        
        return dominant_class, avg_confidence, distribution


if __name__ == '__main__':
    print("Real-time Audio Classification Module")
    print("=" * 50)
    print("Features:")
    print("  - RealTimeClassifier: Real-time classification from microphone")
    print("  - AudioFileClassifier: Classify audio files with sliding window")
    print()
    print("Note: Requires sounddevice library for real-time audio")
    print("Install with: pip install sounddevice")

