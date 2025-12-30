"""
Environmental Sound Dataset - Data Loading Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import os
import numpy as np
import librosa
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Optional


def load_environmental_sounds(
    split: str = 'train',
    data_dir: str = './environmental-sounds',
    sample_rate: int = 22050,
    duration: Optional[float] = None
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load environmental sound audio files from the dataset.
    
    Args:
        split: 'train' or 'test' to specify which split to load
        data_dir: Root directory of the dataset
        sample_rate: Target sample rate for audio files
        duration: Maximum duration in seconds (None for full audio)
    
    Returns:
        Tuple of (audio_data, labels) where:
        - audio_data: List of numpy arrays containing audio waveforms
        - labels: List of class labels (strings)
    """
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise ValueError(f"Directory {split_dir} does not exist")
    
    audio_data = []
    labels = []
    
    # Iterate through class directories
    for class_name in os.listdir(split_dir):
        class_dir = os.path.join(split_dir, class_name)
        
        if not os.path.isdir(class_dir):
            continue
        
        # Load all audio files in this class
        for audio_file in os.listdir(class_dir):
            if audio_file.endswith(('.wav', '.mp3', '.flac')):
                audio_path = os.path.join(class_dir, audio_file)
                
                try:
                    # Load audio file
                    y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
                    audio_data.append(y)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading {audio_path}: {e}")
                    continue
    
    return audio_data, labels


def load_metadata(metadata_path: str = './environmental-sounds/metadata.csv') -> pd.DataFrame:
    """
    Load dataset metadata from CSV file.
    
    Args:
        metadata_path: Path to the metadata CSV file
    
    Returns:
        DataFrame containing metadata
    """
    if os.path.exists(metadata_path):
        return pd.read_csv(metadata_path)
    else:
        print(f"Metadata file not found at {metadata_path}")
        return pd.DataFrame()


def get_class_distribution(labels: List[str]) -> dict:
    """
    Get the distribution of classes in the dataset.
    
    Args:
        labels: List of class labels
    
    Returns:
        Dictionary with class names as keys and counts as values
    """
    from collections import Counter
    return dict(Counter(labels))


def prepare_features(
    audio_data: List[np.ndarray],
    feature_type: str = 'mfcc',
    n_mfcc: int = 13,
    n_mels: int = 128
) -> np.ndarray:
    """
    Extract features from audio data.
    
    Args:
        audio_data: List of audio waveforms
        feature_type: Type of features to extract ('mfcc', 'mel', 'chroma', 'spectral')
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of mel bands
    
    Returns:
        Numpy array of extracted features
    """
    features = []
    
    for audio in audio_data:
        if feature_type == 'mfcc':
            feat = librosa.feature.mfcc(y=audio, n_mfcc=n_mfcc)
            feat = np.mean(feat, axis=1)  # Average over time
        elif feature_type == 'mel':
            feat = librosa.feature.melspectrogram(y=audio, n_mels=n_mels)
            feat = np.mean(feat, axis=1)
        elif feature_type == 'chroma':
            feat = librosa.feature.chroma_stft(y=audio)
            feat = np.mean(feat, axis=1)
        elif feature_type == 'spectral':
            feat = librosa.feature.spectral_centroid(y=audio)
            feat = np.mean(feat)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        
        features.append(feat)
    
    return np.array(features)


if __name__ == '__main__':
    # Example usage
    print("Loading Environmental Sound Dataset...")
    
    try:
        train_data, train_labels = load_environmental_sounds('train')
        print(f"Loaded {len(train_data)} training samples")
        print(f"Classes: {set(train_labels)}")
        print(f"Class distribution: {get_class_distribution(train_labels)}")
        
        test_data, test_labels = load_environmental_sounds('test')
        print(f"Loaded {len(test_data)} test samples")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is properly structured in ./environmental-sounds/")

