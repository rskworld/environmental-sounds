"""
Environmental Sound Dataset - Audio Analysis Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os


def analyze_audio_file(
    audio_path: str,
    sample_rate: int = 22050
) -> Dict:
    """
    Analyze a single audio file and extract various features.
    
    Args:
        audio_path: Path to the audio file
        sample_rate: Target sample rate
    
    Returns:
        Dictionary containing audio analysis results
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Calculate features
    duration = len(y) / sr
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Extract various features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    
    analysis = {
        'file_path': audio_path,
        'duration': duration,
        'sample_rate': sr,
        'tempo': float(tempo),
        'mean_mfcc': np.mean(mfccs, axis=1).tolist(),
        'mean_spectral_centroid': float(np.mean(spectral_centroids)),
        'mean_spectral_rolloff': float(np.mean(spectral_rolloff)),
        'mean_zero_crossing_rate': float(np.mean(zero_crossing_rate)),
        'mean_chroma': np.mean(chroma, axis=1).tolist(),
        'mean_mel_spectrogram': np.mean(mel_spectrogram, axis=1).tolist(),
        'max_amplitude': float(np.max(np.abs(y))),
        'rms_energy': float(np.sqrt(np.mean(y**2)))
    }
    
    return analysis


def plot_audio_features(audio_path: str, save_path: Optional[str] = None):
    """
    Create visualization plots for audio features.
    
    Args:
        audio_path: Path to the audio file
        save_path: Optional path to save the plot
    """
    y, sr = librosa.load(audio_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Waveform
    time = np.linspace(0, len(y) / sr, len(y))
    axes[0].plot(time, y)
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[1])
    axes[1].set_title('Spectrogram')
    axes[1].set_ylabel('Frequency (Hz)')
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=axes[2])
    axes[2].set_title('MFCC')
    axes[2].set_ylabel('MFCC Coefficients')
    axes[2].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def compare_audio_files(audio_paths: list, labels: list = None) -> Dict:
    """
    Compare multiple audio files and return comparative analysis.
    
    Args:
        audio_paths: List of paths to audio files
        labels: Optional list of labels for each file
    
    Returns:
        Dictionary with comparative analysis
    """
    if labels is None:
        labels = [f"File {i+1}" for i in range(len(audio_paths))]
    
    comparisons = {}
    
    for path, label in zip(audio_paths, labels):
        try:
            analysis = analyze_audio_file(path)
            comparisons[label] = analysis
        except Exception as e:
            print(f"Error analyzing {path}: {e}")
    
    return comparisons


def get_dataset_statistics(data_dir: str = './environmental-sounds') -> Dict:
    """
    Get overall statistics about the dataset.
    
    Args:
        data_dir: Root directory of the dataset
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_files': 0,
        'total_duration': 0,
        'classes': {},
        'formats': {},
        'avg_duration': 0,
        'min_duration': float('inf'),
        'max_duration': 0
    }
    
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            if class_name not in stats['classes']:
                stats['classes'][class_name] = 0
            
            for audio_file in os.listdir(class_dir):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    stats['total_files'] += 1
                    stats['classes'][class_name] += 1
                    
                    ext = os.path.splitext(audio_file)[1]
                    stats['formats'][ext] = stats['formats'].get(ext, 0) + 1
                    
                    try:
                        y, sr = librosa.load(os.path.join(class_dir, audio_file))
                        duration = len(y) / sr
                        stats['total_duration'] += duration
                        stats['min_duration'] = min(stats['min_duration'], duration)
                        stats['max_duration'] = max(stats['max_duration'], duration)
                    except:
                        pass
    
    if stats['total_files'] > 0:
        stats['avg_duration'] = stats['total_duration'] / stats['total_files']
    
    if stats['min_duration'] == float('inf'):
        stats['min_duration'] = 0
    
    return stats


if __name__ == '__main__':
    # Example usage
    print("Audio Analysis Module")
    print("=" * 50)
    
    # Get dataset statistics
    try:
        stats = get_dataset_statistics()
        print("\nDataset Statistics:")
        print(f"Total files: {stats['total_files']}")
        print(f"Total duration: {stats['total_duration']:.2f} seconds")
        print(f"Average duration: {stats['avg_duration']:.2f} seconds")
        print(f"Min duration: {stats['min_duration']:.2f} seconds")
        print(f"Max duration: {stats['max_duration']:.2f} seconds")
        print(f"\nClasses: {list(stats['classes'].keys())}")
        print(f"Formats: {stats['formats']}")
    except Exception as e:
        print(f"Error: {e}")

