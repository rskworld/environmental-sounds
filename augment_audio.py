"""
Environmental Sound Dataset - Advanced Audio Augmentation Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides advanced audio augmentation techniques for improving model robustness.
"""

import numpy as np
import librosa
import soundfile as sf
from typing import List, Tuple, Optional
import random
import os


class AudioAugmenter:
    """
    Advanced audio augmentation class with multiple augmentation techniques.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio augmenter.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def time_stretch(self, audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
        """
        Time stretch audio without changing pitch.
        
        Args:
            audio: Audio waveform
            rate: Stretch rate (1.0 = no change, >1.0 = faster, <1.0 = slower)
        
        Returns:
            Time-stretched audio
        """
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, n_steps: float = 0.0) -> np.ndarray:
        """
        Shift pitch of audio.
        
        Args:
            audio: Audio waveform
            n_steps: Number of semitones to shift (positive = higher, negative = lower)
        
        Returns:
            Pitch-shifted audio
        """
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def add_noise(self, audio: np.ndarray, noise_factor: float = 0.005) -> np.ndarray:
        """
        Add Gaussian noise to audio.
        
        Args:
            audio: Audio waveform
            noise_factor: Noise intensity factor
        
        Returns:
            Audio with added noise
        """
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return np.clip(augmented_audio, -1.0, 1.0)
    
    def time_shift(self, audio: np.ndarray, shift_max: float = 0.2) -> np.ndarray:
        """
        Randomly shift audio in time (wrap around).
        
        Args:
            audio: Audio waveform
            shift_max: Maximum shift as fraction of audio length
        
        Returns:
            Time-shifted audio
        """
        shift = int(self.sample_rate * shift_max * random.uniform(-1, 1))
        augmented_audio = np.roll(audio, shift)
        return augmented_audio
    
    def volume_change(self, audio: np.ndarray, db_range: Tuple[float, float] = (-6, 6)) -> np.ndarray:
        """
        Change volume of audio.
        
        Args:
            audio: Audio waveform
            db_range: Range of dB change (min, max)
        
        Returns:
            Volume-adjusted audio
        """
        db_change = random.uniform(db_range[0], db_range[1])
        gain = 10 ** (db_change / 20)
        return np.clip(audio * gain, -1.0, 1.0)
    
    def speed_change(self, audio: np.ndarray, speed_factor: float = 1.0) -> np.ndarray:
        """
        Change speed of audio (affects both time and pitch).
        
        Args:
            audio: Audio waveform
            speed_factor: Speed multiplier (1.0 = no change)
        
        Returns:
            Speed-changed audio
        """
        indices = np.round(np.arange(0, len(audio), speed_factor))
        indices = indices[indices < len(audio)].astype(int)
        return audio[indices]
    
    def apply_reverb(self, audio: np.ndarray, room_size: float = 0.3) -> np.ndarray:
        """
        Apply reverb effect to audio.
        
        Args:
            audio: Audio waveform
            room_size: Room size parameter (0.0 to 1.0)
        
        Returns:
            Audio with reverb
        """
        # Simple reverb using convolution with impulse response
        impulse_length = int(self.sample_rate * room_size)
        impulse = np.exp(-np.linspace(0, 10, impulse_length))
        impulse = impulse / np.max(impulse)
        
        # Convolve with impulse response
        augmented_audio = np.convolve(audio, impulse, mode='same')
        return np.clip(augmented_audio, -1.0, 1.0)
    
    def apply_highpass(self, audio: np.ndarray, cutoff: float = 200.0) -> np.ndarray:
        """
        Apply high-pass filter.
        
        Args:
            audio: Audio waveform
            cutoff: Cutoff frequency in Hz
        
        Returns:
            Filtered audio
        """
        from scipy import signal
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def apply_lowpass(self, audio: np.ndarray, cutoff: float = 3000.0) -> np.ndarray:
        """
        Apply low-pass filter.
        
        Args:
            audio: Audio waveform
            cutoff: Cutoff frequency in Hz
        
        Returns:
            Filtered audio
        """
        from scipy import signal
        nyquist = self.sample_rate / 2
        normal_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, audio)
    
    def augment(self, audio: np.ndarray, techniques: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply random augmentation techniques.
        
        Args:
            audio: Audio waveform
            techniques: List of techniques to use (None = use all)
        
        Returns:
            Augmented audio
        """
        if techniques is None:
            techniques = ['time_stretch', 'pitch_shift', 'add_noise', 'volume_change']
        
        augmented = audio.copy()
        
        # Randomly select and apply techniques
        selected = random.sample(techniques, k=random.randint(1, len(techniques)))
        
        for technique in selected:
            if technique == 'time_stretch':
                rate = random.uniform(0.8, 1.2)
                augmented = self.time_stretch(augmented, rate)
            elif technique == 'pitch_shift':
                n_steps = random.uniform(-2, 2)
                augmented = self.pitch_shift(augmented, n_steps)
            elif technique == 'add_noise':
                noise_factor = random.uniform(0.001, 0.01)
                augmented = self.add_noise(augmented, noise_factor)
            elif technique == 'volume_change':
                augmented = self.volume_change(augmented)
            elif technique == 'time_shift':
                augmented = self.time_shift(augmented)
            elif technique == 'speed_change':
                speed = random.uniform(0.9, 1.1)
                augmented = self.speed_change(augmented, speed)
            elif technique == 'reverb':
                room_size = random.uniform(0.1, 0.5)
                augmented = self.apply_reverb(augmented, room_size)
        
        return augmented
    
    def augment_batch(
        self,
        audio_list: List[np.ndarray],
        labels: List[str],
        n_augmentations: int = 2
    ) -> Tuple[List[np.ndarray], List[str]]:
        """
        Augment a batch of audio files.
        
        Args:
            audio_list: List of audio waveforms
            labels: List of corresponding labels
            n_augmentations: Number of augmented versions per audio
        
        Returns:
            Tuple of (augmented_audio_list, augmented_labels)
        """
        augmented_audio = []
        augmented_labels = []
        
        for audio, label in zip(audio_list, labels):
            # Add original
            augmented_audio.append(audio)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(n_augmentations):
                aug_audio = self.augment(audio)
                augmented_audio.append(aug_audio)
                augmented_labels.append(label)
        
        return augmented_audio, augmented_labels


def create_augmented_dataset(
    source_dir: str,
    output_dir: str,
    n_augmentations: int = 3,
    sample_rate: int = 22050
):
    """
    Create an augmented version of the dataset.
    
    Args:
        source_dir: Source dataset directory
        output_dir: Output directory for augmented dataset
        n_augmentations: Number of augmented versions per file
        sample_rate: Sample rate for audio processing
    """
    augmenter = AudioAugmenter(sample_rate=sample_rate)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'test']:
        split_source = os.path.join(source_dir, split)
        split_output = os.path.join(output_dir, split)
        
        if not os.path.exists(split_source):
            continue
        
        for class_name in os.listdir(split_source):
            class_source = os.path.join(split_source, class_name)
            class_output = os.path.join(split_output, class_name)
            
            if not os.path.isdir(class_source):
                continue
            
            os.makedirs(class_output, exist_ok=True)
            
            for audio_file in os.listdir(class_source):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(class_source, audio_file)
                    
                    try:
                        audio, sr = librosa.load(audio_path, sr=sample_rate)
                        
                        # Save original
                        base_name = os.path.splitext(audio_file)[0]
                        output_path = os.path.join(class_output, audio_file)
                        sf.write(output_path, audio, sample_rate)
                        
                        # Create augmented versions
                        for i in range(n_augmentations):
                            aug_audio = augmenter.augment(audio)
                            aug_filename = f"{base_name}_aug_{i+1}.wav"
                            aug_path = os.path.join(class_output, aug_filename)
                            sf.write(aug_path, aug_audio, sample_rate)
                    
                    except Exception as e:
                        print(f"Error processing {audio_path}: {e}")


if __name__ == '__main__':
    print("Audio Augmentation Module")
    print("=" * 50)
    print("This module provides advanced audio augmentation techniques.")
    print("Use AudioAugmenter class to augment audio samples.")
    print()
    print("Example:")
    print("  augmenter = AudioAugmenter()")
    print("  augmented = augmenter.augment(audio)")

