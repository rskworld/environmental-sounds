"""
Create Sample Dataset Structure and Metadata

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path


def create_sample_audio(filename: str, duration: float = 2.0, sample_rate: int = 22050, 
                        frequency: float = 440.0, noise_level: float = 0.1):
    """
    Create a sample audio file with a simple tone.
    
    Args:
        filename: Output filename
        duration: Duration in seconds
        sample_rate: Sample rate
        frequency: Tone frequency in Hz
        noise_level: Noise level (0-1)
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate tone
    audio = np.sin(2 * np.pi * frequency * t)
    
    # Add some harmonics for more realistic sound
    audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    audio += 0.1 * np.sin(2 * np.pi * frequency * 3 * t)
    
    # Add noise
    noise = np.random.randn(len(audio)) * noise_level
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save
    sf.write(filename, audio, sample_rate)
    return audio


def generate_dataset_structure():
    """Generate the complete dataset structure with sample files."""
    
    base_dir = 'environmental-sounds'
    
    # Define classes and their characteristics
    classes = {
        'bird': {'freq': 2000, 'noise': 0.05},
        'car': {'freq': 200, 'noise': 0.2},
        'dog': {'freq': 500, 'noise': 0.1},
        'rain': {'freq': 100, 'noise': 0.3},
        'wind': {'freq': 50, 'noise': 0.4}
    }
    
    # Create directories
    for split in ['train', 'test']:
        for class_name in classes.keys():
            dir_path = os.path.join(base_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
    
    print("Created dataset directory structure")
    
    # Generate sample audio files
    metadata = []
    
    for split in ['train', 'test']:
        n_samples = 10 if split == 'train' else 3  # More training samples
        
        for class_name, params in classes.items():
            for i in range(n_samples):
                filename = f"{class_name}_{i+1:03d}.wav"
                filepath = os.path.join(base_dir, split, class_name, filename)
                
                # Vary duration slightly
                duration = np.random.uniform(1.5, 3.0)
                
                # Create sample audio
                audio = create_sample_audio(
                    filepath,
                    duration=duration,
                    frequency=params['freq'] * np.random.uniform(0.8, 1.2),
                    noise_level=params['noise']
                )
                
                # Add to metadata
                metadata.append({
                    'filename': filename,
                    'filepath': filepath,
                    'class': class_name,
                    'split': split,
                    'duration': duration,
                    'sample_rate': 22050,
                    'n_samples': len(audio)
                })
                
                print(f"Created: {filepath}")
    
    # Save metadata
    import pandas as pd
    df = pd.DataFrame(metadata)
    metadata_path = os.path.join(base_dir, 'metadata.csv')
    df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"Total files created: {len(metadata)}")
    print(f"  - Training: {len(df[df['split'] == 'train'])}")
    print(f"  - Test: {len(df[df['split'] == 'test'])}")
    
    return metadata


if __name__ == '__main__':
    print("Generating sample dataset...")
    print("=" * 50)
    
    try:
        metadata = generate_dataset_structure()
        print("\n" + "=" * 50)
        print("Sample dataset generation completed!")
        print("\nNote: These are simple synthetic audio files for demonstration.")
        print("For real audio classification, replace with actual environmental sound recordings.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

