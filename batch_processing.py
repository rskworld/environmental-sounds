"""
Environmental Sound Dataset - Batch Processing Utilities

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides batch processing utilities for large-scale audio operations.
"""

import numpy as np
import librosa
import os
from pathlib import Path
from typing import List, Dict, Callable, Optional
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
import json


class BatchProcessor:
    """
    Batch processing utilities for audio files.
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Initialize batch processor.
        
        Args:
            n_workers: Number of worker processes (None = auto)
        """
        self.n_workers = n_workers or cpu_count()
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        process_func: Callable,
        file_extensions: List[str] = ['.wav', '.mp3', '.flac'],
        recursive: bool = True
    ) -> Dict:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            process_func: Function to process each file (filepath) -> result
            file_extensions: List of file extensions to process
            recursive: Process subdirectories recursively
        
        Returns:
            Dictionary with processing results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Collect all files
        files = []
        if recursive:
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    if any(filename.endswith(ext) for ext in file_extensions):
                        files.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(input_dir):
                if any(filename.endswith(ext) for ext in file_extensions):
                    files.append(os.path.join(input_dir, filename))
        
        # Process files
        results = {
            'total': len(files),
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        for filepath in tqdm(files, desc="Processing files"):
            try:
                result = process_func(filepath)
                results['successful'] += 1
            except Exception as e:
                results['failed'] += 1
                results['errors'].append({
                    'file': filepath,
                    'error': str(e)
                })
        
        return results
    
    def extract_features_batch(
        self,
        file_paths: List[str],
        feature_type: str = 'mfcc',
        n_mfcc: int = 13,
        sample_rate: int = 22050
    ) -> np.ndarray:
        """
        Extract features from multiple audio files in parallel.
        
        Args:
            file_paths: List of audio file paths
            feature_type: Type of features to extract
            n_mfcc: Number of MFCC coefficients
            sample_rate: Sample rate
        
        Returns:
            Feature matrix
        """
        def extract_single(filepath):
            try:
                audio, sr = librosa.load(filepath, sr=sample_rate)
                from load_data import prepare_features
                features = prepare_features([audio], feature_type=feature_type, n_mfcc=n_mfcc)
                return features[0]
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                return None
        
        # Process in parallel
        with Pool(self.n_workers) as pool:
            features_list = list(tqdm(
                pool.imap(extract_single, file_paths),
                total=len(file_paths),
                desc="Extracting features"
            ))
        
        # Filter out None values
        features_list = [f for f in features_list if f is not None]
        
        return np.array(features_list)
    
    def classify_batch(
        self,
        file_paths: List[str],
        model,
        scaler,
        label_encoder,
        feature_type: str = 'mfcc'
    ) -> List[Dict]:
        """
        Classify multiple audio files.
        
        Args:
            file_paths: List of audio file paths
            model: Trained classification model
            scaler: Fitted feature scaler
            label_encoder: Fitted label encoder
            feature_type: Type of features to extract
        
        Returns:
            List of prediction dictionaries
        """
        def classify_single(filepath):
            try:
                audio, sr = librosa.load(filepath, sr=22050)
                from load_data import prepare_features
                features = prepare_features([audio], feature_type=feature_type, n_mfcc=13)
                features_scaled = scaler.transform(features)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                else:
                    probabilities = model.predict(features_scaled)[0]
                
                predicted_idx = np.argmax(probabilities)
                predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
                confidence = float(probabilities[predicted_idx])
                
                return {
                    'file': filepath,
                    'prediction': predicted_class,
                    'confidence': confidence,
                    'all_probabilities': probabilities.tolist()
                }
            except Exception as e:
                return {
                    'file': filepath,
                    'error': str(e)
                }
        
        # Process in parallel
        with Pool(self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(classify_single, file_paths),
                total=len(file_paths),
                desc="Classifying files"
            ))
        
        return results
    
    def convert_format_batch(
        self,
        input_dir: str,
        output_dir: str,
        output_format: str = 'wav',
        sample_rate: int = 22050,
        recursive: bool = True
    ) -> Dict:
        """
        Convert audio files to different format.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            output_format: Output format ('wav', 'mp3', 'flac')
            sample_rate: Target sample rate
            recursive: Process subdirectories recursively
        
        Returns:
            Dictionary with conversion results
        """
        import soundfile as sf
        
        def convert_file(input_path):
            try:
                # Load audio
                audio, sr = librosa.load(input_path, sr=sample_rate)
                
                # Create output path
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                output_path = os.path.splitext(output_path)[0] + f'.{output_format}'
                
                # Create output directory
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save in new format
                sf.write(output_path, audio, sample_rate)
                
                return {'input': input_path, 'output': output_path, 'success': True}
            except Exception as e:
                return {'input': input_path, 'success': False, 'error': str(e)}
        
        # Collect files
        files = []
        if recursive:
            for root, dirs, filenames in os.walk(input_dir):
                for filename in filenames:
                    if filename.endswith(('.wav', '.mp3', '.flac')):
                        files.append(os.path.join(root, filename))
        else:
            for filename in os.listdir(input_dir):
                if filename.endswith(('.wav', '.mp3', '.flac')):
                    files.append(os.path.join(input_dir, filename))
        
        # Process in parallel
        with Pool(self.n_workers) as pool:
            results = list(tqdm(
                pool.imap(convert_file, files),
                total=len(files),
                desc="Converting files"
            ))
        
        return {
            'total': len(results),
            'successful': sum(1 for r in results if r.get('success', False)),
            'failed': sum(1 for r in results if not r.get('success', False)),
            'results': results
        }
    
    def generate_metadata(
        self,
        data_dir: str,
        output_path: str = 'metadata.csv'
    ) -> pd.DataFrame:
        """
        Generate metadata CSV for dataset.
        
        Args:
            data_dir: Root directory of dataset
            output_path: Path to save metadata CSV
        
        Returns:
            DataFrame with metadata
        """
        metadata = []
        
        for split in ['train', 'test']:
            split_dir = os.path.join(data_dir, split)
            
            if not os.path.exists(split_dir):
                continue
            
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                
                if not os.path.isdir(class_dir):
                    continue
                
                for audio_file in os.listdir(class_dir):
                    if audio_file.endswith(('.wav', '.mp3', '.flac')):
                        filepath = os.path.join(class_dir, audio_file)
                        
                        try:
                            audio, sr = librosa.load(filepath, sr=22050)
                            duration = len(audio) / sr
                            
                            metadata.append({
                                'filename': audio_file,
                                'filepath': filepath,
                                'class': class_name,
                                'split': split,
                                'duration': duration,
                                'sample_rate': sr,
                                'n_samples': len(audio)
                            })
                        except Exception as e:
                            print(f"Error processing {filepath}: {e}")
        
        df = pd.DataFrame(metadata)
        df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")
        
        return df


if __name__ == '__main__':
    print("Batch Processing Utilities")
    print("=" * 50)
    print("Features:")
    print("  - Process directories in parallel")
    print("  - Extract features from multiple files")
    print("  - Classify multiple files")
    print("  - Convert audio formats")
    print("  - Generate metadata")
    print()
    print("Example usage:")
    print("  processor = BatchProcessor()")
    print("  features = processor.extract_features_batch(file_paths)")

