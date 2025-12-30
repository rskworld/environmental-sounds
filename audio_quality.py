"""
Environmental Sound Dataset - Audio Quality Assessment Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides audio quality assessment and validation tools.
"""

import numpy as np
import librosa
from typing import Dict, Tuple, List
import os


class AudioQualityAssessor:
    """
    Assess audio quality and detect issues.
    """
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize audio quality assessor.
        
        Args:
            sample_rate: Expected sample rate
        """
        self.sample_rate = sample_rate
    
    def assess_audio(self, audio: np.ndarray, sr: int = None) -> Dict:
        """
        Comprehensive audio quality assessment.
        
        Args:
            audio: Audio waveform
            sr: Sample rate (uses self.sample_rate if None)
        
        Returns:
            Dictionary with quality metrics
        """
        if sr is None:
            sr = self.sample_rate
        
        assessment = {
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'n_samples': len(audio),
            'max_amplitude': float(np.max(np.abs(audio))),
            'rms_energy': float(np.sqrt(np.mean(audio**2))),
            'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(audio))),
            'issues': []
        }
        
        # Check for clipping
        if assessment['max_amplitude'] >= 0.99:
            assessment['issues'].append('clipping_detected')
        
        # Check for silence
        if assessment['rms_energy'] < 0.01:
            assessment['issues'].append('likely_silence')
        
        # Check for very short audio
        if assessment['duration'] < 0.1:
            assessment['issues'].append('too_short')
        
        # Check for DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.1:
            assessment['issues'].append('dc_offset')
            assessment['dc_offset'] = float(dc_offset)
        
        # Calculate SNR estimate (simple)
        signal_power = np.mean(audio**2)
        noise_estimate = np.var(audio - np.mean(audio))
        if noise_estimate > 0:
            snr = 10 * np.log10(signal_power / noise_estimate)
            assessment['estimated_snr_db'] = float(snr)
        else:
            assessment['estimated_snr_db'] = float('inf')
        
        # Quality score (0-100)
        quality_score = 100
        if 'clipping_detected' in assessment['issues']:
            quality_score -= 20
        if 'likely_silence' in assessment['issues']:
            quality_score -= 30
        if 'too_short' in assessment['issues']:
            quality_score -= 15
        if 'dc_offset' in assessment['issues']:
            quality_score -= 10
        
        assessment['quality_score'] = max(0, quality_score)
        
        return assessment
    
    def validate_audio_file(self, filepath: str) -> Tuple[bool, Dict]:
        """
        Validate an audio file.
        
        Args:
            filepath: Path to audio file
        
        Returns:
            Tuple of (is_valid, assessment_dict)
        """
        try:
            audio, sr = librosa.load(filepath, sr=self.sample_rate)
            assessment = self.assess_audio(audio, sr)
            
            # Consider valid if no critical issues
            critical_issues = ['too_short', 'likely_silence']
            is_valid = not any(issue in assessment['issues'] for issue in critical_issues)
            
            return is_valid, assessment
        
        except Exception as e:
            return False, {'error': str(e), 'issues': ['load_error']}
    
    def batch_validate(self, file_paths: List[str]) -> Dict:
        """
        Validate multiple audio files.
        
        Args:
            file_paths: List of audio file paths
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'total': len(file_paths),
            'valid': 0,
            'invalid': 0,
            'errors': 0,
            'details': []
        }
        
        for filepath in file_paths:
            is_valid, assessment = self.validate_audio_file(filepath)
            
            result = {
                'file': filepath,
                'valid': is_valid,
                'assessment': assessment
            }
            
            results['details'].append(result)
            
            if 'error' in assessment:
                results['errors'] += 1
            elif is_valid:
                results['valid'] += 1
            else:
                results['invalid'] += 1
        
        return results
    
    def normalize_audio(self, audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target level in dB.
        
        Args:
            audio: Audio waveform
            target_level: Target level in dB
        
        Returns:
            Normalized audio
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # Calculate target RMS
            target_rms = 10 ** (target_level / 20)
            
            # Calculate gain
            gain = target_rms / rms
            
            # Apply gain
            normalized = audio * gain
            
            # Prevent clipping
            if np.max(np.abs(normalized)) > 1.0:
                normalized = normalized / np.max(np.abs(normalized))
            
            return normalized
        
        return audio
    
    def remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from audio.
        
        Args:
            audio: Audio waveform
        
        Returns:
            Audio with DC offset removed
        """
        return audio - np.mean(audio)
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality (normalize, remove DC offset).
        
        Args:
            audio: Audio waveform
        
        Returns:
            Enhanced audio
        """
        enhanced = self.remove_dc_offset(audio)
        enhanced = self.normalize_audio(enhanced)
        return enhanced


class AudioDatasetValidator:
    """
    Validate entire audio dataset.
    """
    
    def __init__(self, data_dir: str = './environmental-sounds'):
        """
        Initialize dataset validator.
        
        Args:
            data_dir: Root directory of dataset
        """
        self.data_dir = data_dir
        self.assessor = AudioQualityAssessor()
    
    def validate_dataset(self) -> Dict:
        """
        Validate entire dataset.
        
        Returns:
            Dictionary with validation results
        """
        results = {
            'train': {'valid': 0, 'invalid': 0, 'total': 0, 'issues': {}},
            'test': {'valid': 0, 'invalid': 0, 'total': 0, 'issues': {}}
        }
        
        for split in ['train', 'test']:
            split_dir = os.path.join(self.data_dir, split)
            
            if not os.path.exists(split_dir):
                continue
            
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                
                if not os.path.isdir(class_dir):
                    continue
                
                for audio_file in os.listdir(class_dir):
                    if audio_file.endswith(('.wav', '.mp3', '.flac')):
                        filepath = os.path.join(class_dir, audio_file)
                        results[split]['total'] += 1
                        
                        is_valid, assessment = self.assessor.validate_audio_file(filepath)
                        
                        if is_valid:
                            results[split]['valid'] += 1
                        else:
                            results[split]['invalid'] += 1
                            
                            # Track issues
                            for issue in assessment.get('issues', []):
                                if issue not in results[split]['issues']:
                                    results[split]['issues'][issue] = 0
                                results[split]['issues'][issue] += 1
        
        return results
    
    def generate_report(self, save_path: str = None) -> str:
        """
        Generate validation report.
        
        Args:
            save_path: Optional path to save report
        
        Returns:
            Report as string
        """
        results = self.validate_dataset()
        
        report = "Audio Dataset Validation Report\n"
        report += "=" * 50 + "\n\n"
        
        for split in ['train', 'test']:
            report += f"{split.upper()} SET:\n"
            report += f"  Total files: {results[split]['total']}\n"
            report += f"  Valid: {results[split]['valid']}\n"
            report += f"  Invalid: {results[split]['invalid']}\n"
            
            if results[split]['invalid'] > 0:
                report += f"  Issues found:\n"
                for issue, count in results[split]['issues'].items():
                    report += f"    - {issue}: {count}\n"
            
            report += "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report


if __name__ == '__main__':
    print("Audio Quality Assessment Module")
    print("=" * 50)
    print("Features:")
    print("  - AudioQualityAssessor: Assess audio quality")
    print("  - AudioDatasetValidator: Validate entire dataset")
    print()
    print("Example usage:")
    print("  assessor = AudioQualityAssessor()")
    print("  is_valid, assessment = assessor.validate_audio_file('audio.wav')")

