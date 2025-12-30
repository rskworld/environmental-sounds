"""
Environmental Sound Dataset - Model Interpretability Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides tools for understanding model predictions and feature importance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from sklearn.inspection import permutation_importance
import librosa
import librosa.display


class ModelInterpreter:
    """
    Interpret and explain model predictions.
    """
    
    def __init__(self, model, scaler, label_encoder, feature_names: Optional[List[str]] = None):
        """
        Initialize model interpreter.
        
        Args:
            model: Trained classification model
            scaler: Fitted feature scaler
            label_encoder: Fitted label encoder
            feature_names: Optional list of feature names
        """
        self.model = model
        self.scaler = scaler
        self.label_encoder = label_encoder
        self.feature_names = feature_names or [f'MFCC_{i+1}' for i in range(13)]
    
    def get_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict:
        """
        Calculate feature importance using permutation importance.
        
        Args:
            X: Feature matrix
            y: True labels
            n_repeats: Number of permutation repeats
            random_state: Random seed
        
        Returns:
            Dictionary with feature importance scores
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            self.model,
            X_scaled,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )
        
        importance_scores = perm_importance.importances_mean
        importance_std = perm_importance.importances_std
        
        # Create importance dictionary
        importance_dict = {
            'features': self.feature_names,
            'scores': importance_scores.tolist(),
            'std': importance_std.tolist()
        }
        
        return importance_dict
    
    def plot_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        top_n: int = 10,
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.
        
        Args:
            X: Feature matrix
            y: True labels
            top_n: Number of top features to show
            save_path: Optional path to save plot
        """
        importance_dict = self.get_feature_importance(X, y)
        
        # Sort by importance
        sorted_indices = np.argsort(importance_dict['scores'])[::-1][:top_n]
        
        features = [importance_dict['features'][i] for i in sorted_indices]
        scores = [importance_dict['scores'][i] for i in sorted_indices]
        stds = [importance_dict['std'][i] for i in sorted_indices]
        
        # Plot
        plt.figure(figsize=(10, 6))
        y_pos = np.arange(len(features))
        plt.barh(y_pos, scores, xerr=stds, align='center')
        plt.yticks(y_pos, features)
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def explain_prediction(
        self,
        audio: np.ndarray,
        top_k: int = 3
    ) -> Dict:
        """
        Explain a single prediction.
        
        Args:
            audio: Audio waveform
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction explanation
        """
        from load_data import prepare_features
        
        # Extract features
        features = prepare_features([audio], feature_type='mfcc', n_mfcc=13)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
        else:
            probabilities = self.model.predict(features_scaled)[0]
        
        # Get top k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        explanation = {
            'predictions': [],
            'feature_values': features[0].tolist(),
            'feature_names': self.feature_names
        }
        
        for idx in top_indices:
            class_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            explanation['predictions'].append({
                'class': class_name,
                'confidence': confidence,
                'probability': confidence
            })
        
        return explanation
    
    def plot_prediction_confidence(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Plot prediction confidence distribution.
        
        Args:
            X: Feature matrix
            y: True labels
            save_path: Optional path to save plot
        """
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = self.model.predict(X_scaled)
        
        # Get max probability (confidence) for each prediction
        confidences = np.max(probabilities, axis=1)
        
        # Plot histogram
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.axvline(np.mean(confidences), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(confidences):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    
    def analyze_misclassifications(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        Analyze misclassified samples.
        
        Args:
            X: Feature matrix
            y_true: True labels
            y_pred: Predicted labels
        
        Returns:
            Dictionary with misclassification analysis
        """
        misclassified = y_true != y_pred
        
        if not np.any(misclassified):
            return {'message': 'No misclassifications found'}
        
        X_mis = X[misclassified]
        y_true_mis = y_true[misclassified]
        y_pred_mis = y_pred[misclassified]
        
        # Calculate statistics
        misclassification_rate = np.sum(misclassified) / len(y_true)
        
        # Confusion pairs
        confusion_pairs = {}
        for true_label, pred_label in zip(y_true_mis, y_pred_mis):
            pair = (true_label, pred_label)
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        analysis = {
            'misclassification_rate': float(misclassification_rate),
            'n_misclassified': int(np.sum(misclassified)),
            'n_total': len(y_true),
            'confusion_pairs': {str(k): v for k, v in confusion_pairs.items()},
            'most_confused': max(confusion_pairs.items(), key=lambda x: x[1])[0] if confusion_pairs else None
        }
        
        return analysis


class AudioVisualizer:
    """
    Visualize audio features and model attention.
    """
    
    @staticmethod
    def plot_audio_features(
        audio: np.ndarray,
        sr: int = 22050,
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive audio features.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            save_path: Optional path to save plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Waveform
        time = np.linspace(0, len(audio) / sr, len(audio))
        axes[0].plot(time, audio)
        axes[0].set_title('Waveform')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[1])
        axes[1].set_title('Spectrogram')
        axes[1].set_ylabel('Frequency (Hz)')
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=axes[2])
        axes[2].set_title('MFCC')
        axes[2].set_ylabel('MFCC Coefficients')
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        librosa.display.specshow(chroma, x_axis='time', sr=sr, ax=axes[3])
        axes[3].set_title('Chroma')
        axes[3].set_ylabel('Chroma')
        axes[3].set_xlabel('Time (s)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    print("Model Interpretability Module")
    print("=" * 50)
    print("Features:")
    print("  - ModelInterpreter: Explain model predictions")
    print("  - AudioVisualizer: Visualize audio features")
    print()
    print("Example usage:")
    print("  interpreter = ModelInterpreter(model, scaler, label_encoder)")
    print("  interpreter.plot_feature_importance(X, y)")

