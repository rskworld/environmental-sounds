"""
Environmental Sound Dataset - Model Training Module

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from typing import Tuple, Optional
import os

from load_data import load_environmental_sounds, prepare_features


def train_classifier(
    train_data: list,
    train_labels: list,
    model_type: str = 'random_forest',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple:
    """
    Train a classifier on the environmental sound dataset.
    
    Args:
        train_data: List of audio waveforms
        train_labels: List of class labels
        model_type: Type of classifier ('random_forest', 'svm', 'mlp')
        test_size: Proportion of data to use for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (trained_model, scaler, label_encoder, accuracy)
    """
    # Extract features
    print("Extracting features...")
    X = prepare_features(train_data, feature_type='mfcc', n_mfcc=13)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_labels)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    print(f"Training {model_type} classifier...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', random_state=random_state, probability=True)
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    
    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
    
    return model, scaler, label_encoder, accuracy


def evaluate_model(
    model,
    scaler,
    label_encoder,
    test_data: list,
    test_labels: list
) -> dict:
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        test_data: List of test audio waveforms
        test_labels: List of test class labels
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Extract features
    X_test = prepare_features(test_data, feature_type='mfcc', n_mfcc=13)
    y_test = label_encoder.transform(test_labels)
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    results = {
        'accuracy': accuracy,
        'predictions': label_encoder.inverse_transform(y_pred),
        'probabilities': y_pred_proba,
        'classification_report': classification_report(
            y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return results


def save_model(
    model,
    scaler,
    label_encoder,
    save_dir: str = './models',
    model_name: str = 'environmental_sound_classifier'
):
    """
    Save the trained model and preprocessors.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        save_dir: Directory to save the model
        model_name: Name for the saved model files
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, f'{model_name}.pkl')
    scaler_path = os.path.join(save_dir, f'{model_name}_scaler.pkl')
    encoder_path = os.path.join(save_dir, f'{model_name}_encoder.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Label encoder saved to {encoder_path}")


def load_model(
    model_dir: str = './models',
    model_name: str = 'environmental_sound_classifier'
) -> Tuple:
    """
    Load a saved model and preprocessors.
    
    Args:
        model_dir: Directory containing the model files
        model_name: Name of the saved model
    
    Returns:
        Tuple of (model, scaler, label_encoder)
    """
    model_path = os.path.join(model_dir, f'{model_name}.pkl')
    scaler_path = os.path.join(model_dir, f'{model_name}_scaler.pkl')
    encoder_path = os.path.join(model_dir, f'{model_name}_encoder.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    
    return model, scaler, label_encoder


def predict_audio(
    model,
    scaler,
    label_encoder,
    audio_data: np.ndarray
) -> Tuple[str, float]:
    """
    Predict the class of a single audio sample.
    
    Args:
        model: Trained classifier
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        audio_data: Audio waveform as numpy array
    
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Extract features
    features = prepare_features([audio_data], feature_type='mfcc', n_mfcc=13)
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    predicted_class = label_encoder.inverse_transform([prediction])[0]
    confidence = float(np.max(probability))
    
    return predicted_class, confidence


if __name__ == '__main__':
    # Example usage
    print("Training Environmental Sound Classifier")
    print("=" * 50)
    
    try:
        # Load data
        print("Loading training data...")
        train_data, train_labels = load_environmental_sounds('train')
        print(f"Loaded {len(train_data)} training samples")
        
        print("Loading test data...")
        test_data, test_labels = load_environmental_sounds('test')
        print(f"Loaded {len(test_data)} test samples")
        
        # Train model
        model, scaler, label_encoder, accuracy = train_classifier(
            train_data, train_labels, model_type='random_forest'
        )
        
        # Evaluate on test set
        print("\nEvaluating on test set...")
        results = evaluate_model(model, scaler, label_encoder, test_data, test_labels)
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        
        # Save model
        save_model(model, scaler, label_encoder)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is properly structured in ./environmental-sounds/")

