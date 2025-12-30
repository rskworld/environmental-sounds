"""
Environmental Sound Dataset - Web API Server

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This module provides a REST API for audio classification.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
import io
import base64
from werkzeug.utils import secure_filename
import os
from typing import Optional
import joblib

app = Flask(__name__)
CORS(app)

# Global variables for model
model = None
scaler = None
label_encoder = None


def load_model(model_dir: str = './models', model_name: str = 'environmental_sound_classifier'):
    """Load the trained model."""
    global model, scaler, label_encoder
    
    try:
        from train_model import load_model as load_model_func
        model, scaler, label_encoder = load_model_func(model_dir, model_name)
        print(f"Model loaded successfully from {model_dir}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict audio class from uploaded file.
    
    Accepts:
    - multipart/form-data with 'audio' file
    - JSON with 'audio_base64' (base64 encoded audio)
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        audio_data = None
        
        # Try to get file upload
        if 'audio' in request.files:
            file = request.files['audio']
            if file.filename:
                audio_bytes = file.read()
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # Try to get base64 encoded audio
        elif request.is_json and 'audio_base64' in request.json:
            audio_base64 = request.json['audio_base64']
            audio_bytes = base64.b64decode(audio_base64)
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        else:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Extract features
        from load_data import prepare_features
        features = prepare_features([audio_data], feature_type='mfcc', n_mfcc=13)
        features_scaled = scaler.transform(features)
        
        # Predict
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            probabilities = model.predict(features_scaled)[0]
        
        # Get top predictions
        top_k = request.args.get('top_k', default=3, type=int)
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        
        predictions = []
        for idx in top_indices:
            class_name = label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])
            predictions.append({
                'class': class_name,
                'confidence': confidence,
                'probability': confidence
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple audio files.
    
    Accepts JSON with list of base64 encoded audio files.
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if not request.is_json:
            return jsonify({'error': 'JSON data required'}), 400
        
        audio_list = request.json.get('audio_list', [])
        
        if not audio_list:
            return jsonify({'error': 'No audio files provided'}), 400
        
        results = []
        
        for i, audio_base64 in enumerate(audio_list):
            try:
                audio_bytes = base64.b64decode(audio_base64)
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
                
                # Extract features and predict
                from load_data import prepare_features
                features = prepare_features([audio_data], feature_type='mfcc', n_mfcc=13)
                features_scaled = scaler.transform(features)
                
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(features_scaled)[0]
                else:
                    probabilities = model.predict(features_scaled)[0]
                
                predicted_idx = np.argmax(probabilities)
                predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
                confidence = float(probabilities[predicted_idx])
                
                results.append({
                    'index': i,
                    'success': True,
                    'prediction': predicted_class,
                    'confidence': confidence
                })
            
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results),
            'successful': sum(1 for r in results if r['success'])
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of available classes."""
    try:
        if label_encoder is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        classes = label_encoder.classes_.tolist()
        return jsonify({
            'success': True,
            'classes': classes,
            'count': len(classes)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """
    Analyze audio file and return features.
    
    Accepts audio file upload or base64 encoded audio.
    """
    try:
        audio_data = None
        
        if 'audio' in request.files:
            file = request.files['audio']
            if file.filename:
                audio_bytes = file.read()
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        elif request.is_json and 'audio_base64' in request.json:
            audio_base64 = request.json['audio_base64']
            audio_bytes = base64.b64decode(audio_base64)
            audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        else:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Analyze audio
        from analyze import analyze_audio_file
        # Create temporary file for analysis
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            import soundfile as sf
            sf.write(tmp.name, audio_data, sr)
            analysis = analyze_audio_file(tmp.name)
            os.unlink(tmp.name)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    if load_model():
        print("Model loaded successfully!")
    else:
        print("Warning: Model not loaded. API will return errors.")
    
    # Run server
    print("Starting API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)

