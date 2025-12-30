# Environmental Sound Dataset

<!--
Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
-->

Environmental sound classification dataset with audio samples of natural and urban sounds for sound event detection and audio scene analysis.

## Description

This dataset contains audio recordings of environmental sounds including nature sounds, urban sounds, and everyday audio events with class labels. Perfect for sound event detection, audio scene classification, and environmental monitoring applications.

## Features

- Environmental sounds
- Multiple sound classes
- Various durations
- Training and test sets
- Ready for audio classification

## Advanced Features

### 🎵 Audio Augmentation
- Time stretching and pitch shifting
- Noise injection and volume adjustment
- Reverb and filtering effects
- Batch augmentation support

### 🧠 Deep Learning Models
- **CNN**: Convolutional Neural Network for spectrogram classification
- **LSTM**: Long Short-Term Memory for sequence modeling
- **Transformer**: Attention-based model for audio classification

### 🔍 Audio Similarity Search
- Find similar audio files using feature embeddings
- Audio clustering (K-means, DBSCAN)
- Duplicate detection
- Fast similarity search engine

### ⚡ Real-time Classification
- Live audio classification from microphone
- Streaming audio file classification
- Sliding window analysis
- Dominant class detection

### 📊 Model Interpretability
- Feature importance analysis
- Prediction explanation
- Misclassification analysis
- Audio feature visualization

### ✅ Audio Quality Assessment
- Automatic quality scoring
- Issue detection (clipping, silence, DC offset)
- Audio normalization and enhancement
- Dataset validation

### 🌐 Web API
- RESTful API for predictions
- Batch prediction support
- Audio analysis endpoints
- Easy integration

### ⚙️ Batch Processing
- Parallel audio processing
- Feature extraction at scale
- Format conversion
- Metadata generation

## Technologies

- WAV
- MP3
- Librosa
- NumPy
- Audio Processing

## Dataset Structure

```
environmental-sounds/
├── train/
│   ├── class1/
│   │   ├── sample1.wav
│   │   ├── sample2.wav
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ...
├── test/
│   ├── class1/
│   ├── class2/
│   └── ...
├── metadata.csv
└── README.md
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Download the dataset from the source link.

## Usage

### Loading the Dataset

```python
from load_data import load_environmental_sounds

# Load training data
train_data, train_labels = load_environmental_sounds('train')

# Load test data
test_data, test_labels = load_environmental_sounds('test')
```

### Analyzing Audio Files

```python
from analyze import analyze_audio_file

# Analyze a single audio file
features = analyze_audio_file('path/to/audio.wav')
print(features)
```

### Training a Model

```python
from train_model import train_classifier

# Train a classifier on the dataset
model = train_classifier(train_data, train_labels)
```

## Advanced Usage Examples

### Audio Augmentation
```python
from augment_audio import AudioAugmenter

augmenter = AudioAugmenter()
augmented_audio = augmenter.augment(audio)
```

### Deep Learning Models
```python
from deep_learning_models import AudioCNN, prepare_features_for_dl

# Prepare features
X_train = prepare_features_for_dl(train_data, feature_type='mel')
X_train = X_train[..., np.newaxis]  # Add channel dimension

# Train CNN
model = AudioCNN(input_shape=(X_train.shape[1], X_train.shape[2], 1), num_classes=10)
model.train(X_train, y_train, X_val, y_val, epochs=50)
```

### Audio Similarity Search
```python
from audio_similarity import AudioSimilaritySearch

search = AudioSimilaritySearch()
search.add_audio(audio1, 'path1.wav')
search.build_index()
results = search.search(query_audio, top_k=5)
```

### Real-time Classification
```python
from realtime_classification import RealTimeClassifier
from train_model import load_model

model, scaler, label_encoder = load_model()
classifier = RealTimeClassifier(model, scaler, label_encoder)
classifier.start(callback=lambda pred, conf: print(f"{pred}: {conf:.2f}"))
```

### Model Interpretability
```python
from model_interpretability import ModelInterpreter

interpreter = ModelInterpreter(model, scaler, label_encoder)
interpreter.plot_feature_importance(X, y)
explanation = interpreter.explain_prediction(audio)
```

### Audio Quality Assessment
```python
from audio_quality import AudioQualityAssessor

assessor = AudioQualityAssessor()
is_valid, assessment = assessor.validate_audio_file('audio.wav')
print(f"Quality score: {assessment['quality_score']}")
```

### Batch Processing
```python
from batch_processing import BatchProcessor

processor = BatchProcessor()
features = processor.extract_features_batch(file_paths)
results = processor.classify_batch(file_paths, model, scaler, label_encoder)
```

### Web API
```bash
# Start API server
python api_server.py

# Make prediction
curl -X POST http://localhost:5000/predict \
  -F "audio=@test.wav"
```

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating:
- Data exploration
- Feature extraction
- Model training
- Evaluation
- Advanced features usage

## License

This dataset is provided by RSK World for educational and research purposes.

## Contact

For questions or support, please contact:
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

---

**RSK World** - Free Programming Resources & Source Code
Founded by Molla Samser, with Designer & Tester Rima Khatun

