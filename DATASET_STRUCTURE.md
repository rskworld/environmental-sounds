# Environmental Sound Dataset - Structure Guide

<!--
Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
-->

## Directory Structure

The Environmental Sound Dataset should be organized as follows:

```
environmental-sounds/
├── train/                          # Training set
│   ├── bird/                        # Class: Bird sounds
│   │   ├── bird_001.wav
│   │   ├── bird_002.wav
│   │   └── ...
│   ├── car/                         # Class: Car sounds
│   │   ├── car_001.wav
│   │   ├── car_002.wav
│   │   └── ...
│   ├── dog/                         # Class: Dog sounds
│   │   └── ...
│   ├── rain/                        # Class: Rain sounds
│   │   └── ...
│   ├── wind/                        # Class: Wind sounds
│   │   └── ...
│   └── ...                          # Additional classes
│
├── test/                            # Test set
│   ├── bird/
│   │   └── ...
│   ├── car/
│   │   └── ...
│   └── ...                          # Same class structure as train
│
├── metadata.csv                     # Optional: Dataset metadata
│
├── README.md                        # This file
│
└── index.html                       # Demo page
```

## File Formats

The dataset supports the following audio formats:
- **WAV** (recommended) - Uncompressed, high quality
- **MP3** - Compressed, smaller file size
- **FLAC** - Lossless compression

## Metadata CSV Format

If you include a `metadata.csv` file, it should have the following structure:

```csv
filename,class,split,duration,source
bird_001.wav,bird,train,3.5,recorded
car_001.wav,car,train,2.1,recorded
...
```

Columns:
- `filename`: Name of the audio file
- `class`: Class label (e.g., bird, car, dog)
- `split`: Dataset split (train or test)
- `duration`: Duration in seconds (optional)
- `source`: Source of the audio (optional)

## Class Labels

Common environmental sound classes include:
- Natural sounds: bird, wind, rain, water, thunder, etc.
- Urban sounds: car, traffic, construction, siren, etc.
- Animal sounds: dog, cat, bird, etc.
- Human sounds: speech, footsteps, door, etc.

## Audio Specifications

Recommended audio specifications:
- **Sample Rate**: 22050 Hz (or 44100 Hz)
- **Bit Depth**: 16-bit
- **Channels**: Mono (1 channel) or Stereo (2 channels)
- **Duration**: Variable (typically 1-10 seconds)

## Usage

1. Download and extract the dataset
2. Ensure the directory structure matches the above
3. Use the provided Python scripts to load and process the data:

```python
from load_data import load_environmental_sounds

train_data, train_labels = load_environmental_sounds('train')
test_data, test_labels = load_environmental_sounds('test')
```

## Notes

- All audio files should be properly labeled and organized by class
- Training and test sets should have similar class distributions
- Ensure audio files are not corrupted and can be loaded by librosa
- Consider data augmentation for imbalanced classes

## Contact

For questions or support:
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

---

**RSK World** - Free Programming Resources & Source Code
Founded by Molla Samser, with Designer & Tester Rima Khatun

