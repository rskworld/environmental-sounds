"""
Create Dataset Structure and Sample Metadata

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import os
import csv
from pathlib import Path


def create_dataset_structure():
    """Create the complete dataset directory structure."""
    
    base_dir = 'environmental-sounds'
    
    # Define classes
    classes = ['bird', 'car', 'dog', 'rain', 'wind']
    
    # Create directories
    for split in ['train', 'test']:
        for class_name in classes:
            dir_path = os.path.join(base_dir, split, class_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created: {dir_path}")
    
    print(f"\nDataset structure created in: {base_dir}/")
    return base_dir, classes


def generate_metadata(base_dir, classes):
    """Generate metadata CSV file."""
    
    metadata = []
    
    for split in ['train', 'test']:
        n_samples = 10 if split == 'train' else 3  # More training samples
        
        for class_name in classes:
            for i in range(n_samples):
                filename = f"{class_name}_{i+1:03d}.wav"
                filepath = os.path.join(base_dir, split, class_name, filename)
                
                # Sample metadata
                duration = 2.0 + (i * 0.1)  # Vary duration
                
                metadata.append({
                    'filename': filename,
                    'filepath': filepath,
                    'class': class_name,
                    'split': split,
                    'duration': f"{duration:.2f}",
                    'sample_rate': '22050',
                    'n_samples': str(int(duration * 22050)),
                    'source': 'synthetic'
                })
    
    # Write metadata CSV
    metadata_path = os.path.join(base_dir, 'metadata.csv')
    
    if metadata:
        fieldnames = ['filename', 'filepath', 'class', 'split', 'duration', 
                     'sample_rate', 'n_samples', 'source']
        
        with open(metadata_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metadata)
        
        print(f"\nMetadata saved to: {metadata_path}")
        print(f"Total entries: {len(metadata)}")
        print(f"  - Training: {len([m for m in metadata if m['split'] == 'train'])}")
        print(f"  - Test: {len([m for m in metadata if m['split'] == 'test'])}")
    
    return metadata_path


def create_readme_file(base_dir):
    """Create a README file in the dataset directory."""
    
    readme_content = """# Environmental Sound Dataset

This directory contains the environmental sound dataset organized by class.

## Structure

```
environmental-sounds/
├── train/
│   ├── bird/
│   ├── car/
│   ├── dog/
│   ├── rain/
│   └── wind/
├── test/
│   ├── bird/
│   ├── car/
│   ├── dog/
│   ├── rain/
│   └── wind/
└── metadata.csv
```

## Usage

Audio files should be placed in their respective class directories.
The metadata.csv file contains information about all audio files.

## Note

This is a sample structure. Replace the directories with actual audio files
for real-world audio classification tasks.

For questions or support:
- Website: https://rskworld.in
- Email: help@rskworld.in
- Phone: +91 93305 39277

---
RSK World - Free Programming Resources & Source Code
Founded by Molla Samser, with Designer & Tester Rima Khatun
"""
    
    readme_path = os.path.join(base_dir, 'README.txt')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"README created: {readme_path}")


if __name__ == '__main__':
    print("Creating dataset structure...")
    print("=" * 50)
    
    try:
        base_dir, classes = create_dataset_structure()
        metadata_path = generate_metadata(base_dir, classes)
        create_readme_file(base_dir)
        
        print("\n" + "=" * 50)
        print("Dataset structure created successfully!")
        print("\nNote: This creates the directory structure and metadata.")
        print("For actual audio files, place WAV/MP3 files in the class directories.")
        print("\nNext steps:")
        print("1. Add your audio files to the respective class directories")
        print("2. Update metadata.csv if needed")
        print("3. Run: python create_zip.py to create the project ZIP file")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

