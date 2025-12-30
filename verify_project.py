"""
Verify Complete Project Structure

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import os
from pathlib import Path


def verify_project():
    """Verify all project files and structure."""
    
    print("Environmental Sound Dataset - Project Verification")
    print("=" * 60)
    print()
    
    # Core Python modules
    core_modules = [
        'load_data.py',
        'analyze.py',
        'train_model.py',
        'augment_audio.py',
        'deep_learning_models.py',
        'audio_similarity.py',
        'realtime_classification.py',
        'model_interpretability.py',
        'audio_quality.py',
        'api_server.py',
        'batch_processing.py'
    ]
    
    # Documentation files
    docs = [
        'README.md',
        'ADVANCED_FEATURES.md',
        'DATASET_STRUCTURE.md',
        'CONTRIBUTING.md',
        'LICENSE',
        'PROJECT_SUMMARY.md'
    ]
    
    # Configuration files
    config = [
        'requirements.txt',
        'setup.py',
        '.gitignore'
    ]
    
    # Utility scripts
    utilities = [
        'example_usage.py',
        'create_dataset_structure.py',
        'create_sample_data.py',
        'create_zip.py',
        'verify_project.py'
    ]
    
    # Web files
    web = [
        'index.html'
    ]
    
    # Examples
    examples = [
        'examples/data_exploration.ipynb'
    ]
    
    # Dataset structure
    dataset_dirs = [
        'environmental-sounds/train/bird',
        'environmental-sounds/train/car',
        'environmental-sounds/train/dog',
        'environmental-sounds/train/rain',
        'environmental-sounds/train/wind',
        'environmental-sounds/test/bird',
        'environmental-sounds/test/car',
        'environmental-sounds/test/dog',
        'environmental-sounds/test/rain',
        'environmental-sounds/test/wind'
    ]
    
    dataset_files = [
        'environmental-sounds/metadata.csv',
        'environmental-sounds/README.txt'
    ]
    
    all_files = {
        'Core Modules': core_modules,
        'Documentation': docs,
        'Configuration': config,
        'Utility Scripts': utilities,
        'Web Files': web,
        'Examples': examples
    }
    
    # Check files
    print("Checking Files:")
    print("-" * 60)
    
    total_files = 0
    missing_files = []
    
    for category, files in all_files.items():
        print(f"\n{category}:")
        for file in files:
            if os.path.exists(file):
                size = os.path.getsize(file)
                print(f"  [OK] {file} ({size:,} bytes)")
                total_files += 1
            else:
                print(f"  [MISSING] {file}")
                missing_files.append(file)
    
    # Check dataset structure
    print("\n" + "-" * 60)
    print("Dataset Structure:")
    print("-" * 60)
    
    dataset_exists = os.path.exists('environmental-sounds')
    if dataset_exists:
        print("  [OK] environmental-sounds/ directory exists")
        
        for dir_path in dataset_dirs:
            if os.path.exists(dir_path):
                print(f"  [OK] {dir_path}/")
            else:
                print(f"  [MISSING] {dir_path}/")
                missing_files.append(dir_path)
        
        for file_path in dataset_files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"  [OK] {file_path} ({size:,} bytes)")
                total_files += 1
            else:
                print(f"  [MISSING] {file_path}")
                missing_files.append(file_path)
    else:
        print("  [MISSING] environmental-sounds/ directory")
        missing_files.append('environmental-sounds/')
    
    # Check ZIP file
    print("\n" + "-" * 60)
    print("ZIP File:")
    print("-" * 60)
    
    zip_file = 'environmental-sounds.zip'
    if os.path.exists(zip_file):
        size = os.path.getsize(zip_file)
        size_mb = size / (1024 * 1024)
        print(f"  [OK] {zip_file} exists")
        print(f"    Size: {size:,} bytes ({size_mb:.2f} MB)")
        total_files += 1
    else:
        print(f"  [MISSING] {zip_file}")
        missing_files.append(zip_file)
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Total files checked: {total_files}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"  - {file}")
    else:
        print("\n[SUCCESS] All files present!")
    
    print("\n" + "=" * 60)
    print("Project Structure Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Add actual audio files to environmental-sounds/train/ and test/ directories")
    print("2. Update metadata.csv if needed")
    print("3. Test the modules: python example_usage.py")
    print("4. Start API server: python api_server.py")
    print("\nFor questions:")
    print("  Website: https://rskworld.in")
    print("  Email: help@rskworld.in")
    print("  Phone: +91 93305 39277")


if __name__ == '__main__':
    verify_project()

