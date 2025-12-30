"""
Environmental Sound Dataset - Example Usage Script

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277

This script demonstrates basic usage of the Environmental Sound Dataset.
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from load_data import load_environmental_sounds, get_class_distribution, prepare_features
from analyze import analyze_audio_file, get_dataset_statistics
from train_model import train_classifier, evaluate_model, save_model


def main():
    """Main example function demonstrating dataset usage."""
    
    print("=" * 60)
    print("Environmental Sound Dataset - Example Usage")
    print("=" * 60)
    print()
    
    # Check if dataset exists
    data_dir = './environmental-sounds'
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        print("Please ensure the dataset is extracted in the current directory.")
        return
    
    # 1. Load dataset statistics
    print("1. Getting dataset statistics...")
    print("-" * 60)
    try:
        stats = get_dataset_statistics(data_dir)
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total duration: {stats['total_duration']:.2f} seconds")
        print(f"   Average duration: {stats['avg_duration']:.2f} seconds")
        print(f"   Classes: {list(stats['classes'].keys())}")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # 2. Load training data
    print("2. Loading training data...")
    print("-" * 60)
    try:
        train_data, train_labels = load_environmental_sounds('train', data_dir=data_dir)
        print(f"   Loaded {len(train_data)} training samples")
        print(f"   Number of classes: {len(set(train_labels))}")
        print(f"   Classes: {sorted(set(train_labels))}")
        
        # Show class distribution
        class_dist = get_class_distribution(train_labels)
        print(f"   Class distribution:")
        for cls, count in sorted(class_dist.items()):
            print(f"     - {cls}: {count} samples")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print("   Please ensure the dataset structure is correct.")
        print("   See DATASET_STRUCTURE.md for details.")
        return
    
    # 3. Load test data
    print("3. Loading test data...")
    print("-" * 60)
    try:
        test_data, test_labels = load_environmental_sounds('test', data_dir=data_dir)
        print(f"   Loaded {len(test_data)} test samples")
        print()
    except Exception as e:
        print(f"   Error: {e}")
        print("   Test set not found or empty.")
        print()
    
    # 4. Extract features (sample)
    print("4. Extracting features from sample data...")
    print("-" * 60)
    try:
        if len(train_data) > 0:
            sample_data = train_data[:10]  # Use first 10 samples
            features = prepare_features(sample_data, feature_type='mfcc', n_mfcc=13)
            print(f"   Extracted features from {len(sample_data)} samples")
            print(f"   Feature shape: {features.shape}")
            print(f"   Features per sample: {features.shape[1]}")
            print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    # 5. Train a simple classifier (if enough data)
    print("5. Training classifier...")
    print("-" * 60)
    try:
        if len(train_data) >= 20:  # Need minimum samples
            print("   Training Random Forest classifier...")
            model, scaler, label_encoder, accuracy = train_classifier(
                train_data, train_labels, model_type='random_forest'
            )
            print(f"   Training completed with validation accuracy: {accuracy:.4f}")
            
            # Evaluate on test set if available
            if len(test_data) > 0:
                print("   Evaluating on test set...")
                results = evaluate_model(model, scaler, label_encoder, test_data, test_labels)
                print(f"   Test accuracy: {results['accuracy']:.4f}")
            
            # Save model
            print("   Saving model...")
            save_model(model, scaler, label_encoder)
            print()
        else:
            print("   Not enough training samples (need at least 20)")
            print()
    except Exception as e:
        print(f"   Error: {e}")
        print()
    
    print("=" * 60)
    print("Example usage completed!")
    print("=" * 60)
    print()
    print("For more examples, see:")
    print("  - examples/data_exploration.ipynb")
    print("  - README.md")
    print()
    print("Contact:")
    print("  Website: https://rskworld.in")
    print("  Email: help@rskworld.in")
    print("  Phone: +91 93305 39277")


if __name__ == '__main__':
    main()

