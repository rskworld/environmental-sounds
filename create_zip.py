"""
Create ZIP file of the complete project

Project: Environmental Sound Dataset
Website: https://rskworld.in
Founded by: Molla Samser
Designer & Tester: Rima Khatun
Email: help@rskworld.in
Phone: +91 93305 39277
"""

import os
import zipfile
from pathlib import Path
import shutil


def create_project_zip(output_filename: str = 'environmental-sounds.zip'):
    """
    Create a ZIP file containing the entire project.
    
    Args:
        output_filename: Name of the output ZIP file
    """
    
    # Files and directories to include
    include_patterns = [
        '*.py',
        '*.md',
        '*.txt',
        '*.html',
        '*.ipynb',
        '*.csv',
        'LICENSE',
        '.gitignore',
        'examples/',
        'environmental-sounds/'
    ]
    
    # Files and directories to exclude
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.git',
        '.DS_Store',
        '*.log',
        'models/',
        '*.pkl',
        '*.h5',
        '*.model'
    ]
    
    base_dir = Path('.')
    
    print("Creating project ZIP file...")
    print("=" * 50)
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all Python files
        for py_file in base_dir.glob('*.py'):
            if py_file.name != 'create_zip.py' and py_file.name != 'create_sample_data.py':
                zipf.write(py_file, py_file)
                print(f"Added: {py_file}")
        
        # Add documentation files
        for md_file in base_dir.glob('*.md'):
            zipf.write(md_file, md_file)
            print(f"Added: {md_file}")
        
        # Add other files
        for file in ['LICENSE', 'requirements.txt', 'setup.py', 'index.html', '.gitignore']:
            if os.path.exists(file):
                zipf.write(file, file)
                print(f"Added: {file}")
        
        # Add examples directory
        if os.path.exists('examples'):
            for root, dirs, files in os.walk('examples'):
                # Skip __pycache__
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path)
                    print(f"Added: {file_path}")
        
        # Add dataset directory (if it exists)
        if os.path.exists('environmental-sounds'):
            for root, dirs, files in os.walk('environmental-sounds'):
                # Skip __pycache__
                dirs[:] = [d for d in dirs if d != '__pycache__']
                
                for file in files:
                    # Skip large model files
                    if file.endswith(('.pkl', '.h5', '.model')):
                        continue
                    
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file_path)
                    print(f"Added: {file_path}")
    
    # Get file size
    file_size = os.path.getsize(output_filename)
    file_size_mb = file_size / (1024 * 1024)
    
    print("=" * 50)
    print(f"ZIP file created: {output_filename}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Location: {os.path.abspath(output_filename)}")
    
    return output_filename


if __name__ == '__main__':
    print("Environmental Sound Dataset - ZIP Creator")
    print("=" * 50)
    
    try:
        zip_file = create_project_zip()
        print("\n[SUCCESS] Project ZIP file created successfully!")
        print(f"\nYou can now distribute: {zip_file}")
    except Exception as e:
        print(f"Error creating ZIP file: {e}")
        import traceback
        traceback.print_exc()

