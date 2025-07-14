#!/usr/bin/env python3
"""Helper script to clean Python cache files and create a zip archive of the project."""

import os
import shutil
import glob
import zipfile
from pathlib import Path
from datetime import datetime

def clean_pycache():
    """Remove all __pycache__ directories and .pyc files."""
    # Remove __pycache__ directories
    for pycache in glob.glob('**/__pycache__', recursive=True):
        print(f"Removing {pycache}")
        shutil.rmtree(pycache)
    
    # Remove .pyc files
    for pyc in glob.glob('**/*.pyc', recursive=True):
        print(f"Removing {pyc}")
        os.remove(pyc)
    
    # Remove .pyo files
    for pyo in glob.glob('**/*.pyo', recursive=True):
        print(f"Removing {pyo}")
        os.remove(pyo)
    
    # Remove .pyd files
    for pyd in glob.glob('**/*.pyd', recursive=True):
        print(f"Removing {pyd}")
        os.remove(pyd)

def create_zip():
    """Create a compressed zip archive of the project."""
    project_root = Path(__file__).parent.parent
    project_name = project_root.name
    
    # Create zip file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_name = f"{project_name}_{timestamp}.zip"
    
    # Define directories to exclude
    exclude_dirs = {
        'venv',  # Virtual environment
        '.venv',  # Alternative virtual environment name
        'env',  # Another common virtual environment name
        '.env',  # Environment directory
        '.git',  # Git directory
        '.idea',  # PyCharm IDE
        '.vscode',  # VS Code
        '__pycache__',  # Python cache
        '.pytest_cache',  # pytest cache
        '.mypy_cache',  # mypy cache
        '.coverage',  # coverage data
        'htmlcov',  # coverage HTML output
        '.ipynb_checkpoints',  # Jupyter notebook checkpoints
    }
    
    # Create a temporary directory for the cleaned project
    temp_dir = Path(f"temp_{timestamp}")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy files to temp directory, excluding specified directories
    print("Copying files to temporary directory...")
    for item in project_root.glob('*'):
        if item.name not in exclude_dirs:
            if item.is_file():
                shutil.copy2(item, temp_dir / item.name)
            elif item.is_dir():
                shutil.copytree(item, temp_dir / item.name, 
                              ignore=shutil.ignore_patterns(*exclude_dirs))
    
    # Create compressed zip archive
    print(f"Creating {zip_name} with compression...")
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up temporary directory
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

def main():
    """Main function to clean cache and create zip."""
    print("Cleaning Python cache files...")
    clean_pycache()
    
    print("\nCreating zip archive...")
    create_zip()
    
    print("\nDone!")

if __name__ == '__main__':
    main() 