#!/usr/bin/env python3
"""
Error Checking Script for Satellite Image Dataset
Created by: RSK World (https://rskworld.in)

This script checks all files for errors and reports issues.
"""

import sys
import importlib
from pathlib import Path
import traceback

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        if package_name:
            mod = importlib.import_module(module_name, package=package_name)
        else:
            mod = importlib.import_module(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def check_file_syntax(file_path):
    """Check if a Python file has syntax errors."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)} at line {e.lineno}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    """Main error checking function."""
    print("=" * 60)
    print("Satellite Image Dataset - Error Checker")
    print("Created by: RSK World (https://rskworld.in)")
    print("=" * 60)
    print()
    
    errors_found = []
    warnings = []
    
    # Check required packages
    print("1. Checking Required Packages...")
    print("-" * 60)
    required_packages = {
        'numpy': 'numpy',
        'cv2': 'opencv-python',
        'matplotlib': 'matplotlib',
        'skimage': 'scikit-image',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'PIL': 'pillow',
        'rasterio': 'rasterio',
        'seaborn': 'seaborn',
        'tqdm': 'tqdm'
    }
    
    for import_name, package_name in required_packages.items():
        success, error = check_import(import_name)
        if success:
            print(f"  [OK] {package_name}")
        else:
            print(f"  [ERROR] {package_name} - {error}")
            errors_found.append(f"Missing package: {package_name}")
    
    print()
    
    # Check optional packages
    print("2. Checking Optional Packages...")
    print("-" * 60)
    optional_packages = {
        'pystac_client': 'pystac-client',
        'planetary_computer': 'planetary-computer',
        'landsatxplore': 'landsatxplore',
        'sentinelsat': 'sentinelsat'
    }
    
    for import_name, package_name in optional_packages.items():
        success, error = check_import(import_name)
        if success:
            print(f"  [OK] {package_name} (optional)")
        else:
            print(f"  [SKIP] {package_name} (optional, not installed)")
            warnings.append(f"Optional package not installed: {package_name}")
    
    print()
    
    # Check Python files for syntax errors
    print("3. Checking Python Files for Syntax Errors...")
    print("-" * 60)
    
    python_files = [
        'data_loader.py',
        'process_images.py',
        'visualize.py',
        'example_usage.py',
        'advanced_processing.py',
        'ml_integration.py',
        'enhanced_real_image_downloader.py',
        'advanced_visualization.py',
        'batch_processing.py',
        'advanced_example.py',
        'batch_processor.py',
        'ml_features.py',
        'real_image_downloader.py'
    ]
    
    for file_name in python_files:
        file_path = Path(file_name)
        if file_path.exists():
            success, error = check_file_syntax(file_path)
            if success:
                print(f"  [OK] {file_name}")
            else:
                print(f"  [ERROR] {file_name} - {error}")
                errors_found.append(f"Syntax error in {file_name}: {error}")
        else:
            print(f"  [SKIP] {file_name} (not found)")
    
    print()
    
    # Try importing main modules
    print("4. Testing Module Imports...")
    print("-" * 60)
    
    modules_to_test = [
        'data_loader',
        'process_images',
        'visualize',
        'advanced_processing',
        'ml_integration',
        'enhanced_real_image_downloader',
        'advanced_visualization',
        'batch_processing'
    ]
    
    for module_name in modules_to_test:
        try:
            mod = importlib.import_module(module_name)
            print(f"  ✓ {module_name}")
        except ImportError as e:
            print(f"  [ERROR] {module_name} - Import error: {e}")
            errors_found.append(f"Import error in {module_name}: {e}")
        except SyntaxError as e:
            print(f"  [ERROR] {module_name} - Syntax error: {e}")
            errors_found.append(f"Syntax error in {module_name}: {e}")
        except Exception as e:
            print(f"  [ERROR] {module_name} - Error: {e}")
            errors_found.append(f"Error in {module_name}: {e}")
    
    print()
    
    # Check for duplicate files
    print("5. Checking for Duplicate Files...")
    print("-" * 60)
    
    potential_duplicates = {
        'batch_processing.py': 'batch_processor.py',
        'ml_integration.py': 'ml_features.py',
        'enhanced_real_image_downloader.py': 'real_image_downloader.py'
    }
    
    for new_file, old_file in potential_duplicates.items():
        new_path = Path(new_file)
        old_path = Path(old_file)
        if new_path.exists() and old_path.exists():
            print(f"  [WARN] Both {new_file} and {old_file} exist")
            warnings.append(f"Duplicate files: {new_file} and {old_file}")
        elif new_path.exists():
            print(f"  [OK] {new_file} (recommended)")
        elif old_path.exists():
            print(f"  [SKIP] {old_file} (older version)")
    
    print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if errors_found:
        print(f"\n[ERRORS] Found {len(errors_found)} error(s):")
        for error in errors_found:
            print(f"  - {error}")
    else:
        print("\n[SUCCESS] No critical errors found!")
    
    if warnings:
        print(f"\n[WARNINGS] {len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  - {warning}")
    
    print()
    
    if errors_found:
        print("=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        print("\n1. Install missing packages:")
        print("   pip install -r requirements.txt")
        print("\n2. For real image downloading (optional):")
        print("   pip install pystac-client planetary-computer")
        print("\n3. Check file syntax errors and fix them")
        print("\n4. Remove duplicate files if any")
        return 1
    else:
        print("=" * 60)
        print("All checks passed! [SUCCESS]")
        print("=" * 60)
        return 0

if __name__ == "__main__":
    sys.exit(main())

