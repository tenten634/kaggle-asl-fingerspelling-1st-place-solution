"""
Environment Check Script for Google Colab
This script checks the current environment and provides information about
GPU/TPU availability, Python version, and installed packages.
"""

import sys
import os
import platform
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print("=" * 60)
    print("PYTHON VERSION")
    print("=" * 60)
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("⚠️  WARNING: Python 3.7+ is recommended")
    else:
        print("✅ Python version is compatible")
    print()

def check_colab():
    """Check if running in Google Colab"""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    try:
        import google.colab
        print("✅ Running in Google Colab")
        return True
    except ImportError:
        print("❌ Not running in Google Colab")
        return False
    print()

def check_gpu():
    """Check GPU availability"""
    print("=" * 60)
    print("GPU CHECK")
    print("=" * 60)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        else:
            print("❌ CUDA is not available")
            print("   Note: You may need to enable GPU in Colab: Runtime -> Change runtime type -> GPU")
    except ImportError:
        print("⚠️  PyTorch not installed yet")
    print()

def check_tpu():
    """Check TPU availability"""
    print("=" * 60)
    print("TPU CHECK")
    print("=" * 60)
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        print("✅ TPU is available")
        print(f"   TPU Device: {xm.xla_device()}")
        print(f"   Number of TPU cores: {xm.xrt_world_size()}")
    except ImportError:
        print("⚠️  TPU libraries not available")
        print("   Note: TPU support requires additional setup")
    except Exception as e:
        print(f"❌ TPU not available: {e}")
    print()

def check_installed_packages():
    """Check if required packages are installed"""
    print("=" * 60)
    print("PACKAGE CHECK")
    print("=" * 60)
    
    required_packages = {
        'torch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'transformers': 'Transformers',
        'neptune': 'Neptune',
        'tqdm': 'tqdm',
        'albumentations': 'Albumentations',
        'rapidfuzz': 'RapidFuzz',
        'timm': 'timm',
        'opencv-python': 'OpenCV',
    }
    
    installed = []
    missing = []
    
    for package, name in required_packages.items():
        try:
            if package == 'opencv-python':
                import cv2
                print(f"✅ {name} (cv2)")
            else:
                __import__(package)
                print(f"✅ {name}")
            installed.append(package)
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(package)
    
    print()
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("   Run the setup script to install them")
    else:
        print("✅ All required packages are installed")
    print()

def check_directories():
    """Check if required directories exist"""
    print("=" * 60)
    print("DIRECTORY CHECK")
    print("=" * 60)
    
    required_dirs = [
        'configs',
        'data',
        'models',
        'postprocess',
        'metrics',
        'datamount',
        'scripts'
    ]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - NOT FOUND")
    print()

def check_data_files():
    """Check if data files exist"""
    print("=" * 60)
    print("DATA FILES CHECK")
    print("=" * 60)
    
    required_files = [
        'datamount/character_to_prediction_index.json',
        'datamount/train_folded.csv',
        'datamount/symmetry.csv',
    ]
    
    optional_files = [
        'datamount/train_folded_oof_supp.csv',
        'datamount/train_landmarks_npy/',
    ]
    
    print("Required files:")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
    
    print("\nOptional files (for full training):")
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"⚠️  {file_path} - NOT FOUND (may need to download data)")
    print()

def check_disk_space():
    """Check available disk space"""
    print("=" * 60)
    print("DISK SPACE CHECK")
    print("=" * 60)
    try:
        stat = os.statvfs('/')
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
        used_gb = total_gb - free_gb
        
        print(f"Total: {total_gb:.2f} GB")
        print(f"Used: {used_gb:.2f} GB")
        print(f"Free: {free_gb:.2f} GB")
        
        if free_gb < 5:
            print("⚠️  WARNING: Low disk space. Consider cleaning up.")
        else:
            print("✅ Sufficient disk space available")
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
    print()

def main():
    """Run all checks"""
    print("\n" + "=" * 60)
    print("GOOGLE COLAB ENVIRONMENT CHECK")
    print("=" * 60)
    print()
    
    check_python_version()
    is_colab = check_colab()
    check_gpu()
    if is_colab:
        check_tpu()
    check_installed_packages()
    check_directories()
    check_data_files()
    check_disk_space()
    
    print("=" * 60)
    print("CHECK COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. If packages are missing, run: python setup_colab.py")
    print("2. If data is missing, download it using Kaggle API")
    print("3. Run training with: python train_colab.py -C cfg_1")

if __name__ == "__main__":
    main()

