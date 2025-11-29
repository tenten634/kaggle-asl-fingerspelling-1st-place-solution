"""
Setup Script for Google Colab
This script installs all required dependencies and configures the environment
for running the ASL Fingerspelling training in Google Colab.
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a shell command and print the result"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Success")
        if result.stdout:
            print(result.stdout)
    else:
        print("❌ Error")
        if result.stderr:
            print(result.stderr)
    return result.returncode == 0

def check_colab():
    """Check if running in Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_colab():
    """Main setup function"""
    print("\n" + "="*60)
    print("GOOGLE COLAB SETUP")
    print("="*60)
    
    is_colab = check_colab()
    if not is_colab:
        print("⚠️  WARNING: This script is designed for Google Colab")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Upgrade pip
    run_command("pip install --upgrade pip", "Upgrading pip")
    
    # Install PyTorch (Colab usually has it, but ensure correct version)
    print("\n" + "="*60)
    print("INSTALLING PYTORCH")
    print("="*60)
    # Check if CUDA is available to determine which PyTorch to install
    try:
        import torch
        if torch.cuda.is_available():
            print("✅ PyTorch with CUDA is already installed")
            print(f"   Version: {torch.__version__}")
        else:
            print("⚠️  PyTorch found but CUDA not available")
            print("   Make sure GPU is enabled in Colab runtime settings")
    except ImportError:
        # Install PyTorch with CUDA support for Colab
        run_command(
            "pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117",
            "Installing PyTorch with CUDA support"
        )
    
    # Install other requirements
    print("\n" + "="*60)
    print("INSTALLING OTHER DEPENDENCIES")
    print("="*60)
    
    requirements = [
        "tensorflow==2.12.0",
        "pandas",
        "numpy",
        "transformers==4.32.1",
        "neptune-client==1.3.1",  # Use neptune-client instead of neptune
        "tqdm",
        "albumentations==1.3.1",
        "rapidfuzz==3.2.0",
        "timm==0.9.6",
        "opencv-python-headless==4.8.0.74",
        "kaggle==1.5.16",
    ]
    
    for package in requirements:
        run_command(f"pip install {package}", f"Installing {package}")
    
    # Install TPU support if needed (optional)
    print("\n" + "="*60)
    print("TPU SUPPORT (OPTIONAL)")
    print("="*60)
    print("If you want to use TPU, uncomment the following line:")
    print("# run_command('pip install cloud-tpu-client==0.10 torch-xla', 'Installing TPU support')")
    
    # Verify installation
    print("\n" + "="*60)
    print("VERIFYING INSTALLATION")
    print("="*60)
    
    try:
        import torch
        import tensorflow as tf
        import pandas as pd
        import numpy as np
        import transformers
        import neptune
        import tqdm
        import albumentations
        import rapidfuzz
        import timm
        import cv2
        
        print("✅ All packages imported successfully")
        
        # Check GPU
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - make sure GPU is enabled")
        
        # Check TensorFlow GPU
        if tf.config.list_physical_devices('GPU'):
            print("✅ TensorFlow GPU available")
        else:
            print("⚠️  TensorFlow GPU not available")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    print("\n" + "="*60)
    print("SETUP COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Run: python check_colab_environment.py")
    print("2. Set up Kaggle API credentials (if needed)")
    print("3. Download data (if not already present)")
    print("4. Run training: python train_colab.py -C cfg_1")
    
    return True

if __name__ == "__main__":
    success = setup_colab()
    sys.exit(0 if success else 1)

