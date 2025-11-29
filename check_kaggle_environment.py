"""
Environment Check Script for Kaggle Notebooks
This script checks the current environment and provides information about
GPU/TPU availability, Python version, and installed packages in Kaggle.
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

def check_kaggle():
    """Check if running in Kaggle"""
    print("=" * 60)
    print("ENVIRONMENT CHECK")
    print("=" * 60)
    # Check for essential Kaggle directories (temp is optional)
    essential_paths = ['/kaggle/input', '/kaggle/working']
    temp_path = '/kaggle/temp'
    is_kaggle = all(os.path.exists(path) for path in essential_paths)
    
    if is_kaggle:
        print("✅ Running in Kaggle Notebook")
        print(f"   Input directory: /kaggle/input")
        print(f"   Working directory: /kaggle/working")
        if os.path.exists(temp_path):
            print(f"   Temp directory: /kaggle/temp")
        else:
            print(f"   Temp directory: /kaggle/temp (optional, not present)")
    else:
        print("❌ Not running in Kaggle Notebook")
        print("   Expected directories: /kaggle/input, /kaggle/working")
    print()
    return is_kaggle

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
            print("   Note: Enable GPU in Kaggle: Settings → Accelerator → GPU")
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
        print("   Install them in a code cell: !pip install package_name")
    else:
        print("✅ All required packages are installed")
    print()

def check_kaggle_directories():
    """Check Kaggle directory structure"""
    print("=" * 60)
    print("KAGGLE DIRECTORY CHECK")
    print("=" * 60)
    
    kaggle_dirs = {
        '/kaggle/input': ('Input (datasets)', True),  # Required
        '/kaggle/working': ('Working (your code)', True),  # Required
        '/kaggle/temp': ('Temp (temporary files)', False),  # Optional
    }
    
    for path, (desc, required) in kaggle_dirs.items():
        if os.path.exists(path):
            print(f"✅ {path} - {desc}")
        else:
            if required:
                print(f"❌ {path} - NOT FOUND")
            else:
                print(f"⚠️  {path} - NOT FOUND (optional)")
    print()

def check_input_datasets():
    """Check if datasets are added to the notebook"""
    print("=" * 60)
    print("INPUT DATASETS CHECK")
    print("=" * 60)
    
    input_dir = '/kaggle/input'
    if os.path.exists(input_dir):
        datasets = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
        if datasets:
            print("✅ Found datasets:")
            for dataset in datasets:
                print(f"   - {dataset}")
        else:
            print("⚠️  No datasets found in /kaggle/input")
            print("   Add datasets: Data → Add input → Search for dataset")
    else:
        print("❌ /kaggle/input directory not found")
    print()

def check_project_files():
    """Check if project files exist"""
    print("=" * 60)
    print("PROJECT FILES CHECK")
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
    
    # Check in working directory
    working_files = [
        'datamount/character_to_prediction_index.json',
        'datamount/train_folded.csv',
        'datamount/symmetry.csv',
    ]
    
    # Check in input directory (if datasets are added)
    input_paths = []
    if os.path.exists('/kaggle/input'):
        for dataset_dir in os.listdir('/kaggle/input'):
            dataset_path = f'/kaggle/input/{dataset_dir}'
            if os.path.isdir(dataset_path):
                # Check common dataset names
                if 'asl-fingerspelling' in dataset_dir.lower() or 'landmarks' in dataset_dir.lower():
                    input_paths.append(dataset_path)
    
    print("Required files (in working directory):")
    for file_path in working_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
    
    print("\nInput datasets:")
    if input_paths:
        for path in input_paths:
            print(f"✅ {path}")
            # Check for landmarks directory
            landmarks_path = os.path.join(path, 'train_landmarks_npy')
            if os.path.exists(landmarks_path):
                print(f"   ✅ train_landmarks_npy/ found")
            else:
                print(f"   ⚠️  train_landmarks_npy/ not found in dataset")
    else:
        print("⚠️  No relevant datasets found in /kaggle/input")
        print("   Add the dataset: Data → Add input → Search 'asl-fingerspelling'")
    print()

def check_disk_space():
    """Check available disk space"""
    print("=" * 60)
    print("DISK SPACE CHECK")
    print("=" * 60)
    try:
        stat = os.statvfs('/kaggle/working')
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
    print("KAGGLE NOTEBOOK ENVIRONMENT CHECK")
    print("=" * 60)
    print()
    
    check_python_version()
    is_kaggle = check_kaggle()
    check_kaggle_directories()
    check_gpu()
    if is_kaggle:
        check_tpu()
    check_installed_packages()
    check_input_datasets()
    check_project_files()
    check_data_files()
    check_disk_space()
    
    print("=" * 60)
    print("CHECK COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. If packages are missing, install them: !pip install package_name")
    print("2. If data is missing, add dataset: Data → Add input")
    print("3. Run training with: !python train_kaggle.py -C cfg_1 --fold 0")

if __name__ == "__main__":
    main()

