"""
Setup packages for TPU environment
Run this after switching to TPU to reinstall manually installed packages
"""
import subprocess
import sys

print("="*60)
print("TPU PACKAGE SETUP")
print("="*60)

# Packages that were manually installed (from KAGGLE_SETUP_GUIDE.md)
manual_packages = [
    'neptune-client==1.3.1',
    'rapidfuzz==3.2.0',
]

# Check what's already installed
print("\n🔍 Checking installed packages...")
installed = {}
missing = []

for package in manual_packages:
    package_name = package.split('==')[0]
    try:
        __import__(package_name.replace('-', '_'))
        installed[package_name] = True
        print(f"   ✅ {package_name} - already installed")
    except ImportError:
        installed[package_name] = False
        missing.append(package)
        print(f"   ❌ {package_name} - needs installation")

# Install missing packages
if missing:
    print(f"\n📦 Installing {len(missing)} missing package(s)...")
    for package in missing:
        print(f"   Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}: {e}")
else:
    print(f"\n✅ All packages already installed!")

# Verify TPU support
print(f"\n🔍 Checking TPU support...")
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    print(f"   ✅ torch_xla available (pre-installed in TPU environment)")
except ImportError:
    print(f"   ⚠️  torch_xla not found")
    print(f"      This should be pre-installed in Kaggle TPU environment")
    print(f"      If missing, you may need to restart the notebook")

print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)

