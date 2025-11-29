# Kaggle Setup & Training Guide

Quick setup guide for ASL Fingerspelling Recognition training in Kaggle Notebooks.

## Prerequisites

1. Create Kaggle Notebook → Enable **TPU v5e8** (Settings → Accelerator → TPU)
2. Add datasets via Data sidebar:
   - `asl-fingerspelling-preprocessing-train-dataset`
   - `asl-fingerspelling-preprocessed-supp-dataset`

## Step 1: Clone Repository

```python
!git clone https://github.com/tenten634/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
```

## Step 2: Install Packages

```python
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
!pip install cloud-tpu-client==0.10 torch-xla[tpu]==2.9.0 -f https://storage.googleapis.com/libtpu-releases/index.html

# Fix protobuf version conflicts (TPU installs old versions, transformers needs newer)
# Note: This will show dependency warnings - they're harmless, the packages will work
!pip install --upgrade --no-deps protobuf>=5.28.0
!pip install --upgrade google-api-core>=2.27.0 google-api-python-client>=2.0.0
```

**Note**: Dependency warnings (e.g., protobuf conflicts) are expected and can be ignored. Installation will complete successfully.

**If TPU test fails with symbol error**, try compatible version:
```python
!pip uninstall torch-xla -y
!pip install torch-xla[tpu]==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html
# Then restart the notebook
```

## Step 3: Prepare Data

**Note**: Only CSV/JSON files are copied. Landmark data uses virtual merge from `/kaggle/input/` (no copying needed).

```python
import os
import shutil

# Make sure we're in the repository directory
repo_dir = '/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution'
os.chdir(repo_dir)

# Create datamount directory in repository
os.makedirs('datamount', exist_ok=True)

# Copy CSV/JSON files from input datasets
input_base = '/kaggle/input'
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith(('.csv', '.json')):
                src = os.path.join(dataset_path, file)
                dst = os.path.join('datamount', file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"✅ Copied {file}")
```

## Step 4: Verify Setup

```python
!python check_kaggle_environment.py
```

**Expected output:**
- ✅ Input datasets found
- ✅ Data files ready (CSV/JSON in datamount/)
- ⚠️ TPU check may show errors (this is OK - test manually below)
- ❌ CUDA not available (expected when using TPU)

**Note**: The check script may crash at the end with "InitializeComputationClient() can only be called once" - this is **normal and harmless**. It's a known torch-xla issue during Python exit. As long as you see:
- ✅ Data files ready
- ✅ Input datasets found  
- ✅ Project files exist

Then you're ready! The manual TPU test below is the reliable way to verify TPU works.

**Note**: The check script may show "TPU libraries not available" even after installation. Test TPU manually:

```python
try:
    import torch_xla
    import torch
    device = torch_xla.device()  # Use new API (no deprecation warning)
    print(f"✅ TPU is working: {device}")
    
    # Quick test
    x = torch.randn(10, 10, device=device)
    print(f"✅ TPU computation test: {x.shape}")
except Exception as e:
    print(f"⚠️  TPU test failed: {e}")
    print("\n🔧 Fix: Try installing compatible torch-xla version:")
    print("   !pip uninstall torch-xla -y")
    print("   !pip install torch-xla[tpu]==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html")
    print("   Then restart the notebook")
```

## Step 5: Start Training

### Quick Path (Recommended)

Skip Round 1 - go directly to Round 2:

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

# Train cfg_2 with 2 seeds
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
```

**Time**: ~3-5 hours per seed on TPU v5e8

**Note**: First iteration on TPU can take 5-15 minutes (JIT compilation). This is normal - be patient!

### Full Pipeline

```python
# Round 1: Train 4 folds
!python train_kaggle.py -C cfg_1 --fold 0 --use_tpu
!python train_kaggle.py -C cfg_1 --fold 1 --use_tpu
!python train_kaggle.py -C cfg_1 --fold 2 --use_tpu
!python train_kaggle.py -C cfg_1 --fold 3 --use_tpu

# Generate OOF predictions
!python scripts/get_train_folded_oof_supp.py

# Round 2: Train fullfit (2 seeds)
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
!python train_kaggle.py -C cfg_2 --fold -1 --use_tpu
```

### Convert to TF-Lite

```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

## Troubleshooting

- **Training stuck on first iteration**: 
  - **Normal**: First TPU iteration takes 5-15 minutes for JIT compilation - be patient!
  - If stuck >20 minutes, restart session and try again
  - Check if progress bar shows any activity (even if 0%)
- **TPU "Device or resource busy" error**: 
  - Restart notebook session (Session → Restart Session)
  - Verify TPU is enabled: Settings → Accelerator → TPU v5e8
  - If still failing, try GPU instead (remove `--use_tpu` flag)
- **Protobuf import error** (`cannot import name 'runtime_version'`): Run `!pip install --upgrade --no-deps protobuf>=5.28.0` then `!pip install --upgrade google-api-core>=2.27.0 google-api-python-client>=2.0.0` after Step 2. Dependency warnings are expected but harmless.
- **TPU symbol error** (`undefined symbol`): Try `torch-xla==2.8.0` instead of 2.9.0, then restart notebook
- **TPU not available**: Settings → Accelerator → TPU v5e8 → Restart notebook
- **TPU libraries not detected**: Restart notebook after Step 2, then test manually (see Step 4)
- **TensorFlow missing**: Not required for training (only needed for TF-Lite conversion)
- **Dependency conflicts**: Warnings about protobuf/API versions are expected - ignore them
- **Out of memory**: Reduce `batch_size` in config or enable `mixed_precision = True`
- **Data not found**: Verify datasets are added in Data sidebar
- **Module not found**: `!pip install package_name`

## Key Points

- **Virtual Merge**: Landmark data accessed directly from `/kaggle/input/` (no copying)
- **Checkpoints**: Saved to `datamount/weights/` (auto-saved to Output tab)
- **Time Limit**: 9 hours per session (free tier)
