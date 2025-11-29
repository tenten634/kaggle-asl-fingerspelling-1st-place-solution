# Complete Kaggle Setup & Training Guide

This is the **single comprehensive guide** for running ASL Fingerspelling Recognition training in Kaggle Notebooks.

---

## 📋 Quick Start (5 Minutes)

If you've already added datasets and enabled GPU P100, jump to **Step 1** below.

**Prerequisites:**
1. Create Kaggle Notebook → Enable **GPU P100** (Settings → Accelerator → GPU)
2. Add datasets via Data sidebar:
   - `asl-fingerspelling-preprocessing-train-dataset`
   - `asl-fingerspelling-preprocessed-supp-dataset`

---

## Step 1: Clone the Repository

In your first code cell, run:

```python
# Clone the repository
!git clone https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
```

**Expected output:**
```
Cloning into 'kaggle-asl-fingerspelling-1st-place-solution'...
/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution
```

**✅ Checkpoint**: Repository cloned and you're in the correct directory.

---

## Step 2: Install Missing Packages

Run this to install any missing dependencies:

```python
# Install required packages
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
```

**Expected output:**
```
Successfully installed neptune-client-1.3.1 rapidfuzz-3.2.0
```

**✅ Checkpoint**: Packages installed without errors.

---

## Step 3: Prepare Data Structure

This step copies data from the input datasets to your working directory:

```python
import os
import shutil

# Create data directory
os.makedirs('/kaggle/working/datamount', exist_ok=True)
print("✅ Created datamount directory")

# Copy CSV and JSON files from input datasets
input_base = '/kaggle/input'
copied_files = []

for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith(('.csv', '.json')):
                src = os.path.join(dataset_path, file)
                dst = os.path.join('/kaggle/working/datamount', file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_files.append(file)
                    print(f"✅ Copied {file}")

print(f"\n📊 Copied {len(copied_files)} files: {copied_files}")

# Create landmarks directory
landmarks_dst = '/kaggle/working/datamount/train_landmarks_npy'
os.makedirs(landmarks_dst, exist_ok=True)
print(f"✅ Created {landmarks_dst}")

# Copy training landmarks
print("\n📦 Copying training landmarks...")
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if 'train' in dataset_dir.lower() and 'preprocessing' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
        if os.path.exists(landmarks_path):
            print(f"   Found training landmarks in: {dataset_dir}")
            !cp -r {landmarks_path}/* {landmarks_dst}/
            print("   ✅ Training landmarks copied")

# Copy supplemental landmarks
print("\n📦 Copying supplemental landmarks...")
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if 'supp' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'supplemental_landmarks')
        if os.path.exists(landmarks_path):
            print(f"   Found supplemental landmarks in: {dataset_dir}")
            !cp -r {landmarks_path}/* {landmarks_dst}/
            print("   ✅ Supplemental landmarks copied and merged")

print("\n" + "="*60)
print("DATA PREPARATION COMPLETE")
print("="*60)
```

**Expected output:**
```
✅ Created datamount directory
✅ Copied train_folded.csv
✅ Copied character_to_prediction_index.json
✅ Copied symmetry.csv

📊 Copied 3 files: ['train_folded.csv', 'character_to_prediction_index.json', 'symmetry.csv']
✅ Created /kaggle/working/datamount/train_landmarks_npy

📦 Copying training landmarks...
   Found training landmarks in: asl-fingerspelling-preprocessing-train-dataset
   ✅ Training landmarks copied

📦 Copying supplemental landmarks...
   Found supplemental landmarks in: asl-fingerspelling-preprocessed-supp-dataset
   ✅ Supplemental landmarks copied and merged

============================================================
DATA PREPARATION COMPLETE
============================================================
```

**✅ Checkpoint**: All data files copied successfully.

---

## Step 3.5: Fix inference_args.json (If Needed)

If you get a JSON parsing error when loading cfg_2, run this to find and copy the file:

```python
import os
import json
import shutil

print("="*60)
print("FIXING inference_args.json")
print("="*60)

# Check if it already exists
working_path = 'datamount/train_landmarks_npy/inference_args.json'
if os.path.exists(working_path):
    try:
        with open(working_path, 'r') as f:
            data = json.load(f)
        print(f"✅ File already exists and is valid: {working_path}")
        print(f"   Keys: {list(data.keys())}")
    except Exception as e:
        print(f"⚠️  File exists but has error: {e}")
        print("   Will try to replace it...")
else:
    print(f"❌ File not found at: {working_path}")

# Find and copy from input datasets
print("\n📦 Searching in input datasets...")
input_base = '/kaggle/input'
found = False

for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
        if os.path.exists(landmarks_path):
            inference_file = os.path.join(landmarks_path, 'inference_args.json')
            if os.path.exists(inference_file):
                print(f"✅ Found in: {dataset_dir}")
                print(f"   Source: {inference_file}")
                
                # Read and verify
                try:
                    with open(inference_file, 'r') as f:
                        data = json.load(f)
                    print(f"   ✅ File is valid JSON")
                    print(f"   Keys: {list(data.keys())}")
                    if 'selected_columns' in data:
                        print(f"   Selected columns: {len(data['selected_columns'])} columns")
                    
                    # Handle symlink case
                    landmarks_dir = 'datamount/train_landmarks_npy'
                    if os.path.islink(landmarks_dir):
                        print(f"   ⚠️  {landmarks_dir} is a symlink")
                        print(f"   Creating real directory and copying file...")
                        # Remove symlink temporarily
                        symlink_target = os.readlink(landmarks_dir)
                        os.unlink(landmarks_dir)
                        os.makedirs(landmarks_dir, exist_ok=True)
                        # Copy the file
                        dst = os.path.join(landmarks_dir, 'inference_args.json')
                        shutil.copy2(inference_file, dst)
                        print(f"   ✅ Copied to: {dst}")
                        # Recreate symlink if needed (optional)
                        # os.symlink(symlink_target, landmarks_dir)
                    else:
                        # Regular directory, just copy
                        dst = working_path
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        shutil.copy2(inference_file, dst)
                        print(f"   ✅ Copied to: {dst}")
                    
                    found = True
                    break
                except Exception as e:
                    print(f"   ❌ Error reading file: {e}")

if not found:
    print("\n⚠️  inference_args.json not found in any dataset")
    print("   The config will use default columns (this should still work)")

print("\n" + "="*60)
```

---

## Step 3.6: Debug Config Loading (If You Get Division by Zero Error)

If you get a "float division by zero" error when loading cfg_2, run this diagnostic:

```python
%cd kaggle-asl-fingerspelling-1st-place-solution

import os
import sys
import traceback
import importlib
import json
import pandas as pd

# Set up paths
BASEDIR = os.getcwd()
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

print("="*60)
print("DEBUGGING CONFIG LOAD")
print("="*60)

try:
    print("\n1. Loading cfg_2...")
    cfg_module = importlib.import_module('cfg_2')
    cfg = cfg_module.cfg
    print("✅ Config loaded successfully")
    
    print(f"\n2. Checking data files...")
    print(f"   inference_args.json: {os.path.exists(cfg.data_folder + 'inference_args.json')}")
    print(f"   train_df: {os.path.exists(cfg.train_df)}")
    print(f"   symmetry_fp: {os.path.exists(cfg.symmetry_fp)}")
    
    print(f"\n3. Checking inference_args.json content...")
    with open(cfg.data_folder + 'inference_args.json', 'r') as f:
        data = json.load(f)
    columns = data['selected_columns']
    print(f"   Number of columns: {len(columns)}")
    
    # Check the division that might cause issues
    num_landmarks = len(columns) // 3
    print(f"\n4. Checking landmark calculation...")
    print(f"   Total columns: {len(columns)}")
    print(f"   Expected landmarks: {num_landmarks}")
    
    if len(columns) == 0:
        print("   ❌ ERROR: columns is empty!")
    elif len(columns) % 3 != 0:
        print(f"   ⚠️  WARNING: {len(columns)} is not divisible by 3")
    else:
        print(f"   ✅ Landmark calculation looks OK")
    
    print(f"\n5. Checking train DataFrame...")
    df = pd.read_csv(cfg.train_df)
    print(f"   DataFrame shape: {df.shape}")
    
    if len(df) == 0:
        print("   ❌ ERROR: DataFrame is empty!")
    else:
        print(f"   ✅ DataFrame has {len(df)} rows")
    
except ZeroDivisionError as e:
    print(f"\n❌ DIVISION BY ZERO ERROR!")
    print(f"   Error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "="*60)
```

This will help identify where the division by zero is occurring.

---

## Step 4: Verify Environment (CRITICAL!)

This is the **most important step** to confirm everything is ready before training:

```python
!python check_kaggle_environment.py
```

### ✅ All Good - You should see:
```
✅ Running in Kaggle Notebook
✅ CUDA is available
   CUDA Version: 11.x or 12.x
   Number of GPUs: 1
   GPU 0: Tesla P100-PCIE-16GB (or similar)
✅ All required packages are installed
✅ Input datasets found
✅ Project files exist
✅ Data files ready
✅ Sufficient disk space
```

### ❌ Issues to Fix:

**If you see "❌ CUDA is not available":**
- Go to Settings → Accelerator → Select "GPU P100"
- Save and restart the notebook

**If you see "❌ Missing packages":**
- Run: `!pip install package_name`
- Re-run the environment check

**If you see "❌ Data files not found":**
- Go back to Step 3 and verify datasets are added
- Check `/kaggle/input/` directory

**If you see "⚠️ Low disk space":**
- Clean up: `!rm -rf __pycache__ *.pyc`
- Or reduce batch size in config

---

## Step 5: Quick Data Verification

Verify the data structure is correct:

```python
import os

# Check required files
required_files = [
    'datamount/character_to_prediction_index.json',
    'datamount/train_folded.csv',
    'datamount/symmetry.csv',
    'datamount/train_landmarks_npy'
]

print("="*60)
print("DATA VERIFICATION")
print("="*60)

for file_path in required_files:
    if os.path.exists(file_path):
        if os.path.isdir(file_path):
            count = len([f for f in os.listdir(file_path) if f.endswith('.npy')])
            print(f"✅ {file_path}/ ({count} .npy files)")
        else:
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"✅ {file_path} ({size:.2f} MB)")
    else:
        print(f"❌ {file_path} - NOT FOUND")

# Count landmarks
landmarks_dir = 'datamount/train_landmarks_npy'
if os.path.exists(landmarks_dir):
    npy_files = [f for f in os.listdir(landmarks_dir) if f.endswith('.npy')]
    print(f"\n📊 Total landmark files: {len(npy_files)}")
    if len(npy_files) > 0:
        print("✅ Landmarks directory is ready")
    else:
        print("⚠️  Landmarks directory is empty - check data copy step")
```

**Expected output:**
```
============================================================
DATA VERIFICATION
============================================================
✅ datamount/character_to_prediction_index.json (0.05 MB)
✅ datamount/train_folded.csv (15.23 MB)
✅ datamount/symmetry.csv (0.01 MB)
✅ datamount/train_landmarks_npy/ (50000+ .npy files)

📊 Total landmark files: 50000+
✅ Landmarks directory is ready
```

**✅ Checkpoint**: All data files present and landmarks directory has files.

---

## Step 6: Test GPU Access

Verify GPU is accessible:

```python
import torch

print("="*60)
print("GPU VERIFICATION")
print("="*60)

if torch.cuda.is_available():
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    print(f"✅ GPU Count: {torch.cuda.device_count()}")
    print(f"✅ GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Test GPU computation
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"✅ GPU computation test: SUCCESS")
    print(f"   Result shape: {z.shape}")
else:
    print("❌ CUDA NOT AVAILABLE")
    print("   Enable GPU in Settings → Accelerator → GPU")
```

**Expected output:**
```
============================================================
GPU VERIFICATION
============================================================
✅ CUDA Available: True
✅ CUDA Version: 11.8
✅ GPU Count: 1
✅ GPU Name: Tesla P100-PCIE-16GB
✅ GPU Memory: 16.00 GB
✅ GPU computation test: SUCCESS
   Result shape: torch.Size([1000, 1000])
```

**✅ Checkpoint**: GPU is working correctly.

---

## Step 7: Final Pre-Training Checklist

Run this to get a final summary:

```python
import os
import torch

print("="*60)
print("PRE-TRAINING CHECKLIST")
print("="*60)

checks = {
    "Repository cloned": os.path.exists('train_kaggle.py'),
    "Configs exist": os.path.exists('configs/cfg_1.py'),
    "Data directory exists": os.path.exists('datamount'),
    "Train CSV exists": os.path.exists('datamount/train_folded.csv'),
    "Landmarks exist": os.path.exists('datamount/train_landmarks_npy'),
    "GPU available": torch.cuda.is_available(),
}

all_passed = True
for check_name, status in checks.items():
    status_icon = "✅" if status else "❌"
    print(f"{status_icon} {check_name}")
    if not status:
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("🎉 ALL CHECKS PASSED - READY TO TRAIN!")
    print("="*60)
    print("\nNext step: Run training with:")
    print("  !python train_kaggle.py -C cfg_1 --fold 0")
else:
    print("⚠️  SOME CHECKS FAILED - FIX ISSUES ABOVE")
    print("="*60)
```

**Expected output:**
```
============================================================
PRE-TRAINING CHECKLIST
============================================================
✅ Repository cloned
✅ Configs exist
✅ Data directory exists
✅ Train CSV exists
✅ Landmarks exist
✅ GPU available

============================================================
🎉 ALL CHECKS PASSED - READY TO TRAIN!
============================================================

Next step: Run training with:
  !python train_kaggle.py -C cfg_1 --fold 0
```

---

## Step 8: Start Training (After All Checks Pass)

**⚠️ IMPORTANT**: Only proceed to this step after **ALL checks in Step 7 pass**!

### Option A: Quick Path (Recommended for First Time)

**Skip Round 1** - The repository already includes `train_folded_oof_supp.csv`, so you can go directly to Round 2:

```python
# Make sure you're in the repository directory
%cd kaggle-asl-fingerspelling-1st-place-solution

# Round 2 - Train Fullfit (2 seeds)
!python train_kaggle.py -C cfg_2 --fold -1

# Seed 2 (run again for second seed)
!python train_kaggle.py -C cfg_2 --fold -1
```

**Training Time Estimates:**
- **Each seed (cfg_2)**: ~6-10 hours on P100 GPU
  - 400 epochs total
  - Validation every 10 epochs
  - Final checkpoint saved automatically at the end

**Checkpoint Saving:**
- ✅ **Automatic**: Final checkpoint is saved when training completes
- ✅ **Location**: `datamount/weights/cfg_2/fold-1/checkpoint_last_seed{seed}.pth`
- ✅ **Kaggle Auto-Save**: All files in `/kaggle/working/` are automatically saved to the **Output** tab
- ✅ **Persistent**: You can download or access them later, even after the notebook session ends

**Running Overnight:**
- ✅ **Safe to run**: If training finishes while you're sleeping, the checkpoint will be saved
- ✅ **Verify in morning**: Check the Output tab or run the verification code below
- ✅ **Continue with Seed 2**: Once Seed 1 is complete, just run the Seed 2 command

**Verify Checkpoint After Training:**
```python
import os

checkpoint_dir = 'datamount/weights/cfg_2/fold-1'
if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    checkpoints = [f for f in files if 'checkpoint' in f]
    print(f"✅ Found {len(checkpoints)} checkpoint(s):")
    for ckpt in checkpoints:
        size = os.path.getsize(os.path.join(checkpoint_dir, ckpt)) / (1024*1024)  # MB
        print(f"   - {ckpt} ({size:.2f} MB)")
else:
    print("❌ Checkpoint directory not found")
```

### Convert to TF-Lite

```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

**This is the fastest way to get the final model weights!**

---

### Option B: Full Training Pipeline (Complete Reproduction)

If you want to reproduce the complete solution from scratch:

#### Round 1 - Train 4 Folds

Train the smaller model (cfg_1) for 4 folds:

```python
# Fold 0
!python train_kaggle.py -C cfg_1 --fold 0

# Fold 1 (run in a new cell or after fold 0 completes)
!python train_kaggle.py -C cfg_1 --fold 1

# Fold 2
!python train_kaggle.py -C cfg_1 --fold 2

# Fold 3
!python train_kaggle.py -C cfg_1 --fold 3
```

**Note**: Each fold will take several hours. You can run them in separate notebook sessions.

#### Generate OOF Predictions

After all 4 folds are complete:

```python
!python scripts/get_train_folded_oof_supp.py
```

This creates `train_folded_oof_supp.csv` needed for round 2.

#### Round 2 - Train Fullfit

Train the larger model (cfg_2) with fullfit:

```python
# Seed 1
!python train_kaggle.py -C cfg_2 --fold -1

# Seed 2 (run again for second seed)
!python train_kaggle.py -C cfg_2 --fold -1
```

#### Convert to TF-Lite

```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

---

## Which Option Should You Choose?

- **Option A (Quick Path)**: Use if you just want the final model weights quickly. Saves ~12+ hours of training time.
- **Option B (Full Pipeline)**: Use if you want to reproduce the complete solution or understand the full training process.

---

## 📁 Directory Structure in Kaggle

```
/kaggle/
├── input/                    # Read-only datasets (added via UI)
│   ├── asl-fingerspelling-preprocessing-train-dataset/
│   └── asl-fingerspelling-preprocessed-supp-dataset/
│
├── working/                  # Your code and outputs (writable)
│   └── kaggle-asl-fingerspelling-1st-place-solution/
│       ├── datamount/        # Data copied here
│       ├── configs/
│       ├── train_kaggle.py
│       └── ...
│
└── temp/                     # Temporary files
```

---

## 🔧 Troubleshooting

### Issue: Environment check fails
**Solution**: Go back to the failing step and fix the issue before proceeding.

### Issue: "CUDA out of memory"
**Solution**: 
- Reduce batch_size in `configs/cfg_1.py` or `configs/cfg_2.py`
- Or enable mixed_precision: set `cfg.mixed_precision = True` in config

### Issue: "Module not found"
**Solution**: Install missing package: `!pip install package_name`

### Issue: "Data file not found"
**Solution**: 
- Verify datasets are added in Data sidebar
- Re-run Step 3 (data preparation)

### Issue: "Dataset not found in /kaggle/input"
**Solution**: Make sure you added the datasets in Data sidebar. Check the Data section.

### Issue: "Permission denied" when copying
**Solution**: `/kaggle/input/` is read-only. Copy files to `/kaggle/working/` instead (which Step 3 does).

---

## 💡 Key Points

- **Data Location**: `/kaggle/input/` (read-only) → Copy to `/kaggle/working/`
- **Output Location**: `/kaggle/working/` (auto-saved to Output tab)
- **GPU**: Enable in Settings → Accelerator → GPU
- **Time Limit**: 9 hours per session (free tier), 30h/week GPU time
- **Save Checkpoints**: Outputs in `/kaggle/working/` are auto-saved

---

## 📋 Summary Checklist

**Before Training:**
- [ ] Created Kaggle Notebook
- [ ] Enabled GPU accelerator (P100)
- [ ] Added training landmarks dataset
- [ ] Added supplemental landmarks dataset
- [ ] Cloned repository (Step 1)
- [ ] Installed packages (Step 2)
- [ ] Prepared data (Step 3)
- [ ] Verified environment (Step 4 - CRITICAL!)
- [ ] Verified data structure (Step 5)
- [ ] Tested GPU (Step 6)
- [ ] Final checklist passed (Step 7)

**Only proceed to Step 8 (Training) after ALL checks pass!**

---

## 🎓 Complete Training Workflow

Once all setup is complete:

1. **Round 1**: Train 4 folds of cfg_1
2. **Generate OOF**: Run `get_train_folded_oof_supp.py`
3. **Round 2**: Train 2 seeds of cfg_2 (fullfit)
4. **Convert**: Export to TF-Lite

---

Good luck with your training! 🚀

