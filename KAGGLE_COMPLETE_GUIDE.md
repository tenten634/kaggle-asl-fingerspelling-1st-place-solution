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

### Round 1 - Train 4 Folds

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

### Generate OOF Predictions

After all 4 folds are complete:

```python
!python scripts/get_train_folded_oof_supp.py
```

This creates `train_folded_oof_supp.csv` needed for round 2.

### Round 2 - Train Fullfit

Train the larger model (cfg_2) with fullfit:

```python
# Seed 1
!python train_kaggle.py -C cfg_2 --fold -1

# Seed 2 (run again for second seed)
!python train_kaggle.py -C cfg_2 --fold -1
```

### Convert to TF-Lite

```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

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

