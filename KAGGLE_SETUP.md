# Complete Step-by-Step Guide: Running in Kaggle Notebooks

This guide will walk you through setting up and running the ASL Fingerspelling Recognition training in Kaggle Notebooks.

---

## Step 1: Create a New Kaggle Notebook

### 1.1. Go to Kaggle
- Visit [kaggle.com](https://www.kaggle.com)
- Log in to your account (or create one if needed)

### 1.2. Create New Notebook
- Click **"Code"** in the top menu
- Click **"New Notebook"** button
- Choose **"Notebook"** (not Script)

### 1.3. Configure Notebook Settings
- **Language**: Python
- **Accelerator**: 
  - For free tier: **GPU P100** (30 hours/week)
  - For Pro: **GPU T4 x2** or **TPU v3-8** (better performance)
- Click **"Create"**

---

## Step 2: Add Required Datasets

Kaggle Notebooks use datasets that you "add" to your notebook. You need to add the preprocessed data.

### 2.1. Add Training Landmarks Dataset
1. In your notebook, look for the **"Data"** section on the right sidebar
2. Click **"+ Add data"** button
3. Search for: `asl-fingerspelling-preprocessing-train-dataset`
4. Click on the dataset by **darraghdog**
5. Click **"Add"** button

**Expected result**: Dataset appears in `/kaggle/input/asl-fingerspelling-preprocessing-train-dataset/`

### 2.2. Add Supplemental Landmarks Dataset
1. Click **"+ Add data"** again
2. Search for: `asl-fingerspelling-preprocessed-supp-dataset`
3. Click on the dataset by **darraghdog**
4. Click **"Add"** button

**Expected result**: Dataset appears in `/kaggle/input/asl-fingerspelling-preprocessed-supp-dataset/`

### 2.3. Verify Datasets
In a code cell, run:
```python
!ls -la /kaggle/input/
```

You should see both datasets listed.

---

## Step 3: Upload Repository Code

You have two options:

### Option A: Clone from GitHub (Recommended)

In a code cell, run:
```python
# Clone the repository
!git clone https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
```

### Option B: Upload Files Manually

1. Click **"File"** → **"Upload"** in the notebook
2. Upload all files from the repository
3. Or use Kaggle Datasets to store the code

**Note**: After cloning/uploading, make sure you're in the right directory:
```python
import os
os.chdir('/kaggle/working/kaggle-asl-fingerspelling-1st-place-solution')
```

---

## Step 4: Install Required Packages

Kaggle has many packages pre-installed, but you may need to install some missing ones.

### 4.1. Check Environment
```python
!python check_kaggle_environment.py
```

### 4.2. Install Missing Packages
Based on the check, install what's missing:

```python
# Install missing packages
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0

# If transformers version is wrong:
!pip install transformers==4.32.1

# If other packages are missing:
!pip install albumentations==1.3.1 timm==0.9.6
```

**Note**: Kaggle usually has PyTorch, TensorFlow, pandas, numpy pre-installed.

---

## Step 5: Prepare Data Structure

### 5.1. Create Data Directory Structure
```python
import os

# Create datamount directory in working folder
os.makedirs('/kaggle/working/datamount', exist_ok=True)

# Create symlinks or copy data from input to working
# Option 1: Create symlinks (saves space, faster)
import shutil

# Copy CSV files
input_base = '/kaggle/input'
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        # Copy CSV and JSON files
        for file in os.listdir(dataset_path):
            if file.endswith(('.csv', '.json')):
                src = os.path.join(dataset_path, file)
                dst = os.path.join('/kaggle/working/datamount', file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"✅ Copied {file}")

# Create symlink for landmarks (saves space)
train_landmarks_src = None
supp_landmarks_src = None

for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if 'train' in dataset_dir.lower() and 'preprocessing' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
        if os.path.exists(landmarks_path):
            train_landmarks_src = landmarks_path
    elif 'supp' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'supplemental_landmarks')
        if os.path.exists(landmarks_path):
            supp_landmarks_src = landmarks_path

# Create train_landmarks_npy directory
landmarks_dst = '/kaggle/working/datamount/train_landmarks_npy'
os.makedirs(landmarks_dst, exist_ok=True)

# Copy or symlink training landmarks
if train_landmarks_src:
    # Copy files (Kaggle input is read-only, so we need to copy)
    print("Copying training landmarks...")
    !cp -r {train_landmarks_src}/* {landmarks_dst}/
    print("✅ Training landmarks copied")

# Copy supplemental landmarks
if supp_landmarks_src:
    print("Copying supplemental landmarks...")
    !cp -r {supp_landmarks_src}/* {landmarks_dst}/
    print("✅ Supplemental landmarks copied and merged")
```

### 5.2. Verify Data Structure
```python
!ls -la /kaggle/working/datamount/
!ls /kaggle/working/datamount/train_landmarks_npy/ | head -10
```

You should see:
- `train_folded.csv`
- `character_to_prediction_index.json`
- `symmetry.csv`
- `train_landmarks_npy/` directory with many `.npy` files

---

## Step 6: Run Environment Check

```python
!python check_kaggle_environment.py
```

**Expected output:**
- ✅ Running in Kaggle Notebook
- ✅ CUDA is available (if GPU enabled)
- ✅ All packages installed
- ✅ Data files found

---

## Step 7: Start Training

### 7.1. Round 1 - Train 4 Folds

Train the smaller model (cfg_1) for 4 folds:

```python
# Fold 0
!python train_kaggle.py -C cfg_1 --fold 0

# Fold 1
!python train_kaggle.py -C cfg_1 --fold 1

# Fold 2
!python train_kaggle.py -C cfg_1 --fold 2

# Fold 3
!python train_kaggle.py -C cfg_1 --fold 3
```

**Note**: Each fold will take several hours. You can run them in separate notebook sessions.

### 7.2. Generate OOF Predictions

After all 4 folds are complete:

```python
!python scripts/get_train_folded_oof_supp.py
```

This creates `train_folded_oof_supp.csv` needed for round 2.

### 7.3. Round 2 - Train Fullfit

Train the larger model (cfg_2) with fullfit:

```python
# Seed 1
!python train_kaggle.py -C cfg_2 --fold -1

# Seed 2 (run again for second seed)
!python train_kaggle.py -C cfg_2 --fold -1
```

### 7.4. Convert to TF-Lite

```python
!python scripts/convert_cfg_2_to_tf_lite.py
```

---

## Step 8: Save Results

### 8.1. Download Output Files

In Kaggle, outputs are automatically saved. To download:

1. Go to **"Output"** tab in your notebook
2. Click on files you want to download
3. Or use the download button

### 8.2. Save to Kaggle Dataset (Optional)

To persist results between sessions:

```python
# Create output dataset
!mkdir -p /kaggle/working/output
!cp -r datamount/weights /kaggle/working/output/

# Then create a new dataset from /kaggle/working/output
```

---

## Complete Setup Code (Copy-Paste Ready)

Here's a complete setup cell you can run:

```python
# ============================================
# COMPLETE KAGGLE SETUP
# ============================================

import os
import shutil

# 1. Clone repository
print("Step 1: Cloning repository...")
!git clone https://github.com/ChristofHenkel/kaggle-asl-fingerspelling-1st-place-solution.git
%cd kaggle-asl-fingerspelling-1st-place-solution
print("✅ Repository cloned")

# 2. Install packages
print("\nStep 2: Installing packages...")
!pip install neptune-client==1.3.1 rapidfuzz==3.2.0
print("✅ Packages installed")

# 3. Prepare data
print("\nStep 3: Preparing data...")
os.makedirs('/kaggle/working/datamount', exist_ok=True)

# Copy CSV and JSON files from input datasets
input_base = '/kaggle/input'
for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if os.path.isdir(dataset_path):
        for file in os.listdir(dataset_path):
            if file.endswith(('.csv', '.json')):
                src = os.path.join(dataset_path, file)
                dst = os.path.join('/kaggle/working/datamount', file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  ✅ Copied {file}")

# Copy landmarks
landmarks_dst = '/kaggle/working/datamount/train_landmarks_npy'
os.makedirs(landmarks_dst, exist_ok=True)

for dataset_dir in os.listdir(input_base):
    dataset_path = os.path.join(input_base, dataset_dir)
    if 'train' in dataset_dir.lower() and 'preprocessing' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'train_landmarks_npy')
        if os.path.exists(landmarks_path):
            print(f"  Copying training landmarks from {dataset_dir}...")
            !cp -r {landmarks_path}/* {landmarks_dst}/
    elif 'supp' in dataset_dir.lower():
        landmarks_path = os.path.join(dataset_path, 'supplemental_landmarks')
        if os.path.exists(landmarks_path):
            print(f"  Copying supplemental landmarks from {dataset_dir}...")
            !cp -r {landmarks_path}/* {landmarks_dst}/

print("✅ Data prepared")

# 4. Check environment
print("\nStep 4: Checking environment...")
!python check_kaggle_environment.py

print("\n" + "="*60)
print("SETUP COMPLETE!")
print("="*60)
print("\nNext: Run training with:")
print("  !python train_kaggle.py -C cfg_1 --fold 0")
```

---

## Kaggle Notebook Features

| Feature | Description |
|---------|-------------|
| **Data Access** | Add datasets via UI (no API needed) |
| **Working Directory** | `/kaggle/working/` (writable) |
| **Input Data** | `/kaggle/input/` (read-only, datasets added via UI) |
| **GPU Access** | P100 (free tier), T4/TPU (Pro tier) |
| **Session Time** | 9 hours per session (free), 30h/week GPU time |
| **Output Persistence** | Auto-saved to Output tab |

---

## Troubleshooting

### Issue: "Dataset not found in /kaggle/input"
**Solution**: Make sure you added the datasets in Step 2. Check the Data sidebar.

### Issue: "Permission denied" when copying
**Solution**: `/kaggle/input/` is read-only. Copy files to `/kaggle/working/` instead.

### Issue: "Out of Memory"
**Solution**: 
- Reduce batch_size in config
- Use cfg_1 instead of cfg_2
- Enable mixed_precision

### Issue: "Module not found"
**Solution**: Install missing packages with `!pip install package_name`

### Issue: "CUDA not available"
**Solution**: Enable GPU in notebook settings (Settings → Accelerator → GPU)

---

## Tips for Kaggle

1. **Save Checkpoints Regularly**: Outputs are auto-saved
2. **Use Version Control**: Kaggle saves notebook versions automatically
3. **Monitor GPU Usage**: Check "Resources" tab
4. **Use Output Tab**: All files in `/kaggle/working/` are saved
5. **Time Limits**: Free tier has 9-hour sessions, 30 hours/week GPU time

---

## Summary Checklist

- [ ] Created Kaggle Notebook
- [ ] Enabled GPU accelerator
- [ ] Added training landmarks dataset
- [ ] Added supplemental landmarks dataset
- [ ] Cloned/uploaded repository code
- [ ] Installed missing packages
- [ ] Prepared data structure
- [ ] Verified environment check
- [ ] Started training

You're ready to train! 🚀

